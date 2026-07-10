// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Demo LLM proxy.
//!
//! HTTP serving and API translation come entirely from switchyard's crates
//! (`switchyard-server` axum router + `switchyard-translation`, reached through
//! the `Profile` runtime). ALL routing is implemented with `libsy`: inbound
//! Codex/Relay headers are normalized into agent metadata, then an LLM classifier
//! assigns each stable agent/subtask to one model from a configured pool. libsy's
//! targets make their model calls through switchyard's OpenAI-compatible backend.
//!
//! The proxy serves all three inbound APIs — OpenAI (`/v1/chat/completions`),
//! Anthropic (`/v1/messages`), and Responses (`/v1/responses`) — and switchyard
//! translates the upstream response back to whichever the caller used.
//!
//! Env:
//!   ANTHROPIC_API_KEY     upstream bearer key (e.g. `$INFERENCE_HUB_SY_API_KEY`)
//!   LIBSY_PROXY_BASE_URL  upstream base url (default https://inference-api.nvidia.com/v1)
//!   LIBSY_PROXY_ADDR      listen address (default 127.0.0.1:4000)
//!
//! Run:
//!   ANTHROPIC_API_KEY=$INFERENCE_HUB_SY_API_KEY cargo run -p libsy-proxy

use std::net::SocketAddr;
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::{json, Value};

use libsy::agentic::{
    metadata_from_headers, AgentAwareOrchAlgoBuilder, AgentRoutingCandidate, AgentRoutingDecision,
};
use libsy::{
    DecisionTrace, LlmClient, LlmRequest, LlmResponse, LlmTarget, LlmTargetSet,
    MultiLlmOrchestrator, OrchestratorRequest, OrchestratorResponse,
};

use switchyard_components::OpenAiPassthroughBackend;
use switchyard_components_v2::{Profile, ProfileInput, ProfileResponse, RoutingMetadata};
use switchyard_core::{
    ChatRequest, ChatResponse, EndpointConfig, LlmBackend, ModelId, ProxyContext, Result,
    SwitchyardError,
};
use switchyard_server::{serve_addr, ProfileRegistry, ServerState};

// Routing configuration (per the demo's inference-hub models).
const CLASSIFIER_MODEL: &str = "nvidia/deepseek-ai/deepseek-v4-flash";
const STRONG_MODEL: &str = "aws/anthropic/bedrock-claude-opus-4-7";
const WEAK_MODEL: &str = "nvidia/deepseek-ai/deepseek-v4-flash";
const CLASSIFIER_TARGET: &str = "classifier";
const FRONTIER_TARGET: &str = "frontier";
const FAST_TARGET: &str = "fast";

const DEFAULT_BASE_URL: &str = "https://inference-api.nvidia.com/v1";
const DEFAULT_ADDR: &str = "127.0.0.1:4000";
/// Model id callers address to reach this proxy (routing picks the real model).
const PROFILE_MODEL_ID: &str = "libsy-agent-aware";

/// A libsy [`LlmClient`] whose model call is performed by switchyard's
/// OpenAI-compatible backend — so libsy owns routing while switchyard owns the
/// HTTP transport. One instance is shared by every routing target.
struct SwitchyardBackendClient {
    backend: Arc<OpenAiPassthroughBackend>,
}

#[async_trait]
impl LlmClient for SwitchyardBackendClient {
    async fn call(
        &self,
        request: OrchestratorRequest,
    ) -> std::result::Result<OrchestratorResponse, Box<dyn std::error::Error + Send + Sync>> {
        let model = request.llm_request.model_name.clone();
        let chat_request = chat_request_for_call(&request, &model);

        let mut ctx = ProxyContext::new();
        let response = self
            .backend
            .call(&mut ctx, &chat_request)
            .await
            .map_err(|e| Box::<dyn std::error::Error + Send + Sync>::from(e.to_string()))?;

        let raw = response.body().cloned().ok_or_else(|| {
            Box::<dyn std::error::Error + Send + Sync>::from(
                "the libsy proxy POC currently requires buffered upstream responses",
            )
        })?;
        let completion = completion_text(&raw).unwrap_or_default();
        Ok(OrchestratorResponse {
            llm_response: LlmResponse {
                completion,
                raw_response: Some(raw),
            },
            metadata: None,
        })
    }
}

/// A switchyard [`Profile`] that routes every request through libsy's LLM
/// classifier. switchyard's router hands us the inbound request and translates
/// our response back to the caller's format.
struct LibsyAgentAwareProfile {
    orchestrator: MultiLlmOrchestrator,
}

#[async_trait]
impl Profile for LibsyAgentAwareProfile {
    async fn run(&self, input: ProfileInput) -> Result<ProfileResponse> {
        // Pull the user's prompt out of whatever inbound wire format we got.
        let prompt = extract_prompt(input.request.body())
            .ok_or_else(|| SwitchyardError::InvalidRequest("no user prompt in request".into()))?;

        let mut metadata = metadata_from_headers(&input.metadata.headers);
        metadata
            .extra_metadata
            .get_or_insert_with(Default::default)
            .insert(
                "inbound_format".to_string(),
                inbound_format(input.request.request_type()).to_string(),
            );
        let orch_request = OrchestratorRequest {
            llm_request: LlmRequest {
                model_name: "auto".to_string(),
                prompt,
            },
            raw_request: Some(input.request.body().clone()),
            metadata: Some(metadata),
        };

        // ALL routing happens here, in libsy: the classifier selects one model
        // from the pool and stable agent/task identities reuse that assignment.
        // libsy's targets perform those calls via the switchyard backend.
        let (trace, response) = self
            .orchestrator
            .orchestrate_direct(orch_request)
            .await
            .map_err(|e| SwitchyardError::Other(e.to_string()))?;

        // Return the routed model's response body (OpenAI chat-completion shape);
        // switchyard reconciles it against the caller's inbound format.
        let body = response.llm_response.raw_response.unwrap_or_else(|| {
            json!({
                "object": "chat.completion",
                "choices": [{
                    "index": 0,
                    "message": { "role": "assistant", "content": response.llm_response.completion },
                    "finish_reason": "stop",
                }],
            })
        });
        let chat_response = ChatResponse::openai_completion(body);
        Ok(ProfileResponse::with_routing_metadata(
            chat_response,
            routing_metadata(&trace),
        ))
    }
}

/// Extract the user prompt from an inbound body, handling OpenAI / Anthropic
/// (`messages[]`) and Responses (`input`) shapes.
fn extract_prompt(body: &Value) -> Option<String> {
    if let Some(messages) = body.get("messages").and_then(Value::as_array) {
        if let Some(user) = messages
            .iter()
            .rev()
            .find(|m| m.get("role").and_then(Value::as_str) == Some("user"))
        {
            if let Some(text) = user.get("content").and_then(content_to_text) {
                return Some(text);
            }
        }
    }
    // Responses API: prefer the latest user message when `input` carries the
    // full item history, then fall back to any directly extractable text.
    let input = body.get("input")?;
    if let Some(items) = input.as_array() {
        if let Some(user) = items
            .iter()
            .rev()
            .find(|item| item.get("role").and_then(Value::as_str) == Some("user"))
        {
            return user
                .get("content")
                .and_then(content_to_text)
                .or_else(|| content_to_text(user));
        }
    }
    content_to_text(input)
}

/// Flatten nested message/content items to the text relevant to classification.
fn content_to_text(content: &Value) -> Option<String> {
    match content {
        Value::String(s) => Some(s.clone()),
        Value::Array(parts) => {
            let text = parts
                .iter()
                .filter_map(content_to_text)
                .collect::<Vec<_>>()
                .join("\n");
            (!text.is_empty()).then_some(text)
        }
        Value::Object(object) => object
            .get("text")
            .and_then(Value::as_str)
            .map(str::to_string)
            .or_else(|| object.get("content").and_then(content_to_text)),
        _ => None,
    }
}

/// Read the assistant text out of an OpenAI chat-completion body.
fn completion_text(body: &Value) -> Option<String> {
    body.get("choices")
        .and_then(Value::as_array)
        .and_then(|choices| choices.first())
        .and_then(|choice| choice.get("message"))
        .and_then(|message| message.get("content"))
        .and_then(Value::as_str)
        .map(str::to_string)
}

/// Preserve the provider-shaped request for the routed call while classifier
/// calls use a small synthetic Chat Completions request.
fn chat_request_for_call(request: &OrchestratorRequest, model: &str) -> ChatRequest {
    let Some(mut body) = request.raw_request.clone() else {
        return ChatRequest::openai_chat(json!({
            "model": model,
            "messages": [{ "role": "user", "content": request.llm_request.prompt }],
            "stream": false,
        }));
    };
    if let Some(object) = body.as_object_mut() {
        object.insert("model".to_string(), Value::String(model.to_string()));
    }
    match request
        .metadata
        .as_ref()
        .and_then(|metadata| metadata.extra_metadata.as_ref())
        .and_then(|extra| extra.get("inbound_format"))
        .map(String::as_str)
    {
        Some("anthropic") => ChatRequest::anthropic(body),
        Some("openai_responses") => ChatRequest::openai_responses(body),
        _ => ChatRequest::openai_chat(body),
    }
}

fn inbound_format(request_type: switchyard_core::ChatRequestType) -> &'static str {
    match request_type {
        switchyard_core::ChatRequestType::OpenAiChat => "openai_chat",
        switchyard_core::ChatRequestType::Anthropic => "anthropic",
        switchyard_core::ChatRequestType::OpenAiResponses => "openai_responses",
    }
}

/// Surface libsy's routing decision as `x-model-router-*` response headers.
fn routing_metadata(trace: &[Arc<dyn DecisionTrace>]) -> RoutingMetadata {
    // A classified trace is [classify, route]; a cached trace is [route].
    let decision = trace.last();
    let agent = decision.and_then(|d| d.as_any().downcast_ref::<AgentRoutingDecision>());
    let selected_target = decision.map(|decision| decision.model_decision());
    RoutingMetadata {
        selected_model: selected_target.map(|target| match target {
            FRONTIER_TARGET => STRONG_MODEL.to_string(),
            FAST_TARGET => WEAK_MODEL.to_string(),
            other => other.to_string(),
        }),
        selected_tier: selected_target.map(str::to_string),
        confidence: agent.and_then(|decision| decision.confidence),
        router_version: Some("libsy-agent-aware-v1".to_string()),
        tolerance: None,
        rationale: decision.and_then(|d| d.reasoning().map(str::to_string)),
    }
}

fn build_orchestrator() -> Result<MultiLlmOrchestrator> {
    let base_url =
        std::env::var("LIBSY_PROXY_BASE_URL").unwrap_or_else(|_| DEFAULT_BASE_URL.to_string());
    let api_key = std::env::var("ANTHROPIC_API_KEY").map_err(|_| {
        SwitchyardError::InvalidConfig(
            "ANTHROPIC_API_KEY must be set to upstream bearer key".into(),
        )
    })?;
    let endpoint = EndpointConfig {
        base_url: Some(base_url),
        api_key: Some(api_key),
        timeout_secs: Some(120.0),
    };
    let backend = Arc::new(OpenAiPassthroughBackend::new(endpoint)?);
    let client = Arc::new(SwitchyardBackendClient { backend }) as Arc<dyn LlmClient>;

    // Logical target names keep routing policy independent of provider model ids;
    // all targets share one upstream client.
    let target = |name: &str, model: &str| LlmTarget {
        name: name.to_string(),
        model: model.to_string(),
        llm_client: Some(client.clone()),
    };
    let targets = LlmTargetSet::new(vec![
        target(CLASSIFIER_TARGET, CLASSIFIER_MODEL),
        target(FRONTIER_TARGET, STRONG_MODEL),
        target(FAST_TARGET, WEAK_MODEL),
    ]);

    let builder = Box::new(AgentAwareOrchAlgoBuilder::new(
        CLASSIFIER_TARGET,
        vec![
            AgentRoutingCandidate::new(
                FRONTIER_TARGET,
                "frontier model for planning, ambiguous implementation, synthesis, and review",
            ),
            AgentRoutingCandidate::new(
                FAST_TARGET,
                "efficient model for bounded exploration, retrieval, and mechanical edits",
            ),
        ],
        FRONTIER_TARGET,
    ));
    Ok(MultiLlmOrchestrator::new(builder, Some(targets)))
}

#[tokio::main]
async fn main() -> Result<()> {
    let orchestrator = build_orchestrator()?;
    let profile = Arc::new(LibsyAgentAwareProfile { orchestrator }) as Arc<dyn Profile>;

    let registry = ProfileRegistry::from_profiles([(
        ModelId::new(PROFILE_MODEL_ID)?,
        profile,
        PROFILE_MODEL_ID.to_string(),
    )])?;
    let state = ServerState::new(registry);

    let addr: SocketAddr = std::env::var("LIBSY_PROXY_ADDR")
        .unwrap_or_else(|_| DEFAULT_ADDR.to_string())
        .parse()
        .map_err(|e: std::net::AddrParseError| SwitchyardError::InvalidConfig(e.to_string()))?;

    println!("libsy-proxy listening on http://{addr}");
    println!("  routing (libsy agent-aware): classifier={CLASSIFIER_MODEL}");
    println!("                               frontier={STRONG_MODEL}");
    println!("                               fast={WEAK_MODEL}");
    println!(
        "  send model \"{PROFILE_MODEL_ID}\" to /v1/chat/completions, /v1/messages, or /v1/responses"
    );

    serve_addr(addr, state).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use libsy::Metadata;
    use std::collections::BTreeMap;

    #[test]
    fn routed_call_preserves_provider_body_and_rewrites_model() {
        let request = OrchestratorRequest {
            llm_request: LlmRequest {
                model_name: "auto".to_string(),
                prompt: "inspect".to_string(),
            },
            raw_request: Some(json!({
                "model": "libsy-agent-aware",
                "input": "inspect",
                "tools": [{"type": "function", "name": "shell"}],
                "stream": false,
            })),
            metadata: Some(Metadata {
                extra_metadata: Some(BTreeMap::from([(
                    "inbound_format".to_string(),
                    "openai_responses".to_string(),
                )])),
                ..Metadata::default()
            }),
        };

        let routed = chat_request_for_call(&request, "provider/model");
        assert_eq!(
            routed.request_type(),
            switchyard_core::ChatRequestType::OpenAiResponses
        );
        assert_eq!(
            routed.body().get("model").and_then(Value::as_str),
            Some("provider/model")
        );
        assert!(routed.body().get("tools").is_some());
    }

    #[test]
    fn extracts_latest_user_text_from_responses_items() {
        let body = json!({
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "first task"}]
                },
                {"type": "function_call_output", "output": "tool result"},
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "current subtask"}]
                }
            ]
        });

        assert_eq!(extract_prompt(&body).as_deref(), Some("current subtask"));
    }

    #[test]
    fn classifier_call_uses_a_buffered_synthetic_request() {
        let request = OrchestratorRequest {
            llm_request: LlmRequest {
                model_name: "classifier".to_string(),
                prompt: "classify this".to_string(),
            },
            raw_request: None,
            metadata: None,
        };

        let classifier = chat_request_for_call(&request, "classifier/model");
        assert_eq!(
            classifier.request_type(),
            switchyard_core::ChatRequestType::OpenAiChat
        );
        assert_eq!(
            classifier.body().get("stream").and_then(Value::as_bool),
            Some(false)
        );
    }

    #[test]
    fn response_metadata_exposes_provider_model_and_logical_tier() {
        let decision: Arc<dyn DecisionTrace> = Arc::new(AgentRoutingDecision {
            selected_model: FAST_TARGET.to_string(),
            reason: "bounded lookup".to_string(),
            task_kind: Some("research".to_string()),
            confidence: Some(0.9),
            agent_id: Some("child-1".to_string()),
            cache_hit: false,
        });

        let metadata = routing_metadata(&[decision]);
        assert_eq!(metadata.selected_model.as_deref(), Some(WEAK_MODEL));
        assert_eq!(metadata.selected_tier.as_deref(), Some(FAST_TARGET));
        assert_eq!(metadata.confidence, Some(0.9));
    }
}
