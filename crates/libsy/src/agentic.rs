// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Agent- and subtask-aware model routing.
//!
//! The module has two deliberately separate responsibilities:
//!
//! - [`metadata_from_headers`] normalizes identity signals emitted by Codex,
//!   NeMo Relay, Dynamo, or an explicit Switchyard adapter into libsy's neutral
//!   [`Metadata`]. It has no dependency on any harness runtime.
//! - [`AgentAwareOrchAlgo`] asks an LLM classifier to select one target from a
//!   configured model pool. A valid selection is retained for the stable
//!   `(session, agent, task)` identity so one spawned agent remains on one model.

use std::collections::{BTreeMap, HashMap};
use std::error::Error;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use serde::Deserialize;

use crate::{
    AgentContext, AgentSysSignals, DecisionTrace, LlmRequest, LlmTargetSet, Metadata, OrchAlgo,
    OrchAlgoBuilder, OrchestratorContext, OrchestratorRequest, OrchestratorResponse,
};

const CODEX_TURN_METADATA_HEADER: &str = "x-codex-turn-metadata";
const MAX_ASSIGNMENTS: usize = 4096;

/// One model available to the agent-aware classifier.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AgentRoutingCandidate {
    /// Target name used to look up the model in [`LlmTargetSet`].
    pub target: String,
    /// Capability/cost guidance shown to the classifier.
    pub description: String,
}

impl AgentRoutingCandidate {
    /// Creates a model-pool entry from a target name and classifier guidance.
    pub fn new(target: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            target: target.into(),
            description: description.into(),
        }
    }
}

/// Inspectable decision emitted for the final routed call.
pub struct AgentRoutingDecision {
    /// Target selected from the configured model pool.
    pub selected_model: String,
    /// Human-readable classifier or fallback explanation.
    pub reason: String,
    /// Classifier-inferred task class, when supplied.
    pub task_kind: Option<String>,
    /// Classifier confidence in `[0, 1]`, when supplied.
    pub confidence: Option<f64>,
    /// Stable agent id used for the assignment, when available.
    pub agent_id: Option<String>,
    /// Whether the selection came from an existing per-agent assignment.
    pub cache_hit: bool,
}

impl DecisionTrace for AgentRoutingDecision {
    fn model_decision(&self) -> &str {
        &self.selected_model
    }

    fn reasoning(&self) -> Option<&str> {
        Some(&self.reason)
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// LLM classifier over a model pool with per-agent/task assignment affinity.
pub struct AgentAwareOrchAlgo {
    classifier_model: String,
    candidates: Vec<AgentRoutingCandidate>,
    fallback_model: String,
    target_set: LlmTargetSet,
    assignments: Mutex<HashMap<AssignmentKey, Assignment>>,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
struct AssignmentKey {
    session_id: Option<String>,
    agent_id: String,
    task_id: Option<String>,
}

#[derive(Clone, Debug)]
struct Assignment {
    model: String,
    reason: String,
    task_kind: Option<String>,
    confidence: Option<f64>,
}

#[derive(Deserialize)]
struct ClassifierOutput {
    model: String,
    #[serde(default)]
    task_kind: Option<String>,
    #[serde(default)]
    confidence: Option<f64>,
    #[serde(default)]
    reason: Option<String>,
}

#[derive(Default, Deserialize)]
struct CodexTurnMetadata {
    session_id: Option<String>,
    thread_id: Option<String>,
    parent_thread_id: Option<String>,
    turn_id: Option<String>,
    subagent_kind: Option<String>,
    agent_role: Option<String>,
    task_id: Option<String>,
    task_kind: Option<String>,
}

#[async_trait]
impl OrchAlgo for AgentAwareOrchAlgo {
    async fn process_request(
        &self,
        ctx: &OrchestratorContext,
        request: OrchestratorRequest,
    ) -> Result<(Vec<Arc<dyn DecisionTrace>>, OrchestratorResponse), Box<dyn Error + Send + Sync>>
    {
        let key = assignment_key(request.metadata.as_ref());
        let cached = key.as_ref().and_then(|key| self.cached_assignment(key));
        let (assignment, cache_hit, classifier_decision) = match cached {
            Some(assignment) => (assignment, true, None),
            None => {
                let (assignment, cacheable, classifier_decision) =
                    self.classify(ctx, &request).await;
                if cacheable {
                    if let Some(key) = key {
                        self.store_assignment(key, assignment.clone());
                    }
                }
                (assignment, false, Some(classifier_decision))
            }
        };

        let routed_target = self.target_set.get_target(&assignment.model)?;
        let route_decision: Arc<dyn DecisionTrace> = Arc::new(AgentRoutingDecision {
            selected_model: assignment.model.clone(),
            reason: if cache_hit {
                format!("reused stable agent/task assignment: {}", assignment.reason)
            } else {
                assignment.reason.clone()
            },
            task_kind: assignment.task_kind.clone(),
            confidence: assignment.confidence,
            agent_id: request
                .metadata
                .as_ref()
                .and_then(|metadata| metadata.agent_id.clone()),
            cache_hit,
        });
        let routed_request = OrchestratorRequest {
            llm_request: LlmRequest {
                model_name: assignment.model,
                prompt: request.llm_request.prompt,
            },
            raw_request: request.raw_request,
            metadata: request.metadata,
        };
        let response = routed_target
            .call(ctx, routed_request, Some(route_decision.clone()))
            .await?;

        let mut trace = Vec::with_capacity(2);
        if let Some(classifier_decision) = classifier_decision {
            trace.push(classifier_decision);
        }
        trace.push(route_decision);
        Ok((trace, response))
    }

    async fn process_signals(
        &self,
        _signals: AgentSysSignals,
    ) -> Result<(), Box<dyn Error + Send + Sync>> {
        Ok(())
    }
}

impl AgentAwareOrchAlgo {
    async fn classify(
        &self,
        ctx: &OrchestratorContext,
        request: &OrchestratorRequest,
    ) -> (Assignment, bool, Arc<dyn DecisionTrace>) {
        let classifier_decision: Arc<dyn DecisionTrace> = Arc::new(AgentRoutingDecision {
            selected_model: self.classifier_model.clone(),
            reason: format!("classifying agent/subtask via {}", self.classifier_model),
            task_kind: request
                .metadata
                .as_ref()
                .and_then(|metadata| metadata.agent_context.as_deref())
                .and_then(|agent| agent.task_kind.clone()),
            confidence: None,
            agent_id: request
                .metadata
                .as_ref()
                .and_then(|metadata| metadata.agent_id.clone()),
            cache_hit: false,
        });
        let classify_request = OrchestratorRequest {
            llm_request: LlmRequest {
                model_name: self.classifier_model.clone(),
                prompt: classifier_prompt(
                    &self.candidates,
                    request.metadata.as_ref(),
                    &request.llm_request.prompt,
                ),
            },
            // The classifier receives the normalized text and metadata, never the
            // provider payload with tools or continuation state.
            raw_request: None,
            metadata: request.metadata.clone(),
        };

        let output = match self.target_set.get_target(&self.classifier_model) {
            Ok(target) => {
                target
                    .call(ctx, classify_request, Some(classifier_decision.clone()))
                    .await
            }
            Err(error) => Err(error),
        };
        match output {
            Ok(response) => {
                match parse_classifier_output(&response.llm_response.completion, &self.candidates) {
                    Some(assignment) => (assignment, true, classifier_decision),
                    None => (
                        self.fallback_assignment(
                            "classifier returned an invalid model-pool decision",
                        ),
                        false,
                        classifier_decision,
                    ),
                }
            }
            Err(error) => (
                self.fallback_assignment(&format!("classifier call failed: {error}")),
                false,
                classifier_decision,
            ),
        }
    }

    fn fallback_assignment(&self, reason: &str) -> Assignment {
        Assignment {
            model: self.fallback_model.clone(),
            reason: format!("{reason}; fell back to {}", self.fallback_model),
            task_kind: None,
            confidence: None,
        }
    }

    fn cached_assignment(&self, key: &AssignmentKey) -> Option<Assignment> {
        self.assignments
            .lock()
            .ok()
            .and_then(|assignments| assignments.get(key).cloned())
    }

    fn store_assignment(&self, key: AssignmentKey, assignment: Assignment) {
        if let Ok(mut assignments) = self.assignments.lock() {
            if !assignments.contains_key(&key) && assignments.len() >= MAX_ASSIGNMENTS {
                if let Some(evicted) = assignments.keys().next().cloned() {
                    assignments.remove(&evicted);
                }
            }
            assignments.insert(key, assignment);
        }
    }
}

/// Builder for [`AgentAwareOrchAlgo`].
pub struct AgentAwareOrchAlgoBuilder {
    classifier_model: String,
    candidates: Vec<AgentRoutingCandidate>,
    fallback_model: String,
    target_set: Option<LlmTargetSet>,
}

impl AgentAwareOrchAlgoBuilder {
    /// Configures the classifier target, candidate model pool, and fail-open target.
    pub fn new(
        classifier_model: impl Into<String>,
        candidates: Vec<AgentRoutingCandidate>,
        fallback_model: impl Into<String>,
    ) -> Self {
        Self {
            classifier_model: classifier_model.into(),
            candidates,
            fallback_model: fallback_model.into(),
            target_set: None,
        }
    }
}

impl OrchAlgoBuilder for AgentAwareOrchAlgoBuilder {
    fn with_target_set(&mut self, target_set: LlmTargetSet) {
        self.target_set = Some(target_set);
    }

    fn build(&mut self) -> Box<dyn OrchAlgo> {
        Box::new(AgentAwareOrchAlgo {
            classifier_model: self.classifier_model.clone(),
            candidates: self.candidates.clone(),
            fallback_model: self.fallback_model.clone(),
            target_set: self
                .target_set
                .take()
                .unwrap_or_else(|| LlmTargetSet::new(vec![])),
            assignments: Mutex::new(HashMap::new()),
        })
    }
}

/// Normalizes harness-specific request headers into libsy metadata.
///
/// Explicit `x-switchyard-*` headers win. NeMo Relay/Dynamo correlation headers
/// are accepted without linking their runtime. Codex's structured turn metadata
/// is preferred over its compatibility projections (`session-id`, `thread-id`,
/// `x-codex-parent-thread-id`, and `x-openai-subagent`).
pub fn metadata_from_headers(headers: &BTreeMap<String, Vec<String>>) -> Metadata {
    let codex = header(headers, CODEX_TURN_METADATA_HEADER)
        .and_then(|value| serde_json::from_str::<CodexTurnMetadata>(&value).ok())
        .unwrap_or_default();

    let agent_context = AgentContext {
        parent_agent_id: first_some([
            header(headers, "x-switchyard-parent-agent-id"),
            header(headers, "x-dynamo-parent-session-id"),
            codex.parent_thread_id,
            header(headers, "x-codex-parent-thread-id"),
        ]),
        agent_kind: first_some([
            header(headers, "x-switchyard-agent-kind"),
            codex.subagent_kind,
            header(headers, "x-openai-subagent"),
        ]),
        agent_role: first_some([header(headers, "x-switchyard-agent-role"), codex.agent_role]),
        task_kind: first_some([header(headers, "x-switchyard-task-kind"), codex.task_kind]),
        turn_id: first_some([header(headers, "x-switchyard-turn-id"), codex.turn_id]),
    };
    let agent_context = (agent_context != AgentContext::default()).then(|| Box::new(agent_context));

    Metadata {
        session_id: first_some([
            header(headers, "x-switchyard-session-id"),
            header(headers, "x-nemo-relay-session-id"),
            codex.session_id,
            header(headers, "session-id"),
        ]),
        agent_id: first_some([
            header(headers, "x-switchyard-agent-id"),
            header(headers, "x-nemo-relay-subagent-id"),
            header(headers, "x-dynamo-session-id"),
            codex.thread_id,
            header(headers, "thread-id"),
        ]),
        task_id: first_some([
            header(headers, "x-switchyard-task-id"),
            codex.task_id,
            header(headers, "x-task-id"),
        ]),
        agent_context,
        correlation_id: first_some([
            header(headers, "x-request-id"),
            header(headers, "x-client-request-id"),
        ]),
        extra_metadata: None,
    }
}

fn assignment_key(metadata: Option<&Metadata>) -> Option<AssignmentKey> {
    let metadata = metadata?;
    Some(AssignmentKey {
        session_id: metadata.session_id.clone(),
        agent_id: metadata.agent_id.clone()?,
        task_id: metadata.task_id.clone(),
    })
}

fn classifier_prompt(
    candidates: &[AgentRoutingCandidate],
    metadata: Option<&Metadata>,
    prompt: &str,
) -> String {
    let metadata = metadata.cloned().unwrap_or_default();
    let agent = metadata
        .agent_context
        .as_deref()
        .cloned()
        .unwrap_or_default();
    let candidate_lines = candidates
        .iter()
        .map(|candidate| format!("- {}: {}", candidate.target, candidate.description))
        .collect::<Vec<_>>()
        .join("\n");
    format!(
        "You route one model call inside an agent hierarchy. Select exactly one target from the \n\
         model pool. Prefer the cheapest candidate likely to complete the subtask correctly. \n\
         Return JSON only: {{\"model\":\"<target>\",\"task_kind\":\"<short label>\",\
         \"confidence\":<0..1>,\"reason\":\"<brief reason>\"}}.\n\n\
         Agent context:\n\
         session_id={:?}\nagent_id={:?}\nparent_agent_id={:?}\nagent_kind={:?}\n\
         agent_role={:?}\ntask_id={:?}\ntask_kind={:?}\nturn_id={:?}\n\n\
         Model pool:\n{}\n\nSubtask:\n<subtask>\n{}\n</subtask>",
        metadata.session_id,
        metadata.agent_id,
        agent.parent_agent_id,
        agent.agent_kind,
        agent.agent_role,
        metadata.task_id,
        agent.task_kind,
        agent.turn_id,
        candidate_lines,
        prompt,
    )
}

fn parse_classifier_output(
    output: &str,
    candidates: &[AgentRoutingCandidate],
) -> Option<Assignment> {
    let trimmed = output.trim();
    if let Some(candidate) = candidates
        .iter()
        .find(|candidate| candidate.target == trimmed)
    {
        return Some(Assignment {
            model: candidate.target.clone(),
            reason: "classifier selected the target by name".to_string(),
            task_kind: None,
            confidence: None,
        });
    }

    let start = trimmed.find('{')?;
    let end = trimmed.rfind('}')?;
    let parsed = serde_json::from_str::<ClassifierOutput>(&trimmed[start..=end]).ok()?;
    let model = parsed.model.trim();
    if !candidates.iter().any(|candidate| candidate.target == model) {
        return None;
    }
    Some(Assignment {
        model: model.to_string(),
        reason: parsed
            .reason
            .filter(|reason| !reason.trim().is_empty())
            .unwrap_or_else(|| "classifier selected the target".to_string()),
        task_kind: parsed.task_kind.filter(|kind| !kind.trim().is_empty()),
        confidence: parsed
            .confidence
            .filter(|value| value.is_finite())
            .map(|value| value.clamp(0.0, 1.0)),
    })
}

fn header(headers: &BTreeMap<String, Vec<String>>, name: &str) -> Option<String> {
    headers
        .iter()
        .find(|(key, _)| key.eq_ignore_ascii_case(name))
        .and_then(|(_, values)| values.iter().find(|value| !value.trim().is_empty()))
        .map(|value| value.trim().to_string())
}

fn first_some<const N: usize>(values: [Option<String>; N]) -> Option<String> {
    values.into_iter().flatten().next()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{LlmClient, LlmResponse, LlmTarget};

    struct RoutingClient {
        classifier_calls: Arc<Mutex<usize>>,
        responses: Arc<Mutex<Vec<String>>>,
    }

    #[async_trait]
    impl LlmClient for RoutingClient {
        async fn call(
            &self,
            request: OrchestratorRequest,
        ) -> Result<OrchestratorResponse, Box<dyn Error + Send + Sync>> {
            let completion = if request.llm_request.model_name == "classifier" {
                let mut calls = self.classifier_calls.lock().map_err(|_| "lock poisoned")?;
                *calls += 1;
                self.responses
                    .lock()
                    .map_err(|_| "lock poisoned")?
                    .remove(0)
            } else {
                request.llm_request.model_name
            };
            Ok(OrchestratorResponse {
                llm_response: LlmResponse {
                    completion,
                    raw_response: None,
                },
                metadata: request.metadata,
            })
        }
    }

    fn algo(responses: Vec<&str>) -> (AgentAwareOrchAlgo, Arc<Mutex<usize>>) {
        let classifier_calls = Arc::new(Mutex::new(0));
        let client = Arc::new(RoutingClient {
            classifier_calls: Arc::clone(&classifier_calls),
            responses: Arc::new(Mutex::new(
                responses.into_iter().map(str::to_string).collect(),
            )),
        }) as Arc<dyn LlmClient>;
        let target = |name: &str| LlmTarget {
            name: name.to_string(),
            model: name.to_string(),
            llm_client: Some(Arc::clone(&client)),
        };
        let target_set = LlmTargetSet::new(vec![
            target("classifier"),
            target("frontier"),
            target("fast"),
        ]);
        (
            AgentAwareOrchAlgo {
                classifier_model: "classifier".to_string(),
                candidates: vec![
                    AgentRoutingCandidate::new("frontier", "complex planning and final review"),
                    AgentRoutingCandidate::new("fast", "bounded research and mechanical edits"),
                ],
                fallback_model: "frontier".to_string(),
                target_set,
                assignments: Mutex::new(HashMap::new()),
            },
            classifier_calls,
        )
    }

    fn request(agent_id: Option<&str>, task_id: Option<&str>, prompt: &str) -> OrchestratorRequest {
        OrchestratorRequest {
            llm_request: LlmRequest {
                model_name: "auto".to_string(),
                prompt: prompt.to_string(),
            },
            raw_request: Some(serde_json::json!({"input": prompt})),
            metadata: Some(Metadata {
                session_id: Some("session-1".to_string()),
                agent_id: agent_id.map(str::to_string),
                task_id: task_id.map(str::to_string),
                agent_context: Some(Box::new(AgentContext {
                    agent_role: Some("explorer".to_string()),
                    ..AgentContext::default()
                })),
                ..Metadata::default()
            }),
        }
    }

    fn decision(model: &str, task_kind: &str, confidence: f64, reason: &str) -> String {
        serde_json::json!({
            "model": model,
            "task_kind": task_kind,
            "confidence": confidence,
            "reason": reason,
        })
        .to_string()
    }

    #[test]
    fn normalizes_codex_turn_metadata_and_compatibility_headers() {
        let mut headers = BTreeMap::new();
        headers.insert("session-id".to_string(), vec!["compat-session".to_string()]);
        headers.insert("thread-id".to_string(), vec!["compat-agent".to_string()]);
        headers.insert(
            CODEX_TURN_METADATA_HEADER.to_string(),
            vec![serde_json::json!({
                "session_id": "root-session",
                "thread_id": "child-agent",
                "parent_thread_id": "root-agent",
                "turn_id": "turn-7",
                "subagent_kind": "collab_spawn",
            })
            .to_string()],
        );

        let metadata = metadata_from_headers(&headers);
        assert_eq!(metadata.session_id.as_deref(), Some("root-session"));
        assert_eq!(metadata.agent_id.as_deref(), Some("child-agent"));
        let agent = metadata.agent_context.as_deref();
        assert_eq!(
            agent.and_then(|value| value.parent_agent_id.as_deref()),
            Some("root-agent")
        );
        assert_eq!(
            agent.and_then(|value| value.agent_kind.as_deref()),
            Some("collab_spawn")
        );
        assert_eq!(
            agent.and_then(|value| value.turn_id.as_deref()),
            Some("turn-7")
        );
    }

    #[test]
    fn explicit_switchyard_headers_override_relay_and_codex() {
        let mut headers = BTreeMap::new();
        headers.insert(
            "x-switchyard-session-id".to_string(),
            vec!["canonical-session".to_string()],
        );
        headers.insert(
            "x-switchyard-agent-id".to_string(),
            vec!["canonical-agent".to_string()],
        );
        headers.insert(
            "x-nemo-relay-session-id".to_string(),
            vec!["relay-session".to_string()],
        );
        headers.insert(
            "x-dynamo-session-id".to_string(),
            vec!["relay-agent".to_string()],
        );

        let metadata = metadata_from_headers(&headers);
        assert_eq!(metadata.session_id.as_deref(), Some("canonical-session"));
        assert_eq!(metadata.agent_id.as_deref(), Some("canonical-agent"));
    }

    #[test]
    fn normalizes_relay_and_dynamo_headers_without_runtime_dependency() {
        let mut headers = BTreeMap::new();
        headers.insert(
            "x-nemo-relay-session-id".to_string(),
            vec!["relay-session".to_string()],
        );
        headers.insert(
            "x-nemo-relay-subagent-id".to_string(),
            vec!["relay-child".to_string()],
        );
        headers.insert(
            "x-dynamo-parent-session-id".to_string(),
            vec!["relay-parent".to_string()],
        );

        let metadata = metadata_from_headers(&headers);
        assert_eq!(metadata.session_id.as_deref(), Some("relay-session"));
        assert_eq!(metadata.agent_id.as_deref(), Some("relay-child"));
        assert_eq!(
            metadata
                .agent_context
                .as_deref()
                .and_then(|agent| agent.parent_agent_id.as_deref()),
            Some("relay-parent")
        );
    }

    #[tokio::test]
    async fn valid_classifier_output_routes_from_the_pool(
    ) -> Result<(), Box<dyn Error + Send + Sync>> {
        let (algo, calls) = algo(vec![&decision("fast", "research", 0.9, "bounded lookup")]);
        let (trace, response) = algo
            .process_request(
                &OrchestratorContext::default(),
                request(Some("agent-a"), None, "survey the API"),
            )
            .await?;

        assert_eq!(response.llm_response.completion, "fast");
        assert_eq!(trace.len(), 2);
        let routed = trace[1]
            .as_any()
            .downcast_ref::<AgentRoutingDecision>()
            .ok_or("expected AgentRoutingDecision")?;
        assert_eq!(routed.task_kind.as_deref(), Some("research"));
        assert_eq!(routed.confidence, Some(0.9));
        assert!(!routed.cache_hit);
        assert_eq!(*calls.lock().map_err(|_| "lock poisoned")?, 1);
        Ok(())
    }

    #[tokio::test]
    async fn stable_agent_assignment_avoids_reclassification(
    ) -> Result<(), Box<dyn Error + Send + Sync>> {
        let (algo, calls) = algo(vec![&decision("fast", "research", 0.8, "explorer")]);
        algo.process_request(
            &OrchestratorContext::default(),
            request(Some("agent-a"), None, "first turn"),
        )
        .await?;
        let (trace, response) = algo
            .process_request(
                &OrchestratorContext::default(),
                request(Some("agent-a"), None, "second turn"),
            )
            .await?;

        assert_eq!(response.llm_response.completion, "fast");
        assert_eq!(*calls.lock().map_err(|_| "lock poisoned")?, 1);
        assert_eq!(trace.len(), 1);
        let routed = trace[0]
            .as_any()
            .downcast_ref::<AgentRoutingDecision>()
            .ok_or("expected AgentRoutingDecision")?;
        assert!(routed.cache_hit);
        Ok(())
    }

    #[tokio::test]
    async fn explicit_task_change_reclassifies_the_same_agent(
    ) -> Result<(), Box<dyn Error + Send + Sync>> {
        let first = decision("fast", "research", 0.8, "lookup");
        let second = decision("frontier", "review", 0.9, "final review");
        let (algo, calls) = algo(vec![&first, &second]);
        let (_, first_response) = algo
            .process_request(
                &OrchestratorContext::default(),
                request(Some("agent-a"), Some("task-1"), "collect evidence"),
            )
            .await?;
        let (_, second_response) = algo
            .process_request(
                &OrchestratorContext::default(),
                request(Some("agent-a"), Some("task-2"), "adjudicate findings"),
            )
            .await?;

        assert_eq!(first_response.llm_response.completion, "fast");
        assert_eq!(second_response.llm_response.completion, "frontier");
        assert_eq!(*calls.lock().map_err(|_| "lock poisoned")?, 2);
        Ok(())
    }

    #[tokio::test]
    async fn invalid_decision_falls_back_without_poisoning_affinity(
    ) -> Result<(), Box<dyn Error + Send + Sync>> {
        let valid = decision("fast", "research", 0.7, "retry succeeded");
        let (algo, calls) = algo(vec!["not-json", &valid]);
        let (_, first) = algo
            .process_request(
                &OrchestratorContext::default(),
                request(Some("agent-a"), None, "first"),
            )
            .await?;
        let (_, second) = algo
            .process_request(
                &OrchestratorContext::default(),
                request(Some("agent-a"), None, "second"),
            )
            .await?;

        assert_eq!(first.llm_response.completion, "frontier");
        assert_eq!(second.llm_response.completion, "fast");
        assert_eq!(*calls.lock().map_err(|_| "lock poisoned")?, 2);
        Ok(())
    }
}
