// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Minimal research agent on `Switchyard` with **client-backed** targets.
//!
//! Every target owns an `LlmClient`, so no call is ever offloaded. That lets the
//! agent use `run_direct` — one request in, the decision trace + final
//! response out, no stream to drive. The multi-step routing (classify -> route) happens inside the
//! classifier algorithm; the agent never sees it. Run with:
//!   cargo run -p libsy --example research_agent

use std::error::Error;
use std::sync::Arc;

use async_trait::async_trait;
use libsy::llm_class::LlmClassifierOrchAlgo;
use libsy::{
    Decision, LlmClient, LlmRequest, LlmResponse, LlmTarget, LlmTargetSet, Request, Response,
    RoutedRequest, Switchyard,
};

const CLASSIFIER: &str = "classifier/model";
const STRONG: &str = "strong/model";
const WEAK: &str = "weak/model";

/// Stub transport. Real integrators implement `LlmClient` over their own HTTP.
struct StubClient;

#[async_trait]
impl LlmClient for StubClient {
    async fn call(&self, routed: RoutedRequest) -> Result<Response, Box<dyn Error + Send + Sync>> {
        // The model to call is the routed decision's selection, not the inbound name.
        let model = routed.decision.selected_model().to_string();
        println!("  -> model call: {model}");
        // The classifier returns a score; other models return an answer.
        let completion = if model == CLASSIFIER {
            "0.9".to_string()
        } else {
            format!("answer from {model}")
        };
        Ok(Response {
            llm_response: LlmResponse {
                completion,
                raw_response: None,
            },
            metadata: None,
        })
    }
}

fn targets() -> LlmTargetSet {
    let client = Arc::new(StubClient) as Arc<dyn LlmClient>;
    let target = |name: &str| LlmTarget {
        semantic_name: name.to_string(),
        llm_client: Some(client.clone()),
    };
    LlmTargetSet::new(vec![target(CLASSIFIER), target(STRONG), target(WEAK)])
}

struct ResearchAgent {
    orchestrator: Switchyard,
}

impl ResearchAgent {
    /// Trivial plan: one lookup per question (stub).
    fn plan(&self, question: &str) -> Vec<String> {
        vec![format!("look up: {question}")]
    }

    async fn run(&self, question: &str) -> Result<String, Box<dyn Error + Send + Sync>> {
        let mut notes = Vec::new();
        for step in self.plan(question) {
            let request = Request {
                llm_request: LlmRequest {
                    inbound_model_name: "auto".to_string(),
                    prompt: step,
                },
                raw_request: None,
                metadata: None,
            };
            // Every target has a client, so nothing is offloaded: run the request
            // and get the decision trace + final response directly, no stream.
            let (trace, response) = self.orchestrator.run_direct(request).await?;
            print_trace(&trace);
            notes.push(response.llm_response.completion);
        }
        Ok(notes.join("\n"))
    }
}

/// Print each decision the algorithm recorded — uniform access via the trait.
fn print_trace(trace: &[Arc<dyn Decision>]) {
    for decision in trace {
        println!(
            "    decision: {} ({})",
            decision.selected_model(),
            decision.reasoning().unwrap_or_default()
        );
    }
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn Error + Send + Sync>> {
    // Configure routing once: an LLM classifier over three named targets. Swapping
    // in `RandomOrchAlgo` needs no change to the agent.
    let algo = Arc::new(LlmClassifierOrchAlgo::new(
        CLASSIFIER,
        STRONG,
        WEAK,
        0.5,
        targets(),
    ));
    let orchestrator = Switchyard::new(algo);

    let agent = ResearchAgent { orchestrator };
    println!("{}", agent.run("what is switchyard?").await?);
    Ok(())
}
