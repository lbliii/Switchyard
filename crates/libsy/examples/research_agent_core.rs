// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Research agent driving the raw `orchestrate` stream with **client-less** targets.
//!
//! With no client, every `target.call` is offloaded as a promise the orchestrator
//! surfaces as a `CallLlm` step. The agent makes the "real" model call itself and
//! fulfills the promise — this is the offload/streaming path ("ask, don't call").
//! The classifier's two steps show up as two `model call:` lines. Run with:
//!   cargo run -p libsy --example research_agent_core

use std::error::Error;
use std::sync::Arc;

use libsy::llm_class::LlmClassifierOrchAlgo;
use libsy::{
    DecisionTrace, LlmRequest, LlmResponse, LlmTarget, LlmTargetSet, MultiLlmOrchestrator,
    OrchestratorRequest, OrchestratorResponse, OrchestratorStep,
};
use tokio_stream::StreamExt;

const CLASSIFIER: &str = "classifier/model";
const STRONG: &str = "strong/model";
const WEAK: &str = "weak/model";

/// The "real" model call the agent makes to fulfill a promise. The core never
/// makes the call itself — it hands back a request and waits for the response.
async fn call_model(request: &OrchestratorRequest) -> OrchestratorResponse {
    let model = &request.llm_request.model_name;
    println!("  -> model call: {model}");
    let completion = if model == CLASSIFIER {
        "0.9".to_string()
    } else {
        format!("answer from {model}")
    };
    OrchestratorResponse {
        llm_response: LlmResponse {
            completion,
            raw_response: None,
        },
        metadata: None,
    }
}

fn targets() -> LlmTargetSet {
    // Client-less targets -> every call is offloaded via a promise.
    let target = |name: &str| LlmTarget {
        name: name.to_string(),
        model: name.to_string(),
        llm_client: None,
    };
    LlmTargetSet::new(vec![target(CLASSIFIER), target(STRONG), target(WEAK)])
}

struct ResearchAgent {
    orchestrator: MultiLlmOrchestrator,
}

impl ResearchAgent {
    /// Trivial plan: one lookup per question (stub).
    fn plan(&self, question: &str) -> Vec<String> {
        vec![format!("look up: {question}")]
    }

    async fn run(&mut self, question: &str) -> Result<String, Box<dyn Error + Send + Sync>> {
        let mut notes = Vec::new();
        for step in self.plan(question) {
            let request = OrchestratorRequest {
                llm_request: LlmRequest {
                    model_name: "auto".to_string(),
                    prompt: step,
                },
                raw_request: None,
                metadata: None,
            };
            let stream = self.orchestrator.orchestrate(request);
            tokio::pin!(stream);
            while let Some(update) = stream.next().await {
                match update? {
                    OrchestratorStep::CallLlm(promises) => {
                        for mut promise in promises {
                            // Perform the model call the algorithm asked for, then fulfill.
                            let response = call_model(promise.get_request()).await;
                            promise.set_response(Ok(response)).await?;
                        }
                    }
                    OrchestratorStep::ReturnToAgent(trace, response) => {
                        print_trace(&trace);
                        notes.push(response.llm_response.completion);
                    }
                }
            }
        }
        Ok(notes.join("\n"))
    }
}

/// Print each decision the algorithm recorded — uniform access via the trait.
fn print_trace(trace: &[Arc<dyn DecisionTrace>]) {
    for decision in trace {
        println!(
            "    decision: {} ({})",
            decision.model_decision(),
            decision.reasoning().unwrap_or_default()
        );
    }
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn Error + Send + Sync>> {
    let algo = Arc::new(LlmClassifierOrchAlgo::new(
        CLASSIFIER,
        STRONG,
        WEAK,
        0.5,
        targets(),
    ));
    let orchestrator = MultiLlmOrchestrator::new(algo);

    let mut agent = ResearchAgent { orchestrator };
    println!("{}", agent.run("what is switchyard?").await?);
    Ok(())
}
