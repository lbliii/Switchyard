// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Random router built on the [`OrchAlgo`] interfaces.
//!
//! Selects one target from the set uniformly at random and calls it. This is the
//! simplest possible routing algorithm and the reference for the single-call
//! shape: one `target.call` inside `process_request`. (Weighted selection could
//! be layered on later; the set defines the candidates.)

use std::error::Error;
use std::sync::Arc;

use async_trait::async_trait;
use rand::seq::SliceRandom;

use crate::{
    AgentSysSignals, DecisionTrace, LlmTargetSet, OrchAlgo, OrchAlgoBuilder, OrchestratorContext,
    OrchestratorRequest, OrchestratorResponse,
};

/// Decision produced by [`RandomOrchAlgo`]: which target was chosen and why.
pub struct RandomDecision {
    /// The randomly selected target/model.
    pub selected_model: String,
    /// Human-readable explanation of the choice.
    pub reasoning: String,
}

impl DecisionTrace for RandomDecision {
    fn model_decision(&self) -> &str {
        &self.selected_model
    }
    fn reasoning(&self) -> Option<&str> {
        Some(&self.reasoning)
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// Uniform random router over a target set.
pub struct RandomOrchAlgo {
    target_set: LlmTargetSet,
}

impl RandomOrchAlgo {
    /// Create a router over `target_set`. Usually built via
    /// [`RandomOrchAlgoBuilder`] + [`MultiLlmOrchestrator`](crate::MultiLlmOrchestrator).
    pub fn new(target_set: LlmTargetSet) -> Self {
        Self { target_set }
    }
}

#[async_trait]
impl OrchAlgo for RandomOrchAlgo {
    async fn process_request(
        &self,
        ctx: &OrchestratorContext,
        request: OrchestratorRequest,
    ) -> Result<(Vec<Arc<dyn DecisionTrace>>, OrchestratorResponse), Box<dyn Error + Send + Sync>>
    {
        // Select a target uniformly at random. Scope the RNG so the non-Send
        // `ThreadRng` is dropped before the await below, keeping the returned
        // future `Send` (required by the `OrchAlgo` bound).
        let target = {
            let mut rng = rand::thread_rng();
            self.target_set
                .targets()
                .choose(&mut rng)
                .ok_or("no targets available")?
                .clone()
        };

        // Route by target name; the target maps it to the provider model id when
        // it serves or offloads the call.
        let selected = target.name.clone();
        let decision: Arc<dyn DecisionTrace> = Arc::new(RandomDecision {
            reasoning: format!("random routing selected target '{selected}'"),
            selected_model: selected,
        });

        let response = target.call(ctx, request, Some(decision.clone())).await?;
        Ok((vec![decision], response))
    }

    async fn process_signals(
        &self,
        _signals: AgentSysSignals,
    ) -> Result<(), Box<dyn Error + Send + Sync>> {
        // Random routing is stateless, so agent-system signals are ignored.
        Ok(())
    }
}

/// Builder that wires a target set into a [`RandomOrchAlgo`].
#[derive(Default)]
pub struct RandomOrchAlgoBuilder {
    target_set: Option<LlmTargetSet>,
}

impl OrchAlgoBuilder for RandomOrchAlgoBuilder {
    fn with_target_set(&mut self, target_set: LlmTargetSet) {
        self.target_set = Some(target_set);
    }

    fn build(&mut self) -> Box<dyn OrchAlgo> {
        Box::new(RandomOrchAlgo::new(
            self.target_set
                .take()
                .unwrap_or_else(|| LlmTargetSet::new(vec![])),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{LlmClient, LlmRequest, LlmResponse, LlmTarget, OrchestratorResponse};
    use std::collections::HashSet;

    /// Echoes back the target name it was called with, so a test can tell which
    /// target the algo selected.
    struct EchoClient;

    #[async_trait]
    impl LlmClient for EchoClient {
        async fn call(
            &self,
            request: OrchestratorRequest,
        ) -> Result<OrchestratorResponse, Box<dyn Error + Send + Sync>> {
            Ok(OrchestratorResponse {
                llm_response: LlmResponse {
                    completion: request.llm_request.model_name,
                    raw_response: None,
                },
                metadata: None,
            })
        }
    }

    fn request() -> OrchestratorRequest {
        OrchestratorRequest {
            llm_request: LlmRequest {
                model_name: "auto".to_string(),
                prompt: "hi".to_string(),
            },
            raw_request: None,
            metadata: None,
        }
    }

    // Client-less-free tests: a channel-less context, since no call offloads.
    fn ctx() -> OrchestratorContext {
        OrchestratorContext::default()
    }

    fn algo(names: &[&str]) -> RandomOrchAlgo {
        let targets: Vec<LlmTarget> = names
            .iter()
            .map(|name| LlmTarget {
                name: name.to_string(),
                model: name.to_string(),
                llm_client: Some(Arc::new(EchoClient)),
            })
            .collect();
        RandomOrchAlgo::new(LlmTargetSet::new(targets))
    }

    #[tokio::test]
    async fn single_target_is_always_selected_and_called(
    ) -> Result<(), Box<dyn Error + Send + Sync>> {
        let algo = algo(&["only/model"]);
        let (trace, response) = algo.process_request(&ctx(), request()).await?;
        assert_eq!(response.llm_response.completion, "only/model");
        assert_eq!(trace.len(), 1);
        assert_eq!(trace[0].model_decision(), "only/model");
        Ok(())
    }

    #[tokio::test]
    async fn selected_target_is_in_the_set_and_matches_the_trace(
    ) -> Result<(), Box<dyn Error + Send + Sync>> {
        let names = ["a/model", "b/model", "c/model"];
        let algo = algo(&names);
        for _ in 0..50 {
            let (trace, response) = algo.process_request(&ctx(), request()).await?;
            let selected = response.llm_response.completion.clone();
            assert!(
                names.contains(&selected.as_str()),
                "selected {selected} not in target set"
            );
            // The trace records the same target that was actually called.
            assert_eq!(trace[0].model_decision(), selected.as_str());
        }
        Ok(())
    }

    #[tokio::test]
    async fn selection_covers_all_targets_over_many_runs(
    ) -> Result<(), Box<dyn Error + Send + Sync>> {
        let algo = algo(&["a/model", "b/model"]);
        let mut seen = HashSet::new();
        for _ in 0..100 {
            let (_, response) = algo.process_request(&ctx(), request()).await?;
            seen.insert(response.llm_response.completion);
        }
        // 100 uniform draws over two targets: both should appear (miss ~ 2^-99).
        assert_eq!(
            seen.len(),
            2,
            "expected both targets to be selected, saw {seen:?}"
        );
        Ok(())
    }

    #[tokio::test]
    async fn empty_target_set_errors() {
        let algo = algo(&[]);
        assert!(algo.process_request(&ctx(), request()).await.is_err());
    }

    #[tokio::test]
    async fn process_signals_is_a_noop() -> Result<(), Box<dyn Error + Send + Sync>> {
        let algo = algo(&["only/model"]);
        algo.process_signals(AgentSysSignals {}).await?;
        Ok(())
    }

    #[tokio::test]
    async fn decision_is_inspectable_and_downcasts() -> Result<(), Box<dyn Error + Send + Sync>> {
        let algo = algo(&["only/model"]);
        let (trace, _) = algo.process_request(&ctx(), request()).await?;
        let decision = &trace[0];
        // Uniform, algo-agnostic access via the trait — no concrete type needed.
        assert_eq!(decision.model_decision(), "only/model");
        assert!(decision
            .reasoning()
            .unwrap_or_default()
            .contains("only/model"));
        // Escape hatch: downcast to the concrete decision when the algo is known.
        let concrete = decision
            .as_any()
            .downcast_ref::<RandomDecision>()
            .ok_or("expected a RandomDecision")?;
        assert_eq!(concrete.selected_model, "only/model");
        Ok(())
    }
}
