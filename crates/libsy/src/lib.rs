// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! # libsy — multi-LLM agent optimization (routing first)
//!
//! `libsy` decides, per request, *how* to serve an LLM call: which model(s) to
//! invoke, in what order, and how to combine the results. Routing is the first
//! and simplest case; the same interfaces also express classifier routing,
//! ensembles, cascades, and other optimizations. The library owns no HTTP client
//! and no provider SDK — it decides, and the host makes (or is asked to make) the
//! actual calls — so it embeds cleanly in a proxy, gateway, or agent runtime.
//!
//! ## The model
//!
//! - An [`OrchAlgo`] is the optimization *algorithm*. Its
//!   [`process_request`](OrchAlgo::process_request) runs once per request and
//!   makes as many model calls as it needs — via [`LlmTargetI::call`], which look
//!   like ordinary calls — then returns a *decision trace* (a list of
//!   [`DecisionTrace`]) plus the final [`OrchestratorResponse`].
//! - An [`LlmTarget`] names a model. If it carries an [`LlmClient`] it *serves*
//!   its own calls; if not, the call is *offloaded* to the host (see below).
//! - A [`MultiLlmOrchestrator`] owns one algorithm and its [`LlmTargetSet`] and
//!   drives requests through it. Construct it with [`MultiLlmOrchestrator::new`]
//!   from an [`OrchAlgoBuilder`] and a target set.
//!
//! ## Two ways to run a request
//!
//! - [`MultiLlmOrchestrator::orchestrate_direct`] — when every target has a
//!   client, libsy makes all the calls itself and returns `(trace, response)`.
//!   The simplest integration; use it when libsy holds the model clients.
//! - [`MultiLlmOrchestrator::orchestrate`] — returns a stream of
//!   [`OrchestratorStep`]s. When the algorithm calls a *client-less* target, the
//!   call is offloaded: the stream yields [`OrchestratorStep::CallLlm`] carrying
//!   promises; the host performs the real model call and fulfills each promise
//!   with [`LlmPromiseTx::set_response`]. When the algorithm finishes the stream
//!   yields [`OrchestratorStep::ReturnToAgent`] with the trace and final response.
//!   This "ask, don't call" mode lets a host that already owns its transport keep
//!   control of every network call.
//!
//! ## Concurrency
//!
//! [`OrchAlgo::process_request`] takes `&self` and the orchestrator holds
//! `Arc<dyn OrchAlgo>` with no lock, so a single orchestrator serves many requests
//! in parallel. Each [`orchestrate`](MultiLlmOrchestrator::orchestrate) call gets
//! its own promise channel, so offloaded calls never cross between concurrent
//! requests. An algorithm is responsible for its own thread-safety — stateless
//! (like the reference routers) or interior mutability over just its own state.
//!
//! ## Reference algorithms
//!
//! - [`rand::RandomOrchAlgo`] — uniform random over the target set (one call).
//! - [`llm_class::LlmClassifierOrchAlgo`] — classify with one model, then route to
//!   a strong/weak model (multi-step).
//! - [`ensemble::EnsembleOrchAlgo`] — fan out to several models, then judge and
//!   commit (stateful).
//!
//! See the `examples/` directory for runnable agents built on both run modes.

pub mod ensemble;
pub mod llm_class;
pub mod rand;

use std::{error::Error, sync::Arc};
use tokio::sync::oneshot;

use async_trait::async_trait;
use tokio_stream::wrappers::ReceiverStream;

/// Correlation and routing metadata attached to a request or response.
///
/// All fields are optional; algorithms and observers use whichever are present
/// (e.g. to key per-session state or emit correlated telemetry). `extra_metadata`
/// is a free-form escape hatch for host-specific keys.
#[derive(Clone)]
pub struct Metadata {
    /// Stable id for a multi-request session/conversation.
    pub session_id: Option<String>,
    /// Id of the agent making the request.
    pub agent_id: Option<String>,
    /// Id of the task the request belongs to.
    pub task_id: Option<String>,
    /// External trace/request id for joining with the host's telemetry.
    pub correlation_id: Option<String>,
    /// Arbitrary host-defined key/value metadata.
    pub extra_metadata: Option<std::collections::BTreeMap<String, String>>,
}

/// The neutral model request an algorithm reasons over and hands to a target.
///
/// Deliberately minimal: a target model name and the user prompt. The full
/// provider-shaped request (messages, params, tools) rides on
/// [`OrchestratorRequest::raw_request`] when a host needs to forward it losslessly.
#[derive(Clone)]
pub struct LlmRequest {
    /// The model to call. Algorithms rewrite this as they route.
    pub model_name: String,
    /// The user prompt an algorithm inspects (e.g. to classify) and sends.
    pub prompt: String,
}

/// A request entering the orchestrator: the neutral [`LlmRequest`] plus the
/// original provider payload and correlation [`Metadata`].
#[derive(Clone)]
pub struct OrchestratorRequest {
    /// The neutral request an algorithm routes.
    pub llm_request: LlmRequest,
    /// The original provider-shaped request body, if the host wants to forward it
    /// verbatim (e.g. a proxy preserving messages/params). libsy does not read it.
    pub raw_request: Option<serde_json::Value>,
    /// Correlation metadata carried through the request.
    pub metadata: Option<Metadata>,
}

/// Agentic-stack events fed to an algorithm out of band via
/// [`MultiLlmOrchestrator::process_signals`] (e.g. tool results, budget updates).
///
/// A placeholder today; a stateful algorithm can begin consuming signals as the
/// enum grows without changing the orchestrator contract.
#[derive(Clone)]
pub struct AgentSysSignals {}

/// The neutral model response returned by a target.
#[derive(Clone)]
pub struct LlmResponse {
    /// The model's completion text — what an algorithm inspects (e.g. a
    /// classifier score) or returns.
    pub completion: String,
    /// Optional raw provider response body, so a host (e.g. a proxy) can forward
    /// the upstream response losslessly instead of rebuilding it from `completion`.
    pub raw_response: Option<serde_json::Value>,
}

/// A response leaving the orchestrator: the neutral [`LlmResponse`] plus optional
/// correlation [`Metadata`].
#[derive(Clone)]
pub struct OrchestratorResponse {
    /// The neutral model response.
    pub llm_response: LlmResponse,
    /// Correlation metadata carried through the response.
    pub metadata: Option<Metadata>,
}

/// A decision/trace object produced by an algorithm.
///
/// Carried as a trait object (not a generic parameter) so a stream consumer can
/// inspect any algorithm's decision through this common interface without
/// knowing the concrete type. `as_any` is the escape hatch for a consumer that
/// *does* know the algo and wants to downcast to the concrete decision.
pub trait DecisionTrace: Send + Sync {
    /// The model this decision selected (e.g. the routed target's name).
    fn model_decision(&self) -> &str;
    /// A human-readable explanation of the decision, for logs and traces.
    fn reasoning(&self) -> Option<&str>;
    /// Downcast handle: a consumer that knows the algorithm can recover the
    /// concrete decision type via `as_any().downcast_ref::<ConcreteDecision>()`.
    fn as_any(&self) -> &dyn std::any::Any;
}

/// The result the caller sends back for an offloaded call: the model response, or
/// the error from the failed model call so it can propagate into the algorithm.
pub type PromiseResult = Result<OrchestratorResponse, Box<dyn Error + Send + Sync>>;

/// The host-facing half of an offloaded model call.
///
/// Yielded to the host inside [`OrchestratorStep::CallLlm`]. The host reads the
/// request it should perform ([`get_request`](Self::get_request)) and the
/// decision behind it ([`get_decision`](Self::get_decision)), makes the real model
/// call, and fulfills the promise with [`set_response`](Self::set_response). That
/// unblocks the algorithm's [`LlmTargetI::call`] on the other side.
pub struct LlmPromiseTx {
    request: OrchestratorRequest,
    decision: Option<Arc<dyn DecisionTrace>>,
    tx: Option<oneshot::Sender<PromiseResult>>,
}

/// The algorithm-facing half of an offloaded model call.
///
/// Held inside a client-less [`LlmTargetI::call`], which awaits
/// [`get_response`](Self::get_response) until the host fulfills the paired
/// [`LlmPromiseTx`]. Host code normally does not touch this type directly.
pub struct LlmPromiseRx {
    rx: Option<oneshot::Receiver<PromiseResult>>,
    response: Option<OrchestratorResponse>,
}

impl LlmPromiseTx {
    /// The model call the host should perform to fulfill this promise.
    pub fn get_request(&self) -> &OrchestratorRequest {
        &self.request
    }

    /// The decision that led to this call, if the algorithm attached one.
    pub fn get_decision(&self) -> Option<&dyn DecisionTrace> {
        self.decision.as_deref()
    }

    /// Fulfill the promise with the caller's model-call result. Pass `Err(..)` to
    /// propagate a failed model call back to the algorithm that made the call.
    pub async fn set_response(
        &mut self,
        result: PromiseResult,
    ) -> Result<(), Box<dyn Error + Send + Sync>> {
        let tx = match self.tx.take() {
            Some(tx) => tx,
            None => return Ok(()), // already fulfilled
        };
        tx.send(result).map_err(|_| "Failed to send result")?;
        Ok(())
    }
}

impl LlmPromiseRx {
    /// Await the response the host provides via [`LlmPromiseTx::set_response`].
    ///
    /// Returns the fulfilled response, propagates the host's `Err` if the model
    /// call failed, or errors if the promise was dropped unfulfilled. The result
    /// is memoized, so repeated calls return the same response.
    pub async fn get_response(
        &mut self,
    ) -> Result<OrchestratorResponse, Box<dyn Error + Send + Sync>> {
        if let Some(response) = &self.response {
            return Ok(response.clone());
        }
        let rx = self.rx.take().ok_or("Result already received")?;
        // Outer error: the sender was dropped without fulfilling the promise.
        // Inner error: the caller's model call failed — propagate it to the algo.
        let response = match rx.await {
            Ok(result) => result?,
            Err(_) => return Err("Failed to receive result".into()),
        };
        self.response = Some(response.clone());
        Ok(response)
    }
}

/// Create a paired promise for one offloaded model call: the [`LlmPromiseTx`] is
/// handed to the host, the [`LlmPromiseRx`] is awaited by the algorithm's target.
/// Used internally by the offload path; exposed for algorithms/hosts that build
/// their own offloading.
pub fn llm_promise(
    request: OrchestratorRequest,
    decision: Option<Arc<dyn DecisionTrace>>,
) -> (LlmPromiseTx, LlmPromiseRx) {
    let (tx, rx) = oneshot::channel();
    let tx = LlmPromiseTx {
        request,
        decision,
        tx: Some(tx),
    };
    let rx = LlmPromiseRx {
        rx: Some(rx),
        response: None,
    };
    (tx, rx)
}

/// One item in the stream returned by [`MultiLlmOrchestrator::orchestrate`].
pub enum OrchestratorStep {
    /// The algorithm needs these model calls performed. The host fulfills each
    /// promise with [`LlmPromiseTx::set_response`]; only client-less targets
    /// produce these steps.
    CallLlm(Vec<LlmPromiseTx>),
    /// The algorithm finished: its decision trace and the final response. This is
    /// the last step of a successful run.
    ReturnToAgent(Vec<Arc<dyn DecisionTrace>>, OrchestratorResponse),
}

/// A stream item from [`MultiLlmOrchestrator::orchestrate`]: an
/// [`OrchestratorStep`], or an error if the algorithm (or an offloaded call) failed.
pub type OrchestratorStepResult = Result<OrchestratorStep, Box<dyn Error + Send + Sync>>;

/// Performs the actual model call for a target. This is the one piece of I/O
/// `libsy` does not own — a host implements it over its own transport (HTTP SDK,
/// in-process model, mock). A target that carries an `LlmClient` *serves* its own
/// calls; one without offloads them (see [`MultiLlmOrchestrator::orchestrate`]).
#[async_trait]
pub trait LlmClient: Send + Sync {
    /// Call `model_name` (falling back to `request.llm_request.model_name`) with
    /// the given request, returning the model's response.
    async fn call(
        &self,
        request: OrchestratorRequest,
        model_name: Option<String>,
    ) -> Result<OrchestratorResponse, Box<dyn Error + Send + Sync>>;
}

/// A routing target: a named model an algorithm can call. Implemented by
/// [`LlmTarget`] (the value type hosts build) and [`WrappedLlmTarget`] (the
/// offload-capable wrapper the orchestrator applies internally).
#[async_trait]
pub trait LlmTargetI: Send + Sync {
    /// The target's name — usually the upstream model id an algorithm routes to.
    fn get_name(&self) -> &str;
    /// The client that serves this target's calls, or `None` if calls are offloaded.
    fn get_client(&self) -> Option<Arc<dyn LlmClient>>;
    /// Whether this target can serve its own call (has a client). Used by
    /// [`LlmTargetSet::all_have_clients`] to decide if `orchestrate_direct` is safe.
    fn has_client(&self) -> bool {
        self.get_client().is_some()
    }
    /// Perform (or offload) the model call, tagging it with the algorithm's
    /// `decision` so a host driving the stream can see *why* the call was made.
    async fn call(
        &self,
        request: OrchestratorRequest,
        decision: Option<Arc<dyn DecisionTrace>>,
    ) -> Result<OrchestratorResponse, Box<dyn Error + Send + Sync>>;
}

/// A named routing target, optionally backed by an [`LlmClient`].
///
/// With a client, [`call`](LlmTargetI::call) invokes it directly. Without one,
/// `call` errors *unless* the target has been wrapped by
/// [`MultiLlmOrchestrator::new`] (as a [`WrappedLlmTarget`]), which turns a
/// client-less call into an offloaded promise.
#[derive(Clone)]
pub struct LlmTarget {
    /// The target/model name.
    pub name: String,
    /// The client that serves calls, or `None` to offload them.
    pub llm_client: Option<Arc<dyn LlmClient>>,
}

#[async_trait]
impl LlmTargetI for LlmTarget {
    fn get_name(&self) -> &str {
        &self.name
    }
    fn get_client(&self) -> Option<Arc<dyn LlmClient>> {
        self.llm_client.clone()
    }

    async fn call(
        &self,
        request: OrchestratorRequest,
        _decision: Option<Arc<dyn DecisionTrace>>,
    ) -> Result<OrchestratorResponse, Box<dyn Error + Send + Sync>> {
        match &self.llm_client {
            Some(client) => client.call(request, Some(self.name.clone())).await,
            None => Err(format!("No LLM client for target {}", self.name).into()),
        }
    }
}

tokio::task_local! {
    /// The current request's promise sender, set by `orchestrate` in the task
    /// scope around `process_request`. A wrapped target routes an offloaded call
    /// to *this* channel, so concurrent requests never share a promise channel —
    /// which is what lets many requests be orchestrated in parallel.
    static PROMISE_TX: tokio::sync::mpsc::Sender<LlmPromiseTx>;
}

/// A target that offloads calls it cannot serve directly. With no client, `call`
/// makes a promise and hands it to the current request's promise channel (the
/// `PROMISE_TX` task-local); the orchestrator surfaces it as a `CallLlm` step for
/// the caller to fulfill.
pub struct WrappedLlmTarget {
    target: LlmTarget,
}

#[async_trait]
impl LlmTargetI for WrappedLlmTarget {
    fn get_name(&self) -> &str {
        self.target.get_name()
    }

    fn get_client(&self) -> Option<Arc<dyn LlmClient>> {
        self.target.get_client()
    }

    async fn call(
        &self,
        request: OrchestratorRequest,
        decision: Option<Arc<dyn DecisionTrace>>,
    ) -> Result<OrchestratorResponse, Box<dyn Error + Send + Sync>> {
        match self.get_client() {
            Some(client) => {
                client
                    .call(request, Some(self.get_name().to_string()))
                    .await
            }
            None => {
                // No direct client: offload the call via a promise on the current
                // request's channel (task-local), attaching the decision so the
                // orchestrator can surface it on its stream.
                let promise_tx = PROMISE_TX
                    .try_with(|tx| tx.clone())
                    .map_err(|_| "offloading a target call requires an orchestrate() context")?;
                let (tx, mut rx) = llm_promise(request, decision);
                promise_tx
                    .send(tx)
                    .await
                    .map_err(|_| "Failed to send promise")?;
                rx.get_response().await
            }
        }
    }
}

/// The set of targets an algorithm may route among. An algorithm receives it via
/// its [`OrchAlgoBuilder`] and picks targets by position ([`targets`](Self::targets))
/// or by name ([`get_target`](Self::get_target)).
#[derive(Clone)]
pub struct LlmTargetSet {
    targets: Vec<Arc<dyn LlmTargetI>>,
}

impl LlmTargetSet {
    /// Build a target set from a list of targets.
    pub fn new(targets: Vec<Arc<dyn LlmTargetI>>) -> Self {
        Self { targets }
    }

    /// All targets in the set — e.g. for an algorithm to select among.
    pub fn targets(&self) -> &[Arc<dyn LlmTargetI>] {
        &self.targets
    }

    /// Look up a target by name; errors if no target has that name.
    pub fn get_target(
        &self,
        name: &str,
    ) -> Result<Arc<dyn LlmTargetI>, Box<dyn Error + Send + Sync>> {
        self.targets
            .iter()
            .find(|t| t.get_name() == name)
            .cloned()
            .ok_or(format!("Target {} not found", name).into())
    }

    /// Whether every target can serve its own call — the precondition for
    /// [`MultiLlmOrchestrator::orchestrate_direct`].
    pub fn all_have_clients(&self) -> bool {
        self.targets.iter().all(|t| t.has_client())
    }
}

/// A stateful optimization algorithm. `process_request` is called once per
/// request; inside it the algorithm makes as many `LlmTargetI::call`s as it needs
/// (each may be served directly or offloaded), and returns a decision trace plus
/// the final response. `process_signals` feeds it agentic-stack events.
///
/// Both methods take `&self`, not `&mut self`: the orchestrator shares one
/// algorithm (`Arc<dyn OrchAlgo>`) across all requests and calls it concurrently,
/// so an algorithm is responsible for its own thread-safety. Stateless algorithms
/// (like the reference routers) get this for free; a stateful one must use
/// interior mutability (e.g. a `Mutex`/`RwLock`/atomics over just its own state)
/// rather than a coarse lock over the whole algorithm.
#[async_trait]
pub trait OrchAlgo: Send + Sync {
    /// Run one request to completion: make the model calls the algorithm decides
    /// on (via [`LlmTargetI::call`]) and return the decision trace plus the final
    /// response. Called concurrently for many requests, so it takes `&self`.
    async fn process_request(
        &self,
        request: OrchestratorRequest,
    ) -> Result<(Vec<Arc<dyn DecisionTrace>>, OrchestratorResponse), Box<dyn Error + Send + Sync>>;
    /// Feed the algorithm agentic-stack events (tool results, budgets, etc.). The
    /// reference algorithms ignore signals; a stateful algorithm updates its own
    /// (interior-mutable) state.
    async fn process_signals(
        &self,
        signals: AgentSysSignals,
    ) -> Result<(), Box<dyn Error + Send + Sync>>;
}

/// Builds an [`OrchAlgo`] once its target set is known. The orchestrator wraps
/// the targets (to route offloaded calls to its promise channel) before handing
/// them to the builder, so an algorithm never constructs its own targets.
pub trait OrchAlgoBuilder: Send + Sync {
    /// Supply the (wrapped) target set the built algorithm should route among.
    /// Called by [`MultiLlmOrchestrator::new`] before [`build`](Self::build).
    fn with_target_set(&mut self, target_set: LlmTargetSet);
    /// Construct the algorithm. Called once, after `with_target_set`.
    fn build(&mut self) -> Box<dyn OrchAlgo>;
}

/// Drives one [`OrchAlgo`] over a [`LlmTargetSet`], the main entry point for a
/// host. Cheap to share (`Arc<dyn OrchAlgo>` internally, no lock) and safe to call
/// from many threads at once. Run requests with
/// [`orchestrate_direct`](Self::orchestrate_direct) (targets serve their own
/// calls) or [`orchestrate`](Self::orchestrate) (offloaded calls stream back for
/// the host to fulfill).
pub struct MultiLlmOrchestrator {
    // Shared, lock-free: `process_request` takes `&self`, so one algorithm serves
    // every request concurrently.
    algo: Arc<dyn OrchAlgo>,
    target_set: LlmTargetSet,
}

impl MultiLlmOrchestrator {
    /// Build an orchestrator from an algorithm `builder` and an optional target
    /// set. Each target is wrapped so a client-less call offloads to the current
    /// request's promise channel, then the wrapped set is handed to the builder.
    pub fn new(mut builder: Box<dyn OrchAlgoBuilder>, target_set: Option<LlmTargetSet>) -> Self {
        let mut wrapped_targets = Vec::new();
        for target in target_set.unwrap_or(LlmTargetSet::new(vec![])).targets {
            let wrapped_target = WrappedLlmTarget {
                target: LlmTarget {
                    name: target.get_name().to_string(),
                    llm_client: target.get_client(),
                },
            };
            wrapped_targets.push(Arc::new(wrapped_target) as Arc<dyn LlmTargetI>);
        }

        let target_set = LlmTargetSet::new(wrapped_targets);
        builder.with_target_set(target_set.clone());
        let algo: Arc<dyn OrchAlgo> = Arc::from(builder.build());
        MultiLlmOrchestrator { algo, target_set }
    }

    /// Run a request as a stream of [`OrchestratorStep`]s.
    ///
    /// Use this when some targets are client-less (calls are offloaded). Drive the
    /// stream: on [`OrchestratorStep::CallLlm`], perform each promised model call
    /// and fulfill it with [`LlmPromiseTx::set_response`]; the run ends with a
    /// single [`OrchestratorStep::ReturnToAgent`] carrying the trace and response
    /// (or an `Err` item if the algorithm or a call failed). Every call gets its
    /// own promise channel, so concurrent `orchestrate` runs never interfere.
    pub fn orchestrate(
        &self,
        request: OrchestratorRequest,
    ) -> impl futures::stream::Stream<Item = OrchestratorStepResult> {
        let (stream_tx, stream_rx) = tokio::sync::mpsc::channel(10);
        // Per-request promise channel: concurrent requests get independent channels
        // (via the PROMISE_TX task-local), so offloaded promises never cross between
        // requests. Nothing here is shared or locked, so many `orchestrate` calls
        // run in parallel.
        let (promise_tx, mut promise_rx) = tokio::sync::mpsc::channel::<LlmPromiseTx>(10);
        let algo = self.algo.clone();

        // One driver task races two things: forwarding each offloaded promise to
        // the caller as a `CallLlm` step, and the algorithm finishing (which
        // yields the final `ReturnToAgent`). `process_request` runs on its own task
        // — inside the PROMISE_TX scope so its wrapped targets offload to *this*
        // request's channel — because it blocks awaiting promise responses the
        // caller only produces after it receives the `CallLlm` step; the two must
        // run concurrently, so a single receiver-loop cannot do both.
        tokio::spawn(async move {
            let mut algo_handle = tokio::spawn(PROMISE_TX.scope(promise_tx, async move {
                algo.process_request(request).await
            }));

            let mut recv_closed = false;
            loop {
                tokio::select! {
                    maybe = promise_rx.recv(), if !recv_closed => match maybe {
                        Some(promise) => {
                            if stream_tx
                                .send(Ok(OrchestratorStep::CallLlm(vec![promise])))
                                .await
                                .is_err()
                            {
                                // Consumer dropped the stream; stop the algo too.
                                algo_handle.abort();
                                return;
                            }
                        }
                        None => recv_closed = true,
                    },
                    result = &mut algo_handle => {
                        let step = match result {
                            Ok(Ok((trace, response))) => {
                                Ok(OrchestratorStep::ReturnToAgent(trace, response))
                            }
                            Ok(Err(err)) => Err(err),
                            Err(join_err) => Err(Box::new(join_err) as Box<dyn Error + Send + Sync>),
                        };
                        let _ = stream_tx.send(step).await;
                        return;
                    }
                }
            }
        });

        ReceiverStream::new(stream_rx)
    }

    /// Run a request without the stream: run the algorithm and return its decision
    /// trace plus the final response. Only valid when every target has a client —
    /// otherwise the algorithm may offload a call (there is no `orchestrate` task
    /// scope here, so a promise would have nowhere to go), so this errors up front.
    pub async fn orchestrate_direct(
        &self,
        request: OrchestratorRequest,
    ) -> Result<(Vec<Arc<dyn DecisionTrace>>, OrchestratorResponse), Box<dyn Error + Send + Sync>>
    {
        if !self.target_set.all_have_clients() {
            return Err(
                "Cannot orchestrate directly: some targets lack clients and require offloading"
                    .into(),
            );
        }
        // All targets have clients, so the algorithm never offloads a call. Run it
        // directly (no lock — `process_request` takes `&self`) and return.
        self.algo.process_request(request).await
    }

    /// Feed agentic-stack signals to the algorithm (see [`OrchAlgo::process_signals`]).
    pub async fn process_signals(
        &self,
        signals: AgentSysSignals,
    ) -> Result<(), Box<dyn Error + Send + Sync>> {
        self.algo.process_signals(signals).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio_stream::StreamExt;

    /// Mock client that echoes back the target name it was called with.
    struct EchoClient;

    #[async_trait]
    impl LlmClient for EchoClient {
        async fn call(
            &self,
            _request: OrchestratorRequest,
            model_name: Option<String>,
        ) -> Result<OrchestratorResponse, Box<dyn Error + Send + Sync>> {
            Ok(OrchestratorResponse {
                llm_response: LlmResponse {
                    completion: model_name.unwrap_or_default(),
                    raw_response: None,
                },
                metadata: None,
            })
        }
    }

    /// Trivial decision + algo used only to exercise the orchestrator: calls the
    /// first target and returns its response with a one-item trace.
    struct TestDecision {
        model: String,
    }

    impl DecisionTrace for TestDecision {
        fn model_decision(&self) -> &str {
            &self.model
        }
        fn reasoning(&self) -> Option<&str> {
            None
        }
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    }

    struct TestAlgo {
        target_set: LlmTargetSet,
    }

    #[async_trait]
    impl OrchAlgo for TestAlgo {
        async fn process_request(
            &self,
            request: OrchestratorRequest,
        ) -> Result<(Vec<Arc<dyn DecisionTrace>>, OrchestratorResponse), Box<dyn Error + Send + Sync>>
        {
            let target = self
                .target_set
                .targets()
                .first()
                .ok_or("no targets")?
                .clone();
            let decision: Arc<dyn DecisionTrace> = Arc::new(TestDecision {
                model: target.get_name().to_string(),
            });
            let response = target.call(request, Some(decision.clone())).await?;
            Ok((vec![decision], response))
        }

        async fn process_signals(
            &self,
            _signals: AgentSysSignals,
        ) -> Result<(), Box<dyn Error + Send + Sync>> {
            Ok(())
        }
    }

    #[derive(Default)]
    struct TestBuilder {
        target_set: Option<LlmTargetSet>,
    }

    impl OrchAlgoBuilder for TestBuilder {
        fn with_target_set(&mut self, target_set: LlmTargetSet) {
            self.target_set = Some(target_set);
        }
        fn build(&mut self) -> Box<dyn OrchAlgo> {
            Box::new(TestAlgo {
                target_set: self
                    .target_set
                    .take()
                    .unwrap_or_else(|| LlmTargetSet::new(vec![])),
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

    /// `(name, has_client)` — a client-less target offloads via a promise.
    fn target_set(names: &[(&str, bool)]) -> LlmTargetSet {
        let targets = names
            .iter()
            .map(|(name, has_client)| {
                Arc::new(LlmTarget {
                    name: name.to_string(),
                    llm_client: has_client.then(|| Arc::new(EchoClient) as Arc<dyn LlmClient>),
                }) as Arc<dyn LlmTargetI>
            })
            .collect();
        LlmTargetSet::new(targets)
    }

    #[tokio::test]
    async fn orchestrate_offloads_via_promise_then_returns_to_agent(
    ) -> Result<(), Box<dyn Error + Send + Sync>> {
        // A client-less target -> its call is offloaded via a promise the
        // orchestrator surfaces as a `CallLlm` step for us to fulfill.
        let orch = MultiLlmOrchestrator::new(
            Box::new(TestBuilder::default()),
            Some(target_set(&[("offload/model", false)])),
        );

        let stream = orch.orchestrate(request());
        tokio::pin!(stream);

        let mut saw_call = false;
        let mut final_completion = None;
        while let Some(step) = stream.next().await {
            match step? {
                OrchestratorStep::CallLlm(promises) => {
                    saw_call = true;
                    for mut promise in promises {
                        // The decision rode along with the promise.
                        assert_eq!(
                            promise.get_decision().map(|d| d.model_decision()),
                            Some("offload/model")
                        );
                        // Fulfilling the promise is the "real" model call the caller makes.
                        promise
                            .set_response(Ok(OrchestratorResponse {
                                llm_response: LlmResponse {
                                    completion: "fulfilled".to_string(),
                                    raw_response: None,
                                },
                                metadata: None,
                            }))
                            .await?;
                    }
                }
                OrchestratorStep::ReturnToAgent(trace, response) => {
                    assert_eq!(trace.len(), 1);
                    assert_eq!(trace[0].model_decision(), "offload/model");
                    final_completion = Some(response.llm_response.completion);
                }
            }
        }

        assert!(saw_call, "expected a CallLlm step before ReturnToAgent");
        assert_eq!(
            final_completion.ok_or("no ReturnToAgent step")?,
            "fulfilled"
        );
        Ok(())
    }

    #[tokio::test]
    async fn orchestrate_with_direct_client_returns_without_offload(
    ) -> Result<(), Box<dyn Error + Send + Sync>> {
        // A target with a client serves its own call, so no promise is emitted.
        let orch = MultiLlmOrchestrator::new(
            Box::new(TestBuilder::default()),
            Some(target_set(&[("direct/model", true)])),
        );

        let stream = orch.orchestrate(request());
        tokio::pin!(stream);

        let mut steps = Vec::new();
        while let Some(step) = stream.next().await {
            steps.push(step?);
        }

        assert_eq!(
            steps.len(),
            1,
            "direct client should not emit a CallLlm step"
        );
        match &steps[0] {
            OrchestratorStep::ReturnToAgent(trace, response) => {
                assert_eq!(trace[0].model_decision(), "direct/model");
                // EchoClient echoes the model name back as the completion.
                assert_eq!(response.llm_response.completion, "direct/model");
            }
            OrchestratorStep::CallLlm(_) => return Err("expected ReturnToAgent".into()),
        }
        Ok(())
    }

    #[tokio::test]
    async fn orchestrate_direct_returns_the_response_when_all_targets_have_clients(
    ) -> Result<(), Box<dyn Error + Send + Sync>> {
        // Every target has a client -> no call is ever offloaded, so we can skip
        // the stream and get the final response directly.
        let orch = MultiLlmOrchestrator::new(
            Box::new(TestBuilder::default()),
            Some(target_set(&[("direct/model", true)])),
        );

        let (trace, response) = orch.orchestrate_direct(request()).await?;
        // TestAlgo calls the first target; EchoClient echoes its name.
        assert_eq!(response.llm_response.completion, "direct/model");
        assert_eq!(trace[0].model_decision(), "direct/model");
        Ok(())
    }

    #[tokio::test]
    async fn orchestrate_direct_errors_when_a_target_lacks_a_client(
    ) -> Result<(), Box<dyn Error + Send + Sync>> {
        // A client-less target could offload a call via a promise that nobody
        // would fulfill, so orchestrate_direct refuses up front.
        let orch = MultiLlmOrchestrator::new(
            Box::new(TestBuilder::default()),
            Some(target_set(&[("offload/model", false)])),
        );

        assert!(orch.orchestrate_direct(request()).await.is_err());
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn requests_are_processed_in_parallel() -> Result<(), Box<dyn Error + Send + Sync>> {
        use std::time::Duration;
        use tokio::sync::Barrier;

        const N: usize = 4;

        // A client that blocks until all N concurrent calls have arrived. If
        // requests were serialized (one algorithm behind a `Mutex`), only one
        // call could be in flight, the barrier would never reach N, and the test
        // would time out. It passes only because `orchestrate_direct` runs the
        // shared algorithm concurrently.
        struct BarrierClient {
            barrier: Arc<Barrier>,
        }

        #[async_trait]
        impl LlmClient for BarrierClient {
            async fn call(
                &self,
                _request: OrchestratorRequest,
                model_name: Option<String>,
            ) -> Result<OrchestratorResponse, Box<dyn Error + Send + Sync>> {
                self.barrier.wait().await;
                Ok(OrchestratorResponse {
                    llm_response: LlmResponse {
                        completion: model_name.unwrap_or_default(),
                        raw_response: None,
                    },
                    metadata: None,
                })
            }
        }

        let barrier = Arc::new(Barrier::new(N));
        let targets = LlmTargetSet::new(vec![Arc::new(LlmTarget {
            name: "m".to_string(),
            llm_client: Some(Arc::new(BarrierClient {
                barrier: barrier.clone(),
            })),
        }) as Arc<dyn LlmTargetI>]);
        // One orchestrator shared (by `&`) across many concurrent requests.
        let orch = Arc::new(MultiLlmOrchestrator::new(
            Box::new(TestBuilder::default()),
            Some(targets),
        ));

        let mut handles = Vec::new();
        for _ in 0..N {
            let orch = orch.clone();
            handles.push(tokio::spawn(async move {
                orch.orchestrate_direct(request())
                    .await
                    .map(|(_, response)| response.llm_response.completion)
            }));
        }

        for handle in handles {
            // The timeout turns a serialization deadlock into a failure, not a hang.
            let completion = tokio::time::timeout(Duration::from_secs(5), handle).await???;
            assert_eq!(completion, "m");
        }
        Ok(())
    }

    #[tokio::test]
    async fn offload_error_propagates_back_to_the_algorithm(
    ) -> Result<(), Box<dyn Error + Send + Sync>> {
        // A client-less target offloads its call; we fulfill the promise with an
        // Err, which must flow back through `target.call` into the algorithm and
        // out as an error step — not a response.
        let orch = MultiLlmOrchestrator::new(
            Box::new(TestBuilder::default()),
            Some(target_set(&[("offload/model", false)])),
        );

        let stream = orch.orchestrate(request());
        tokio::pin!(stream);

        let mut saw_error = false;
        while let Some(step) = stream.next().await {
            match step {
                Ok(OrchestratorStep::CallLlm(promises)) => {
                    for mut promise in promises {
                        promise
                            .set_response(Err("upstream model call failed".into()))
                            .await?;
                    }
                }
                Ok(OrchestratorStep::ReturnToAgent(..)) => {
                    return Err("expected the offload error to propagate, got a response".into());
                }
                Err(err) => {
                    // The algorithm's `target.call` saw the error via the promise.
                    assert!(err.to_string().contains("upstream model call failed"));
                    saw_error = true;
                }
            }
        }

        assert!(saw_error, "expected an error step");
        Ok(())
    }
}
