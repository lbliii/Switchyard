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
//! - An [`Algorithm`] is the optimization *algorithm*. Its
//!   [`process_request`](Algorithm::process_request) runs once per request and
//!   makes as many model calls as it needs — via [`LlmTarget::call`], which look
//!   like ordinary calls — then returns a *decision trace* (a list of
//!   [`DecisionTrace`]) plus the final [`Response`].
//! - An [`LlmTarget`] names a model. If it carries an [`LlmClient`] it *serves*
//!   its own calls; if not, the call is *offloaded* to the host (see below).
//! - A [`Switchyard`] drives requests through one algorithm (which owns
//!   its [`LlmTargetSet`]). Construct the algorithm, then wrap it with
//!   [`Switchyard::new`].
//!
//! ## Two ways to run a request
//!
//! - [`Switchyard::run_direct`] — when every target has a
//!   client, libsy makes all the calls itself and returns `(trace, response)`.
//!   The simplest integration; use it when libsy holds the model clients.
//! - [`Switchyard::run`] — returns a stream of
//!   [`Step`]s. When the algorithm calls a *client-less* target, the
//!   call is offloaded: the stream yields [`Step::CallLlm`] carrying
//!   promises; the host performs the real model call and fulfills each promise
//!   with [`CallLlmRequest::respond`]. When the algorithm finishes the stream
//!   yields [`Step::ReturnToAgent`] with the trace and final response.
//!   This "ask, don't call" mode lets a host that already owns its transport keep
//!   control of every network call.
//!
//! ## Concurrency
//!
//! [`Algorithm::process_request`] takes `&self` and the orchestrator holds
//! `Arc<dyn Algorithm>` with no lock, so a single orchestrator serves many requests
//! in parallel. Each [`run`](Switchyard::run) call gets
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
/// [`Request::raw_request`] when a host needs to forward it losslessly.
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
pub struct Request {
    /// The neutral request an algorithm routes.
    pub llm_request: LlmRequest,
    /// The original provider-shaped request body, if the host wants to forward it
    /// verbatim (e.g. a proxy preserving messages/params). libsy does not read it.
    pub raw_request: Option<serde_json::Value>,
    /// Correlation metadata carried through the request.
    pub metadata: Option<Metadata>,
}

/// Agentic-stack events fed to an algorithm out of band via
/// [`Switchyard::process_signals`] (e.g. tool results, budget updates).
///
/// A placeholder today; a stateful algorithm can begin consuming signals as the
/// enum grows without changing the orchestrator contract.
#[derive(Clone)]
pub struct Signals {}

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
pub struct Response {
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

/// The host-facing half of an offloaded model call.
///
/// Yielded to the host inside [`Step::CallLlm`]. The host reads the
/// request it should perform ([`get_request`](Self::get_request)) and the
/// decision behind it ([`get_decision`](Self::get_decision)), makes the real model
/// call, and fulfills the promise with [`respond`](Self::respond). That
/// unblocks the algorithm's [`LlmTarget::call`] on the other side.
pub struct CallLlmRequest {
    request: Request,
    decision: Option<Arc<dyn DecisionTrace>>,
    // Fulfilled exactly once — `respond` consumes `self` to send, so no `Option`.
    tx: oneshot::Sender<Result<Response, Box<dyn Error + Send + Sync>>>,
}

/// The algorithm-facing half of an offloaded model call.
///
/// Held inside a client-less [`LlmTarget::call`], which awaits
/// [`get_response`](Self::get_response) until the host fulfills the paired
/// [`CallLlmRequest`]. Host code normally does not touch this type directly.
struct CallLlmRequestRx {
    // Awaited exactly once — `get_response` consumes `self`, so no `Option` and no
    // memoized response.
    rx: oneshot::Receiver<Result<Response, Box<dyn Error + Send + Sync>>>,
}

impl CallLlmRequest {
    /// The model call the host should perform to fulfill this promise.
    pub fn get_request(&self) -> &Request {
        &self.request
    }

    /// The decision that led to this call, if the algorithm attached one.
    pub fn get_decision(&self) -> Option<&dyn DecisionTrace> {
        self.decision.as_deref()
    }

    /// Fulfill the promise with the caller's model-call result. Pass `Err(..)` to
    /// propagate a failed model call back to the algorithm that made the call.
    /// Consumes the promise: it can only be fulfilled once.
    /// respond -> rename
    pub async fn respond(
        self,
        result: Result<Response, Box<dyn Error + Send + Sync>>,
    ) -> Result<(), Box<dyn Error + Send + Sync>> {
        self.tx.send(result).map_err(|_| "Failed to send result")?;
        Ok(())
    }
}

impl CallLlmRequestRx {
    /// Await the response the host provides via [`CallLlmRequest::respond`].
    ///
    /// Returns the fulfilled response, propagates the host's `Err` if the model
    /// call failed, or errors if the promise was dropped unfulfilled. Consumes
    /// `self`: awaited exactly once.
    async fn get_response(self) -> Result<Response, Box<dyn Error + Send + Sync>> {
        // Outer error: the sender was dropped without fulfilling the promise.
        // Inner error: the caller's model call failed — propagate it to the algo.
        match self.rx.await {
            Ok(result) => result,
            Err(_) => Err("Failed to receive result".into()),
        }
    }
}

/// Create a paired promise for one offloaded model call: the [`CallLlmRequest`] is
/// handed to the host, the [`CallLlmRequestRx`] is awaited by the algorithm's target.
/// Used internally by the offload path; exposed for algorithms/hosts that build
/// their own offloading.
fn llm_promise(
    request: Request,
    decision: Option<Arc<dyn DecisionTrace>>,
) -> (CallLlmRequest, CallLlmRequestRx) {
    let (tx, rx) = oneshot::channel();
    (
        CallLlmRequest {
            request,
            decision,
            tx,
        },
        CallLlmRequestRx { rx },
    )
}

/// Per-request state the orchestrator threads to each [`LlmTarget::call`].
///
/// Its job is to carry the current request's offload channel, so a client-less
/// target hands its promise to *this* request's channel and never to another's.
/// Each [`run`](Switchyard::run) run builds its own
/// context and passes it (by reference) down through the algorithm — which is what
/// lets many requests run concurrently with no shared or global state. In
/// [`run_direct`](Switchyard::run_direct) (or when an
/// algorithm is called directly, e.g. in tests) the context carries no channel, so
/// an unexpected offload errors instead of going nowhere.
#[derive(Clone, Default)]
pub struct Context {
    // The current request's promise sender, or `None` when offloading is
    // unavailable (direct mode). Private: only the orchestrator populates it, and
    // only `LlmTarget::call` reads it to offload. Unbounded so an algorithm can
    // hand off a promise without awaiting channel capacity (it then awaits the
    // response); the driver drains it promptly.
    promise_tx: Option<tokio::sync::mpsc::UnboundedSender<CallLlmRequest>>,
}

/// One item in the stream returned by [`Switchyard::run`].
pub enum Step {
    /// The algorithm needs these model calls performed. The host fulfills each
    /// promise with [`CallLlmRequest::respond`]; only client-less targets
    /// produce these steps.
    CallLlm(Vec<CallLlmRequest>),
    /// The algorithm finished: its decision trace and the final response. This is
    /// the last step of a successful run.
    ReturnToAgent(Vec<Arc<dyn DecisionTrace>>, Response),
}

/// Performs the actual model call for a target. This is the one piece of I/O
/// `libsy` does not own — a host implements it over its own transport (HTTP SDK,
/// in-process model, mock). A target that carries an `LlmClient` *serves* its own
/// calls; one without offloads them (see [`Switchyard::run`]).
#[async_trait]
pub trait LlmClient: Send + Sync {
    /// Call the model named in `request.llm_request.model_name` with the given
    /// request, returning the model's response. An algorithm sets that name to
    /// the target it is routing to before the call, so the client always knows
    /// which model to hit.
    async fn call(&self, request: Request) -> Result<Response, Box<dyn Error + Send + Sync>>;
}

/// A named routing target, optionally backed by an [`LlmClient`].
///
/// An algorithm selects a target by its [`name`](Self::name) and calls it. With a
/// client, [`call`](Self::call) invokes it directly. Without one, `call` offloads
/// to the promise channel in the [`Context`] it is given — which exists
/// only inside an [`run`](Switchyard::run) run; a
/// client-less call made with a channel-less context (e.g. `run_direct`)
/// errors.
#[derive(Clone)]
pub struct LlmTarget {
    /// The routing name an algorithm selects this target by (a logical tier like
    /// `"strong"`, or the model id when they coincide).
    pub name: String,
    /// The provider model id the client actually calls (e.g. `"openai/gpt-4o"`).
    /// Set equal to `name` when the routing label *is* the model id.
    pub model: String,
    /// The client that serves calls, or `None` to offload them.
    pub llm_client: Option<Arc<dyn LlmClient>>,
}

impl LlmTarget {
    /// Whether this target can serve its own call (has a client). Used by
    /// [`LlmTargetSet::all_have_clients`] to decide if `run_direct` is safe.
    pub fn has_client(&self) -> bool {
        self.llm_client.is_some()
    }

    /// Perform (or offload) the model call, tagging it with the algorithm's
    /// `decision` so a host driving the stream can see *why* the call was made.
    ///
    /// A client-backed target serves the call directly; a client-less one offloads
    /// via `ctx`'s promise channel. An algorithm gets `ctx` from its
    /// `process_request` and passes it straight through.
    pub async fn call(
        &self,
        ctx: &Context,
        mut request: Request,
        decision: Option<Arc<dyn DecisionTrace>>,
    ) -> Result<Response, Box<dyn Error + Send + Sync>> {
        // Translate the routing name into the provider model id the client (or the
        // host fulfilling an offload) calls, so the algorithm can route by a label.
        request.llm_request.model_name = self.model.clone();
        match &self.llm_client {
            Some(client) => client.call(request).await,
            None => {
                // No client: offload via a promise on this request's channel,
                // attaching the decision so the orchestrator can surface it on its
                // stream. The context has no channel outside an run() run.
                let promise_tx = ctx.promise_tx.as_ref().ok_or_else(|| {
                    format!(
                        "target '{}' has no client and no offload channel",
                        self.name
                    )
                })?;
                let (tx, rx) = llm_promise(request, decision);
                promise_tx.send(tx).map_err(|_| "Failed to send promise")?;
                rx.get_response().await
            }
        }
    }
}

/// The set of targets an algorithm may route among. An algorithm is constructed
/// with one and picks targets by position ([`targets`](Self::targets)) or by name
/// ([`get_target`](Self::get_target)).
#[derive(Clone)]
pub struct LlmTargetSet {
    targets: Vec<LlmTarget>,
}

impl LlmTargetSet {
    /// Build a target set from a list of targets.
    pub fn new(targets: Vec<LlmTarget>) -> Self {
        Self { targets }
    }

    /// All targets in the set — e.g. for an algorithm to select among.
    pub fn targets(&self) -> &[LlmTarget] {
        &self.targets
    }

    /// Look up a target by name; errors if no target has that name.
    pub fn get_target(&self, name: &str) -> Result<LlmTarget, Box<dyn Error + Send + Sync>> {
        self.targets
            .iter()
            .find(|t| t.name == name)
            .cloned()
            .ok_or(format!("Target {} not found", name).into())
    }

    /// Whether every target can serve its own call — the precondition for
    /// [`Switchyard::run_direct`].
    pub fn all_have_clients(&self) -> bool {
        self.targets.iter().all(|t| t.has_client())
    }
}

/// A stateful optimization algorithm. `process_request` is called once per
/// request; inside it the algorithm makes as many `LlmTarget::call`s as it needs
/// (each may be served directly or offloaded), and returns a decision trace plus
/// the final response. `process_signals` feeds it agentic-stack events.
///
/// Both methods take `&self`, not `&mut self`: the orchestrator shares one
/// algorithm (`Arc<dyn Algorithm>`) across all requests and calls it concurrently,
/// so an algorithm is responsible for its own thread-safety. Stateless algorithms
/// (like the reference routers) get this for free; a stateful one must use
/// interior mutability (e.g. a `Mutex`/`RwLock`/atomics over just its own state)
/// rather than a coarse lock over the whole algorithm.
#[async_trait]
pub trait Algorithm: Send + Sync {
    /// Run one request to completion: make the model calls the algorithm decides
    /// on (via [`LlmTarget::call`]) and return the decision trace plus the final
    /// response. Called concurrently for many requests, so it takes `&self`. Pass
    /// `ctx` straight through to every [`LlmTarget::call`] — it carries the
    /// per-request offload channel; the algorithm never inspects it.
    async fn process_request(
        &self,
        ctx: &Context,
        request: Request,
    ) -> Result<(Vec<Arc<dyn DecisionTrace>>, Response), Box<dyn Error + Send + Sync>>;
    /// Feed the algorithm agentic-stack events (tool results, budgets, etc.). The
    /// reference algorithms ignore signals; a stateful algorithm updates its own
    /// (interior-mutable) state.
    async fn process_signals(&self, signals: Signals) -> Result<(), Box<dyn Error + Send + Sync>>;

    /// The target set this algorithm routes among, if it has one. Lets a host (or
    /// the orchestrator) introspect the candidate models — e.g. to check that every
    /// target can serve its own call before running `run_direct`.
    fn get_target_set(&self) -> &LlmTargetSet;
}

/// Drives one [`Algorithm`] over a [`LlmTargetSet`], the main entry point for a
/// host. Cheap to share (`Arc<dyn Algorithm>` internally, no lock) and safe to call
/// from many threads at once. Run requests with
/// [`run_direct`](Self::run_direct) (targets serve their own
/// calls) or [`run`](Self::run) (offloaded calls stream back for
/// the host to fulfill).
pub struct Switchyard {
    // Shared, lock-free: `process_request` takes `&self`, so one algorithm serves
    // every request concurrently. The target set lives on the algorithm; reach it
    // via `algo.get_target_set()`.
    algo: Arc<dyn Algorithm>,
}

impl Switchyard {
    /// Build an orchestrator over an already-constructed algorithm. The algorithm
    /// carries its own [`LlmTargetSet`] (see [`Algorithm::get_target_set`]);
    /// offloading is wired per request by the [`Context`], not here.
    pub fn new(algo: Arc<dyn Algorithm>) -> Self {
        Switchyard { algo }
    }

    /// Run a request as a stream of [`Step`]s.
    ///
    /// Use this when some targets are client-less (calls are offloaded). Drive the
    /// stream: on [`Step::CallLlm`], perform each promised model call
    /// and fulfill it with [`CallLlmRequest::respond`]; the run ends with a
    /// single [`Step::ReturnToAgent`] carrying the trace and response
    /// (or an `Err` item if the algorithm or a call failed). Every call gets its
    /// own promise channel, so concurrent `run` runs never interfere.
    pub fn run(
        &self,
        request: Request,
    ) -> impl futures::stream::Stream<Item = Result<Step, Box<dyn Error + Send + Sync>>> {
        let (stream_tx, stream_rx) = tokio::sync::mpsc::channel(10);
        // Per-request promise channel, carried in a fresh context: concurrent
        // requests get independent channels, so offloaded promises never cross
        // between requests. Nothing here is shared or locked, so many `run`
        // calls run in parallel.
        let (promise_tx, mut promise_rx) = tokio::sync::mpsc::unbounded_channel::<CallLlmRequest>();
        let ctx = Context {
            promise_tx: Some(promise_tx),
        };
        let algo = self.algo.clone();

        // One driver task races two things: forwarding each offloaded promise to
        // the caller as a `CallLlm` step, and the algorithm finishing (which
        // yields the final `ReturnToAgent`). `process_request` runs on its own task
        // — owning `ctx` so its targets offload to *this* request's channel —
        // because it blocks awaiting promise responses the caller only produces
        // after it receives the `CallLlm` step; the two must run concurrently, so a
        // single receiver-loop cannot do both.
        tokio::spawn(async move {
            let mut algo_handle =
                tokio::spawn(async move { algo.process_request(&ctx, request).await });

            let mut recv_closed = false;
            loop {
                tokio::select! {
                    maybe = promise_rx.recv(), if !recv_closed => match maybe {
                        Some(promise) => {
                            if stream_tx
                                .send(Ok(Step::CallLlm(vec![promise])))
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
                                Ok(Step::ReturnToAgent(trace, response))
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
    /// otherwise the algorithm may offload a call, and the channel-less context used
    /// here gives a promise nowhere to go, so this errors up front.
    pub async fn run_direct(
        &self,
        request: Request,
    ) -> Result<(Vec<Arc<dyn DecisionTrace>>, Response), Box<dyn Error + Send + Sync>> {
        if !self.algo.get_target_set().all_have_clients() {
            return Err(
                "Cannot run directly: some targets lack clients and require offloading".into(),
            );
        }
        // All targets have clients, so the algorithm never offloads a call. Run it
        // directly (no lock — `process_request` takes `&self`) with a channel-less
        // context, since no offload can occur.
        self.algo
            .process_request(&Context::default(), request)
            .await
    }

    /// Feed agentic-stack signals to the algorithm (see [`Algorithm::process_signals`]).
    pub async fn process_signals(
        &self,
        signals: Signals,
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
        async fn call(&self, request: Request) -> Result<Response, Box<dyn Error + Send + Sync>> {
            // Echo back the model the algorithm routed to (the target's name).
            Ok(Response {
                llm_response: LlmResponse {
                    completion: request.llm_request.model_name,
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
    impl Algorithm for TestAlgo {
        async fn process_request(
            &self,
            ctx: &Context,
            request: Request,
        ) -> Result<(Vec<Arc<dyn DecisionTrace>>, Response), Box<dyn Error + Send + Sync>> {
            let target = self
                .target_set
                .targets()
                .first()
                .ok_or("no targets")?
                .clone();
            let decision: Arc<dyn DecisionTrace> = Arc::new(TestDecision {
                model: target.name.clone(),
            });
            let response = target.call(ctx, request, Some(decision.clone())).await?;
            Ok((vec![decision], response))
        }

        async fn process_signals(
            &self,
            _signals: Signals,
        ) -> Result<(), Box<dyn Error + Send + Sync>> {
            Ok(())
        }

        fn get_target_set(&self) -> &LlmTargetSet {
            &self.target_set
        }
    }

    /// Build an orchestrator over `TestAlgo` for the given target set.
    fn orch(target_set: LlmTargetSet) -> Switchyard {
        Switchyard::new(Arc::new(TestAlgo { target_set }))
    }

    fn request() -> Request {
        Request {
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
            .map(|(name, has_client)| LlmTarget {
                name: name.to_string(),
                model: name.to_string(),
                llm_client: has_client.then(|| Arc::new(EchoClient) as Arc<dyn LlmClient>),
            })
            .collect();
        LlmTargetSet::new(targets)
    }

    #[tokio::test]
    async fn run_offloads_via_promise_then_returns_to_agent(
    ) -> Result<(), Box<dyn Error + Send + Sync>> {
        // A client-less target -> its call is offloaded via a promise the
        // orchestrator surfaces as a `CallLlm` step for us to fulfill.
        let orch = orch(target_set(&[("offload/model", false)]));

        let stream = orch.run(request());
        tokio::pin!(stream);

        let mut saw_call = false;
        let mut final_completion = None;
        while let Some(step) = stream.next().await {
            match step? {
                Step::CallLlm(promises) => {
                    saw_call = true;
                    for promise in promises {
                        // The decision rode along with the promise.
                        assert_eq!(
                            promise.get_decision().map(|d| d.model_decision()),
                            Some("offload/model")
                        );
                        // Fulfilling the promise is the "real" model call the caller makes.
                        promise
                            .respond(Ok(Response {
                                llm_response: LlmResponse {
                                    completion: "fulfilled".to_string(),
                                    raw_response: None,
                                },
                                metadata: None,
                            }))
                            .await?;
                    }
                }
                Step::ReturnToAgent(trace, response) => {
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
    async fn run_with_direct_client_returns_without_offload(
    ) -> Result<(), Box<dyn Error + Send + Sync>> {
        // A target with a client serves its own call, so no promise is emitted.
        let orch = orch(target_set(&[("direct/model", true)]));

        let stream = orch.run(request());
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
            Step::ReturnToAgent(trace, response) => {
                assert_eq!(trace[0].model_decision(), "direct/model");
                // EchoClient echoes the model name back as the completion.
                assert_eq!(response.llm_response.completion, "direct/model");
            }
            Step::CallLlm(_) => return Err("expected ReturnToAgent".into()),
        }
        Ok(())
    }

    #[tokio::test]
    async fn run_direct_returns_the_response_when_all_targets_have_clients(
    ) -> Result<(), Box<dyn Error + Send + Sync>> {
        // Every target has a client -> no call is ever offloaded, so we can skip
        // the stream and get the final response directly.
        let orch = orch(target_set(&[("direct/model", true)]));

        let (trace, response) = orch.run_direct(request()).await?;
        // TestAlgo calls the first target; EchoClient echoes its name.
        assert_eq!(response.llm_response.completion, "direct/model");
        assert_eq!(trace[0].model_decision(), "direct/model");
        Ok(())
    }

    #[tokio::test]
    async fn run_direct_errors_when_a_target_lacks_a_client(
    ) -> Result<(), Box<dyn Error + Send + Sync>> {
        // A client-less target could offload a call via a promise that nobody
        // would fulfill, so run_direct refuses up front.
        let orch = orch(target_set(&[("offload/model", false)]));

        assert!(orch.run_direct(request()).await.is_err());
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
        // would time out. It passes only because `run_direct` runs the
        // shared algorithm concurrently.
        struct BarrierClient {
            barrier: Arc<Barrier>,
        }

        #[async_trait]
        impl LlmClient for BarrierClient {
            async fn call(
                &self,
                request: Request,
            ) -> Result<Response, Box<dyn Error + Send + Sync>> {
                self.barrier.wait().await;
                Ok(Response {
                    llm_response: LlmResponse {
                        completion: request.llm_request.model_name,
                        raw_response: None,
                    },
                    metadata: None,
                })
            }
        }

        let barrier = Arc::new(Barrier::new(N));
        let targets = LlmTargetSet::new(vec![LlmTarget {
            name: "m".to_string(),
            model: "m".to_string(),
            llm_client: Some(Arc::new(BarrierClient {
                barrier: barrier.clone(),
            })),
        }]);
        // One orchestrator shared (by `&`) across many concurrent requests.
        let orch = Arc::new(orch(targets));

        let mut handles = Vec::new();
        for _ in 0..N {
            let orch = orch.clone();
            handles.push(tokio::spawn(async move {
                orch.run_direct(request())
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
        let orch = orch(target_set(&[("offload/model", false)]));

        let stream = orch.run(request());
        tokio::pin!(stream);

        let mut saw_error = false;
        while let Some(step) = stream.next().await {
            match step {
                Ok(Step::CallLlm(promises)) => {
                    for promise in promises {
                        promise
                            .respond(Err("upstream model call failed".into()))
                            .await?;
                    }
                }
                Ok(Step::ReturnToAgent(..)) => {
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
