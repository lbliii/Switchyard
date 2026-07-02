# Switchyard-Lib Design

Status: design draft. The Rust crate in `src/` (`lib.rs` core, `rand.rs`, `llm_class.rs`,
`ensemble.rs`) is the design of record; the type names below match the code. The model is
language-independent — "trait" means interface/contract — but it is worked out against the Rust code.

## Summary

`libsy` is a lightweight library for multi-LLM agent optimization, with **routing** as the first
case. An algorithm sits where a request is about to hit a model and decides — statefully, using more
than the request — which model(s) to call and how.

The core never performs a model call itself. An algorithm makes normal-looking `target.call`s; a
target either **serves** the call (it has a client) or **offloads** it as a promise the caller
fulfills. That keeps the core provider- and transport-agnostic.

## The algorithm + the orchestrator

An `OrchAlgo` is called once per request; inside `process_request` it makes as many `target.call`s as
it needs (a router makes one; a classifier makes two — classify, then route) and returns a decision
trace plus the final response.

```rust
#[async_trait]
pub trait OrchAlgo: Send + Sync {
    // `&self`, not `&mut self`: one shared algorithm serves requests in parallel.
    async fn process_request(&self, request: OrchestratorRequest)
        -> Result<(Vec<Arc<dyn DecisionTrace>>, OrchestratorResponse), Box<dyn Error + Send + Sync>>;
    async fn process_signals(&self, signals: AgentSysSignals)
        -> Result<(), Box<dyn Error + Send + Sync>>;
}
```

A `MultiLlmOrchestrator` drives one algorithm and exposes each request as a **stream** of steps:

```rust
pub enum OrchestratorStep {
    CallLlm(Vec<LlmPromiseTx>),                                        // fulfill these offloaded calls
    ReturnToAgent(Vec<Arc<dyn DecisionTrace>>, OrchestratorResponse),  // done
}
// orchestrate(&self, request) -> impl Stream<Item = OrchestratorStepResult>
```

When every target has a client, no call is ever offloaded, so there is nothing to fulfill on the
stream. `orchestrate_direct(&self, request) -> (Vec<Arc<dyn DecisionTrace>>, OrchestratorResponse)`
is the shortcut for that case: it runs the algorithm and returns the decision trace plus the final
response, skipping the stream. It errors up front if any target lacks a client (which could
otherwise offload a call that nobody fulfills, deadlocking).

## Targets and offloading

A target is decision-agnostic and reused across requests:

```rust
#[async_trait]
pub trait LlmTargetI: Send + Sync {
    fn get_name(&self) -> &str;
    fn get_client(&self) -> Option<Arc<dyn LlmClient>>;
    async fn call(&self, request: OrchestratorRequest, decision: Option<Arc<dyn DecisionTrace>>)
        -> Result<OrchestratorResponse, Box<dyn Error + Send + Sync>>;
}
```

- **`LlmTarget` with a client** serves its own call (`LlmClient::call`).
- **A client-less target**, once wrapped by the orchestrator (`WrappedLlmTarget`), **offloads**: it
  makes a promise (`llm_promise`), sends it on the orchestrator's channel, and awaits the response.
  The orchestrator surfaces the promise as a `CallLlm` step; the caller performs the real model call
  and fulfills it with `set_response(Ok(response))` — or `set_response(Err(..))` to propagate a failed
  call back into the algorithm. To the algorithm, `target.call` just returned (or errored) — the
  offload is invisible.

The decision the algorithm attached rides along on the promise (`get_decision`), so the stream
consumer knows *why* each call is being made.

## Decisions are trait objects, not a generic

`DecisionTrace` is a trait object, not a `<D>` parameter. That lets a stream consumer inspect any
algorithm's decision uniformly — without knowing the concrete type — with a downcast escape hatch:

```rust
pub trait DecisionTrace: Send + Sync {
    fn model_decision(&self) -> &str;        // the model chosen
    fn reasoning(&self) -> Option<&str>;      // human-readable "why"
    fn as_any(&self) -> &dyn std::any::Any;   // downcast when the algo is known
}
```

Using a trait (not a generic) keeps `OrchAlgo`, `OrchestratorStep`, and the target non-generic, and
gives the consumer one uniform way to read decisions across algorithms.

## Construction

```rust
pub trait OrchAlgoBuilder: Send + Sync {
    fn with_target_set(&mut self, target_set: LlmTargetSet);
    fn build(&mut self) -> Box<dyn OrchAlgo>;
}
// MultiLlmOrchestrator::new(builder, Some(target_set)) wraps the targets (so offloaded calls reach
// its promise channel), hands them to the builder, and builds the algo.
```

The algorithm never constructs its own targets; the orchestrator wraps them and passes them via the
builder. An algorithm selects among targets with `LlmTargetSet::targets()` / `get_target(name)`.

## Reference algorithms

- **`RandomOrchAlgo`** (`rand.rs`) — one `target.call`: pick a target uniformly at random, call it.
- **`LlmClassifierOrchAlgo`** (`llm_class.rs`) — two `target.call`s: call the classifier target for a
  score, then call the strong/weak target. Fail-open — an unparseable score routes strong.
- **`EnsembleOrchAlgo`** (`ensemble.rs`) — **stateful**: fan out to several candidate models
  concurrently, call a judge to pick the best, and — after a configurable number of exploration turns
  — commit to the winningest model and route straight to it. Its win tally/turn counter live behind a
  `Mutex` (interior mutability over just its own state), the pattern the `&self` contract expects.

All three drive through the same orchestrator; an algorithm's extra rounds and state are its own
control flow, not orchestrator machinery.

## Examples

- **`examples/research_agent.rs`** — client-backed targets: they self-serve, so the agent uses
  `orchestrate_direct` (one call in, the response out). The simplest usage.
- **`examples/research_agent_core.rs`** — client-less targets: each call is offloaded, so the agent
  fulfills `CallLlm` promises with its own model calls. The offload/streaming path.
- **`demo/libsy-proxy`** (workspace crate) — a real HTTP proxy: switchyard's crates serve the
  OpenAI/Anthropic/Responses APIs and translate formats, while *all* routing is the classifier
  algorithm; libsy's targets make their upstream calls through switchyard's backend. Shows libsy
  embedded in a production-shaped I/O stack.

## Concurrency

- **Requests run in parallel.** `process_request`/`process_signals` take `&self`, and the
  orchestrator holds one shared `Arc<dyn OrchAlgo>` with no lock, so many threads can call
  `orchestrate` / `orchestrate_direct` at once. An algorithm is responsible for its own thread-safety
  — stateless (like the reference routers), or interior mutability over just its own state.
- **Per-request promise channel.** `orchestrate` makes a fresh channel per call and passes the sender
  to `process_request` via a task-local (`PROMISE_TX`), so concurrent requests never share a channel
  and their offloaded promises cannot cross.
- `process_request` runs on its own task so it can block awaiting a promise response while the driver
  forwards `CallLlm` steps to the caller — the driver `select!`s promise-forwarding against
  algorithm-completion.

## Not yet built

- **Signals** — `process_signals` / `AgentSysSignals` exist but carry no events yet (tool/task/budget/
  telemetry).
- **Observability** — spans + a metrics sink for token counts / timings / failures; `DecisionTrace`
  is the hook it will build on.
- **Config-driven construction** — a registry that builds an orchestrator from config.
- **Typed errors** instead of `Box<dyn Error + Send + Sync>`; a richer request (params, tools) beyond
  a single prompt.
- **Weighted random** — the random router is uniform over the target set today.
