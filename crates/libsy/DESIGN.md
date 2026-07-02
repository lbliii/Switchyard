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
    // `ctx` carries this request's offload channel; the algorithm just threads it
    // through to each `target.call`.
    async fn process_request(&self, ctx: &OrchestratorContext, request: OrchestratorRequest)
        -> Result<(Vec<Arc<dyn DecisionTrace>>, OrchestratorResponse), Box<dyn Error + Send + Sync>>;
    async fn process_signals(&self, signals: AgentSysSignals)
        -> Result<(), Box<dyn Error + Send + Sync>>;
    fn get_target_set(&self) -> &LlmTargetSet;   // targets this algo routes among
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
stream. `orchestrate_direct(&self, request)` is the shortcut for that case: it runs the algorithm and
returns `Result<(decision trace, final response)>`, skipping the stream. It errors up front if any
target lacks a client (which could otherwise offload a call that nobody fulfills, deadlocking).

## Targets and offloading

A target is a plain value — a name, the provider model id, and an optional client:

```rust
pub struct LlmTarget {
    pub name: String,                           // routing label the algorithm selects by ("strong")
    pub model: String,                          // provider model id the client calls ("openai/gpt-4o")
    pub llm_client: Option<Arc<dyn LlmClient>>, // `None` -> the call is offloaded
}

impl LlmTarget {
    // `ctx` carries the per-request offload channel a client-less target uses.
    pub async fn call(&self, ctx: &OrchestratorContext, request: OrchestratorRequest,
                      decision: Option<Arc<dyn DecisionTrace>>)
        -> Result<OrchestratorResponse, Box<dyn Error + Send + Sync>> { /* serve or offload */ }
}
```

An algorithm routes by **name** — a logical label like `"strong"`/`"weak"` (or a model id when they
coincide). Before the call reaches the client, the target stamps its **model** (the provider id) onto
`request.llm_request.model_name`, so `LlmClient::call` — which takes only the request — always knows
the concrete model to hit. Name and model let routing stay abstract while the client stays concrete.

- **`LlmTarget` with a client** serves its own call (`LlmClient::call`).
- **A client-less target** **offloads**: it makes a promise (`llm_promise`), sends it on the promise
  channel carried in the per-request `OrchestratorContext`, and awaits the response. The orchestrator
  surfaces the promise as a `CallLlm` step; the caller performs the real model call and fulfills it
  with `set_response(Ok(response))` — or `set_response(Err(..))` to propagate a failed call back into
  the algorithm. To the algorithm, `target.call` just returned (or errored) — the offload is invisible.

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
// Each algorithm's `new` takes its config plus the target set it routes among; it
// owns that set and exposes it via `OrchAlgo::get_target_set`.
let algo = Arc::new(LlmClassifierOrchAlgo::new(classifier, strong, weak, threshold, target_set));
let orch = MultiLlmOrchestrator::new(algo);   // just wraps the Arc<dyn OrchAlgo>
```

The algorithm owns its target set and selects among targets with `LlmTargetSet::targets()` /
`get_target(name)`. `MultiLlmOrchestrator::new` takes the constructed `Arc<dyn OrchAlgo>` directly —
there is no builder; offloading is wired per request by the `OrchestratorContext`, not at
construction.

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
  in an `OrchestratorContext` threaded (by reference) through `process_request` to each `target.call`,
  so concurrent requests never share a channel and their offloaded promises cannot cross — no global
  or task-local state.
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
