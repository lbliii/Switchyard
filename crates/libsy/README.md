# libsy — Switchyard-Lib

A lightweight library for multi-LLM agent optimization, with **routing** as the first case. An
algorithm decides — statefully, using more than the request — which model(s) to call and how, and
never performs the call itself ("ask, don't call"), which keeps it provider- and transport-agnostic.
This README shows the scope through code; the narrative design is in [`DESIGN.md`](DESIGN.md).

## Core: algorithm + orchestrator

An `OrchAlgo` is run once per request and makes as many normal-looking `target.call`s as it needs; a
`MultiLlmOrchestrator` drives it and streams the result.

```rust
#[async_trait]
pub trait OrchAlgo: Send + Sync {
    // Makes target.call()s, returns a decision trace + the final response.
    // `&self` (not `&mut self`): one algorithm serves many requests concurrently.
    // `ctx` carries this request's offload channel; thread it into each target.call.
    async fn process_request(&self, ctx: &OrchestratorContext, request: OrchestratorRequest)
        -> Result<(Vec<Arc<dyn DecisionTrace>>, OrchestratorResponse), Box<dyn Error + Send + Sync>>;
    async fn process_signals(&self, signals: AgentSysSignals)
        -> Result<(), Box<dyn Error + Send + Sync>>;
}

// orchestrate(request) yields a stream of steps:
pub enum OrchestratorStep {
    CallLlm(Vec<LlmPromiseTx>),                                        // fulfill these offloaded calls
    ReturnToAgent(Vec<Arc<dyn DecisionTrace>>, OrchestratorResponse),  // done
}
```

## Targets: serve or offload

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

- `LlmTarget` **with** a client serves its own call.
- `LlmTarget` **without** a client is offloaded: it hands a promise to the per-request
  `OrchestratorContext`'s channel, which the orchestrator surfaces as a `CallLlm` step the caller
  fulfills. The algorithm sees a plain response either way.

## Decisions are trait objects (not a generic)

So a stream consumer can read *any* algorithm's decision uniformly, with a downcast escape hatch:

```rust
pub trait DecisionTrace: Send + Sync {
    fn model_decision(&self) -> &str;        // the model chosen
    fn reasoning(&self) -> Option<&str>;      // human-readable "why"
    fn as_any(&self) -> &dyn std::any::Any;   // downcast when the algo is known
}
```

## Implementing a router (LLM classifier, minimized)

Two `target.call`s inside one `process_request`: classify, then route. (Full version in
[`src/llm_class.rs`](src/llm_class.rs); the one-call random router is in [`src/rand.rs`](src/rand.rs),
and a stateful fan-out → judge → commit ensemble in [`src/ensemble.rs`](src/ensemble.rs).)

```rust
#[async_trait]
impl OrchAlgo for LlmClassifierOrchAlgo {
    async fn process_request(&self, request: OrchestratorRequest)
        -> Result<(Vec<Arc<dyn DecisionTrace>>, OrchestratorResponse), Box<dyn Error + Send + Sync>> {
        let user_prompt = request.llm_request.prompt.clone();

        // 1. Classify: call the classifier target for a score.
        let classifier = self.target_set.get_target(&self.classifier_model)?;
        let classify_req = /* OrchestratorRequest with preamble + user_prompt */;
        let score = classifier.call(classify_req, Some(classify_decision.clone())).await?
            .llm_response.completion.trim().parse::<f64>().ok();

        // 2. Route: strong if score >= threshold, else weak (fail open on None).
        let model = if score.map_or(true, |s| s >= self.threshold) { &self.strong_model } else { &self.weak_model };
        let routed = self.target_set.get_target(model)?;
        let response = routed.call(routed_req, Some(route_decision.clone())).await?;

        Ok((vec![classify_decision, route_decision], response))   // trace: classify + route
    }
    async fn process_signals(&self, _s: AgentSysSignals) -> Result<(), Box<dyn Error + Send + Sync>> { Ok(()) }
}
```

Build it with an `OrchAlgoBuilder`; the orchestrator hands the target set to `build`:

```rust
pub trait OrchAlgoBuilder: Send + Sync {
    fn with_target_set(&mut self, target_set: LlmTargetSet);
    fn build(&mut self) -> Box<dyn OrchAlgo>;
}
```

## Using it — self-serving targets (`orchestrate_direct`)

When every target has a client, no call is ever offloaded, so there is no stream to drive. Use
`orchestrate_direct` — one request in, the decision trace + final response out. Runnable:
[`examples/research_agent.rs`](examples/research_agent.rs).

```rust
let builder = Box::new(LlmClassifierOrchAlgoBuilder::new(CLASSIFIER, STRONG, WEAK, 0.5));
let orch = MultiLlmOrchestrator::new(builder, Some(client_backed_targets()));

// Runs the algorithm and returns (trace, response), skipping the stream.
// Errors up front if any target lacks a client (it could offload a call nobody fulfills).
let (trace, response) = orch.orchestrate_direct(request).await?;
```

## Using it — offloaded targets (research agent, core)

Client-less targets offload every call, so the agent fulfills each `CallLlm` promise with its own
model call. Runnable: [`examples/research_agent_core.rs`](examples/research_agent_core.rs).

```rust
while let Some(step) = stream.next().await {
    match step? {
        OrchestratorStep::CallLlm(promises) => {
            for mut p in promises {
                let response = call_model(p.get_request()).await;  // the "real" call the caller owns
                p.set_response(Ok(response)).await?;               // or Err(..) to propagate a failed call
            }
        }
        OrchestratorStep::ReturnToAgent(trace, response) => { /* … */ }
    }
}
```

## Building a target set

Both usages above take a target set. Implement `LlmClient` over your transport, wrap each model as an
`LlmTarget`, and collect them into an `LlmTargetSet` for an algorithm to route among:

```rust
use libsy::{LlmClient, LlmRequest, LlmResponse, LlmTarget, LlmTargetSet,
            OrchestratorRequest, OrchestratorResponse};
use std::sync::Arc;

// 1. A client that performs the actual model call over your own transport.
struct MyClient { http: reqwest::Client, base_url: String, api_key: String }

#[async_trait::async_trait]
impl LlmClient for MyClient {
    async fn call(&self, request: OrchestratorRequest)
        -> Result<OrchestratorResponse, Box<dyn std::error::Error + Send + Sync>> {
        let model = request.llm_request.model_name;   // the provider id, stamped by the target
        // POST {base_url}/chat/completions with { model, messages:[{user, prompt}] } ...
        let completion = /* extract the assistant text */;
        Ok(OrchestratorResponse {
            llm_response: LlmResponse { completion, raw_response: None },
            metadata: None,
        })
    }
}

// 2. One client, shared by every target that hits the same endpoint.
// `name` is the label an algorithm routes by; `model` is the provider id the
// client actually calls. They can differ ("strong" -> "openai/gpt-4o") or coincide.
let client = Arc::new(MyClient { /* .. */ }) as Arc<dyn LlmClient>;
let target = |name: &str, model: &str| LlmTarget {
    name: name.to_string(),
    model: model.to_string(),
    llm_client: Some(client.clone()),       // `None` instead -> the call is offloaded
};

// 3. The set an algorithm routes among (it selects targets by `name`).
let targets = LlmTargetSet::new(vec![
    target("strong", "openai/gpt-4o"),
    target("weak", "openai/gpt-4o-mini"),
]);
```

## A built-in client: `SwitchyardClient` (planned, optional feature)

Writing an `LlmClient` means doing your own HTTP and provider-format handling. For integrators on the
core lib who'd rather not, we plan to ship a **`SwitchyardClient`** — an `LlmClient` that maps in
Switchyard's provider **translations** (OpenAI / Anthropic / Responses) and talks to an upstream
endpoint for you. It will sit behind an optional Cargo feature, so the core crate stays
transport-agnostic and integrators who write their own client pay nothing for it. (The
[`demo/libsy-proxy`](../../demo/libsy-proxy) binary hand-rolls this today; `SwitchyardClient` will be
the built-in version.)

```toml
# Cargo.toml
libsy = { version = "0.1", features = ["switchyard-client"] }
```

```rust
use libsy::client::SwitchyardClient;   // behind the `switchyard-client` feature

// One client points at an OpenAI-compatible endpoint; it handles transport + translation.
let client = Arc::new(
    SwitchyardClient::openai("https://api.openai.com/v1", std::env::var("OPENAI_API_KEY")?)
) as Arc<dyn LlmClient>;

let target = |name: &str, model: &str| LlmTarget {
    name: name.to_string(),
    model: model.to_string(),
    llm_client: Some(client.clone()),
};

let targets = LlmTargetSet::new(vec![
    target("strong", "openai/gpt-4o"),
    target("weak", "openai/gpt-4o-mini"),
]);
```

## Not yet built

- **`Signal` events** — `process_signals` / `AgentSysSignals` exist but carry nothing yet (tool/task/budget/telemetry).
- **Observability** — spans + a metrics sink for token counts / timings / failures (`DecisionTrace` is the hook).
- **Config-driven construction** — a registry that builds an orchestrator from config.
- **Typed errors** instead of `Box<dyn Error + Send + Sync>`, and a richer request (params, tools) beyond a single prompt.
- **Weighted random** — the random router is uniform over the target set today.
