# libsy — Switchyard-Lib

A lightweight library for multi-LLM agent optimization, with **routing** as the first case.
The library is provider agnostic and gives the user control of the llm api calls.

## Example

Build a target set, pick an algorithm, run a request:

```rust
use libsy::llm_class::LlmClassifierOrchAlgo;
use libsy::{LlmRequest, LlmTarget, LlmTargetSet, MultiLlmOrchestrator, OrchestratorRequest};
use std::sync::Arc;

// Targets the algorithm routes among, each backed by your LlmClient (see below).
let client = Arc::new(MyClient { /* .. */ }) as Arc<dyn LlmClient>;
let t = |name: &str, model: &str| LlmTarget {
    name: name.into(), model: model.into(), llm_client: Some(client.clone()),
};

let algo = Arc::new(LlmClassifierOrchAlgo::new(
    "classifier", "strong", "weak", 0.5,
    LlmTargetSet::new(vec![
        t("classifier", "openai/gpt-4o-mini"),
        t("strong",     "openai/gpt-4o"),
        t("weak",       "openai/gpt-4o-mini"),
    ]),
));
let orch = MultiLlmOrchestrator::new(algo);

let req = OrchestratorRequest {
    llm_request: LlmRequest { model_name: "auto".into(), prompt: "explain tail latency".into() },
    raw_request: None, metadata: None,
};
let (trace, response) = orch.orchestrate_direct(req).await?;   // one call in, trace + response out
println!("routed to {}", trace.last().unwrap().model_decision());
```

Runnable: [`examples/research_agent.rs`](examples/research_agent.rs).

## Requests & responses

```rust
pub struct OrchestratorRequest {
    pub llm_request: LlmRequest,                // normalized:  { model_name, prompt }
    pub raw_request: Option<serde_json::Value>, // original provider body, forwarded verbatim if present
    pub metadata: Option<Metadata>,             // correlation: session / agent / task / correlation_id / extra
}

pub struct OrchestratorResponse {
    pub llm_response: LlmResponse,              // normalized:  { completion, raw_response? }
    pub metadata: Option<Metadata>,
}
```

## Building a target set / using a client

An `LlmTarget` pairs a routing `name` with a provider `model` id and an optional `LlmClient`.
Although `libsy` provides a `SwitchyardClient` and a reference implementation, `LlmClient`
is designed to be implemented by the user.

```rust
struct MyClient { /* http client, base url, key */ }

#[async_trait::async_trait]
impl LlmClient for MyClient {
    async fn call(&self, request: OrchestratorRequest)
        -> Result<OrchestratorResponse, Box<dyn std::error::Error + Send + Sync>> {
        let model = request.llm_request.model_name;   // provider id, stamped by the target
        // ... POST to your endpoint, read the completion ...
        Ok(OrchestratorResponse { llm_response: LlmResponse { completion, raw_response: None }, metadata: None })
    }
}
```

`name` is the label an algorithm routes by; `model` is the id the client calls — they can differ
(`"strong"` -> `"openai/gpt-4o"`) or coincide. A target with `llm_client: None` is **offloaded**
instead of served (streaming, below). The planned optional `SwitchyardClient` will provide an
`LlmClient` that maps in Switchyard's OpenAI / Anthropic / Responses translations, so you needn't
hand-roll one.

## The orchestrator — `MultiLlmOrchestrator`

The entry point. Cheap to share (`Arc<dyn OrchAlgo>` inside, no lock) and safe to drive from many
threads at once:

```rust
impl MultiLlmOrchestrator {
    pub fn new(algo: Arc<dyn OrchAlgo>) -> Self;   // the algorithm owns its target set

    // libsy holds every client: makes the calls, returns (trace, response).
    // Errors up front if any target is client-less.
    pub async fn orchestrate_direct(&self, request: OrchestratorRequest)
        -> Result<(Vec<Arc<dyn DecisionTrace>>, OrchestratorResponse), Box<dyn Error + Send + Sync>>;

    // "Ask, don't call": client-less targets stream back as `CallLlm` promises you
    // fulfill, then a final `ReturnToAgent`.
    pub fn orchestrate(&self, request: OrchestratorRequest)
        -> impl Stream<Item = OrchestratorStepResult>;

    pub async fn process_signals(&self, signals: AgentSysSignals)   // out-of-band events
        -> Result<(), Box<dyn Error + Send + Sync>>;
}
```

## Streaming — you own the model calls (`orchestrate`)

Build the targets client-less (`llm_client: None`) and every `target.call` is offloaded: `orchestrate`
yields `CallLlm` promises you fulfill with your own transport, then a final `ReturnToAgent`. Runnable:
[`examples/research_agent_core.rs`](examples/research_agent_core.rs).

```rust
let stream = orch.orchestrate(request);
tokio::pin!(stream);
while let Some(step) = stream.next().await {
    match step? {
        OrchestratorStep::CallLlm(promises) => for mut p in promises {
            let response = call_model(p.get_request()).await;   // your real call
            p.set_response(Ok(response)).await?;                // or Err(..) to propagate a failure
        },
        OrchestratorStep::ReturnToAgent(trace, response) => { /* done */ }
    }
}
```

## Building an algorithm (`OrchAlgo`)

Implement `OrchAlgo` to add a strategy. `process_request` runs once per request and makes as many
`LlmTarget::call`s as it needs.

```rust
#[async_trait]
pub trait OrchAlgo: Send + Sync {
    // `&self` (not `&mut`): one algorithm serves requests concurrently — use interior
    // mutability for state. Thread `ctx` into every target.call (it carries the offload channel).
    async fn process_request(&self, ctx: &OrchestratorContext, request: OrchestratorRequest)
        -> Result<(Vec<Arc<dyn DecisionTrace>>, OrchestratorResponse), Box<dyn Error + Send + Sync>>;
    async fn process_signals(&self, signals: AgentSysSignals)
        -> Result<(), Box<dyn Error + Send + Sync>>;
    fn get_target_set(&self) -> &LlmTargetSet;
}
```

Give it a `new(config.., target_set)` constructor and `Arc`-wrap it for `MultiLlmOrchestrator::new` —
there is no builder. A `target.call` serves the call if the target has a client, else offloads it via
`ctx` — invisible to the algorithm. Attach a `DecisionTrace` to each call so a consumer can see *why*
(it's a trait object, read uniformly, downcast when the algo is known):

```rust
pub trait DecisionTrace: Send + Sync {
    fn model_decision(&self) -> &str;         // the model chosen
    fn reasoning(&self) -> Option<&str>;       // human-readable "why"
    fn as_any(&self) -> &dyn std::any::Any;    // downcast to the concrete decision
}
```

Example — the LLM classifier (classify, then route; full version in
[`src/llm_class.rs`](src/llm_class.rs)):

```rust
#[async_trait]
impl OrchAlgo for LlmClassifierOrchAlgo {
    async fn process_request(&self, ctx: &OrchestratorContext, request: OrchestratorRequest)
        -> Result<(Vec<Arc<dyn DecisionTrace>>, OrchestratorResponse), Box<dyn Error + Send + Sync>> {
        // 1. Classify: ask the classifier target for a score.
        let classifier = self.target_set.get_target(&self.classifier_model)?;
        let score = classifier.call(ctx, classify_req, Some(classify_decision.clone())).await?
            .llm_response.completion.trim().parse::<f64>().ok();

        // 2. Route: strong if score >= threshold, else weak (fail open on None).
        let model = if score.map_or(true, |s| s >= self.threshold) { &self.strong_model } else { &self.weak_model };
        let response = self.target_set.get_target(model)?
            .call(ctx, routed_req, Some(route_decision.clone())).await?;

        Ok((vec![classify_decision, route_decision], response))
    }
    async fn process_signals(&self, _s: AgentSysSignals) -> Result<(), Box<dyn Error + Send + Sync>> { Ok(()) }
    fn get_target_set(&self) -> &LlmTargetSet { &self.target_set }
}
```

## Explore

**Reference algorithms** — implementations to read and route with:

- [`src/rand.rs`](src/rand.rs) — `RandomOrchAlgo`: uniform random over the set (one call).
- [`src/llm_class.rs`](src/llm_class.rs) — `LlmClassifierOrchAlgo`: classify, then route strong/weak;
  fail open to strong.
- [`src/ensemble.rs`](src/ensemble.rs) — `EnsembleOrchAlgo`: stateful — fan out to candidates, judge
  the best, commit to the winner after N exploration turns 

**Runnable examples** (`cargo run -p libsy --example <name>`):

- [`examples/research_agent.rs`](examples/research_agent.rs) — client-backed targets, `orchestrate_direct`
  (libsy makes the calls).
- [`examples/research_agent_core.rs`](examples/research_agent_core.rs) — client-less targets,
  `orchestrate` stream (the agent makes the calls).

**Demo proxy** — [`demo/libsy-proxy`](../../demo/libsy-proxy): a real HTTP proxy where switchyard's
crates serve the OpenAI / Anthropic / Responses APIs and translate formats, while *all* routing is
libsy's classifier calling upstream through switchyard's backend.

## Not yet built

- **`Signal` events** — `process_signals` / `AgentSysSignals` exist but carry nothing yet.
- **Observability** — spans + a metrics sink (`DecisionTrace` is the hook).
- **Config-driven construction**, **typed errors** (vs `Box<dyn Error>`), **weighted random**.
