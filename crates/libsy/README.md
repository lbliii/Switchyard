# libsy — Switchyard-Lib

A lightweight library for multi-LLM agent optimization, with **routing** as the first case.
The library is provider agnostic and gives the user control of the llm api calls.

## Example

Build a target set, pick an algorithm, run a request:

```rust
use libsy::llm_class::LlmClassifierOrchAlgo;
use libsy::{LlmRequest, LlmTarget, LlmTargetSet, Request, Switchyard};
use std::sync::Arc;

// Targets the algorithm routes among, each backed by your LlmClient (see below).
let client = Arc::new(MyClient { /* .. */ }) as Arc<dyn LlmClient>;
let t = |name: &str| LlmTarget {
    semantic_name: name.into(), llm_client: Some(client.clone()),
};

let algo = Arc::new(LlmClassifierOrchAlgo::new(
    "classifier", "strong", "weak", 0.5,
    LlmTargetSet::new(vec![
        t("classifier"),
        t("strong"),
        t("weak"),
    ]),
));
let orch = Switchyard::new(algo);

let req = Request {
    llm_request: LlmRequest { inbound_model_name: "auto".into(), prompt: "explain tail latency".into() },
    raw_request: None, metadata: None,
};
let (trace, response) = orch.run_direct(req).await?;   // one call in, trace + response out
println!("routed to {}", trace.last().unwrap().selected_model());
```

Runnable: [`examples/research_agent.rs`](examples/research_agent.rs).

## Requests & responses

```rust
pub struct Request {
    pub llm_request: LlmRequest,                // normalized:  { inbound_model_name, prompt }
    pub raw_request: Option<serde_json::Value>, // original provider body, forwarded verbatim if present
    pub metadata: Option<Metadata>,             // correlation: session / agent / task / correlation_id / extra
}

pub struct Response {
    pub llm_response: LlmResponse,              // normalized:  { completion, raw_response? }
    pub metadata: Option<Metadata>,
}
```

## Building a target set / using a client

An `LlmTarget` pairs a routing `semantic_name` with an optional `LlmClient`. Mapping that
name to a provider model id is the client's job, not the algorithm's. Although `libsy`
provides a `SwitchyardClient` and a reference implementation, `LlmClient` is designed to be
implemented by the user.

```rust
struct MyClient { /* http client, base url, key */ }

#[async_trait::async_trait]
impl LlmClient for MyClient {
    async fn call(&self, routed: RoutedRequest)
        -> Result<Response, Box<dyn std::error::Error + Send + Sync>> {
        let model = routed.decision.selected_model();   // the routed target — map it to a provider id
        // routed.request.llm_request.inbound_model_name is the agent's original name (not a call target)
        // ... POST to your endpoint, read the completion ...
        Ok(Response { llm_response: LlmResponse { completion, raw_response: None }, metadata: None })
    }
}
```

A `RoutedRequest` bundles the `request` with the routing `decision`; the model to call is
`decision.selected_model()`, never a mutated request field. `semantic_name` is the label an
algorithm routes by; the client maps it to the id it calls —
they can differ (`"strong"` -> `"openai/gpt-4o"`) or coincide. A target with `llm_client: None` is **offloaded**
instead of served (streaming, below). The planned optional `SwitchyardClient` will provide an
`LlmClient` that maps in Switchyard's OpenAI / Anthropic / Responses translations, so you needn't
hand-roll one.

## The orchestrator — `Switchyard`

The entry point. Cheap to share (`Arc<dyn Algorithm>` inside, no lock) and safe to drive from many
threads at once:

```rust
impl Switchyard {
    pub fn new(algo: Arc<dyn Algorithm>) -> Self;   // the algorithm owns its target set

    // libsy holds every client: makes the calls, returns (trace, response).
    // Errors up front if any target is client-less.
    pub async fn run_direct(&self, request: Request)
        -> Result<(Vec<Arc<dyn Decision>>, Response), Box<dyn Error + Send + Sync>>;

    // "Ask, don't call": client-less targets stream back as `CallLlm` promises you
    // fulfill, then a final `ReturnToAgent`.
    pub fn run(&self, request: Request)
        -> impl Stream<Item = Result<Step, Box<dyn Error + Send + Sync>>>;

    pub async fn process_signals(&self, signals: Signals)   // out-of-band events
        -> Result<(), Box<dyn Error + Send + Sync>>;
}
```

## Streaming — you own the model calls (`run`)

Build the targets client-less (`llm_client: None`) and every `target.call` is offloaded: `run`
yields `CallLlm` promises you fulfill with your own transport, then a final `ReturnToAgent`. Runnable:
[`examples/research_agent_core.rs`](examples/research_agent_core.rs).

```rust
let stream = orch.run(request);
tokio::pin!(stream);
while let Some(step) = stream.next().await {
    match step? {
        Step::CallLlm(promises) => for p in promises {
            let model = p.get_decision().selected_model();  // which target to call
            let response = call_model(model, p.get_request()).await;   // your real call
            p.respond(Ok(response)).await?;                // or Err(..) to propagate a failure
        },
        Step::ReturnToAgent(trace, response) => { /* done */ }
    }
}
```

## Building an algorithm (`Algorithm`)

Implement `Algorithm` to add a strategy. `process_request` runs once per request and makes as many
`LlmTarget::call`s as it needs.

```rust
#[async_trait]
pub trait Algorithm: Send + Sync {
    // `&self` (not `&mut`): one algorithm serves requests concurrently — use interior
    // mutability for state. Thread `ctx` into every target.call (it carries the offload channel).
    async fn process_request(&self, ctx: &Context, request: Request)
        -> Result<(Vec<Arc<dyn Decision>>, Response), Box<dyn Error + Send + Sync>>;
    async fn process_signals(&self, signals: Signals)
        -> Result<(), Box<dyn Error + Send + Sync>>;
    fn get_target_set(&self) -> &LlmTargetSet;
}
```

Give it a `new(config.., target_set)` constructor and `Arc`-wrap it for `Switchyard::new` —
there is no builder. A `target.call` serves the call if the target has a client, else offloads it via
`ctx` — invisible to the algorithm. Attach a `Decision` to each call so a consumer (and the client
serving the call) can see *which* model and *why* (it's a trait object, read uniformly, downcast
when the algo is known):

```rust
pub trait Decision: Send + Sync {
    fn selected_model(&self) -> &str;          // the model chosen — the client's call target
    fn reasoning(&self) -> Option<&str>;       // human-readable "why"
    fn as_any(&self) -> &dyn std::any::Any;    // downcast to the concrete decision
}
```

Example — the LLM classifier (classify, then route; full version in
[`src/llm_class.rs`](src/llm_class.rs)):

```rust
#[async_trait]
impl Algorithm for LlmClassifierOrchAlgo {
    async fn process_request(&self, ctx: &Context, request: Request)
        -> Result<(Vec<Arc<dyn Decision>>, Response), Box<dyn Error + Send + Sync>> {
        // 1. Classify: ask the classifier target for a score.
        let classifier = self.target_set.get_target(&self.classifier_model)?;
        let score = classifier.call(ctx, classify_req, classify_decision.clone()).await?
            .llm_response.completion.trim().parse::<f64>().ok();

        // 2. Route: strong if score >= threshold, else weak (fail open on None).
        let model = if score.map_or(true, |s| s >= self.threshold) { &self.strong_model } else { &self.weak_model };
        let response = self.target_set.get_target(model)?
            .call(ctx, routed_req, route_decision.clone()).await?;

        Ok((vec![classify_decision, route_decision], response))
    }
    async fn process_signals(&self, _s: Signals) -> Result<(), Box<dyn Error + Send + Sync>> { Ok(()) }
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

- [`examples/research_agent.rs`](examples/research_agent.rs) — client-backed targets, `run_direct`
  (libsy makes the calls).
- [`examples/research_agent_core.rs`](examples/research_agent_core.rs) — client-less targets,
  `run` stream (the agent makes the calls).

**Demo proxy** — [`demo/libsy-proxy`](../../demo/libsy-proxy): a real HTTP proxy where switchyard's
crates serve the OpenAI / Anthropic / Responses APIs and translate formats, while *all* routing is
libsy's classifier calling upstream through switchyard's backend.

## Not yet built

- **`Signal` events** — `process_signals` / `Signals` exist but carry nothing yet.
- **Observability** — spans + a metrics sink (`Decision` is the hook).
- **Config-driven construction**, **typed errors** (vs `Box<dyn Error>`), **weighted random**.
