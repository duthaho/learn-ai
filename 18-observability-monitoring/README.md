# Module 18: Observability & Monitoring

**What you'll learn:**
- Why LLM applications need observability that classical APM cannot provide
- The three signals (logs, metrics, traces) and why traces are load-bearing for LLM apps
- The span schema that captures what matters: tokens, cost, prompt hash, finish reason, errors
- Nested spans -- why one logical user request becomes 5-15 LLM calls and how parent-child grouping rescues the debugging story
- Cost telemetry as a first-class signal -- cost per run, cost per user, cost regressions
- Sensitive data in traces -- default-redact policies, hash-not-store, the M16 cross-link
- Failure modes -- sampling decisions, async export, distributed-trace boundaries, alerting (deferred to Module 20)
- The ecosystem -- Langfuse, Arize Phoenix, Helicone, Datadog LLM Observability, LangSmith, OTel GenAI semantic conventions

| Detail        | Value                                                                                                |
|---------------|------------------------------------------------------------------------------------------------------|
| Level         | Intermediate--Advanced                                                                               |
| Time          | ~3.5 hours                                                                                           |
| Prerequisites | Module 04 (AI API Layer), Module 11 (Building AI Agents), Module 15 (Evaluation & Testing), Module 17 (Caching & Cost Optimization) |

---

## Table of Contents

1. [Why LLM Observability Is Different From APM](#1-why-llm-observability-is-different-from-apm)
2. [The Three Signals: Logs, Metrics, Traces](#2-the-three-signals-logs-metrics-traces)
3. [What Goes In an LLM Span](#3-what-goes-in-an-llm-span)
4. [Nested Spans: The Agent / Chain Case](#4-nested-spans-the-agent--chain-case)
5. [Cost Telemetry as a First-Class Citizen](#5-cost-telemetry-as-a-first-class-citizen)
6. [Sensitive Data in Traces](#6-sensitive-data-in-traces)
7. [Failure Modes & What This Module Doesn't Cover](#7-failure-modes--what-this-module-doesnt-cover)
8. [The Ecosystem](#8-the-ecosystem)

---

## 1. Why LLM Observability Is Different From APM

Classical APM -- Datadog, New Relic, Dynatrace -- was designed for a world where the unit of work is an HTTP request, the input is a JSON body, the output is a status code plus a JSON body, and the interesting question is whether the request was fast and whether it succeeded. That design fits almost every web service built in the last twenty years: the service receives a request, queries a database, calls a downstream API, serializes a response, and returns. If something went wrong, you read the database query from the trace, replay it, and reproduce the behavior. The trace is a perfect audit of the computation because the computation is deterministic given its inputs and the inputs are compact and self-contained.

LLM applications break this model on four dimensions, each of which requires the observability layer to capture something that classical APM never had to think about. Understanding each dimension is not academic table-setting -- it is why the project in this module ships a custom trace recorder rather than a wrapper around an existing APM agent.

### The prompt/response dimension

An HTTP span in Datadog captures `POST /api/v1/completions`, the request duration, and the HTTP response code. A 200 and a 2.3-second duration. That tells you the call succeeded and how long it took. It tells you nothing about what was actually computed.

An LLM span has to capture the *input* -- the rendered system prompt, the full messages list, any retrieved context stuffed into the user turn -- because that input *is* the computation in a way that a route path is not. There is no SQL string to read back and replay. There is no RPC payload to retry. The prompt is the complete specification of what the model was asked to do, and without the prompt in the trace, the question "why did the model say that?" has no tractable answer. A classical APM span showing `llm.completion` and a 3.2-second duration tells you the operation name and the latency. It tells you nothing that matters. A span with the full rendered system prompt, the messages list, and the response text tells you everything.

This is a storage problem that classical APM tooling was not designed to absorb. A Postgres span carries a query string, maybe 500 bytes. An LLM span carries the rendered prompt, which can be 2,000-10,000 bytes of system instructions, few-shot examples, retrieved documents, and conversation history, plus the response, which can be another 500-3,000 bytes. At a million requests per month, that is gigabytes of trace data, and managing that storage (the hash-not-store policy in Section 6, the retention window, the preview truncation) is part of the observability design, not an operational afterthought.

### The cost dimension

A Postgres query costs micro-cents on the AWS bill, distributed across millions of queries in a monthly aggregate that only finance looks at. An LLM call costs between $0.001 and $0.05 per invocation, and the exact cost is returned in the API response the moment the call completes. The usage object in the provider's response carries input token count and output token count; multiply those counts by the published per-token rate and you have the precise dollar cost of the call, attributable to a specific span, available the moment the response returns.

A customer-support assistant answering 500 questions per hour is spending between $0.50 and $25.00 per hour on inference alone -- before infrastructure, before developer time, before support. The per-call cost varies with the length of the system prompt, the retrieved context, the response, and the model tier, and the variance is large. A system that routes simple queries to Haiku and complex ones to Sonnet has a bimodal cost distribution; a system that stuffs 8,000 tokens of retrieved context into every prompt has a much higher baseline cost than one that retrieves only 2,000 tokens. None of this variation is visible in a classical APM trace. Cost-per-call is a first-class telemetry signal that sits alongside latency and error rate, and capturing it requires storing the token counts and the computed cost in the span the moment the response returns.

### The non-determinism dimension

A given HTTP request with a given JSON body produces the same response on every retry, within a narrow envelope of infrastructure variance. A caching layer can safely serve the same response for the same request because the response is deterministic. A given LLM call with the same prompt at any temperature above zero produces a *different* response on every call. The input bounds the distribution but does not fix the sample.

Classical APM is built on the assumption that "reproduce from the logs" is possible because the input fully determines the output. In LLM applications, that assumption is false. A bug where the model occasionally produces the wrong output cannot be reproduced by replaying the request; it can only be investigated by reading what the model actually said in the failing call. The response text is a first-class field in the span, not an artifact you can omit and reconstruct. The trace that stores the model's actual output is the only evidence you have of what happened.

This is a quality-analysis requirement that classical APM never had. In a deterministic web service, you debug by replaying. In an LLM application, you debug by reading the trace -- the actual prompt that was sent, the actual response that was received, the actual finish reason, the actual token counts. Every field in the span schema exists because of this requirement: without the prompt, you cannot understand what was asked; without the response, you cannot understand what was answered; without the finish reason, you cannot understand why generation stopped.

### The "agent took 7 steps" dimension

A web application's request span covers a shallow tree: one HTTP handler, one or two database calls, one cache lookup, one downstream API call. The tree is four levels deep at most and usually three. The causal chain within a single request is short, predictable, and rarely interesting as a structure -- what matters is whether each step succeeded and how long it took, not the relationship between steps.

An agent answering a user question runs a retrieve-then-decide loop, calls one or more tools, reflects on tool results, possibly loops back for another retrieval, and produces a final answer. That is 5-15 LLM calls under the hood for a single user-visible request, with the exact count depending on how many retrieval cycles the agent runs and how many tools it calls. A document-summarization chain runs a map call over each retrieved chunk and a reduce call over the summaries -- three or more LLM invocations per user request, each with its own prompt, its own token counts, its own cost, its own possible failure mode.

Flat per-call logs leave you with 30 interleaved lines from three concurrent users and no way to ask "which calls belong to user A?" You can filter on a timestamp range, but agent loops are slow (seconds per call) and the ranges overlap across concurrent users. You can sort by cost, but cost does not identify the parent request. You can search for a specific prompt fragment, but that finds only one of the seven calls that belong to user A's request, not the other six. Flat logs cannot answer "what did user A's agent actually do?" That question requires traces -- nested span trees -- and the nested-span pattern is the core engineering contribution of this module.

### Why "observable" means something specific here

The phrase "observable system" in classical APM means: given the outputs of the system (metrics, logs, request counts), can you infer the internal state well enough to debug a failure? For a deterministic web service, the answer is usually yes -- the logs tell you what function was called with what arguments, and you can replay the call. For a non-deterministic LLM application, the internal state *is* the stochastic sample the model drew from its output distribution, and that state is only accessible by having stored the actual output text.

An observable LLM system is one where the following questions have tractable answers without reading source code or replaying traffic:

- Why did the model say X on this specific call? (requires the prompt and the response in the span)
- What did this agent invocation cost, and which step drove the cost? (requires `cost_usd` on children and `total_cost_usd` on the run span)
- Did the per-run cost change after Tuesday's deploy? (requires `total_cost_usd` indexed by `started_at`)
- Which model calls finished with `finish_reason="length"` this week? (requires `finish_reason` in the span)
- What percentage of agent invocations errored, and at which step? (requires `status` on both run and llm spans)

A system that cannot answer these questions from its trace data is not observable, regardless of how many dashboards it has. The span schema in Section 3 is designed to make every one of these questions a structured query, not a log-file archaeology project.

### Why this module ships a trace recorder, not a logger

All four dimensions converge on the same design requirement: the observability layer must capture the prompt, the response, the token counts, the cost, and the parent-child relationships between calls as first-class structured fields, not as afterthoughts in a log line. A logger that appends "called claude-sonnet, status 200, 2.4s" to a file is not observability for an LLM application. A trace recorder that writes a structured JSON span with every field the Section 3 schema specifies is. The gap between these two is not a gap in tooling quality -- it is a gap in what the tool was designed to answer. That is why this module builds from scratch rather than wrapping an existing logger.

---

## 2. The Three Signals: Logs, Metrics, Traces

Observability practitioners organize signals into three categories. Every monitoring primer covers them in the same order: logs, metrics, traces. The framing applies to LLM workloads, but the weight of each category shifts in ways that are not obvious until you try to debug an agent failure using only logs and metrics and discover that the causal structure you needed was never captured.

### Logs

Logs are discrete, timestamped records of events. A log line is emitted when something happens: a request arrives, an error is raised, a guardrail fires, a tool call returns, a cache hit occurs. Logs are cheap to write, easy to emit from any point in the stack, and good at capturing the *what* of individual events. A well-structured log line is a compact, atomic record of fact: at time T, thing X happened with properties Y and Z.

Their structural weakness is that they carry no inherent link between one event and another. A log file is a flat sequence of lines, and the causal chain between events is implicit in the order and in the correlation IDs that the developer remembered to include. When a log contains three concurrent agent runs, the lines interleave by timestamp, and grouping them back into the three logical invocations requires string-searching on IDs -- if the IDs were added. If they were not, the events are permanently mixed and cannot be reconstructed without reading the whole file and reasoning from timestamps and prompt shapes.

In LLM applications, logs are the right medium for hard facts you cannot afford to lose: `llm_call_failed` with the error message and the span ID, `guardrail_blocked` with the block reason, `tool_call_timeout` with the tool name and the timeout threshold, `cache_hit` with the hit type and the lookup latency. These are atomic events that belong in a log. They are not the primary debugging surface for complex agent behavior; they are the safety net for events that must be captured regardless of whether the trace layer is functioning.

A concrete illustration of the gap: an agent fails on its third tool call. The log contains three lines: `llm_call: ok (2.1s)`, `llm_call: ok (3.4s)`, `llm_call: error (RateLimitError, 1.2s)`. The log tells you three calls happened and the third failed. It does not tell you the three calls were part of the same agent invocation, what the agent was trying to do, what the total cost was before the failure, or which user triggered the run. All of that context requires the trace. The log line that says `RateLimitError` is still useful -- it is the starting point for "what went wrong?" -- but the trace is what answers "what was the agent doing when it went wrong, and at what cost?"

### Metrics

Metrics are numeric aggregates computed over time windows. A counter (`llm_calls_total`), a histogram (`llm_call_duration_ms`), a gauge (`cache_entries`), a running sum (`cost_usd_total`) -- these are the signals that feed dashboards and alerting rules. Metrics are compact, cheap to store at any scale, and good at answering trend questions that span minutes to months.

Their structural weakness is that they throw away all per-call detail in the aggregation step. You know the p99 latency was 8.3 seconds yesterday and is 12.1 seconds today, but the metric alone does not tell you which requests were slow, which model calls contributed to the slow ones, what the prompts looked like, or whether the slowdown is concentrated in a specific agent step or spread across all of them. The aggregate is the signal; the individual events that produced it are gone. Metrics tell you *that* something changed; only traces can tell you *why*.

For LLM applications, metrics are the right medium for dashboards and SLO burn rates: `llm_error_rate`, `llm_calls_per_minute`, `cost_per_run_p95`, `cache_hit_rate`. They are not the right medium for incident investigation, quality attribution, or cost debugging at the per-invocation level. A useful operational pattern: derive metrics from trace data rather than instrumenting them separately. If every LLM span carries `cost_usd`, `tokens_in`, `tokens_out`, `duration_ms`, and `status`, then a daily job that reads the trace file and computes histograms, counters, and gauges from those fields gives you all the metrics you need without maintaining a separate instrumentation layer. The trace is the ground truth; the metrics are summaries of it.

### Traces

Traces are records of the causal chain within a single unit of work. A trace is a tree of spans; each span is a timed, structured record of one step in the computation; the tree shape encodes parent-child relationships and lets you follow the flow of time and cost through the full call graph, from the user's initial request to the model's final response, through every intermediate step in between.

Traces are the most expensive signal to collect and store. In classical web applications they matter primarily at the margins of the request lifecycle -- the slow database call buried in the middle of an otherwise-fast request, the downstream API that added 400ms to the p99. The tree is shallow, the steps are fast, and logs and metrics together answer most of the interesting questions.

LLM applications invert this priority order completely. The causal chain within a single user request is the *most* interesting object in the system -- more interesting than the aggregate trends, more interesting than the individual log events. Which prompts ran, in what order, at what cost, with what results, where did the agent branch, where did it fail and recover, why did this invocation cost $0.12 when the typical invocation costs $0.003? These questions are only answerable from the trace. Logs have the individual facts but not the relationships. Metrics have the aggregates but not the per-invocation detail. Traces have both.

### The same event, seen through three lenses

A single failing LLM call looks different through each signal type, and the comparison makes the priority ordering concrete.

**The log view.** A structured log line for the same event:

```json
{
  "timestamp": "2024-05-20T14:23:06.841Z",
  "level": "error",
  "event": "llm_call_failed",
  "model": "anthropic/claude-sonnet-4-20250514",
  "error": "RateLimitError: max_tokens exceeded",
  "duration_ms": 1302
}
```

This tells you the call failed, with which error, at what time. It does not tell you what the call was trying to do, what it cost before failing, which user triggered it, or which agent run it belongs to.

**The metric view.** A counter increment: `llm_calls_total{status="error", model="anthropic/claude-sonnet-4-20250514"} += 1`. This tells you errors are happening. It does not tell you anything about the specific call.

**The trace view.** The span from Section 3's example, with `status="error"` and the `finish_reason`, the `cost_usd`, the `tokens_in`, the `parent_id` linking to the agent run. This tells you the call, what it cost, what step it was, and which user invocation it belongs to.

The log gives you the fact. The metric gives you the trend. The trace gives you the story. All three are useful; the story is what debugging requires.

### The inversion in investment priority

Most web application teams ship logs first, metrics second, and traces as a finishing touch. Traces are added once the application is well-understood and the team wants visibility into the deeper structure of request handling. The logs are sufficient for debugging most failures because the causal chain is short and reproducible.

The right investment order for LLM applications is the reverse: traces first, logs and metrics as complements to the trace data. The argument is practical. A well-designed trace gives you logs (each span is a structured event record with a timestamp and all relevant fields) and the basis for metrics (aggregate the numeric span fields across all spans in a time window to get call counts, cost totals, error rates). A logging-first approach gives you the flat stream with no causal structure, and retrofitting parent-child IDs onto an existing log stream is the first step toward rebuilding the tracer that this module ships from the start.

The practical test: if you are shipping a RAG chain today and add structured logging, you will have per-call log lines with model names, durations, and token counts. Six months later, when you need to answer "which agent invocations from last week cost more than $0.10, and which step drove the cost?", you will discover that the log lines do not connect to each other -- you have per-call records but not per-invocation records. Adding the connection at that point is the first step toward implementing the tracer in this module. Building the tracer from the start avoids the retrofit.

---

## 3. What Goes In an LLM Span

The span schema is the contract between the tracer and the system reading the traces. Get the schema right and every debugging session, cost analysis, and quality investigation is a structured query over well-defined fields -- the kind you run in a tool's UI, not in your head. Get it wrong and every investigation requires reading raw text and reconstructing what happened by hand, which is the logging-only failure mode dressed up with extra storage costs.

This section walks the schema field by field, using a concrete single-call example: a `kind="llm"` span wrapping one `client.messages.create()` call inside an agent loop. The span is a JSON object. Every field is present on every span of the matching kind, with consistent types, and the schema does not change between runs unless the schema version changes.

### Identity: span_id, parent_id, run_id

Three identity fields give every span a position in the trace tree and make the tree recoverable from a flat list of span records stored in a JSONL file.

`span_id` is a UUID (e.g., `"7f3c2a1b-4e8d-4b9a-8f2c-1a3b5c7d9e0f"`) generated at span creation time. It is the unique identifier for this specific invocation. No two spans ever share a `span_id`, including spans from the same model, the same agent run, or the same user session.

`parent_id` is the `span_id` of the enclosing span in the call tree. A top-level run span has `parent_id = null`; every child span carries the `span_id` of its parent. In a three-level tree -- run span, retrieval child, generate grandchild -- the generate span's `parent_id` points at the retrieval span, and the retrieval span's `parent_id` points at the run span.

`run_id` is the `span_id` of the root span of the whole invocation. The root span sets `run_id = span_id`; every descendant copies the root's `run_id` at creation time. This shortcut matters operationally: without `run_id`, finding all spans belonging to a single user request requires a traversal from each leaf up to the root, which is multiple index lookups or a recursive query on the JSONL file. With `run_id`, a single filter -- `[s for s in spans if s["run_id"] == target_run_id]` -- returns the complete tree in one pass. The cost is one extra UUID per span; the payoff is that the most common trace-debugging query -- "show me everything that happened in this user's request" -- is a single equality filter instead of a graph traversal.

### Labels and timing: kind, name, started_at, ended_at, duration_ms

`kind` classifies the span's role in the application's call graph. The teaching project uses two values: `"run"` for a span wrapping a logical unit of orchestration work (the agent loop, the RAG pipeline, the chain), and `"llm"` for a span wrapping a single model call. Production systems extend this with `"tool"` for spans wrapping tool executions, `"retrieval"` for vector-search or keyword-search calls, `"rerank"` for reranking steps, and `"chain"` for named pipeline stages. The extension is additive: `kind` is a string, and the consuming system can filter on values it knows about and group unknown values under "other".

`name` is a human-readable label for the specific operation within its kind: `"agent_loop"`, `"tool_call:web_search"`, `"completion:decide_tool"`, `"retrieval:vector_db"`. The name is what the on-call engineer reads when scanning a list of spans in a trace viewer. A good name uses the application's vocabulary, not the infrastructure's: `"completion:synthesize"` says what the call was *for*; `"anthropic.messages.create"` says only which SDK method was called. Both are true; only the first is useful when reading 30 spans in sequence.

`started_at` and `ended_at` are Unix timestamps in float seconds: `1716217384.293`. `duration_ms` is the derived value: `(ended_at - started_at) * 1000`. Storing the derived value avoids recomputing it in every downstream query and makes the field immediately readable when a human eyeballs a JSONL trace file. Float-seconds Unix timestamps are timezone-free, sort lexicographically, and are natively handled by every data platform from SQLite to BigQuery.

### Status and error handling

`status` is either `"ok"` or `"error"`. `error` is the string representation of the exception when `status == "error"`, and null otherwise. The error string carries the exception class and message -- `"anthropic.RateLimitError: Request rate limit exceeded. Retry in 45s."` -- but not the full traceback. Tracebacks are long, often redundant across retries, and belong in the log. The span's `error` field is the compressed form that lets a downstream filter -- `WHERE status = 'error'` -- find all failures and read the failure reason without opening a separate log file.

Status propagation on parent spans is a deliberate design choice. A run span's status is `"error"` if any of its child spans errored, even if the agent recovered from the failure and ultimately produced a successful answer. This is the conservative choice: the run is marked failed if anything inside it failed. The alternative -- propagate only unrecovered errors -- is more precise but requires the tracer to know whether the application treated the error as fatal, which breaks the tracer's separation of concerns. Mark the run errored; let the application layer record the recovery in a separate log event. The on-call engineer can see both the error in the run span and the successful final response in the application log.

### LLM-specific attributes

These fields appear under an `attributes` key on spans with `kind="llm"`. They capture what makes an LLM call different from every other kind of remote call.

**model.** The full model identifier including the provider prefix: `"anthropic/claude-sonnet-4-20250514"`. Not `"claude-sonnet"` and not `"claude"`. The provider prefix matters when the application routes to multiple providers via LiteLLM (as [Module 04 (AI API Layer)](../04-ai-api-layer/) does), because a query "which provider had a higher error rate last week?" is unanswerable if `model` carries only the model family. The version suffix matters for debugging quality regressions after a provider silently updates a model behind a stable display name.

**prompt_hash and prompt_preview.** `prompt_hash` is the SHA-256 of the canonicalized messages list: the list serialized to JSON with sorted keys, then hashed to a hex digest. `prompt_preview` is the first 80 characters of the serialized string concatenated with `" ... "` and the last 80 characters, or the full string if its total length is under 160 characters. Both fields travel together in the span because they serve different purposes.

The hash is for machine-readable operations: dedup ("have we seen this exact prompt configuration before?"), change detection ("the prompt hash changed on all tool-decide spans after Tuesday's deploy"), and grouping ("find every invocation that used this system prompt hash"). The preview is for human-readable operations: the on-call engineer scanning spans can see at a glance what the prompt was about -- the first 80 characters typically contain the system role and the start of the instruction, and the last 80 characters often contain the user's actual question.

Using only the hash makes spans opaque to humans reading a trace file. Using only the full text makes spans expensive to store (potentially 10KB per span) and creates the PII risk addressed in Section 6. The hash-plus-preview combination costs roughly 250 bytes per span rather than 10KB, gives machine-readable dedup, and gives human-readable context.

**prompt_chars and response_chars.** The full character lengths of the input and output, present even when the preview is truncated. These are cheap to compute and serve as a sanity check on the token counts: a `response_chars` of 12 on a span that was supposed to produce a paragraph-length summary is the first hint that the response was truncated or that the model was content-filtered.

**tokens_in and tokens_out.** The authoritative token counts from the provider's usage object in the API response. These are what the provider billed; they are not estimated from the character count. A call with 2,100 input tokens and 320 output tokens at current Sonnet input and output rates costs approximately $0.009. Store the raw counts and compute the cost from them; do not store only the cost, because the token counts are needed separately for debugging. A `tokens_in` of 8,500 on a span that should have a 600-token prompt means retrieved context leaked or the conversation history was not being trimmed at the right boundary.

**cost_usd.** The result of `litellm.completion_cost()` called immediately after the API response returns, passing the response object and the model identifier. LiteLLM maintains a rate table for all major providers and models; for most models it computes the exact cost from the token counts and the published per-token rates. When LiteLLM does not have a rate for the model -- a new model, a private deployment, a fine-tuned variant -- `cost_usd` defaults to 0.0. The fallback is explicit and non-fatal. A tracer that raises an exception because it cannot compute a cost is a tracer that crashes the application; 0.0 is the right sentinel, and a `cost_usd` of 0.0 with a non-zero token count is a queryable signal that the rate is missing.

**finish_reason.** The value the provider returns to explain why generation stopped: `"stop"` for a normal completion, `"length"` for a max-tokens truncation, `"tool_calls"` when the model decided to invoke a tool rather than produce a final text response, `"content_filter"` when the provider's own safety layer blocked the output. Finish reason is the first field to check when an agent behaves unexpectedly. A `"length"` on the tool-decide step means the model's chain-of-thought was truncated mid-reasoning, which explains the missing or malformed tool call that follows. A `"content_filter"` on a prompt that looks benign means the provider's moderation fired, which is the kind of signal the application-side guardrails in [Module 16 (AI Safety & Guardrails)](../16-ai-safety-guardrails/) should have been designed to either catch upstream or handle gracefully downstream.

### A concrete span in JSONL

Putting the schema fields together, a single LLM span in the teaching project's JSONL format looks like this:

```json
{
  "span_id": "7f3c2a1b-4e8d-4b9a-8f2c-1a3b5c7d9e0f",
  "parent_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "run_id":    "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "kind":      "llm",
  "name":      "completion:decide_tool",
  "started_at": 1716217384.293,
  "ended_at":   1716217386.841,
  "duration_ms": 2548.0,
  "status": "ok",
  "error": null,
  "attributes": {
    "model": "anthropic/claude-sonnet-4-20250514",
    "prompt_hash": "sha256:3f4a2b8c1d9e7f6a5b4c3d2e1f0a9b8c7d6e5f4a3b2c1d0e9f8a7b6c5d4e3f2",
    "prompt_preview": "[{\"role\": \"system\", \"content\": \"You are a research as ... tools available: [web_search, fetch_paper].\"}]",
    "prompt_chars": 1843,
    "response_chars": 312,
    "tokens_in":  612,
    "tokens_out": 98,
    "cost_usd":   0.00387,
    "finish_reason": "tool_calls"
  }
}
```

Reading this span tells the on-call engineer: the model was asked to decide which tool to use (name `completion:decide_tool`), it belongs to the agent run with `parent_id` / `run_id` `a1b2c3d4...`, it took 2.5 seconds, it succeeded (`status: "ok"`), it used 612 input tokens and 98 output tokens on Sonnet, it cost $0.00387, and it finished because the model decided to call a tool (`finish_reason: "tool_calls"`) rather than producing a final text answer. The prompt preview shows the beginning of the system prompt and the tool list at the end. The full prompt -- 1,843 characters, about 460 tokens of system instructions -- is represented only by its hash and the preview.

That is the span. One object. Everything needed to answer "what did this call do, how long did it take, what did it cost, and why did it stop?" is a field lookup, not a log-file search.

A `kind="run"` span captured at the close of a three-call agent invocation looks like this — note the rollup fields in `attributes` that summarize the children.

```json
{
  "span_id": "f47ac10b58cc4372",
  "run_id":  "f47ac10b58cc4372",
  "parent_id": null,
  "kind": "run",
  "name": "research_agent",
  "started_at": 1716143240.114,
  "ended_at":   1716143242.038,
  "duration_ms": 1924,
  "status": "ok",
  "error": null,
  "attributes": {
    "child_span_count": 3,
    "total_cost_usd": 0.0042,
    "total_tokens_in":  810,
    "total_tokens_out": 215
  }
}
```

### What production systems add

The fields above are the minimum viable schema for a teaching project. Production tracers extend this schema with fields that require more application-level infrastructure than the teaching project provides.

`user_id` turns cost and error queries into per-user queries: "which users had agent runs exceeding $0.10 this week?" is a `GROUP BY user_id` on the run spans rather than a manual audit of raw traces. `session_id` lets you group multiple user requests into a conversation and trace quality and cost across a multi-turn exchange rather than per single request.

Time-to-first-token (TTFT) is the latency metric that matters for streaming responses. The user's perceived latency is the gap between sending the message and seeing the first visible token, not the total response time; p95 TTFT is the number that determines whether the product feels fast or slow in the user's perception. Without TTFT in the span, the `duration_ms` field lumps TTFT and total generation time together and cannot distinguish "model is slow to start" from "model generates quickly but the response is long."

Retry count distinguishes a span that failed three times before succeeding from one that succeeded on the first attempt -- a meaningful signal for detecting provider instability or for identifying prompt patterns that consistently trigger moderation. An A/B variant tag and a prompt template version field let you attribute cost and quality differences to specific experimental changes, so the experiment result is readable from the trace data without a separate experiment-tracking system. Each of these is a natural extension of the schema; the extension is additive and does not break any existing trace consumer.

### The schema as a discipline

The span schema is worth treating the same way the cache-key schema is treated in [Module 17 (Caching & Cost Optimization)](../17-caching-cost-optimization/): it should live in one place, be well-named, be tested, and be treated as a contract change when it changes. The schema in the teaching project is defined as a dataclass or TypedDict; a change to it is a change to every trace consumer that reads the JSONL file. Adding a new optional field is safe; removing an existing field or changing its type breaks any consumer that was reading it.

The practical discipline: bump a schema version constant whenever a field is removed, renamed, or changes type. Consumers that read a schema version they do not recognize can skip those records rather than silently misinterpreting them. The same versioning discipline applies to the trace file itself: `.traces/traces.jsonl` should have a companion `.traces/schema_version` file (or a header line in the JSONL) that any consumer can check before reading. The teaching project omits this for simplicity; a production tracer should include it from the first deploy.

---

## 4. Nested Spans: The Agent / Chain Case

Consider an agent answering "find me 3 papers on diffusion models and summarize each." The agent does not make one LLM call. It runs a tool-decide call to determine whether and how to search, calls the search tool, makes a second tool-decide call to evaluate whether the results are sufficient, calls the fetch tool three times for the top three results, makes a reflect call to evaluate the fetched content for relevance, and makes a synthesize call to produce the final answer with per-paper summaries. That is seven to twelve LLM calls depending on whether the reflect step decides a second search is needed. Every call has its own prompt, its own token counts, its own cost, its own possible error.

Now three users submit questions simultaneously. The tracer emits spans as each LLM call completes, interleaved across the three running [agent loops](../11-building-ai-agents/). After two minutes, the trace file contains 30-plus spans with overlapping timestamps, mixed prompt previews, and combined token counts that add up to someone's request but cannot be partitioned into three without reading every span by hand.

A flat view of those 30 spans is not a debugging surface -- it is a puzzle. Sorting by timestamp does not separate the invocations because concurrent agents produce spans in interleaved time order. Sorting by cost does not identify the parent request because cost is a property of each call, not of the invocation. Filtering on a specific prompt fragment finds only one of the seven calls that belong to user A, not the other six. Flat logs and flat span streams cannot answer "what did user A's agent actually do?" That question requires the full tree -- all seven of user A's spans, grouped under a single parent, with their collective cost and their collective status visible at the top level.

### The parent-child structure

Nested spans solve this by making the tree structure a first-class property of the data. A `run` span is opened at the start of the agent invocation. Its `span_id` is immediately copied to `run_id` and propagated to every span created during the invocation. Each LLM call opens a child `llm` span with `parent_id` pointing at the enclosing `run` span's `span_id`. If the agent uses sub-agents or multi-step chains, those create intermediate `run` spans with the same `parent_id` discipline, and the LLM calls under them carry `parent_id` pointing at the sub-agent run and `run_id` pointing at the top-level root.

When the outer `run` span closes, it computes rollups over all descendant spans:

- `child_span_count` -- the number of direct child spans (a proxy for agent complexity).
- `total_cost_usd` -- the sum of every child `llm` span's `cost_usd`.
- `total_tokens_in` and `total_tokens_out` -- the sums of input and output tokens across all child calls.

The run span's `status` propagates to `"error"` if any descendant span errored; no separate error count is needed — error state lives in `status`.

The pseudocode pattern that produces this structure:

```python
with tracer.run("agent_loop", metadata={"user_id": user_id}) as run:
    # Step 1: decide which tool to call
    decide_result = tracer.wrap_llm_call(
        client.messages.create,
        messages=messages,
        model=model,
        name="completion:decide_tool",
    )

    # Step 2: call the chosen tool, observe results
    search_results = tool_search(decide_result.tool_input["query"])
    messages = append_tool_result(messages, search_results)

    # Step 3: reflect on results, decide whether to continue
    reflect_result = tracer.wrap_llm_call(
        client.messages.create,
        messages=messages,
        model=model,
        name="completion:reflect",
    )

    # Step 4: synthesize the final answer
    answer = tracer.wrap_llm_call(
        client.messages.create,
        messages=final_messages,
        model=model,
        name="completion:synthesize",
    )
```

When the `with` block exits, the `run` span closes with its rollups. The trace file now contains four spans: one `run` parent and three `llm` children, all sharing the same `run_id`. Any query filtering on `run_id = "abc123"` returns all four spans in one pass. Any filter on the parent span's `total_cost_usd > 0.10` identifies expensive invocations before touching the child spans. "Which invocations errored on the synthesize step?" is `kind='llm' AND name='completion:synthesize' AND status='error'` with a join to the run span via `run_id` to retrieve the invocation's context.

### What the nesting unlocks across application shapes

The same pattern generalizes across every common LLM application architecture without requiring changes to the span schema.

A RAG chain from [Module 07 (RAG)](../07-rag/) produces a run span with three children: a retrieval span (`kind="retrieval"`), an optional rerank span (`kind="rerank"`), and a generate span (`kind="llm"`). The run span's `total_cost_usd` gives the combined retrieval-plus-generation cost; the individual child spans show the per-step breakdown. When the reranking step adds $0.002 per query at a throughput of 10,000 queries per day, that is $20 per day attributable to reranking -- a number that only becomes visible when the reranking span carries its own `cost_usd`.

A multi-agent handoff from [Module 12 (Multi-Agent Systems)](../12-multi-agent-systems/) produces a parent run span, one child `run` span per sub-agent, and `llm` grandchild spans under each sub-agent run. The grandchildren carry the sub-agent's `run_id` as their own parent run and the top-level `run_id` in a separate field (or the traversal is done in two steps). The tree at any depth is recoverable by following `parent_id` links, and the root's rollups still give the total cost and status for the top-level user request even when the tree is three levels deep.

A chain from [Module 13 (Workflows & Chains)](../13-workflows-chains/) that runs a map step followed by a reduce step produces a run span, N map child spans (one per retrieved chunk, each wrapping one LLM call), and one reduce child span. The run span's `span_count` is N+1 -- a natural measure of how many chunks were processed. A `span_count` of 18 on a request that typically produces 4 is the signal that the retrieval step returned far more chunks than expected, which explains why that run cost $0.09 instead of the usual $0.02.

Three tree shapes appear repeatedly across LLM applications, and seeing them explicitly helps calibrate what the trace data should look like:

```
-- RAG chain (typical) --
run: rag_pipeline | 4.2s | $0.011 | ok
  +-- retrieval:vector_db  | 0.3s | $0.000 | ok
  +-- llm: completion:generate | 3.9s | $0.011 | ok

-- Agent loop (two tool calls) --
run: agent_loop | 9.1s | $0.032 | ok
  +-- llm: completion:decide_tool  | 2.1s | $0.006 | ok (finish_reason: tool_calls)
  +-- llm: completion:reflect      | 3.3s | $0.008 | ok (finish_reason: tool_calls)
  +-- llm: completion:synthesize   | 3.7s | $0.018 | ok (finish_reason: stop)

-- Map-reduce chain (3 chunks) --
run: summarize_docs | 11.4s | $0.027 | ok
  +-- llm: map:chunk_0  | 3.2s | $0.009 | ok
  +-- llm: map:chunk_1  | 3.6s | $0.009 | ok
  +-- llm: map:chunk_2  | 3.1s | $0.007 | ok
  +-- llm: reduce       | 1.5s | $0.002 | ok
```

Each tree's root span carries the rollup cost and status. Each child span carries the per-step detail. The query "which RAG pipelines cost more than $0.02" is a filter on the root spans of the RAG tree; the query "which map calls produced errors" is a filter on the `kind='llm' AND name LIKE 'map:%' AND status='error'` across all trees. The tree shape is the same regardless of the application pattern; only the names and the depth differ.

### What the trace tree looks like

The tree structure that nested spans produce is worth visualizing explicitly, because the ASCII view of a trace is what every trace UI shows and what the teaching project's `--show-tree` CLI flag renders.

```
run_id: a1b2c3d4 | agent_loop | 12.3s | $0.18 | status: error | span_count: 5
  |
  +-- llm | completion:decide_tool | 2.5s | $0.01 | ok | finish_reason: tool_calls
  |
  +-- llm | completion:reflect     | 3.1s | $0.03 | ok | finish_reason: stop
  |
  +-- llm | completion:fetch-p1    | 2.8s | $0.04 | ok | finish_reason: stop
  |
  +-- llm | completion:fetch-p2    | 2.6s | $0.04 | ok | finish_reason: stop
  |
  +-- llm | completion:synthesize  | 1.3s | $0.06 | ERROR | finish_reason: length
            error: "RateLimitError: max_tokens=1024 exceeded; response truncated"
```

Reading the tree top-down: the root `agent_loop` run span took 12.3 seconds and cost $0.18 total. It is marked `error` because the last child span errored. Four of the five children succeeded; the synthesize call hit the `max_tokens` limit and returned a truncated response, which the application treated as a failure. The fix is clear: raise `max_tokens` on the synthesize call, or summarize each paper individually before the final synthesis step so the input to synthesis is shorter.

That diagnosis took reading one tree. Without nested spans: 30 interleaved log lines from three concurrent agent runs, manual timestamp-based grouping to identify which lines belong to this invocation, locating the error line, determining whether a retry happened by scanning forward for a matching prompt hash, and manually summing the costs. A 25-minute investigation instead of a 2-minute one, on every incident.

### The debugging story in concrete numbers

Being specific about what nested spans make possible is more useful than asserting the principle.

The same tree view that shows the single failing invocation above also answers fleet-level questions when run as an aggregate:

- "Which run spans cost more than $0.10?" -- filter on `kind='run' AND total_cost_usd > 0.10`. Returns the set of expensive invocations sorted by cost descending. Click any result to see its child tree.
- "Which users had an error in their last 10 runs?" -- filter on `kind='run' AND status='error'`, group by `user_id`, count per user. The users with the highest error counts appear first.
- "Did the per-run cost change after Tuesday's deploy?" -- filter on `kind='run'`, partition by `started_at < deploy_timestamp` vs `started_at >= deploy_timestamp`, compare `total_cost_usd` distributions. If the post-deploy p50 is 4x the pre-deploy p50, someone changed the prompt.
- "What is the most expensive individual LLM call in the past week?" -- filter on `kind='llm'`, sort by `cost_usd` descending, take the top 10. The result is a list of specific calls with prompt previews that explain why they were expensive.

Every one of these queries is a simple filter or aggregation on the structured fields of the spans. None of them requires reading log text or manually reconstructing causal chains. That is the observability value of the span schema and the nested-span structure together.

---

## 5. Cost Telemetry as a First-Class Citizen

Cost-as-telemetry does not exist in classical APM, and the reason is structural. The cost of a web application's requests is not observable at the span level. A Postgres query costs micro-cents distributed across millions of queries in a monthly aggregate. A Redis lookup costs sub-micro-cents. A downstream SaaS call costs money, but it is billed on a monthly subscription or a usage tier, not returned as a per-call figure in the API response. Classical APM tooling was not designed around per-request cost because per-request cost was not accessible at the point of the request.

LLM APIs are different. The cost of a model call is returned in the response: the usage object carries input token count and output token count, the per-token rates are published and stable for months, and the arithmetic is one multiplication per token type. `cost_usd = (tokens_in * input_rate) + (tokens_out * output_rate)`. The cost is available at span-close time, attributable to that specific call, accurate to the cent. There is no other domain in software engineering where per-request cost is this accessible, this precise, and this variable call-to-call.

The variation is what makes cost a first-class observability signal rather than a billing curiosity. A "what is the capital of France?" query costs under $0.001. A "summarize these 8 research papers and identify the methodological disagreements" query on the same model costs $0.05 or more. A multi-step agent run that retrieves documents, calls three tools, and synthesizes a report costs $0.08-0.25 depending on how many retrieval cycles the agent runs and how long each retrieved document is. The cost of a single endpoint can vary by two orders of magnitude depending on the user's input and the prompt's structure. Without per-call cost telemetry, that variance is invisible until the billing aggregate arrives -- too late to catch a regression, too aggregated to identify the cause.

### Cost per call

Cost per call is the unit-economics signal. It answers "is this specific model invocation cheap enough to include in a high-volume endpoint?"

A single tool-decide call in an agent loop at $0.006 can scale to a million calls per day without putting the budget under meaningful pressure. A single summarize call at $0.04 can scale to only 25,000 calls per day at the same budget -- a difference of 40x in supportable volume. The right model tier for a given use case is determined by this number: Haiku at $0.001 for simple classifiers, Sonnet at $0.012 for mid-complexity tasks, Opus at $0.075 for reasoning-heavy tasks. Cost per call makes that routing decision measurable rather than intuited.

Cost per call is also the right level for evaluating prompt changes. If the new few-shot examples raised the per-call cost from $0.008 to $0.011 -- a 37% increase -- the relevant question is whether the quality improvement justifies the cost increase. Without per-call cost in the span, the comparison is impossible during the evaluation phase and only becomes visible in the billing aggregate after the change has been live for a month.

### Cost per run

Cost per run is the user-experience-tied signal. When a user types one question and gets one answer, the cost of that answer is the run span's `total_cost_usd` -- the sum of all model calls made to produce the response.

A run that costs $0.003 can be priced at $0.01 per query and still produce a 70% gross margin. A run that costs $0.12 requires a pricing model that either charges the user $0.50 per query or accepts a loss on the inference line item until scale unlocks a better model tier or a caching layer (see [Module 17](../17-caching-cost-optimization/)). Cost per run is the number the product manager needs to set pricing and the number the capacity planner needs to estimate the monthly inference budget. At $0.003 per run and a $6,000 per month inference budget, the system can serve 2,000,000 runs per month. At $0.012 per run and the same budget, it serves 500,000. Cost per call is the number the engineer needs; cost per run is the number the business needs.

Cost per run is also the natural alert threshold. An alerting rule that fires when any run span has `total_cost_usd > 0.50` catches runaway agent loops -- the agent that entered an infinite tool-call cycle, or the agent that was fed a 40,000-token document and tried to process it in one pass. Without per-run cost telemetry, these pathological cases are invisible until the billing alert fires at the end of the month.

### Cost per user per day

Cost per user per day is the abuse and heavy-user signal. The median user of a customer-facing assistant runs a handful of queries per day at a few cents per run. The tail user who runs 4,000 queries per day at $0.02 per run spends $80 of inference budget in a single day -- $2,400 per month from one user.

Without per-user cost aggregation in the trace data, this user is invisible. The billing aggregate goes up, the engineering team debates whether traffic grew or a prompt got more expensive, and nobody traces the increase to one user until the data is manually inspected. With `user_id` in the span and a daily aggregate over `total_cost_usd` grouped by `user_id`, the heavy user appears as the top row in the cost report on the first day they run 4,000 queries. The application can rate-limit them, move them to a capacity-constrained tier, or flag them for investigation -- before the cost compounds.

The same aggregation identifies the structural difference between a heavy user and a bot. A human power user runs 200-400 queries per day in recognizable patterns: bursts during working hours, pauses at night, queries that vary in topic and length. A bot runs 4,000 queries in a six-hour window with identical query lengths and no variation in time-of-day pattern. Both are visible in the per-user cost aggregate; distinguishing them requires looking at the span distribution within the day, which the trace data also provides.

### Cost regression detection

Cost regression detection is the deploy-safety signal and the most directly actionable of the four aggregations.

A prompt change ships on Tuesday. The per-run cost on the affected endpoint climbs from $0.003 to $0.012 because someone added a 4KB retrieved-context block that is now included in every request -- a change that felt like a quality improvement during manual testing but that quadrupled the inference cost in production. The error rate did not change. The quality metrics on the eval set did not obviously change because the eval set was small and did not cover the edge cases that the new context block was meant to address. The latency increased slightly but within the normal variance. The only clean signal is the per-run cost delta, and that signal is only visible if `total_cost_usd` on run spans is being tracked and compared across deploys.

Without per-run cost tracking: the regression is discovered when the monthly billing aggregate arrives, four weeks after the change shipped, at which point the change has been built upon by three subsequent deploys and reverting it is a significant engineering operation. With per-run cost tracking: the regression appears in the trace data within an hour of the deploy, the alerting rule that checks for a greater-than-50% increase in per-run cost fires, the on-call engineer compares the prompt hashes before and after the deploy, and the change is rolled back the same day.

### Token bloat as a cost signal

A secondary pattern worth naming: per-span token counts surface prompt bloat that is otherwise invisible. Prompt bloat is the gradual accumulation of tokens in the system prompt or in the retrieved context that each prompt-engineering iteration adds without removing. A system prompt that started at 300 tokens after the initial design, grew to 800 tokens after six months of edge-case handling, and now sits at 2,100 tokens is spending $0.006 more per call in input-token cost than the original -- on every single call, forever, with no benefit to the typical user (who never hits the edge cases that motivated each addition).

`tokens_in` in the span is the diagnostic. A filter on `kind='llm' AND tokens_in > 3000` on a system that was designed with 1,500-token prompts identifies the calls where the system prompt or the retrieved context ballooned. Sorting those results by prompt hash isolates whether the bloat comes from a specific system prompt variant (consistent `prompt_hash`, high `tokens_in`) or from retrieved context that varies by query (varied `prompt_hash`, high `tokens_in`). The fix for the first case is prompt compression; the fix for the second is retrieval-chunk truncation, which [Module 19 (Advanced RAG)](../19-advanced-rag/) covers in detail. Cost telemetry is the meter that makes the bloat visible; the fix lives elsewhere in the stack.

### The caching-and-tracing pair

The relationship between [Module 17 (Caching & Cost Optimization)](../17-caching-cost-optimization/) and this module is reciprocal in a way that is worth stating directly. Caching is the lever -- the mechanism that reduces inference costs by serving repeated queries from a local store rather than from the model. Tracing is the meter -- the mechanism that measures costs precisely enough to know whether the lever is doing its job, and by how much.

Without the meter, the cache's savings are invisible at the span level. The monthly billing aggregate goes down after the cache ships, but the aggregate cannot tell you whether the reduction came from the cache, from a reduction in traffic, from a model price drop, or from a change in the query distribution toward cheaper prompts. With the meter, a trace query comparing spans tagged `cache_hit=true` to spans tagged `cache_hit=false` gives you the per-call savings in real time and the cache hit rate as a ratio of span counts. The cache's economic impact is measurable on the day it ships, not in the next billing cycle.

The same meter that measures the cache's savings measures cost regressions, heavy-user behavior, per-step unit economics, and every other cost signal the application needs. It is one instrumentation investment with many payoffs. The lever and the meter belong in the same production deployment; shipping one without the other is shipping half a system.

---

## 6. Sensitive Data in Traces

Prompts and responses carry more sensitive material than any other field that a web application's trace has historically stored. They can contain the user's exact words -- their name, their medical condition, their financial situation, their emotional state, the specifics of a dispute they are trying to resolve. They can contain retrieved context from documents the user uploaded or from an internal knowledge base that is not intended to be publicly readable. They can contain an API key or access token that the user accidentally pasted into a chat window. They can contain the operator's system prompt, which is proprietary intellectual property -- the engineered persona, the few-shot examples, the output format instructions that the team spent weeks refining.

A naive tracer that stores the full prompt text and the full response text for every call is a data-retention liability that most teams do not consciously choose. It is one misconfigured IAM policy away from exposing every user's inputs and outputs to anyone with read access to the logging system. It is one overly broad "view logs for debugging" permission away from a developer reading an unrelated user's medical questions while investigating a latency issue. It is one third-party logging integration away from that text appearing on a vendor's servers without a data processing agreement that covers it.

The threat is proportional to how much text the tracer stores and how long it retains it. The mitigations in the span schema address the retention and the content separately, and they compose: you can apply all three to the same tracer, and the combined effect is a trace file that is useful for debugging and safe to treat as a semi-public operational artifact.

### Hash-not-store as the default

The teaching project stores `prompt_hash` (the SHA-256 of the canonicalized messages list) and `prompt_preview` (first 80 + last 80 characters), not the full prompt text. The hash gives machine-readable operations: dedup, change detection, grouping. The preview gives human-readable context. The full text -- the multi-kilobyte rendered prompt with the system instructions, the few-shot examples, the retrieved documents, and the conversation history -- never lands in the trace file.

An adversary who reads the trace file learns the prompt's SHA-256 hash and its first and last 80 characters. The hash is not reversible without a preimage attack. The preview reveals the structural shape of the prompt (the beginning of the system instruction and the end of the user turn) but not the full content. The sensitive middle -- the retrieved documents, the few-shot examples, the detailed system prompt body -- is never stored and cannot be leaked.

This is not a complete solution. The preview's last 80 characters often include the user's actual question, which may contain PII. The `redact=True` flag addresses this case.

### Per-tracer redact flag

`Tracer(redact=True)` suppresses the preview entirely, storing only the hash. Use this in any environment where prompts may contain user-supplied PII that should not appear anywhere in the logging infrastructure: a medical assistant where users describe symptoms and medications, a legal document analyzer where users paste contract text with names and deal terms, a financial assistant where users describe their account situation with specific figures.

The redact flag is a single parameter in the tracer constructor and applies uniformly to every span the tracer writes. It cannot be selectively disabled per span. This is a design choice, not an oversight. A system with partial redaction -- "redact medical-topic spans but not general-topic spans" -- offers the same privacy guarantees as no redaction if an adversary knows which categories are unredacted and can probe those categories. Uniform redaction is the only guarantee that is robust to topic misclassification.

In practice: use `redact=False` (the default) in development and in staging environments where prompts are synthetic test data. Use `redact=True` in production environments where real users type real questions. The trace file in production is then safe to ship to a third-party observability tool like Langfuse or Arize Phoenix without a data processing agreement that covers the full text of user inputs.

### Response-side scrubbing

The response text is more dangerous to retain than the prompt text because it is less predictable. The prompt is composed from operator-authored templates and user-typed input; both parties have some awareness of what they put in. The response is model-generated and can volunteer information the prompt did not explicitly request -- a model discussing a topic adjacent to personal information may produce a real email address, a real phone number, or a real person's name from its training data, unprompted by the user and unintended by the operator.

[Module 16 (AI Safety & Guardrails)](../16-ai-safety-guardrails/) ships a `redact_pii()` function that pattern-matches and scrubs email addresses, phone numbers, credit card numbers, SSNs, and IBANs from text. The correct integration with the tracer is to pass the model's response through `redact_pii()` before handing it to the tracer, not after. The span's `response_chars` is then the character count of the scrubbed response, and any response preview field contains the scrubbed text. The original model output is handled by the application layer and never reaches the trace file.

The two modules compose cleanly: the guardrail layer in Module 16 owns what the application does with sensitive content in the response before it reaches the user; this module's tracer owns what gets stored, and it stores only what Module 16 has already screened. The result is that the trace file and the user-facing response have the same PII exposure profile -- which is the right property, because they are both downstream of the same model call.

### Response preview and the same policy

The same hash-not-store discipline should apply to the response text as to the prompt text. The span schema as described in Section 3 stores only `response_chars` and no response preview by default, which is the correct starting point. A response preview (first 80 + last 80 characters of the response) is useful for debugging -- it lets the engineer see at a glance whether the model produced a tool call, a prose answer, or an error -- but it should be gated on the same `redact` flag. When `redact=True`, neither prompt preview nor response preview appears in the trace.

The hash-of-response is useful for dedup on the response side: "did the model produce the same answer twice?" is a question about output quality that the response hash can answer without storing the full text. Whether to include the response hash in the teaching schema is a judgment call; the teaching project omits it and focuses the dedup mechanism on the prompt side, where the change-detection use case is more pressing.

### Retention windows

Even a well-redacted trace file accumulates data over time. A JSONL file in `.traces/traces.jsonl` that grows unbounded over months contains a high-fidelity historical record of every prompt shape, every cost, every error, and every finish reason in the system -- more than most security policies intend to retain and more than most compliance requirements demand.

A 7-day rolling window is appropriate for operational debugging: the on-call engineer investigating an incident that happened three days ago can access the traces, but last month's production traffic is not retained indefinitely. A 30-day window is appropriate for trend analysis and cost attribution: the team can compare cost distributions week-over-week for a month, but not quarter-over-quarter. Anything older should be either archived to cheaper, access-controlled cold storage or deleted.

The retention decision belongs to the deployment layer, not the tracer. The tracer writes; the deployment layer decides when to stop keeping what it wrote. Log rotation (a cron job that compresses and archives JSONL files older than the retention window) and access-controlled archival (the archive is stored in a bucket that requires explicit approval to access) are the standard operational tools. [Module 20 (Deployment Patterns)](../20-deployment-patterns/) covers the mechanics in the context of a full production deployment.

What goes wrong without a retention policy: the trace file grows without bound, eventually consuming all available disk space on the server (a production outage), or growing large enough that it attracts security review attention when an audit reveals the company is indefinitely retaining detailed records of every user's inputs and the model's outputs. Both outcomes are avoidable with a single cron job and a retention constant; neither is avoidable retroactively once the file is large and the audit has started.

The three defenses -- hash-not-store, per-tracer redact, response scrubbing -- and the retention window are not redundant. They address different threat surfaces. Hash-not-store limits what is in each span. Redact suppresses even the preview in high-sensitivity environments. Response scrubbing removes PII that the model might volunteer. The retention window limits how long any of it is kept. The full defense is all four in sequence; partial implementation leaves one surface exposed.

---

## 7. Failure Modes & What This Module Doesn't Cover

The tracer in this module makes four design choices that are correct for learning and incorrect for production at any scale that puts meaningful load on the instrumentation layer. Each choice was deliberate: the simpler version teaches the span schema and the nesting pattern without the operational complexity that would obscure both. This section names the choices, explains the production failure mode each one creates, and identifies where each is addressed.

### Sampling at scale

The teaching project traces every request. This is pedagogically correct: every call is visible in the trace file, every exercise produces observable output, the span schema is visible on every invocation rather than on a sampled subset. It is operationally wrong for a production system at meaningful volume.

At 1,000 LLM calls per minute, a single LLM span is roughly 1-3KB in JSONL format. The trace file grows by 60-180MB per hour, 1.5-4GB per day. At 10,000 calls per minute, the trace file grows by 15-40GB per day. Beyond the storage cost, synchronous writes at this rate become a bottleneck: the operating system's page cache absorbs short bursts, but sustained high-write-rate JSONL appends produce contention when multiple threads compete for the same file handle.

Production tracers use sampling: a configurable fraction of normal traffic (1-10% is typical) and 100% of error traffic. The distinction between head-based and tail-based sampling matters for LLM workloads. Head-based sampling makes the trace/no-trace decision when the root span opens -- before any calls run. A 5% head-based sampler traces 5% of normal traffic and 5% of error traffic; error traces are statistically under-represented if the error rate is low. Tail-based sampling defers the decision until the root span closes -- after all calls have run -- and can guarantee 100% of error invocations are traced regardless of the normal-traffic sample rate. The cost is that tail-based sampling must buffer span data for the entire invocation duration before deciding whether to keep it, which requires more memory than head-based sampling. The right choice depends on whether error-trace coverage is a hard requirement or a nice-to-have.

### Async export

The teaching project writes span data to disk synchronously. The `with tracer.run(...)` block exits, the run span closes, the JSON is serialized, the file handle is acquired, the bytes are appended, and only then does the caller get control back. On a modern SSD, a synchronous write of a 2KB JSON span adds roughly 50-200 microseconds of latency to the call. At 1,000 calls per minute, this is invisible -- 200 microseconds on a call that took 2,000 milliseconds is 0.01% overhead. At 100,000 calls per minute, the writes become a wall-clock bottleneck: the file write queue backs up, the OS-level write buffering is insufficient to absorb all the pressure, and the tracer's I/O cost competes with the LLM calls for the process's available resources.

Production tracers decouple span-close from disk-write by buffering completed spans in an in-memory queue and flushing them in a background thread on a configurable schedule: flush every 100ms, or flush when the queue reaches 50 spans, whichever comes first. The queue adds a small delay between when a span closes and when it appears in the trace file -- spans are committed to disk a few seconds late rather than immediately -- in exchange for removing the I/O cost from the hot path entirely. The application's observed latency cost for tracing drops to the queue insertion cost: a few microseconds of dict construction and queue push, with no file I/O.

The async pattern also enables batched writes: instead of one file-system call per span, the background thread accumulates 50 spans and writes them in one `file.write()` call. Fewer file-system calls at higher payload sizes is the more efficient I/O shape for both local disk and remote logging endpoints.

### OTel-compatible export

The OpenTelemetry GenAI semantic conventions define standard field names for LLM spans: `gen_ai.system` for the provider name (`"anthropic"`, `"openai"`), `gen_ai.request.model` for the model identifier, `gen_ai.usage.input_tokens` and `gen_ai.usage.output_tokens` for token counts, `gen_ai.operation.name` for the operation type (`"chat"` for chat completion calls). The teaching project uses its own field names -- `model`, `tokens_in`, `tokens_out` -- because they are more readable in a teaching context and because standing up the full OTel stack (SDK, exporter, collector sidecar, visualization backend) is its own module's worth of infrastructure work.

A production system should emit spans in the OTel format. The reason is portability: an application that emits OTel-format LLM spans can ship those spans to Langfuse, Arize Phoenix, Datadog LLM Observability, Honeycomb, Grafana Tempo, or any other OTel-native backend by changing the exporter configuration rather than the span-construction code. The migration from the teaching schema to the OTel schema is a mechanical field-rename: `tokens_in` becomes `gen_ai.usage.input_tokens`, `model` becomes `gen_ai.request.model`. The span architecture -- tree structure, rollups, hash-not-store -- transfers without changes because it describes relationships and computation logic, not field names.

### Distributed traces across HTTP boundaries

The tracer in this module lives in a single process. All spans in a trace share a process heap, a single JSONL file, and a single `run_id` namespace. If the agent calls a tool that is implemented as a separate microservice over HTTP, the trace context -- the `run_id` and `parent_id` that give each span its position in the tree -- does not automatically cross the HTTP boundary. The downstream service creates spans with a new `run_id`, and the trace breaks into two disconnected trees: the upstream agent's tree and the downstream tool service's tree, with no data link between them.

Distributed tracing across HTTP boundaries requires two things: injecting a `traceparent` header (the W3C Trace Context standard: `traceparent: 00-{trace_id}-{parent_span_id}-{flags}`) into every outbound HTTP call, and extracting it in every inbound HTTP handler and using the extracted `parent_span_id` to set the `parent_id` of the first span created in the downstream service. The OTel SDK handles context propagation automatically through its `propagators` API; a homegrown tracer does not. The teaching project is single-process, so this gap does not surface in any exercise -- but it is the first failure that appears when the monolithic teaching project is decomposed into a microservice architecture.

### Alerting and SLOs

Trace data is the input to alerting; alerting is not part of the tracer. "Send a Slack notification when any run span has `total_cost_usd > 0.50`" requires a trace-querying process, a threshold rule, and a Slack webhook -- infrastructure outside the tracer and outside this module. Service Level Objectives on LLM quality (error rate below 1%, p95 cost below $0.05 per run, p95 latency below 5 seconds) require a metrics pipeline that reads from the trace data on a schedule, computes the SLO burn rate, and emits the burn rate to a metrics store that drives dashboards and alerting rules.

The teaching project produces trace files that a human can read and query manually. The operational infrastructure that makes trace data drive automated decisions -- alerting rules, SLO error budgets, capacity-planning dashboards, anomaly detection -- is covered in [Module 20 (Deployment Patterns)](../20-deployment-patterns/), which treats the observability stack from the deployment and operations perspective rather than from the instrumentation perspective.

### The right mental model for the gap

The four limitations above share a shape. Each one is a place where the teaching project chose the simpler implementation -- synchronous, unbounded, single-process, no sampling -- for pedagogical clarity, and each simplification comes with a scaling cliff. The cliff is not at the "a few developers calling the API" scale; it is at the "10,000 calls per minute" scale. For the vast majority of this curriculum's projects, the teaching tracer is appropriate. It becomes inappropriate exactly when the project transitions from a prototype to a production service, which is the transition [Module 20](../20-deployment-patterns/) is designed to support.

The span schema and the nesting architecture do not change at that transition. The identity fields, the LLM-specific attributes, the rollup logic, the hash-not-store policy -- all of these are as correct at 10,000 calls per minute as they are at 10 calls per minute. What changes is the plumbing around the schema: how spans are exported (async queue instead of synchronous write), what fraction is stored (sampled instead of all), where they go (OTel exporter instead of local JSONL), and how they are queried (a hosted tool's UI instead of a hand-written Python filter). The schema is the stable part; the plumbing is what the next module addresses.

---

## 8. The Ecosystem

The teaching project ships its own tracer because building one is the most direct way to understand what the span schema must capture and why each field exists. The act of writing the `cost_usd` field makes it concrete that cost is available at span-close time; the act of writing the `run_id` propagation makes it concrete why a shortcut to the root is needed; the act of writing the `prompt_hash` makes it concrete why hash-not-store is the right default. Production teams adopt an existing tool rather than maintaining a custom tracer; the ecosystem has matured enough that the tools are well-differentiated and the choice depends meaningfully on the team's infrastructure.

### Langfuse

Langfuse is the most widely deployed open-source LLM observability tool. It provides a trace UI, prompt management -- versioning and A/B testing system prompts from a web console rather than from code -- and an eval suite that runs evaluators against collected traces and surfaces the results alongside the span data. The self-hosted version runs on Postgres and Redis, is straightforward to deploy on a VPS or Kubernetes cluster, and is free for self-hosted usage without seat limits. A managed hosted version is available for teams that do not want to operate the infrastructure themselves.

The Python and TypeScript SDKs instrument OpenAI and Anthropic clients via a callback-based decorator that adds minimal overhead and requires changing roughly three lines of existing code. The `run_id` equivalent in Langfuse is called a `trace_id`; the nesting model maps directly onto the span tree described in Section 4. Langfuse is the right default for teams starting from scratch who want an open-source tool with a full-featured UI, built-in prompt management, and a growing eval ecosystem, and who do not have an existing observability stack that LLM spans must integrate with.

### Arize Phoenix

Arize Phoenix is open-source, built natively on OpenTelemetry, and strongest in the eval-to-trace feedback loop. Phoenix ingests OTel spans and surfaces them in a trace UI designed around the debug-then-eval workflow: find a bad trace in the UI, annotate it with a label, add it to an eval dataset, run an evaluator (an LLM-judge or a mechanical scorer from [Module 15](../15-evaluation-testing/)), and see the results as span-level annotations alongside the original trace. The OTel-native architecture means Phoenix works with any OTel exporter without vendor-specific SDK wrapping -- any application that can emit OTel spans can feed Phoenix.

Phoenix's strongest use case is the team that wants tight coupling between runtime tracing and offline evaluation: the traces from production feed the eval dataset, the eval dataset drives the evaluators, and the evaluator results are visible back in the trace UI as quality signals. That loop -- trace, annotate, evaluate, surface -- is the same loop that Module 15's harness implements for offline development; Phoenix extends it to the runtime data.

### Helicone

Helicone is proxy-based. You change the `base_url` parameter in your OpenAI or Anthropic client to point at Helicone's endpoint, and Helicone captures every request and response transparently without any SDK integration, any decorator, or any span construction in application code. The entire setup is a one-line URL change.

The cost of that simplicity is flexibility and latency. Helicone controls what is captured, what fields are available, how long data is retained, and what the UI shows. Custom span fields (the `user_id`, the `session_id`, the A/B variant tag) require Helicone's custom properties API, which is less flexible than writing them directly into a span. The proxy adds a network round-trip (20-40ms from typical cloud deployment locations to Helicone's endpoints) to every LLM call, which is small relative to the LLM call itself (1-6 seconds) but is a non-zero tax on every request. The right pick for teams that want the lowest possible integration friction and have not yet determined whether they need custom span fields or a tight eval integration.

### Datadog LLM Observability

Datadog LLM Observability is the managed-service option for teams already running in Datadog's APM ecosystem. LLM spans appear in the same Datadog dashboards, the same trace viewer, and the same alerting system as the rest of the application's telemetry. A web request span that calls an LLM can show the LLM child spans inline in the same waterfall view as the database calls and the downstream HTTP calls. The LLM-specific metrics (token usage, cost, finish reason distribution) appear in dashboards alongside the standard web-service metrics.

A service already instrumented with the Datadog Agent can add LLM observability by adding the `ddtrace` Python library and a few lines of configuration; the LLM spans are then correlated with surrounding web-request spans automatically via Datadog's trace propagation. The vendor lock-in is the same as for any Datadog feature: moving away requires migrating the entire observability stack, not just the LLM piece. The right pick for enterprise teams whose platform organization has standardized on Datadog and for whom the tight APM integration outweighs the lock-in.

### LangSmith

LangSmith is LangChain's hosted tracing and eval product. Its strongest integration is with applications that use LangChain primitives -- chains, agents, retrievers, prompts -- because LangSmith can auto-instrument them via a `LANGCHAIN_TRACING_V2=true` environment variable, with no manual span construction required. Every LangChain chain call, every LangChain agent step, every LangChain retriever call is automatically wrapped in a span and sent to LangSmith.

For applications built without LangChain -- including the applications built throughout this curriculum using the Anthropic SDK directly -- LangSmith requires the same kind of manual span construction as Langfuse or Phoenix, and the auto-instrumentation advantage disappears. The right pick for teams already invested in the LangChain ecosystem, where the automatic instrumentation of LangChain objects removes a real instrumentation burden that would otherwise require manual `with tracer.run()` wrapping on every chain step.

### OpenTelemetry GenAI semantic conventions

The OpenTelemetry GenAI semantic conventions are the standardization layer that all of the above tools are converging toward. The conventions are maintained by the OpenTelemetry project and define canonical attribute names for LLM spans: `gen_ai.system` (the provider name), `gen_ai.request.model` (the model identifier), `gen_ai.usage.input_tokens` and `gen_ai.usage.output_tokens` (token counts), `gen_ai.operation.name` (the operation type), `gen_ai.request.max_tokens` (the max tokens parameter), and a set of event types for prompt and completion content.

Any application that emits spans with these field names can be read by any OTel-compatible backend -- Langfuse, Phoenix, Datadog, Honeycomb, Grafana Tempo, and tools that do not yet exist -- by changing the exporter configuration. The conventions are the right north star for any team building tracer infrastructure: use the standard field names from the beginning, because the migration to any OTel-native backend then becomes a routing change rather than a field-rename across the codebase. The teaching project uses readable shorthand names for pedagogical clarity; a production project should adopt the conventions and gain the portability they provide.

### The OTel migration path in practice

The migration from the teaching schema to the OTel GenAI schema deserves a concrete mapping, because the field names are the main surface-area difference and seeing the translation makes clear that the conceptual model is identical.

```
Teaching schema field        OTel GenAI attribute
--------------------         ----------------------------
model                   -->  gen_ai.request.model
tokens_in               -->  gen_ai.usage.input_tokens
tokens_out              -->  gen_ai.usage.output_tokens
finish_reason           -->  gen_ai.response.finish_reasons (array)
kind = "llm"            -->  gen_ai.operation.name = "chat"
(provider prefix)       -->  gen_ai.system = "anthropic" | "openai"
```

The span identity fields (`span_id`, `parent_id`, `run_id`, `started_at`, `ended_at`) are standard OTel span fields, not LLM-specific, and map directly to the OTel SDK's `Span` object. The cost field (`cost_usd`) is not yet in the GenAI semantic conventions -- it is a common addition that individual tools implement as a custom attribute. The prompt hash and preview are also custom; the OTel conventions define event-based prompt recording (a `gen_ai.content.prompt` event) rather than a hash-based approach, and the tradeoffs (full content vs. hash-not-store) are the same design decision, just made in a different layer of the stack.

The practical migration order is: adopt OTel field names for the standardized fields, keep custom attributes for the non-standardized ones (cost, hash, preview), and emit to an OTel exporter once the field names are aligned. The span tree structure and the rollup logic do not change at all.

### Module cross-references

**Module 11 (Building AI Agents)** ([`../11-building-ai-agents/`](../11-building-ai-agents/)) is where the agent loops that generate the nested-span shape live. The tool-decide-act-observe cycles in Module 11 are the source of 7-12 LLM calls per user request; the tracer in this module is what makes those calls debuggable, attributable to a specific user request, and analyzable for cost and quality.

**Module 15 (Evaluation & Testing)** ([`../15-evaluation-testing/`](../15-evaluation-testing/)) produces eval rows offline; the tracer produces trace rows at runtime. They are siblings in the quality-engineering stack: eval rows capture what the model does on a curated, labelled dataset under controlled conditions; trace rows capture what the model does on real user traffic under real conditions. The labelled dataset that Module 15's harness produces is the right seed for calibrating cost regression alert thresholds and for seeding an offline eval run when a production trace surfaces a failure case worth investigating in a controlled environment.

**Module 16 (AI Safety & Guardrails)** ([`../16-ai-safety-guardrails/`](../16-ai-safety-guardrails/)) is where `redact_pii()` lives, and it is the right pre-processing step before passing any model response to the tracer in environments where model outputs may contain personal information. The cross-module dependency is intentional: the guardrail layer owns what is safe to emit into downstream systems; the tracer owns what gets stored; the two compose cleanly when the response passes through the guardrail before reaching the tracer.

**Module 17 (Caching & Cost Optimization)** ([`../17-caching-cost-optimization/`](../17-caching-cost-optimization/)) is the lever; this module is the meter. The cache's hit rate, the per-call savings, and the aggregate cost delta are all measurable from trace data if the span schema captures `cost_usd` on every call and the cache layer tags each span with whether the result was served from cache. Without the meter, the cache's impact is a black box visible only in billing aggregates. With it, the cache's performance is a first-class queryable signal available from the day the cache is deployed.

**Module 19 (Advanced RAG)** introduces multi-step retrieval pipelines -- hybrid search, query expansion, contextual compression -- that produce nested-span shapes with retrieval, rerank, and generate children under each run span. The tracer in this module handles these shapes without modification; per-step cost and latency become observable signals alongside generation cost and latency, which is where you discover that the reranking step adds 800ms and $0.002 per query and decide whether that trade-off is worth the retrieval quality improvement.

**Module 20 (Deployment Patterns)** ([`../20-deployment-patterns/`](../20-deployment-patterns/)) is where sampling strategies, async export, OTel collector configuration, log rotation, retention policies, and alerting all land. The tracer in this module writes the spans; Module 20 covers the operational infrastructure that manages, queries, monitors, and acts on what the tracer writes -- the deployment layer that turns a teaching artifact into production observability.
