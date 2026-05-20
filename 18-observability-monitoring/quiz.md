# Module 18 Quiz: Observability & Monitoring

Self-assessment questions for Module 18. Test your understanding before revealing each answer.

---

### Q1: Why is classical APM insufficient for LLM applications?

<details>
<summary>Answer</summary>

Classical APM (Datadog, New Relic) was built for a world where the unit of work is an HTTP request whose input is a JSON body and whose output is a JSON body — and where you can read the SQL from the trace and recover what happened. LLM apps break this on four dimensions: (1) the *input is the computation* (the rendered prompt has to be captured to make sense of what happened), (2) cost-per-call is observable and non-trivial (cents, not micro-cents), (3) outputs are non-deterministic so "repro from logs" requires the captured prompt, and (4) one user request becomes 5-15 LLM calls that form a causal chain a flat per-request log cannot represent. Traces are the only signal that captures the input *and* the causal structure.

</details>

---

### Q2: What goes in an LLM span that doesn't go in an HTTP span?

<details>
<summary>Answer</summary>

`model` (full provider/model identifier), `prompt_hash` (SHA-256 of the canonicalized messages for dedup), `prompt_preview` (first 80 + last 80 chars for human reading), `prompt_chars` / `response_chars` (full lengths even when the preview is truncated), `tokens_in` / `tokens_out` (usage from the provider), `cost_usd` (best-effort from a price map), and `finish_reason` (stop / length / tool_calls / content_filter). Production tracers add `user_id`, `session_id`, time-to-first-token for streaming, retry count, A/B variant, and prompt-template version. None of these have meaningful analogues in an HTTP span — they are LLM-specific.

</details>

---

### Q3: Why nest spans? What does the parent-child relationship buy you?

<details>
<summary>Answer</summary>

An agent loop, a RAG pipeline, or a multi-agent handoff turns one logical user request into 5-15 LLM calls. Flat per-call logs interleave those calls with calls from other concurrent users, and reconstruction becomes a needle-in-haystack grep. Nested spans wrap the logical request in a parent `run` span whose children are the individual LLM calls; the relationship is preserved by `parent_id` and `run_id`. With nesting you can ask "what did *user A* actually do" (one tree lookup), "which agent invocation cost more than $0.10" (filter on parent's `total_cost_usd`), and "which user request errored" (filter on parent's `status`) — all impossible against flat logs.

</details>

---

### Q4: Cost-per-call vs cost-per-run — when does each matter?

<details>
<summary>Answer</summary>

Cost-per-call is unit economics: is this prompt cheap enough to put in a high-volume endpoint? You use it when deciding whether to ship a prompt at all, or whether to drop to a cheaper model tier. Cost-per-run is UX-tied: when a user types one question and the agent makes 12 calls to answer it, the run cost is what the user is "worth" to you in compute terms. You use it for pricing decisions, capacity planning, and abuse detection. Cost-per-call without cost-per-run leaves you blind to the multiplier that agent loops introduce; cost-per-run without cost-per-call leaves you unable to attribute spend to specific prompts when something regresses.

</details>

---

### Q5: What's the default redaction policy for prompts and responses in a sane tracer, and why?

<details>
<summary>Answer</summary>

Hash-not-store as the default for prompts: the SHA-256 of the canonicalized messages gives you dedup ("have we seen this prompt before?") and a configurable preview (first 80 + last 80 chars) gives you human readability — but the full text never lands in the trace. For responses the same preview is sane by default; a `Tracer(redact=True)` flag should suppress even the preview for environments where prompts may contain user-supplied PII. Response-side PII (emails, phone numbers, card numbers) belongs to a redactor like Module 16's `redact_pii()`, applied before the response reaches the tracer. The default is conservative because the alternative is one careless log-export request away from a data-leak incident.

</details>

---

### Q6: Name three failure modes a naive tracer introduces and one mitigation for each.

<details>
<summary>Answer</summary>

1. **Synchronous-write contention.** Every LLM call's return blocks on a JSONL append; at scale this is a hot lock and a write-amplification problem. Mitigation: buffer + batch + async-flush (with a bounded queue so an exporter failure doesn't OOM the app).
2. **Unbounded disk growth.** "Trace everything" fills disk on a busy service. Mitigation: tail-based sampling (keep 1-10% of normal traffic, 100% of errors) plus a rotation/retention window (7-30 days).
3. **PII leakage into the trace store.** A trace file is a data-store the team forgot to threat-model. Mitigation: default hash-not-store for prompts, `redact=True` for responses, and per-tenant access controls on the trace store itself.

</details>

---

### Q7: What's the canonical way to capture an LLM call failure in trace data — status, error, parent rollup?

<details>
<summary>Answer</summary>

Close the LLM span with `status="error"` and `error="<ExceptionType>: <first line of message>"` (one line, not a full stack trace — stacks belong in logs, not in spans). Re-raise the exception so user code observes the failure. The parent run span sets a `had_error` flag during child close and, on its own close, propagates `status="error"` if any descendant errored. This gives you the right query shapes: filter LLM spans by `status="error"` for failure-mode analysis; filter run spans by `status="error"` to find user requests that didn't complete cleanly. The error string is for triage; the structured `status` is for filtering.

</details>

---

### Q8: Name three real-world LLM observability tools and what each specializes in.

<details>
<summary>Answer</summary>

- **Langfuse** — open-source (self-host or hosted). Trace UI + prompt management + eval suite. The most popular OSS option; tight integration with the major Python frameworks.
- **Arize Phoenix** — open-source, OpenTelemetry-native. Strongest in eval workflows; the pick if your dev loop is eval-driven and you want traces and evals in one place.
- **Helicone** — proxy-based; you change your base URL and it captures everything. Lowest friction to adopt; less flexibility than instrumenting your own tracer.

Honorable mentions: Datadog LLM Observability (managed, fits orgs already on Datadog), LangSmith (LangChain's hosted offering), and the OpenTelemetry GenAI semantic conventions (the standardization layer everyone is converging on — `gen_ai.system`, `gen_ai.request.model`, etc.).

</details>
