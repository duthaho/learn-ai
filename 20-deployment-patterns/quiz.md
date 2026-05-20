# Module 20 Quiz

Eight self-assessment questions. Each answer is 2–4 sentences and is answerable from the module README.

---

### Q1. Why is the policy order `budget → fallback → circuit → retry → timeout`?

<details>
<summary>Answer</summary>

Each pair has a defensible reason. Budget is outermost because if you're over cap, no provider in the chain helps — reject before spending. Fallback wraps the per-provider state (circuit, retry) because those state machines belong to individual providers. Circuit comes before retry so we fail fast against an open breaker instead of burning retry attempts on it. Retry comes before timeout because timeout governs each attempt, not the whole operation; inverting them would let a single attempt consume the entire operation budget on retries.

</details>

---

### Q2. When should you NOT retry an LLM call?

<details>
<summary>Answer</summary>

Retry only transient errors: rate limits (`429`), timeouts, and server errors (`5xx`). Never retry caller bugs (`400 BadRequest` — a malformed prompt will fail the same way next time), authentication errors (`401`/`403` — your key is bad or expired), content-filter rejections (the provider refused on policy grounds), or model-not-found errors (the model name is wrong). Retrying any of these wastes time, money, and rate-limit budget while producing the same failure.

</details>

---

### Q3. What is the half-open state's job, and why exactly one probe?

<details>
<summary>Answer</summary>

After `recovery_seconds` in the OPEN state, the circuit transitions to HALF_OPEN to test whether the provider has recovered. It admits exactly one probe call: success transitions to CLOSED and resumes normal traffic; failure goes back to OPEN with the timer reset. Admitting more than one probe would let multiple clients hit the recovering provider simultaneously, doubling load right when it's weakest — defeating the breaker's purpose of giving the upstream room to recover.

</details>

---

### Q4. If `RateLimitError.retry_after_s` is set, which wins: the provider's value or the computed backoff?

<details>
<summary>Answer</summary>

The provider's `retry_after_s` always wins. The provider is telling you precisely when to come back; respecting that value is both more polite and more accurate than our exponential-backoff guess. The retry loop logs the difference (`honored retry_after` vs `computed backoff`) so debugging traces show which behavior fired.

</details>

---

### Q5. Why does a `BadRequestError` not hop to the fallback chain?

<details>
<summary>Answer</summary>

A 400 is a caller bug — the request was malformed in a way the provider can't process. The fallback provider would receive the same malformed request and return the same 400. Hopping to the fallback adds latency and burns the secondary provider's rate-limit budget for no benefit. The chain raises immediately so the caller sees the bug instead of a misleading "all providers failed."

</details>

---

### Q6. When would you use a window-based budget cap vs a lifetime cap?

<details>
<summary>Answer</summary>

Window caps (rolling N seconds) allow recovery between bursts — useful when you want to permit normal traffic patterns but prevent sustained overspend. Lifetime caps are absolute and never reset — useful for one-shot experiments, customer pay-as-you-go quotas, or any context where there is a hard ceiling that should never be exceeded. The wrapper defaults to lifetime because it's the conservative choice; window mode is opt-in via `window_seconds=`.

</details>

---

### Q7. Why does each alert rule have an independent cool-down?

<details>
<summary>Answer</summary>

Different signals deserve different re-fire policies. An error-rate alert during a sustained outage shouldn't re-page every second once the on-call engineer is awake — a 60s cool-down is plenty. The same logic applies to latency. But `BudgetBurnAlert` is a one-shot crossing (80% of cap is reached exactly once per process), so its cool-down is effectively infinite. Bundling all rules under a single cool-down would either spam (too short) or silence real signals (too long).

</details>

---

### Q8. Why is a mock provider better than live chaos engineering for this module's eval harness?

<details>
<summary>Answer</summary>

Live chaos against real providers is non-deterministic (you can't reliably force a 429 at call 3), costs real money on the success paths, and produces failure shapes that differ across providers (OpenAI's `RateLimitError` looks nothing like Anthropic's). The mock is fully scripted: scenario "circuit_trip" returns 503 exactly twelve times. That determinism is the only way the README's expected output can match what the reader sees. Production should absolutely include real chaos testing; the teaching module uses the mock for repeatability.

</details>
