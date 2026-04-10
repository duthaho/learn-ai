# Module 04: The AI API Layer — Quiz

Test your understanding. Try answering before revealing the answer.

---

### Q1: What fields are in an API response's usage object, and why do they matter?

<details>
<summary>Answer</summary>

The usage object contains `prompt_tokens` (input), `completion_tokens` (output), and `total_tokens` (sum). OpenAI also includes `prompt_tokens_details.cached_tokens` and `completion_tokens_details.reasoning_tokens`. Anthropic uses `input_tokens` and `output_tokens` plus cache-related fields. These matter because billing is calculated directly from these numbers: `cost = (input_tokens × input_price / 1M) + (output_tokens × output_price / 1M)`.
</details>

---

### Q2: Your API response has `finish_reason: "length"`. What happened and what should you do?

<details>
<summary>Answer</summary>

The model hit the `max_tokens` limit before finishing its response — the output is truncated. The response may be cut off mid-sentence or mid-JSON. Fix: increase `max_tokens`, reduce input length, or make a follow-up call to continue generation. Always check `finish_reason` — treating a truncated response as complete leads to broken JSON, incomplete answers, and subtle bugs.
</details>

---

### Q3: You get a 429 error and a 401 error. Which do you retry and why?

<details>
<summary>Answer</summary>

Retry the 429 (rate limit exceeded) — it's transient. The server is temporarily overloaded or you've exceeded your per-minute quota. Wait with exponential backoff and the request will likely succeed. Do NOT retry the 401 (authentication error) — your API key is invalid, expired, or missing. Retrying the same bad key will always fail. Fix: check your API key in `.env` and verify it's valid in the provider's dashboard.
</details>

---

### Q4: Why does exponential backoff need jitter? What happens without it?

<details>
<summary>Answer</summary>

Without jitter, when a server recovers from an outage, all clients retry at the exact same intervals (1s, 2s, 4s) — creating synchronized waves that immediately re-overwhelm the server. This is the thundering herd problem. Jitter adds randomness (`delay = random(0, base × 2^attempt)`) so clients spread their retries across time, letting the server recover gradually. AWS research showed full jitter produces fewer total retries and faster overall completion.
</details>

---

### Q5: Calculate the cost: 2000 input tokens + 800 output tokens on a model charging $3/M input, $15/M output.

<details>
<summary>Answer</summary>

Input cost: 2,000 / 1,000,000 × $3.00 = $0.006. Output cost: 800 / 1,000,000 × $15.00 = $0.012. Total: $0.018 per request. At scale: 100,000 requests/day = $1,800/day = ~$54,000/month. Output tokens cost 5x more per token than input in this case, so optimizing output length has a bigger impact on cost than optimizing input.
</details>

---

### Q6: Why should you always set `max_tokens`? What happens if you don't?

<details>
<summary>Answer</summary>

Without `max_tokens`, the model can generate until it decides to stop — potentially thousands of tokens. This creates unpredictable costs (a model going on a tangent generates expensive output tokens), unpredictable latency (generation time is proportional to output length), and possible context window overflow (input + output must fit). Anthropic actually requires `max_tokens` — you can't skip it. Even with OpenAI where it's optional, always set it as a safety net.
</details>

---

### Q7: Name something LiteLLM abstracts across providers and something it can't abstract.

<details>
<summary>Answer</summary>

**Abstracts:** Response format normalization — LiteLLM converts Anthropic's `stop_reason: "end_turn"` to `finish_reason: "stop"`, maps `input_tokens`/`output_tokens` to `prompt_tokens`/`completion_tokens`, and converts content blocks to plain strings. All responses look like OpenAI format regardless of provider. **Cannot abstract:** Provider-specific features like Anthropic's prompt caching (`cache_control` blocks), OpenAI's structured output (`response_format` with JSON schema), or model capability differences (tool calling behavior varies between providers).
</details>

---

### Q8: What's the difference between TPM and RPM rate limits? Which can a single large request exhaust?

<details>
<summary>Answer</summary>

RPM (Requests Per Minute) limits the number of API calls. TPM (Tokens Per Minute) limits the total tokens processed (input + output). They're independent — whichever you hit first applies. A single large request can exhaust TPM: one request with 100K input tokens uses 100K of your TPM budget in one call, even though it only uses 1 RPM. Conversely, many tiny requests (e.g., "say ok") can exhaust RPM while barely touching TPM.
</details>
