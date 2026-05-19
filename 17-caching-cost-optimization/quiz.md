# Module 17 Quiz: Caching & Cost Optimization

Self-assessment questions for Module 17. Test your understanding before revealing each answer.

---

### Q1: Why is caching usually the highest-leverage cost lever after model selection?

<details>
<summary>Answer</summary>

LLM per-call cost is non-trivial (cents, not micros) and real production traffic is full of duplicates and near-duplicates. A 60% hit-rate on a 1M-call/month system at $0.02 per call cuts monthly spend from $20k to $8k for essentially zero per-request work after the cache is built. No other single change after switching models comes close on dollar-per-line-of-code. Caching also doubles as a latency win: a hit returns in milliseconds instead of seconds, so the user-facing speedup ships alongside the cost reduction.

</details>

---

### Q2: What must go into a cache key and what must stay out? Give one consequence of getting each wrong.

<details>
<summary>Answer</summary>

The key must include the prompt, the model identifier, the system prompt, and any sampling parameters that change the output (temperature, top_p, max_tokens, response_format). It must NOT include timestamps, request IDs, or per-call randomness.

Get the model wrong (omit it) and you serve a Haiku response when the caller asked for Sonnet — a correctness bug. Get the timestamp wrong (include it) and you have a 0% hit-rate because every key is unique — a complete defeat of the cache.

</details>

---

### Q3: When does semantic caching pay off, and when does it actively hurt?

<details>
<summary>Answer</summary>

Pays off when traffic is paraphrase-heavy — FAQ pages, customer support, knowledge bases — and the same underlying question is asked in many surface forms. Hurts when prompts are deterministic computations (each prompt is genuinely unique, so all the semantic layer does is waste an embedding) or when small wording differences should produce meaningfully different outputs (code generation, structured extraction) — there the false-positive cost of a semantic hit is higher than the savings.

</details>

---

### Q4: What is the canonical default similarity threshold for semantic caching, and what does loosening it cost you?

<details>
<summary>Answer</summary>

0.95 cosine similarity is the canonical default. Loosening to 0.90 catches more paraphrases (higher recall) but increases the false-positive rate: queries semantically near a cached entry but with a meaningfully different intent get the wrong cached answer. Tightening to 0.98 is the conservative direction — higher precision, fewer hits, more LLM calls. The threshold is a hyperparameter; tune it on your own traffic with labeled true-pair / false-pair queries rather than treating 0.95 as a constant.

</details>

---

### Q5: Why is the production-standard eviction stack TTL primary + LRU as overflow guard? What does each policy alone fail to do?

<details>
<summary>Answer</summary>

TTL keeps entries fresh — without it, the cache happily serves week-old answers after a model upgrade or knowledge change. LRU bounds the disk/memory footprint — without it, the cache grows unboundedly.

TTL alone fails on size: a sufficiently busy system fills the disk before any entry expires. LRU alone fails on staleness: hot entries are never evicted, so a hot wrong answer is the most persistent wrong answer. Combining them gives you a cache that is both bounded and fresh, which is what production needs.

</details>

---

### Q6: Name three failure modes a cache introduces and one mitigation for each.

<details>
<summary>Answer</summary>

1. **Stale answers** — mitigation: TTL + a `cache_version` constant bumped on every meaningful prompt or model change.
2. **Cross-tenant leakage** — include `tenant_id` in the cache key whenever response content is tenant-scoped; treat it as part of the key invariant, not as an add-on.
3. **PII in cache values** — redact PII at the response layer before storing (or before the response reaches the cache) so the cache never receives raw PII; cross-link to Module 16's PII redactor.

</details>

---

### Q7: Why is hit-rate a misleading metric on its own?

<details>
<summary>Answer</summary>

Hit-rate counts hits; what matters is dollars and seconds saved. A 90% hit-rate on $0.0001 calls saves less money than a 40% hit-rate on $0.02 calls. A 90% hit-rate on calls that return in 100ms saves less wall-clock than a 40% hit-rate on calls that return in 5s. Always pair hit-rate with the underlying call-cost and call-latency distributions; otherwise you optimize the wrong number.

</details>

---

### Q8: Name three real-world LLM caching tools or features and what each one specializes in.

<details>
<summary>Answer</summary>

- **GPTCache** — open-source LLM cache with exact + semantic layers, pluggable backends (FAISS, Redis, Milvus), swap-in similarity evaluators.
- **LangChain `set_llm_cache`** — pluggable cache wrapping any LangChain LLM, backends include SQLite, Redis, Cassandra, Momento.
- **Anthropic prompt caching** (and OpenAI cached prompts, Bedrock prompt caching) — provider-managed prefix caches that bill the cached prefix tokens at a steep discount; the highest-impact win on prompts with long system messages or large retrieved context.

</details>
