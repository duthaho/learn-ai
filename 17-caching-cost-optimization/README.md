# Module 17: Caching & Cost Optimization

**What you'll learn:**
- Why caching is the highest-leverage cost lever after model selection
- The three cache strategies — exact-match, semantic, and provider-side prompt-prefix
- Cache-key invariants — what must go in (prompt, model, system, temperature) and what must stay out (timestamps, request IDs)
- Semantic caching in depth — local embeddings, cosine similarity, the threshold dial, and the false-positive failure mode
- Eviction policies — TTL primary, LRU as overflow guard, plus manual versioning as the kill-switch
- The rest of cost optimization — model-tier routing, prompt compression, streaming, batching, token budgets
- Failure modes — stale answers, cross-tenant leakage, PII in cache values, hit-rate vs cost-saved
- The ecosystem — GPTCache, LangChain `set_llm_cache`, Helicone, provider-side prompt caching

| Detail        | Value                                                                                                |
|---------------|------------------------------------------------------------------------------------------------------|
| Level         | Intermediate–Advanced                                                                                |
| Time          | ~3.5 hours                                                                                           |
| Prerequisites | Module 03 (Embeddings & Vector Search), Module 04 (AI API Layer), Module 15 (Evaluation & Testing), Module 16 (AI Safety & Guardrails) |

---

## Table of Contents

1. [Why LLM Cost Is a Production Problem](#1-why-llm-cost-is-a-production-problem)
2. [The Three Cache Strategies](#2-the-three-cache-strategies)
3. [Cache Keys & Invariants](#3-cache-keys--invariants)
4. [Semantic Caching in Depth](#4-semantic-caching-in-depth)
5. [Eviction Policies](#5-eviction-policies)
6. [The Rest of Cost Optimization](#6-the-rest-of-cost-optimization)
7. [Failure Modes & Cache Hygiene](#7-failure-modes--cache-hygiene)
8. [Ecosystem & Module Cross-Reference](#8-ecosystem--module-cross-reference)

---

## 1. Why LLM Cost Is a Production Problem

Every module before this one has worked under an assumption it never quite stated: that calling the model is cheap and fast enough that the question is whether the model produces the right output. The prompt is iterated, the few-shot is rebalanced, the retrieval is tuned, the evals are run — and across all of it, the cost of the LLM call sits in the background as a line item nobody worries about. That assumption is correct in the prototype and incorrect in production, and the gap between the two is wider than most engineers expect.

In production, a single LLM call costs *cents*, not micro-dollars. The unit isn't comparable to a database query or a Redis lookup; it is comparable to a payments-API call or a third-party SaaS request. A typical mid-tier model on a typical prompt-and-response shape lands somewhere between $0.001 and $0.05 per call, and the variance within that range is set by the size of the system prompt, the retrieved context, the response length, and the model tier. On a single user's session those numbers feel invisible. On a million sessions a month they are the line item the CFO calls about.

Latency tells the same story from the user's perspective. A typical chat-style completion takes between 1 and 6 seconds, and a reasoning-heavy completion with a long output can take 20 seconds or more. The unit is *seconds*, not milliseconds, and it is human-visible in a way nothing else in a normal web stack is. The user types a question, the model thinks for four seconds, the user starts to wonder if the page broke. The same call from a cache returns in microseconds — fast enough that the response feels instant, and slow enough by comparison that the user notices on every miss.

### The economic case in concrete numbers

Pick a single grounding example and keep returning to it. A system that handles 1,000,000 LLM calls per month at an average cost of $0.02 per call spends $20,000 per month on inference alone. Add a cache that catches 60% of those calls — a realistic hit rate for a customer-facing assistant with a long tail of repeated questions — and the bill drops to $8,000. That is $12,000 per month, $144,000 per year, on a single configuration change that touches one wrapper function.

The math scales linearly in both directions. The same system at 10 million calls a month with the same hit rate saves $120,000 a month. The same system at 100,000 calls a month with the same hit rate saves $1,200 a month. The point is not the absolute number but the *ratio*: the cache strips out roughly the same fraction of cost no matter how big the deployment is. Latency follows the same ratio — 60% of calls return in microseconds instead of seconds, and the user-visible p50 drops by an amount that no other engineering change can match for the same effort.

This is why, after the safety layer ships, caching is the first lever every team reaches for. Nothing else has the same dollar-per-line-of-code. Switching to a cheaper model tier saves money but costs quality (Section 6). Compressing prompts saves money but takes prompt-engineering work. Batching saves money but only on offline workloads. Caching saves money on the request path the user is already paying for, without touching the model or the prompt or the application logic. The wrapper around `completion()` is two hundred lines of Python; the savings are five figures a month at modest scale. The ratio is the highest in the engineer's toolkit.

The reason the ratio is so high is that *real LLM traffic is repetitive*. Users ask the same question many ways. Onboarding flows replay the same FAQ. Tooling calls the model on the same templates across many runs. Knowledge-base queries cluster around the few topics users actually care about. The first time the assistant sees "what is the capital of France?" it pays the LLM call; the next thousand times, it could have served the same answer for free. Production traffic is not a uniform distribution of unique questions; it is a long tail with a fat head, and caching is the way the system gets paid for the redundancy.

### Cost and latency are correlated

The second framing matters because it changes who wants the cache shipped. A cache hit returns in milliseconds, not seconds — the same lookup that saves money saves the wait. The user-facing speedup is often what gets the cache shipped *before* the cost spreadsheet does, because the product team can see it in the latency dashboards and the support team can see it in the user surveys. The cost win arrives in the next billing cycle; the latency win arrives the moment the wrapper is deployed.

The correlation is not coincidental. The expensive operation and the slow operation are the same operation: the network round-trip to the model provider, the model's own forward pass over tokens, the response stream coming back. Skipping that operation skips both the dollars and the seconds. Whatever motivates a team to ship the cache — the CFO's quarterly review or the product manager's latency complaint — both stakeholders are happy with the same change.

### What this module asks

The earlier modules in this curriculum asked one question over and over: *does the model produce the right output?* Prompt design, model selection, structured output, RAG, agents, eval harnesses — all of it converged on the quality of the artifact the model produces. Module 16 (AI Safety & Guardrails) was the first module to widen the frame: it asked whether the system around the model is safe to ship. This module widens it again: does the system around the model produce the right output *cheaply enough to ship at scale*?

The shift is the same one every production engineer makes once their prototype meets real traffic. The prototype optimised for capability; the production system optimises for capability *per dollar* and capability *per second*. The cache is the first place that optimisation lands. The rest of Phase 4 — observability, advanced retrieval, deployment — continues the same widening: the engineering work is now about the system, not about the model.

### Why "two orders of magnitude" is the right frame

A single number is worth pinning down. The gap between *prototype cost* and *production cost* is rarely a percentage; it is rarely even a doubling. It is two orders of magnitude in the typical case, and the reason is multiplicative. Prototypes run on a few dozen test prompts a day during development; the same system, exposed to production traffic, runs on a few hundred thousand or a few million prompts a month. The per-call cost stays the same; the call volume changes by 10,000x. A $20 monthly bill becomes a $20,000 monthly bill on the same code, and that is *before* any of the production-only multipliers (long retrieved context for RAG, multi-turn conversations, agent loops with multiple tool calls per request).

The two-orders-of-magnitude gap is what turns caching from "nice optimisation" into "precondition for shipping." A prototype can ignore the bill; a production system that ignores the bill is a production system that does not get a second budget approval. The cache closes the gap by absorbing the redundancy that always exists in real traffic — the same questions asked thousands of times — and turning the multiplicative growth in call volume into a multiplicative growth in cache hits rather than a multiplicative growth in costs.

The latency story has the same shape. A prototype that takes four seconds per call is acceptable in a developer demo; a production system that takes four seconds per call loses every user who came expecting a chat experience. The cache reshapes the latency distribution: the median request now returns in milliseconds because most of the requests find their answer cached, and only the long tail of genuinely-new requests pays the full latency. The p50 drops by the cache hit rate; the p99 (the tail of misses) stays where it was, but the median user never experiences it.

---

## 2. The Three Cache Strategies

There is no single way to cache an LLM call. The shape of the cache depends on what counts as a "hit," and there are three coherent answers, each with its own implementation, its own sweet spot, and its own failure mode. A production stack often runs more than one of them simultaneously; the project in this module composes the first two in a single wrapper. The third is provider-managed and arrives as a configuration flag rather than as code.

### Exact-match cache

The simplest cache. Canonicalize the prompt (strip whitespace, normalize case), concatenate it with the model identifier, the system prompt, and the sampling parameters that affect the output, take the SHA-256 of the result, and look up the hex digest in a key-value store. If the key exists, return the stored response; if not, call the model, store the response under the key, and return.

The hit semantics are exact. Every byte of the inputs must match. "What is the capital of France?" and "What is the capital of France" (no question mark) are different keys and miss each other. "What is the capital of France?" run at `temperature=0` and `temperature=0.7` are different keys. The same prompt against `claude-sonnet` and `claude-haiku` are different keys.

The performance is unbeatable. A SHA-256 over a few kilobytes of text takes microseconds; a lookup in an in-memory dict or a local key-value store is another microsecond. There are zero false positives — the hash collision probability on real prompts is astronomically low — and zero embedding cost. The whole layer can be implemented in under fifty lines of Python and ships in any environment that supports a dict.

Where it fits: FAQ pages, deterministic prompts, internal tools where the prompt set is small and stable. Any workload whose prompts come from a finite list — a fixed set of summarisation templates, a fixed set of intent classifiers, a fixed set of customer-service responses — has near-100% hit rate on the second pass and pays only the hash cost on the first. Where it misses: any traffic where users phrase the same question differently. "Who wrote Hamlet?" and "Who is the author of Hamlet?" are different keys, and the exact-match cache is blind to the relationship between them.

### Semantic cache

The second strategy answers the obvious objection: most of the traffic the exact-match cache misses is *semantically equivalent* to traffic it has already seen. Users paraphrase. The cache should catch the paraphrase.

Embed the prompt with a small local encoder (`all-MiniLM-L6-v2` is the canonical default — 384-dim outputs, ~80MB on disk, ~50ms to encode on CPU). Store the embedding alongside the cached response. On a new prompt, embed it the same way, run cosine similarity against the array of all stored embeddings, take the highest score. If that score is above a threshold (0.95 is the canonical default), return the cached response for the matching entry; otherwise, miss, call the model, store the new embedding and response, return.

The hit semantics are *approximate*. "Who wrote Hamlet?" and "Who is the author of Hamlet?" embed into nearby points in the semantic space; their cosine similarity is high enough to clear the threshold; the second query hits the first one's cached answer. The same applies to "what is the capital of France?", "tell me the capital of France", and "France's capital is what?" — all of them collapse to the first cached entry.

The cost is real but bounded. The encoder runs once per prompt on the cold path, adds ~50ms of CPU latency, and produces a unit-length 384-dim vector. Cosine similarity becomes a dot product (`all_embs @ query_emb` for a 2D embeddings array and a 1D query); the math runs in milliseconds even at tens of thousands of stored entries. Sentence-transformers ships the model; numpy does the rest.

Where it fits: customer-facing chatbots, knowledge bases, support flows — any traffic where users phrase the same question differently. The hit rate jumps from "exact-match-only" levels (often 10–30%) to "semantic-cache" levels (often 50–80%), and the savings climb proportionally. Where it misses: deterministic computation prompts, code-generation prompts, and any traffic where small variations in the input must produce different outputs (Section 4 covers the false-positive case in detail).

### Prompt-prefix cache

The third strategy lives on the provider's side, not the application's. Anthropic prompt caching, OpenAI cached prompts, Bedrock prompt caching — each provider has shipped a feature with slightly different ergonomics and the same underlying idea: the provider caches the *prefix tokens* of a long system prompt or context block and bills subsequent calls at a steep discount on those tokens.

The wire shape is a flag on the API call. The application marks a segment of the prompt as cacheable; the provider keeps the model's internal state at the end of that segment in its own cache; the next call that arrives with the same prefix loads the state from the cache rather than recomputing it from the tokens. The discount is significant — Anthropic's prompt cache, for example, bills cached input tokens at 10% of the normal rate after a small write-once premium.

The hit semantics are *prefix-exact*. The cached portion must match byte-for-byte; the tail of the prompt (the user's question, the small variable part) can be anything. The provider does the bookkeeping; the application does nothing beyond setting the flag.

Where it fits: long system prompts (the few-thousand-token persona prompts that customer-service bots use), large retrieved-context blocks (RAG queries where the same five documents stay attached across many user turns), few-shot prompts where the example set is stable. Any workload where the *prompt prefix* is reused even though the *prompt suffix* varies. The win compounds with semantic and exact caching: the application-side caches catch fully repeated prompts; the provider-side prefix cache catches the long-prefix-with-variable-suffix shape that the application-side caches cannot.

The economic shape of the discount is worth understanding. The provider charges a small write premium the first time a prefix is cached (Anthropic's is 25% above the normal input rate), in exchange for a steep discount on subsequent reads (10% of the normal rate). The break-even point is two cache hits: a prompt prefix that gets reused twice has already paid back the write premium and is now net-cheaper than uncached. For high-frequency prefixes — a system prompt every request shares, a context block every retrieval-augmented request includes — the break-even is reached in the first few seconds of traffic, and every subsequent request runs at the discounted rate.

### Where each strategy sits in the request lifecycle

```text
   ┌────────────────────────────────────────────────────────────────┐
   │                       CLIENT-SIDE CACHES                       │
   │                                                                │
   │  user prompt ──→ Exact-match ──→ Semantic ──→ (miss)           │
   │                    │                │                          │
   │                    └── hit ─→ return cached                    │
   │                                                                │
   └────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
   ┌────────────────────────────────────────────────────────────────┐
   │                  PROVIDER-SIDE PREFIX CACHE                    │
   │       (Anthropic / OpenAI / Bedrock — bills input              │
   │        tokens for the cached prefix at a discount)             │
   └────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
                            ┌───────────┐
                            │  the LLM  │
                            └───────────┘
```

Read the diagram top to bottom and the cost ordering becomes the strategy ordering. Exact-match catches the cheapest cases at zero cost. Semantic catches paraphrases at the cost of an embedding. Prefix caching catches the long-prefix shapes at the cost of a flag and a small first-call premium. By the time the request reaches the model, every cheaper strategy has had its turn; only the genuinely novel requests pay the full LLM bill.

The three strategies are not exclusive. The application-side caches and the provider-side cache target different shapes of repetition; running all three is the production-canonical configuration. The project in this module ships the exact-match and semantic layers, with a forward pointer to provider-side caching as the next configuration change to make.

### Why the ordering is exact-then-semantic

A subtle but load-bearing detail: the exact-match layer runs *before* the semantic layer, not after. The reason is the same cost-ordering argument that puts mechanical evaluators before LLM-judge in [Module 15 (Evaluation & Testing)](../15-evaluation-testing/) and cheap regex before expensive moderation in [Module 16 (AI Safety & Guardrails)](../16-ai-safety-guardrails/): the cheaper check goes first because it short-circuits the expensive check whenever it hits.

An exact-match lookup costs microseconds and zero dollars. A semantic-cache lookup costs an embedding (~50ms on CPU) plus a dot product. On a workload where 30% of traffic is byte-identical repeats and 50% is paraphrases, putting exact-match first means 30% of requests skip the embedding cost entirely. Reversing the order would pay the embedding on every request, including the 30% that didn't need it. The savings on a million-request month are real: 300,000 embeddings avoided, ~15,000 CPU-seconds saved, encoder warm path kept cooler.

The ordering also matches the precision ordering. Exact-match has zero false positives by construction. Semantic has a non-zero false-positive rate proportional to how loose the threshold is. Trying exact-match first means the cache only falls through to the fuzzy layer when the precise layer can't help — the safer-when-possible ordering. Reversing this gives the fuzzy layer a chance to collapse two byte-identical requests into the same entry when there was no need to, which is harmless in practice but a category-error of design.

---

## 3. Cache Keys & Invariants

The cache key is the most important design decision in the whole layer. Get it right and the cache works invisibly; get it wrong and you ship a class of bugs that look like flaky model behavior — wrong answers, missing personalities, stale knowledge — and that nobody traces back to the cache because the cache is supposed to be transparent. This section walks through what belongs in the key, what doesn't, what is negotiable, and what each mistake looks like in production.

### What must go in the key

Four things must always go in the key, because each of them, if varied, would produce a different correct response:

- **The prompt text.** The user's actual question or instruction. Canonicalized — see below — but otherwise the load-bearing input.
- **The model identifier.** Sonnet and Haiku and Opus produce different responses to the same prompt. So do GPT-4o and GPT-4o-mini. So do two different versions of the same model from the same provider. The cache key must distinguish them.
- **The system prompt.** The system prompt shapes everything the model says — tone, refusal behavior, output format. A cache that serves a Sonnet-system-prompt response in answer to a Haiku-system-prompt request returns text the operator did not intend. The system prompt belongs in the key.
- **The sampling parameters that affect the output.** Temperature, top_p, max_tokens, and any structured-output mode (response_format, tool schema, JSON mode). A `temperature=0` call and a `temperature=1` call to the same model on the same prompt should produce different responses; collapsing them in the cache is a correctness bug, not an optimisation.

The rule is simple: anything that, if changed, would make the model produce a different correct output must be part of the key. If the operator wanted both variants to share a cache entry, they would have used the same value for the variant in question.

### What must stay out of the key

Equally important, certain things must *never* go in the key, because they would prevent the cache from ever hitting:

- **Timestamps.** Putting the current time in the key guarantees every request has a unique key, which means every request misses. The cache is then a write-only data structure: it grows, it stores, it never serves. This sounds obvious; it is one of the most common bugs in new cache implementations because timestamps creep in through serialised request objects.
- **Request IDs.** Same reason. Every request gets a unique correlation id, and if the id ends up in the key the cache stops working.
- **Per-call randomness.** A random nonce, a UUID generated in the caller, a session token attached to the prompt for some unrelated reason — all of them salt the key into uniqueness and kill the hit rate. The cache key derivation function must consume only the *semantic* inputs to the request, not the operational metadata.

The rule is the inverse of the previous one: anything that, if changed, would make the model produce the *same* correct output must not be part of the key.

### What may go in the key

Some inputs are correctly part of the key only in certain deployments. They live in a third bucket: include them if your topology demands the isolation, omit them if it doesn't.

- **`tenant_id` or `user_id`.** In a multi-tenant system where different tenants have different data, different policies, or different prompts that customise the assistant's behavior per tenant, the cache key must include the tenant identifier. Omitting it serves one tenant's cached response to another tenant — the single most dangerous cache bug, treated in detail in Section 7's cross-tenant-leakage discussion.
- **`session_id` or conversation_id.** When the prompt depends on conversation history that varies per session — Module 09's memory pattern — the conversation id (or, more precisely, a hash of the recent turns) must be part of the key. Otherwise, two users having the same in-the-moment exchange share each other's prior context.
- **Locale, region, language flag.** When the same prompt produces different correct responses based on the user's locale (currency, units, date format, legal jurisdiction), the locale belongs in the key.

The rule for the optional inputs is *scope alignment*: include any identifier that scopes the response. If two users in the same scope would correctly receive the same response, they should share a cache entry; if they would correctly receive different responses, they must not.

### Canonicalization

Before the cache key is computed, the prompt text passes through a canonicalization step that collapses cosmetically-different prompts into the same key. The exact rules are a configuration choice; the canonical defaults:

- **Strip leading and trailing whitespace.** `"  What is X?  "` and `"What is X?"` are the same prompt for caching purposes.
- **Normalize internal whitespace.** Runs of spaces, tabs, or newlines collapse to a single space.
- **Optionally lower-case.** For case-insensitive workloads (most chatbots), `"What is X?"` and `"what is x?"` collapse to the same key. For case-sensitive workloads (code-generation, legal text), this rule is skipped.

The canonicalization function is the *only* place that knows the prompt-normalisation policy. Both the lookup and the store paths call it; both must agree on the normalised form, or the store-side never produces keys the lookup-side can find. Centralising the rule is what keeps the cache consistent across refactors.

### The discipline link to Module 16

Both this module and [Module 16 (AI Safety & Guardrails)](../16-ai-safety-guardrails/) wrap the same `completion()` call. Both impose discipline on what passes through that wrapper. Module 16's discipline is *what reaches the model* and *what reaches the user*; this module's discipline is *what produces the same answer*. The two wrappers compose: the safety layer enforces what is legal, the cache layer enforces what is reusable, and a production stack runs both layers around the same model call. The shape is the same — a function around `completion()` that owns a single, narrow responsibility — and the techniques transfer: deterministic checks first, structured logging on every decision, fail-loudly on configuration drift.

### What each mistake looks like in production

Three concrete failure stories worth keeping in mind:

**Omitting the model from the key.** The application originally ran on Haiku to keep costs down. The team upgrades to Sonnet for the user-facing product but leaves an internal eval harness on Haiku. The cache, keyed only on prompt + system + temperature, starts serving Haiku responses to Sonnet callers — and serving Sonnet responses to Haiku callers. Each consumer gets responses from the *other* model. Quality drops on both sides; the team spends a week debugging "Sonnet got worse" before someone notices the cache key.

**Omitting the system prompt.** A customer-service bot ships with two personalities — a formal one for the support flow and a casual one for the marketing flow. The same prompt produces different answers from the two system prompts, but the cache key only contains the prompt and the model. A user in the marketing flow gets a formal-tone response; a user in the support flow gets the casual one. The personalities cross-pollinate in proportion to the hit rate.

**Including the timestamp.** A new engineer adds the current ISO timestamp to the cache key derivation because "we should be able to tell when each entry was cached" — confusing the *value* (which can include the timestamp) with the *key* (which must not). The cache deploys; the hit rate is zero; the cache grows by one entry per request; the bill does not drop. Nobody notices for a week because the cache *runs* — it just never serves. The fix is one line; the embarrassment is the time spent staring at dashboards wondering why the savings never showed up.

The pattern across all three is the same: the cache key is *invisible* in production, and the bug shows up downstream as quality degradation, not as a cache error. The discipline is therefore to write the key derivation function once, name it explicitly, document the schema, and review changes to it like any other contract change.

### The single-place rule

The cache-key derivation should live in *one* function, called `_cache_key`, and nothing else in the codebase should construct cache keys. This is not aesthetic; it is operational. The schema of the key is the contract that binds the lookup side and the store side. If two places construct keys, two places have to agree on the schema, and the moment they drift the cache becomes a write-only data structure (the lookup side computes one key, the store side computes another, the lookup never hits the store's entries).

The single-place rule also makes the key schema *testable*. A unit test that calls `_cache_key` with the same inputs from two different code paths and asserts the outputs match catches drift at commit time rather than in production. A unit test that pins the key for a fixed example (asserts that `_cache_key("hello", "claude-sonnet", "", 0.0)` equals a specific hex digest) catches schema changes that would invalidate the entire on-disk cache.

The same discipline applies to canonicalization. The canonicalization function lives in one place, is called from `_cache_key` only, and changes to it are treated as cache-schema changes that warrant a version bump. A new engineer who quietly adds "strip Unicode combining characters" to the canonicalization is making a backwards-incompatible change to the cache schema; the version bump is what tells the operator the existing entries are now unreachable.

### What the audit log should record on every key

When a request flows through the cache layer, the cache-key inputs and the resulting key should both end up in the structured log. The minimum useful record:

```json
{
  "request_id": "req_a1b2c3",
  "timestamp": "2026-05-13T18:23:04Z",
  "layer": "cache_lookup",
  "model": "anthropic/claude-sonnet-4-20250514",
  "system_hash": "sha256:f3a8...",
  "temperature": 0.0,
  "canonical_prompt": "what is the capital of france?",
  "cache_key": "sha256:7f4c2a...",
  "verdict": "exact_hit",
  "lookup_latency_ms": 1
}
```

The `system_hash` (instead of the full system prompt) keeps the log compact and avoids leaking the operator's prompt engineering to whoever has log access. The `canonical_prompt` is logged in full because reproducing a miss requires seeing exactly what the lookup saw. The `verdict` is one of `exact_hit`, `semantic_hit`, or `miss`, with the semantic-hit case also carrying a `similarity` score and a `matched_key` field. The on-call engineer debugging "why didn't this hit?" reads four fields and answers the question in seconds rather than minutes.

---

## 4. Semantic Caching in Depth

Semantic caching is the part of the cache layer that does the most interesting work and creates the most interesting failure modes. Exact-match is mechanically obvious — if the bytes match, return the answer — and the only design questions are about canonicalization. Semantic is the layer where genuine taste, threshold tuning, and risk management decisions live, because the layer's whole purpose is to return a cached answer for a prompt that is *not byte-identical* to anything it has seen. That power is also the layer's failure mode: the same fuzziness that catches paraphrases also catches things that look like paraphrases but aren't.

### When semantic caching pays off

The workloads where semantic caching is a multiplier on top of exact-match share a shape: the underlying *intent space* is small relative to the *expression space*. Users want one of a few hundred things; they phrase each want a thousand different ways.

- **Paraphrase-heavy traffic.** General chatbot traffic, where the same question arrives as "explain X," "what is X," "tell me about X," "I want to understand X," and so on. The exact-match cache misses each of these; the semantic cache catches all of them from a single stored entry.
- **FAQ-style queries.** Help-centre bots, customer-onboarding flows, internal Q&A systems. The space of legitimate questions is bounded; the space of phrasings is open. The semantic layer collapses the phrasings.
- **Customer-support flows.** Users describe the same problem many ways — "my login isn't working," "I can't sign in," "the password page is broken," "I keep getting kicked out." All four point at the same underlying intent and should land on the same cached response (assuming the response is generic enough to serve all four cases).
- **Knowledge-base queries.** Searches over a fixed body of documentation, where many users ask many phrasings of the same handful of underlying questions.

The shared signal: the operator can predict, in advance, that the traffic will cluster around a small number of underlying intents. The semantic cache is what makes that clustering visible to the system.

### When semantic caching doesn't pay off

The shape that breaks the assumption is *high-uniqueness traffic*, where small variations in the input must produce different correct outputs:

- **Deterministic computation prompts.** "Compute the SHA-256 of the string `hello world`." "Compute the SHA-256 of the string `hello world!`." The two prompts are semantically similar (high cosine score), but they must produce different responses. A semantic cache that collapses them is broken — the response stored for the first prompt is the wrong answer for the second.
- **Code-generation prompts.** "Write a Python function that returns the sum of a list." "Write a Python function that returns the product of a list." High semantic similarity, wildly different correct outputs. A semantic cache that hits here returns code that does the wrong thing.
- **Per-user personalised prompts.** "What did I do last Tuesday?" "What did Jamie do last Tuesday?" Same shape, same embedding-space neighbourhood, completely different correct answers because the *referent* differs even though the *expression* is similar.
- **Math and unit-conversion prompts.** "What's 47 times 83?" and "What's 47 times 84?" are near-identical in embedding space and far apart in correct output.

The shared signal: the inputs are syntactically similar but their *truth conditions* are independent. A semantic cache trades off precision for recall, and these workloads cannot afford the precision loss. Leave them on exact-match only, or skip the cache entirely if the entire workload is unique-per-call.

### The similarity threshold dial

The threshold is the single most important hyperparameter in the semantic cache. It controls the trade-off between recall (catching more paraphrases) and precision (avoiding wrong-answer collapses). The canonical operating points:

- **0.95 — the default.** A safe, well-calibrated threshold for `all-MiniLM-L6-v2` embeddings on general English text. Paraphrases of the same question (different word order, synonyms, polite phrasings) typically score 0.95–0.99; semantically related but distinct questions typically score 0.85–0.94. The 0.95 line is high enough to keep the precision good and low enough to catch most legitimate paraphrases.
- **0.98 — conservative.** Tighter precision, lower recall. Used when the cost of a wrong-answer hit is high (medical, legal, financial — wherever a wrong cached answer has real consequences). At 0.98 the cache catches near-exact paraphrases ("what is X" vs "what's X") but misses looser phrasings ("can you explain X" vs "what is X").
- **0.90 — aggressive.** Looser precision, higher recall. Used when the wrong-answer cost is low (casual chat, recommendation framing, "tell me something interesting about X") and the savings are large. At 0.90 the cache catches more paraphrases but also starts collapsing questions that have related but meaningfully different answers.

There is no universally correct threshold. The threshold belongs to your workload, not to the encoder, and it should be tuned on your own traffic with a labelled dataset of true paraphrase pairs and false paraphrase pairs. Treat the threshold as a hyperparameter, not a constant.

### How the project computes similarity

The mechanic is straightforward once the embeddings are normalised at encode time:

```python
def _embed(text: str, encoder) -> np.ndarray:
    # normalize_embeddings=True makes each vector unit-length, so
    # cosine similarity is just a dot product downstream.
    return encoder.encode(text, normalize_embeddings=True)


def _cosine_similarity_search(
    query_emb: np.ndarray,
    all_embs: np.ndarray,
    threshold: float,
) -> tuple[int, float] | None:
    if all_embs.size == 0:
        return None
    scores = all_embs @ query_emb          # shape (N,)
    idx = int(scores.argmax())
    score = float(scores[idx])
    return (idx, score) if score >= threshold else None
```

Two design points worth pulling out. The encoder is loaded *lazily* — the first prompt that needs an embedding triggers the model download (~80MB) and the import of `sentence_transformers`. A single-prompt run that hits the exact-match layer never pays the encoder cost; a benchmark run pays it once and reuses the encoder for the rest of the queries. The embedding matrix is kept in memory as a single `(N, 384)` float32 array, which fits comfortably (1.5MB for 1000 entries) and lets the similarity search be a single numpy dot product without any index structure.

The mechanics of cosine similarity, dot-product equivalence on normalised vectors, and the encoder's behaviour are all upstream of this module. The patterns are introduced in [Module 03 (Embeddings & Vector Search)](../03-embeddings-vector-search/), and the same `all-MiniLM-L6-v2` encoder that the project uses there is the encoder this module's cache loads. The semantic-cache implementation is essentially a vector index with a similarity threshold and a key-value lookup attached.

### The false-positive failure mode

The risk that motivates threshold tuning is precise and worth naming. Two queries can be *semantically similar* in the embedding space and *behaviourally distinct* in their correct answers. The canonical example:

- **Query A:** "How do I cancel my subscription?"
- **Query B:** "How do I pause my subscription?"

The two queries share most of their tokens, share their grammatical structure, and embed into nearby points in the semantic space. A typical cosine score lands around 0.93–0.96. At threshold 0.95, on the wrong side of the dice roll, Query B hits Query A's cached response — and the user who wanted to *pause* their subscription gets instructions for *cancelling* it instead. The financial consequence is direct: the user follows the instructions, cancels their subscription, and the operator just lost a customer to a cache bug.

The mitigation is *threshold calibration on your traffic*. Take a sample of recent user queries. Manually label pairs as "should-collapse" (same intent, same answer should serve both) or "should-stay-separate" (different intent, different answer required). Run the semantic cache at several thresholds on the dataset. Compute the false-collapse rate and the false-miss rate at each threshold. Pick the threshold whose error rates match the cost asymmetry of your workload — same discipline as Module 16's input-guardrail tuning, applied to a different layer.

The other mitigation is *layered defenses* on the cached response itself. If the cache is the only layer between the user and the response, a false-positive hit serves a wrong answer with no further check. Adding an LLM-judge that compares the new query and the cached response and confirms the response answers the query is one option (turns the semantic cache into an "approximate retrieve + LLM verify" pattern, at the cost of an LLM call on every semantic hit). Adding a *response-level versioning* check that lets the operator invalidate specific entries when they discover a wrong-answer collapse is another. Both are appropriate when the cost of a false positive is high enough to justify the per-hit overhead.

The third mitigation, less commonly discussed but worth naming, is *separate caches per topic*. If the workload has a few obvious topic clusters (account-management vs product-questions vs general-knowledge), running a separate semantic cache per topic lowers the chance that an account-cancellation query and an account-pause query end up in the same neighbourhood. The cost is operational complexity and a slightly lower aggregate hit rate; the benefit is sharper precision within each topic.

### The encoder choice matters less than people think

A reasonable second instinct is to chase the encoder. Maybe `all-MiniLM-L6-v2` isn't the right model; maybe `all-mpnet-base-v2` would produce sharper similarity scores; maybe a dedicated reranker would close the gap. The instinct is wrong for most workloads. The improvement from a bigger encoder is usually a few percentage points of cosine score; the improvement from threshold tuning, response-level versioning, or topic-separated caches is usually an order of magnitude bigger. Start with the canonical default encoder, calibrate the threshold on your traffic, and only reach for a more expensive encoder if the calibrated numbers are not good enough on your specific workload.

The same applies to the embedding dimension. 384 dimensions is the all-MiniLM-L6-v2 output; 768 and 1024 are common alternatives. Bigger embeddings give marginally better similarity scores at the cost of more memory and slower dot-product math. For client-side semantic caching with hundreds or thousands of entries, the 384-dim default is the right starting point and the place to stay until the precision/recall numbers demand otherwise.

### Treating the threshold as a hyperparameter, not a constant

The discipline worth pulling out explicitly. Most teams ship the cache with the canonical 0.95 threshold, see good-enough hit rates, and never revisit. The teams that get the most out of the cache treat the threshold the same way an ML team treats a model hyperparameter: it has a value, the value was chosen, the value is measured periodically, the value can change. The change cadence is slow (monthly, quarterly), and each change is gated on a measurement: take a sample of recent shadow-mode lookups, label them as correct-hit / wrong-hit / correct-miss / wrong-miss, compute the four-cell confusion matrix at the current threshold and at a candidate threshold, ship the candidate if the trade-off improves on the business's cost asymmetry. The same eval-harness pattern from [Module 15 (Evaluation & Testing)](../15-evaluation-testing/) — labelled dataset, mechanical scorer, scorecard output — applies directly: the threshold-tuning eval is just another evaluator with a different SUT.

The same discipline applies to the *threshold per scope*. A general-knowledge cache might want 0.95; an account-management cache might want 0.97 (because the cancel-vs-pause distinction matters); a casual-chat cache might want 0.92 (because a slightly looser collapse is harmless). The project's `--threshold` flag is the lever; production deployments often go further and configure the threshold per cache instance, with the cache layer carrying a typed `threshold: float` parameter rather than a global constant.

### A worked numerical example

Make the threshold dial concrete with three labelled pairs. The encoder is `all-MiniLM-L6-v2`; the scores are typical of what real workloads see.

- "What is the capital of France?" vs "What's France's capital?" — same intent, same correct answer. Typical cosine score: **0.96**. At threshold 0.95, this hits (correct). At 0.98, it misses (lost recall, no harm).
- "Who wrote Hamlet?" vs "Who is the author of Hamlet?" — same intent, same correct answer. Typical cosine score: **0.95**. At threshold 0.95, this is on the boundary (sometimes hits, sometimes not). At 0.90, it hits consistently.
- "How do I cancel my subscription?" vs "How do I pause my subscription?" — *different* intents, *different* correct answers. Typical cosine score: **0.93**. At threshold 0.95, this misses (correct — the two queries get separate entries). At 0.90, this hits incorrectly (a paused user is told how to cancel; a customer-success disaster).

The 0.90 threshold buys an extra 5% of hit rate (the "who wrote Hamlet" pair starts hitting reliably) at the cost of admitting the "cancel vs pause" collision. The 0.95 default refuses both kinds of fuzziness: the borderline-paraphrase costs a miss, the borderline-distinct stays separated. The 0.98 threshold buys precision at a real recall cost: only the closest paraphrases hit, and the cache effectively functions as exact-match-plus-typo-tolerance. Which is right depends on what the cache is for, which is why the threshold is a hyperparameter, not a default to leave on.

### Encoder warm-up and lazy loading

A practical operational detail. The `sentence-transformers` library lazily downloads the model weights the first time `SentenceTransformer("all-MiniLM-L6-v2")` is called. The download is ~80MB; the load-into-memory takes another second or two; the first encode call is ~5x slower than subsequent calls because the model's internal buffers haven't warmed up. The project structures this carefully: the encoder is loaded only when the semantic layer is about to fire, the load is a one-time cost per process, and the warm encoder is reused for the rest of the session.

This matters for the request-path latency budget. A request that hits the exact-match layer never pays the encoder cost — the cache returns in microseconds because no encoder is loaded. A request that falls through to the semantic layer on a cold encoder pays the load cost (a one-time penalty) plus the encode cost (~50ms ongoing). A request that falls through to the semantic layer on a warm encoder pays only the encode cost. In a long-running service, the encoder warms once and stays warm; in a serverless deployment that spins up cold, the first semantic-layer request pays the load every time, which is a real argument for either persistent workers or a warm-up routine that pre-loads the encoder at process start.

---

## 5. Eviction Policies

A cache is a bounded resource pretending to be unbounded. Disk fills up. Memory fills up. The set of stored entries grows until something has to give. The eviction policy is the rule that decides what to remove when the cache hits its limit, and the right policy depends on what failure mode you can least afford: stale entries, unbounded size, or wrong-answer collapses. The production-canonical pattern uses more than one policy at once.

### TTL (time-based eviction)

Time-to-live is the policy that addresses *staleness*. Every entry carries an expiration timestamp (typically `created_at + ttl_seconds`); entries past their expiration are removed on the next access or on a periodic sweep. The user never sees a stale entry; the operator never has to remember to invalidate manually.

The right TTL depends on how fast the underlying data changes:

- **24 hours — the general default.** Catches the case where the model itself was updated, or where the operator quietly tuned a prompt template, or where the underlying knowledge had a routine refresh. Most production caches start here.
- **1 hour — fast-moving knowledge bases.** When the cache sits on top of a knowledge source that updates frequently (a help centre with daily edits, a price list, a status dashboard), the TTL should be short enough that the cache doesn't outlive a typical edit.
- **Indefinite — purely computational prompts.** When the response depends only on the prompt and the model, and both are stable across the deployment, the entry can live forever. "What is 47 times 83?" against a fixed model has the same right answer today and a year from now; the cache should keep the entry until something else evicts it.

TTL is not free. Every access has to check the entry's age before serving it; every miss path that finds an expired entry has to remove it before falling through to the model. The cost is negligible in practice (a single integer comparison), but it must be there. A cache that stores `last_modified` and never checks it is not a TTL cache; it is a cache with a `last_modified` field.

### LRU (least-recently-used eviction)

Time isn't the only bound. The cache also has a *size* bound — disk space, memory, or both — and once it fills, something has to be removed even if no entry has expired. The LRU policy is the standard answer: when the cache hits its capacity, the entry that was accessed longest ago gets evicted.

The mechanic requires tracking `last_accessed_at` on every entry and updating it on every hit. On a store, the new entry takes a slot; if the cache is over capacity, the entries sort by `last_accessed_at` and the oldest few are dropped. The bet behind LRU is that *recency predicts re-use* — entries the system touched recently are more likely to be touched again than entries it hasn't seen in days. The bet is correct for most workloads; the exceptions (workloads where popularity is what matters, not recency) are what LFU is for.

LRU's cost is the bookkeeping. Every hit writes `last_accessed_at`; the storage layer absorbs the extra write traffic. For an in-memory cache, the cost is invisible; for a disk-backed cache, the cost is real and worth measuring. The project in this module updates `last_accessed_at` in memory and persists it on `_save()`, so a long-running session amortises the write cost across many hits.

### LFU (least-frequently-used eviction)

LFU is the less common alternative. Instead of dropping the entry that was accessed *longest ago*, drop the entry that has been accessed the *fewest times*. The bet is that *popularity predicts re-use* — entries that have been hit a thousand times are more likely to be hit again than entries that have been hit twice, regardless of when those hits happened.

LFU has a known weakness: new entries start with a frequency of one and have to compete with old, well-established entries that have been hit many times. A genuinely-new popular query can take a long time to displace an old-stalwart entry, and during that time the new query is paying the LLM cost it should have stopped paying after the second hit. The mitigations (windowed LFU, aging factors, two-level frequency tracking) all add complexity. For most workloads, LRU is simpler and good enough, and that is why LRU is the production default and LFU lives in textbooks.

### Manual versioning

The fourth policy is the one that addresses what TTL and LRU cannot: *the operator changed something*. The system prompt was rewritten. The retrieval index was rebuilt. The team upgraded the model. The cached responses, all of them, are now wrong — or at least, all of them are now potentially wrong, and the operator doesn't know which specific entries are affected.

The kill-switch is a `cache_version` constant. The version is part of the cache-key derivation (either appended to the input string before hashing, or stored alongside entries and compared on lookup). When the operator changes anything that affects what the model would produce — a new prompt template, a new model version, a major knowledge update — they bump the version. Every existing entry's effective key changes; every new request misses; the cache rebuilds itself with fresh entries from the new configuration.

Manual versioning is the most aggressive eviction policy in the toolkit. It throws away the whole cache. The operational cost is real — the first wave of requests after the bump all pay the LLM cost again — but the alternative is worse: serving stale answers from before the configuration change and not knowing which ones to invalidate. Versioning is the *correct response* to "the cache is serving stale answers and I'm not sure which entries are wrong." Use it when in doubt; the cost of a few hours of cold-cache traffic is far below the cost of a week of subtly-wrong cached responses.

### The production-canonical stack

The stack that ships in most production systems is *TTL primary + LRU as overflow guard*:

- **TTL is the freshness policy.** Every entry expires after a chosen interval. The cache cannot serve a response older than the TTL, no matter what.
- **LRU is the size policy.** Every store call, after appending the new entry, checks whether the cache exceeds its size cap. If it does, the oldest-accessed entries are evicted until it fits.

Each policy alone fails in a way the other catches. TTL alone has no size bound: a workload with high diversity and a long TTL grows the cache without limit, eventually running out of disk or memory. LRU alone has no staleness bound: a workload with low diversity and stable popularity keeps the same entries forever, serving year-old responses long after the underlying knowledge has changed. Composing them is what makes both bounds enforced.

The project's eviction routine implements both in two passes:

```python
def _evict(self) -> None:
    now = time.time()
    # TTL pass: drop expired entries.
    self._entries = {
        k: e
        for k, e in self._entries.items()
        if (now - e.created_at) < self.ttl_seconds
    }
    # LRU pass: bound the total count.
    if len(self._entries) > self.max_size:
        ordered = sorted(
            self._entries.values(),
            key=lambda e: e.last_accessed_at,
        )
        keep = ordered[-self.max_size:]
        self._entries = {e.key: e for e in keep}
```

Two passes, both linear in the number of entries, both running on every store. The cost is bounded by `max_size` (a few thousand entries in the project's defaults), so the eviction routine runs in milliseconds even at the upper end. Both passes leave the embeddings array untouched; the next `_save()` compacts the embeddings file to match the surviving entries, reassigning `embedding_idx` so the on-disk layout stays dense.

Versioning sits outside this loop. When the operator bumps `cache_version`, every entry's cache key derivation produces a different hash, and the lookups miss. The old entries linger until the TTL drops them or LRU squeezes them out — a slow garbage collection that costs nothing because nothing is reading them. A `--flush` mode (which the project ships) is the more aggressive option: delete the cache directory outright and start over.

### What this looks like in a long-running deployment

Three concrete trajectories to keep in mind:

- **Hot, narrow, stable workload.** A bot answering ~50 distinct FAQ questions a thousand times each per day. The 50 entries get repeatedly refreshed by hits; they survive TTL because every hit updates `last_accessed_at` (or, more precisely, they refresh through repeated stores after expiration). LRU never fires because the entry count is well below cap. Hit rate stabilises near 100% after the first hour.
- **Cold, wide, evolving workload.** A general-purpose assistant where users ask thousands of different questions a day, each once. New entries arrive constantly; existing entries get evicted by LRU long before TTL fires. Hit rate stabilises low (10–20% from the exact-match layer, 30–50% from semantic) because the long tail of unique queries dominates. The cache pays for itself but only barely.
- **Versioning event.** The team upgrades from Sonnet to Opus. They bump `cache_version` as part of the deploy. The next morning, the cache reports a 0% hit rate for the first hour (everything misses against the new version), climbs back to ~40% by midday, and stabilises near the pre-upgrade rate by the end of the day. The cost spreadsheet shows a one-day spike; the freshness audit shows no stale answers.

Each trajectory is normal. The eviction policy is the mechanism that keeps the cache useful across the trajectory shapes the workload can take.

### Manual versioning in practice

The mechanics of a `cache_version` bump deserve a worked walkthrough because the discipline only works if the operator can do it confidently. The version constant lives in the same module as the cache implementation, with a comment explaining what triggers a bump:

```python
# Bump this when the prompt template, system prompt, or model
# semantics change in a way that should invalidate cached responses.
# See docs/cache-versioning.md for the change-log of past bumps.
CACHE_VERSION = "v7"
```

The version is included in the cache-key derivation, either by appending it to the input string before hashing or by storing it alongside each entry and comparing on lookup. Either approach makes a bump behave as a global invalidation: every existing entry's effective key is now different, every lookup misses, the cache rebuilds itself from fresh LLM calls on the new configuration.

The change-log of past bumps is a small but useful artifact. A list of rows — "v3: switched from Haiku to Sonnet, 2026-02-14; v4: rewrote the customer-service system prompt, 2026-03-02; v5: enabled provider-side prompt caching, 2026-04-19" — gives the team a record of why the cache flushed at each historical point, and a way to debug "why did our bill spike on the 14th?" by checking whether a version bump landed that day. The discipline is the same as keeping a database-migration log: cheap to maintain, valuable when something breaks.

The bump is not the same as a flush. A bump *invalidates* but doesn't *delete*. The old entries linger in the cache files until TTL drops them or LRU squeezes them out. Nothing reads them, so they cost only disk space. A `--flush` operation, by contrast, deletes the cache directory outright. The project ships both: the version bump is the day-to-day discipline; the flush is the nuclear option for "I want to start clean."

### Eviction is not deletion: the embeddings layout

A persistence-layer detail worth understanding. When TTL or LRU drops an entry, the entry is removed from the in-memory dictionary, but the entry's row in the embeddings file is not immediately reclaimed — the embeddings array stays the same shape with the now-orphaned row still in place. The reason is that compacting the embeddings file every time an entry is evicted would be expensive (a full rewrite of a `(N, 384)` array on every store call). The cheaper pattern: leave the orphans in place during the request path, and *compact on save*.

The project's `_save()` routine walks the live entries (after eviction), rebuilds the embeddings array in entry order, reassigns `embedding_idx` on each entry to match the new row positions, then writes both files atomically (write to `.tmp`, rename to the live filename). The compaction runs once per save, in milliseconds even at the project's `max_size`, and keeps the on-disk file from growing unboundedly across long-running deployments. The cost asymmetry — fast on the request path, periodic compaction on the save path — is the same pattern many storage engines use, applied to the project's much smaller scale.

The atomic-rename detail matters because a save that crashes halfway through is a save that corrupts the cache. The `.tmp` write followed by `os.replace` is the standard idiom on POSIX and Windows alike: either the new file is fully written and replaces the old one, or the old one stays in place. The cache always reflects a complete state, never a half-written one.

---

## 6. The Rest of Cost Optimization

Caching is the highest-leverage lever, but it is not the only one. The full cost-optimisation toolkit has five members; caching is the first because of the ratio of impact to effort, but a system that ships caching alone and stops is leaving real money on the table. This section walks through the rest of the toolkit, one paragraph each, with cross-links to the modules that own them.

### Model-tier routing

Not every query needs the most capable model. A short factual question — "what's the capital of France?" — produces the same correct answer from Haiku, Sonnet, and Opus, and only the cheapest of the three is the right one to call. A mid-complexity question — summarise this email, classify this support ticket, extract these fields — is right for Sonnet's tier: capable enough to do the job correctly, cheap enough that the per-call cost is reasonable. A genuinely hard reasoning task — multi-step plans, deep code generation, anything that benefits from extended thinking — is what Opus is for, and Opus's per-call cost is justified by the quality it produces on those tasks.

A simple router decides per-query which tier to call. The router can be heuristic (string length, keyword presence, presence of code blocks) or LLM-based (a cheap classifier judges complexity and picks the tier). The router itself costs a small amount — a few milliseconds for heuristics, a few cents per thousand calls for an LLM classifier — and the savings come from routing the easy cases away from the expensive tiers. A workload that ran 100% on Sonnet might route 60% to Haiku, 30% to Sonnet, and 10% to Opus, with savings on the bulk and quality preserved on the hard cases. Model-tier routing is the natural extension of this module's project: wrap the cache in a router, and you have the two highest-leverage cost levers chained.

### Prompt compression

The model bills on tokens. Fewer tokens means less money. Once the model is producing the desired behaviour, the engineering work shifts to *removing* tokens without losing the behaviour. The classic targets: the few-shot examples that taught the model what good output looks like (which can be trimmed once the model is reliable), the verbose system-prompt sections that document edge cases nobody hits any more, the retrieved-context blocks that include more chunks than the answer actually uses. Lowering `max_tokens` is the simplest compression — most responses don't need 4096 tokens of room — and pruning retrieved context to the top-k most relevant chunks (see [Module 19 (Advanced RAG)](../19-advanced-rag/)) is the highest-leverage on RAG workloads. Compression is the second-highest-leverage cost lever after caching, with the trade-off that it requires careful re-evaluation of the model's quality after each change.

### Streaming

Streaming doesn't reduce cost. The model produces the same number of tokens; the bill is identical to a non-streaming call. What streaming reduces is *perceived* latency: the user sees the first few tokens arrive within a couple of hundred milliseconds, the rest fills in as it generates, and the wait-feeling is over by the time the model has actually finished producing. Worth doing whenever the user is waiting on the response; not worth doing for batch workloads or for outputs that need to pass through an output guardrail that buffers the full response (Module 16, Section 5). The patterns and protocol details live in [Module 05 (Streaming & Realtime AI)](../05-streaming-realtime-ai/); the relevant observation here is that streaming and caching are orthogonal — a streamed response can still be cached as a complete string after the stream finishes, and a cache hit can be replayed as if it were streamed for UI consistency.

### Batching

Batching applies to *offline* workloads, where many prompts are processed together rather than each one being sent the moment a user clicks a button. Many providers offer a batch API at a discount (Anthropic's batch API is 50% off, with multi-hour latency); even without a batch API, parallelising many independent calls with a thread pool is a real cost optimisation because it concentrates the network overhead and lets the provider keep its hot path warm. [Module 15 (Evaluation & Testing)](../15-evaluation-testing/) demonstrated the pattern with `ThreadPoolExecutor` running an eval harness across hundreds of examples in parallel; the same pattern applies to nightly summarisation jobs, periodic re-classification runs, and any workload that doesn't have a user waiting on the per-call response. Batching is not usable for the interactive request path, but it is the right pattern for everything that runs on a schedule.

### Token budget tracking

The hardest cost-optimisation problem is the one you can't see. Every LLM call should emit `{input_tokens, output_tokens, cost, model, prompt_id, tenant_id}` into a metrics pipeline. Without this telemetry, every other cost decision is shadow-boxing: you change a prompt and don't know whether the bill went up or down; you flip a router and don't know how much it saved; you tune the cache and don't know which prompts contribute most to the remaining cost. The Pydantic call records from earlier modules already produce most of this data; the work is wiring them into a metrics backend (StatsD, OpenTelemetry, a per-tenant dashboard) so the team has live visibility. [Module 18 (Observability & Monitoring)](../18-observability-monitoring/) is the dedicated treatment of the observability stack; the short version is that *measurement precedes optimisation*, and a system without per-call cost telemetry cannot meaningfully optimise its costs.

### The toolkit as a stack

The five levers compose. A request arrives. The router picks a tier. The cache (this module) checks for a hit. On a hit, the response returns from the cache layer with the cached tier's cost zeroed out. On a miss, the prompt — already compressed by template hygiene — is sent to the model. The response streams back to the user while the safety layer (Module 16) buffers and checks. The cost telemetry records the actual tokens and dollars spent. Each lever cuts a different slice of the bill: the router routes away from expensive tiers, the cache eliminates calls entirely, the compression trims the per-call token count, the batching applies to the workloads where it's appropriate, the telemetry is what tells you whether the whole stack is paying off. The order of the levers in the request lifecycle is the order in which they save money: routing first (decides the tier), caching second (decides whether to call at all), compression third (decides how many tokens the call is), streaming fourth (decides the user-perceived shape of the response), and observability wrapping everything so the team can see the numbers.

### Why caching is first among equals

A useful framing for prioritising the levers: caching is the lever that saves money on calls *the system has already proven it knows the answer to*. The other levers save money on calls the system *is* making. The asymmetry matters because the cost of doing caching wrong is bounded (a stale answer, a missed hit) while the cost of doing routing wrong is unbounded (a hard query routed to a model that can't handle it returns a wrong response that ships to the user). Caching's downside is the smallest; its upside is the largest; its position at the top of the stack is not accidental.

The same logic applies in reverse to provider-side prefix caching, which sits at the *bottom* of the application-side stack but is often the highest-impact win for RAG-heavy workloads. A typical RAG request has a 5,000-token retrieved-context block and a 50-token user question; the prefix cache catches the 5,000-token portion at a 90% discount and the application pays full freight only on the 50-token tail. The savings on the input side often exceed what the application-side caches save on the output side, for the cost of a configuration flag. A team that has shipped exact-match + semantic caching and still sees high inference costs should look at provider-side prefix caching as the next move.

---

## 7. Failure Modes & Cache Hygiene

A cache layer in production is also a *state* layer in production. State in production is the source of most of the worst bugs in the engineering trade — race conditions, stale data, leaked identifiers, retention violations, debugging mysteries. This section walks through the five failure modes that come specifically with LLM caches and the operational hygiene that prevents each.

### Stale answers

The most common failure. The cache holds onto an answer that was correct when it was stored and is wrong now. The underlying knowledge changed: the prices updated, the policy was revised, the product was renamed. The system prompt changed: the operator improved the persona, fixed an output-format bug, tightened a refusal rule. The model upgraded: the provider released a new version of the same model name, or the team migrated from one model to another. The cache, unaware of any of this, keeps serving yesterday's response to today's user.

The mitigation is *TTL + versioning, both*. TTL guarantees no entry survives longer than the configured interval — even without operator intervention, knowledge that's older than the TTL gets refreshed by the next miss. Versioning is the kill-switch for the targeted case: when the operator knows something changed, bumping `cache_version` invalidates every entry at once, no matter how recently it was stored. Each policy alone is incomplete. TTL alone takes hours or days to flush; versioning alone is forgotten until the staleness shows up in support tickets. The pair is the practice: a 24-hour TTL by default, plus a version constant the operator bumps on every meaningful change.

### Cross-tenant leakage

The most *dangerous* failure. A multi-tenant system runs the same code path for many customers; the cache key omits the tenant identifier; one tenant's request hits another tenant's cached response. The user reading the response sees something written for a different account — perhaps with a different account's data interpolated into it, perhaps with a different account's policies applied. The consequences range from "embarrassing customer-support escalation" to "regulatory notification under data-protection law," and the bug is rarely caught in testing because it requires a shared cache and at least two tenants exercising the same prompt at the same time.

The mitigation is mechanical: include any identifier that scopes the response in the cache key. `tenant_id` in B2B SaaS. `user_id` in consumer products with per-user customisation. `session_id` or `conversation_id` when the prompt depends on recent turns. Locale and region when the response varies by jurisdiction. The rule from Section 3 — *if two callers would correctly receive different responses, they must not share a cache entry* — exists specifically to prevent this failure mode. The check is best implemented as a typed function that consumes the request object directly, so a refactor that adds a new scoping dimension forces a corresponding cache-key change.

A reinforcing discipline: run the cache layer's audit log past the security team. Any cache that stores responses that were generated under one tenant's context is a potential leak vector for that tenant's data. The security review should ask, explicitly, *what is the worst response in this cache that could be served to the wrong user?* If the answer is "nothing — the responses are generic" the risk is bounded. If the answer is "any response that interpolates account data" the cache must be tenant-scoped.

### PII in cache values

A subtler failure with a long compliance tail. The cache stores model responses; the responses sometimes contain personally-identifying information that the user supplied, that the retrieval layer surfaced, or that the model hallucinated. The cache becomes a PII store — and PII stores carry obligations the engineer is rarely thinking about when designing a cache: encryption at rest, retention limits, audit logs, deletion-on-request support, jurisdictional restrictions on cross-border storage.

The first mitigation is to *not put PII in the cache in the first place*. The PII-redaction layer from [Module 16 (AI Safety & Guardrails)](../16-ai-safety-guardrails/) redacts emails, phone numbers, credit cards, and similar structured PII before the response leaves the model wrapper. If the cache sits *after* the redaction layer in the response pipeline, it never receives raw PII; the stored response is the already-redacted version. This is the cleanest way to keep the cache out of compliance scope: it never sees the regulated data.

The second mitigation, for the cases where some PII must persist (a personalised response that depends on the user's name, for example), is to treat the cache like any other PII store. Encryption at rest. Retention limits enforced by a sweep. Audit logs of every read and write. A deletion API that locates a user's entries by `user_id` and removes them on request (which is the GDPR Right to Erasure and the CCPA equivalent, applied to your cache). Cross-region replication policies that comply with the relevant jurisdictional rules. These obligations are not exotic — they are the same obligations every other PII store carries — but the cache is the place they get forgotten because the cache was designed as a performance optimisation, not a database.

### Hit-rate vs cost-saved confusion

The most common *metric* failure. A team ships the cache, the dashboard shows a 90% hit rate, the team celebrates — and the bill barely moves. The reason: the 90% hit rate is on cheap calls. The expensive calls (long context, complex reasoning, the workload's real cost centre) all miss, and the cache's headline number reflects the easy traffic, not the costly traffic.

A worked comparison makes the point. Workload A has 1,000,000 calls a month at an average cost of $0.001 per call, with a 90% hit rate — total savings, $900. Workload B has 100,000 calls a month at an average cost of $0.05 per call, with a 40% hit rate — total savings, $2,000. Workload B's hit rate is less than half of Workload A's; its dollar savings are more than double. Hit rate alone is misleading; cost saved is the metric that matches the business outcome.

The mitigation is to emit *both* numbers from the cache layer. Hit count, miss count, cumulative cost saved (sum of the original costs of the calls that hit), cumulative cost paid (sum of the costs of the calls that missed). The dashboard shows both. The team optimises against the cost-saved number, not the hit-rate number, and the conversation with the CFO is about dollars rather than ratios.

### Debugging "why didn't this hit?"

The most *infuriating* failure mode in practice. A user hits a prompt that should have been cached; the cache misses; the engineer trying to debug it has no visibility into why. The cache key derivation is invisible — it's a function of inputs the engineer doesn't have a record of. Was the system prompt different? Was the temperature different? Was a stray character in the canonicalization? Did a recent code change introduce a new field into the key derivation? Without instrumentation, the answer is unknowable.

The mitigation is structured logging on every cache decision:

- Log the *canonicalized prompt* (the string after whitespace stripping, normalisation, case-folding).
- Log the *full set of key inputs* (model, system, temperature, any optional scoping fields).
- Log the *resulting key* (the SHA-256 hex).
- Log the *verdict* (exact hit, semantic hit, miss) and, on semantic hits, the *score* and the *index* of the matched entry.

With these four fields, the on-call engineer can reproduce a miss by hand. They can find the closest stored entry and see what differs. They can spot the stray character, the wrong temperature, the missing scoping field. Without these fields, debugging a cache miss is guesswork. The cost of logging is small — a few hundred bytes per request, far smaller than the response itself — and the savings on incident-response time are large.

### The shadow-mode rollout for cache changes

Borrowing the pattern from Module 16's safety-layer rollout: a candidate change to the cache (a new threshold, a new canonicalization rule, a new eviction policy) ships in *shadow mode* before it ships as the production behaviour. The shadow path runs the candidate logic alongside the production logic, emits both verdicts to the log, and compares them offline. If the candidate would have produced a different hit/miss/score on the request, the log captures the divergence. After a few days of shadow data, the team has a real measurement of what the change would do in production, and the decision to flip the switch is grounded in numbers rather than vibes.

The pattern transfers cleanly. The cache's lookup function is small enough to run twice — once with the production parameters, once with the candidate — for a fraction of a millisecond of extra latency. The store path stays on the production behaviour; only the lookup is doubled. The cost is invisible; the visibility into "what would change?" is the entire point.

### Cache poisoning and the trust boundary

A failure mode that becomes visible only when caches are shared across processes or across users: an attacker who can write to the cache can poison the responses other users read from it. In the project's single-process file-backed cache this is mostly a theoretical concern — only the operator's own code writes to the cache, and the cache lives in the operator's filesystem. In a distributed cache (Redis, Memcached, a shared NFS mount) the attack surface widens: any process with write access to the backend can plant a response that other processes will serve.

The mitigation is the same one most shared-state systems apply: treat the cache as a *trusted internal store*, not as an attacker-controllable input. The cache backend lives behind the same authentication and network controls as the rest of the application's internal state. The values written to the cache come only from successful, logged LLM calls — not from user input directly. The cache layer never deserialises untrusted JSON into Python objects without schema validation. And, in a defence-in-depth posture, the output guardrail from [Module 16 (AI Safety & Guardrails)](../16-ai-safety-guardrails/) runs on cache *reads* as well as on cache *writes*, so a planted toxic response is caught on its way out even if the cache layer was compromised.

### A worked end-to-end audit trail

To make the hygiene story concrete, here is the audit record shape the project's `cached_chat` emits on every request, with one example of each verdict:

```json
{
  "request_id": "req_20260513_182304_a1b2c3",
  "verdict": "exact_hit",
  "cache_key": "sha256:7f4c2a...",
  "model": "anthropic/claude-sonnet-4-20250514",
  "lookup_latency_ms": 1,
  "llm_latency_ms": 0,
  "total_latency_ms": 1,
  "cost_paid_usd": 0.0,
  "cost_saved_usd": 0.0036,
  "hit_count_after": 17
}
```

```json
{
  "request_id": "req_20260513_182306_b2c3d4",
  "verdict": "semantic_hit",
  "cache_key": "sha256:7f4c2a...",
  "matched_key": "sha256:a1b2c3...",
  "similarity": 0.967,
  "lookup_latency_ms": 54,
  "llm_latency_ms": 0,
  "total_latency_ms": 54,
  "cost_paid_usd": 0.0,
  "cost_saved_usd": 0.0036
}
```

```json
{
  "request_id": "req_20260513_182310_c3d4e5",
  "verdict": "miss",
  "cache_key": "sha256:f3a8b7...",
  "best_semantic_score": 0.873,
  "lookup_latency_ms": 53,
  "llm_latency_ms": 2147,
  "total_latency_ms": 2200,
  "cost_paid_usd": 0.0041,
  "cost_saved_usd": 0.0
}
```

Three things to notice. Every record carries the same shape so a downstream aggregator can compute hit rate and cost saved without conditional logic per verdict. The miss record carries `best_semantic_score` (the highest similarity that *didn't* clear the threshold), which is the diagnostic the team uses to decide whether the threshold needs tuning — a stream of 0.94-near-misses on prompts the operator would have wanted to hit is the signal to consider loosening to 0.93. The hit records carry `cost_saved_usd` populated from the *original* call's cost, so the rolling sum across hits is the dollars-saved metric the dashboard reports.

### The hygiene loop

The five failure modes form a single discipline. Set a TTL so entries don't outlive their truth. Include scoping identifiers in the key so tenants don't see each other's responses. Run the safety layer's redaction *before* the cache so PII doesn't accumulate. Emit cost-saved as a metric so the dashboard tells the truth. Log the cache decision shape so debugging is possible. None of these is exotic; all of them are habits the team builds once and then runs forever, and the cost of building them up front is far below the cost of fixing each failure in production after it's already shipped harm to a user.

---

## 8. Ecosystem & Module Cross-Reference

A custom tiered cache like the one in this module's project is the right teaching vehicle and a reasonable starting point for production. It is also the *floor* of what production systems use. As the workload scales — multi-region traffic, multi-tenant compliance, per-prompt cost dashboards, provider-side cache savings — most teams end up combining a hand-written cache like this one with one or more off-the-shelf tools. This section is a tour of the major players, followed by a cross-reference table mapping the project's components back to the modules they extend.

### GPTCache

GPTCache is the open-source LLM cache that most closely resembles the project in this module — a tiered cache with exact-match and semantic layers, designed to wrap an LLM call. The library ships with pluggable similarity evaluators (cosine similarity is the default; other evaluators including LLM-judges are available), pluggable embedding backends (sentence-transformers, OpenAI embeddings, ONNX-hosted models), and pluggable storage backends (FAISS, Milvus, Redis, SQLite). A team that needs the same shape as this module's project but at a different scale — millions of entries, distributed deployment, a particular embedding model — can swap in GPTCache's components without rewriting the orchestration layer.

The bet behind GPTCache is *swap-in components*. The same tiered-cache pattern works with any reasonable choice of encoder, similarity function, and storage backend; the library is the integration layer that keeps those choices interchangeable. The downside is the framework-shaped weight: configuration is YAML-driven, the component graph has its own vocabulary, and the failure modes (a misconfigured component returning the wrong shape) are different from the failure modes of hand-written code. For teams that need the scale, GPTCache pays its weight; for teams at the project's scale, the hand-written version is simpler.

### LangChain `set_llm_cache`

LangChain's cache integration is a single function call: `set_llm_cache(cache_instance)` swaps in a cache for every LLM call in the LangChain runtime. The backends include in-memory (`InMemoryCache`), SQLite (`SQLiteCache`), Redis (`RedisCache`), Cassandra (`CassandraCache`), Momento (`MomentoCache`), and a semantic variant (`SemanticCache`) that uses a vector store for similarity lookup. The shape is the cache pattern from this module applied to LangChain's LLM abstraction.

The bet behind `set_llm_cache` is *cache-as-decorator*. The application code is unchanged; the cache layer is configured globally and intercepts every call. The shape is convenient when the application is already a LangChain application, and uncomfortable when it isn't — the global-state pattern is hard to reason about in multi-tenant or per-request configurations. The library is the right fit for LangChain-native teams who want the cache benefit without restructuring; the project in this module shows the same pattern written explicitly, which is what most production teams end up with even if they reach for the library first.

### Helicone and LangSmith

Helicone and LangSmith are observability platforms that include cache layers as built-in features. The shape is a proxy in front of the model provider: every LLM call passes through the platform, every call is logged, every call has a cost-saved-by-cache metric attached, and the team gets a per-tenant dashboard out of the box. The cache itself is similar to the patterns in this module (exact-match + semantic + provider-prefix coordination); the differentiator is that it ships *with the observability* — Module 18's concern — already wired in.

The bet behind these platforms is *cache-as-observability-feature*. A team that wants both a cache and the per-call cost dashboard, and that is willing to pipe traffic through a third-party proxy, can adopt the platform and skip much of both this module's project and Module 18's. The downside is the architecture commitment (traffic now flows through the platform's proxy) and the data-residency conversation (the platform sees every prompt and response). These trade-offs are acceptable in many deployments and unacceptable in others; the choice is operational rather than technical.

### Provider-side prompt caching

Section 2 introduced the third strategy. The implementations matter:

- **Anthropic prompt caching.** Each `messages` block can carry a `cache_control` marker that flags it as cacheable. The first call writes the prefix to the cache (with a small write premium); subsequent calls that hit the cache pay 10% of the normal input-token rate on the cached portion. Cache TTL is short (a few minutes for the standard tier, longer with a dedicated tier), so the workload pattern that benefits is high-frequency repeated-prefix traffic.
- **OpenAI cached prompts.** Similar shape with a different ergonomic — prefix caching is enabled by passing a stable prefix at the start of the prompt; the API automatically detects and bills the cached portion at a discount.
- **Bedrock prompt caching.** AWS's implementation, with provider-specific configuration but the same underlying mechanic: prefix tokens cached server-side, billed at a discount on the second-plus call.

The win compounds with the application-side caches. The application-side caches catch fully repeated prompts (entire prompts that match a prior request). The provider-side prefix cache catches the long-prefix-with-variable-suffix shape (same system prompt and same retrieved context, different user question). A production stack with long system prompts and retrieved context — typical for any RAG-backed assistant — leaves significant money on the table by skipping provider-side prefix caching. The configuration cost is a one-line change; the savings are real.

### Module cross-reference

This module composes patterns from earlier modules into the cache layer. The mapping:

| This module's component | Prior module it builds on |
|---|---|
| Sentence-transformers encoder, normalised embeddings, cosine-similarity-as-dot-product | [Module 03 (Embeddings & Vector Search)](../03-embeddings-vector-search/) — the same encoder, the same math, applied as a cache lookup rather than a retrieval index |
| `completion()` wrapper that owns a single narrow responsibility | [Module 04 (AI API Layer)](../04-ai-api-layer/) — the layered-wrapper pattern, now wrapping for cost rather than for retries or fallbacks |
| Pydantic models for cache entries, hits, and results | [Module 08 (Structured Output)](../08-structured-output/) — typed records as the contract between caller and cache |
| JSON + numpy persistence with atomic rewrites and version metadata | [Module 15 (Evaluation & Testing)](../15-evaluation-testing/) — the same persistence pattern the eval harness used for scorecard storage |
| Wrapper-around-`completion()` shape with deterministic checks before the model call | [Module 16 (AI Safety & Guardrails)](../16-ai-safety-guardrails/) — the cache wrapper and the safety wrapper share the same shape and compose around the same model call |
| Threshold tuning on labelled paraphrase pairs | [Module 15 (Evaluation & Testing)](../15-evaluation-testing/) — the eval-harness discipline applied to the semantic-cache threshold as a hyperparameter |
| Audit-trail logging on every decision | [Module 16 (AI Safety & Guardrails)](../16-ai-safety-guardrails/) — every verdict logged with timestamp, inputs, and outcome |
| Batching pattern as the cost lever for offline workloads | [Module 15 (Evaluation & Testing)](../15-evaluation-testing/) — the ThreadPoolExecutor pattern the eval harness used for parallel grading |

The shared theme is *composition*. The cache layer is not a new abstraction; it is the wrapper-around-`completion()` shape from Module 16 with a different policy, the embedding-and-similarity machinery from Module 03 with a different goal, the typed persistence from Module 15 with a different schema. The engineering work is in the integration — what goes in the key, what triggers eviction, what counts as a hit — and the components themselves are familiar.

### A decision matrix for the ecosystem

The honest summary: there is no single winner. The right combination is the one that matches the team's constraints. A rough decision matrix:

**Skip the frameworks and write your own (like this module's project) when:** the team is small, the workload's threat surface is well-understood, the deployment is single-region single-process, the latency budget is tight, and the operational story does not require multi-tenant cost dashboards. The custom layer is the lowest-friction option — no framework to learn, no DSL to manage, no third-party rate limits to worry about — and it teaches the most, which is why this module's project takes the custom route.

**Reach for GPTCache when:** the cache pattern in this module's project is right, but the scale demands distributed storage (Redis, Milvus) or an embedding model the project's defaults don't offer. The library is essentially the project's pattern with swappable components, and the migration is straightforward.

**Reach for LangChain `set_llm_cache` when:** the application is already a LangChain application, the cache backend is supported (in-memory, SQLite, Redis, Cassandra, Momento), and the team wants the cache benefit without restructuring code paths. Less flexible than GPTCache or a custom layer, more convenient when the host stack is already LangChain.

**Reach for Helicone or LangSmith when:** the team wants the cache *and* the observability dashboard *and* the per-tenant cost breakdown all wired in by default, and is comfortable routing traffic through a third-party proxy. Often the right choice for early-stage teams that need both layers and lack the engineering bandwidth to build them.

**Enable provider-side prompt caching when:** the workload has long system prompts (multi-thousand-token personas), large retrieved-context blocks (RAG with stable document sets), or stable few-shot prefixes. The configuration cost is a flag; the savings on input tokens are large; this is the highest-impact change available once application-side caching is shipped.

**Combine multiple tools when:** you need application-side caching *and* provider-side prefix caching *and* per-tenant cost dashboards. Most production stacks end up here. The custom layer this module teaches is the *integration* point — the place where application-side decisions get made and where the provider-side flags get set. The tools shorten parts of the integration; they do not eliminate the need for it.

### Forward pointer: Module 18

The signals every cache layer must emit live in the next module. Cache hit-rate, cost saved per request, cost saved per tenant, per-prompt latency, embedding latency, lookup latency, eviction events — these are the canonical observability metrics for the cache, and they belong to [Module 18 (Observability & Monitoring)](../18-observability-monitoring/). The cache layer is the *producer* of those metrics; Module 18 is the *consumer* and the dashboard.

The relationship is symmetric with Module 16's. Just as the safety layer's audit trail needs an observability stack to surface the shadow-mode verdicts, the cache layer's hit/miss/cost record needs an observability stack to surface the dollars-saved-over-time and the per-tenant hit-rate breakdown. Both layers emit structured records on every decision; both layers depend on Module 18 to make those records visible.

### What this module deliberately doesn't cover

Three adjacent topics are out of scope, by design:

- **Distributed caches.** Redis, Memcached, DynamoDB-backed caches — all of these are the right answer at scale, and all of them have operational concerns (consistency, partitioning, eviction across nodes, network latency on lookup) that the project's on-disk single-process layout avoids. The patterns this module teaches transfer cleanly to a distributed backend; the project keeps it on-disk to teach the mechanics without an external service.
- **Cache warming and pre-population.** Some workloads benefit from warming the cache with a known set of common queries before user traffic arrives — running a batch job over the FAQ list at deploy time, for example, so the first user never hits a cold cache. The pattern is straightforward (call `cached_chat` over a query list at startup) but operational rather than mechanical, and the project leaves it as a follow-up.
- **Multi-tenant isolation beyond the cache-key discussion.** Tenant-scoped caches, per-tenant encryption keys, per-tenant rate limits, per-tenant TTLs — all are real concerns in B2B SaaS deployments and all live downstream of the basic cache-key hygiene this module teaches. The cache-key rule (include any identifier that scopes the response) is the entry point; the rest is operational tenancy work that the curriculum's later modules and the team's own infrastructure choices fill in.

The omissions are the same pattern as Module 16's: cover the application-engineering layer cleanly, point at the adjacent concerns, leave the deeper engineering for the modules and operational practices that own each concern. The cache layer this module builds is the *integration point* where the application-side strategies — exact-match, semantic, key hygiene, eviction policy, audit logging — meet the surrounding system.

### One final reminder

The mindset shift Module 16 demanded was *the system around the model is the product*; this module's corollary is *the system around the model is the cost centre*. Every cent the deployment spends on inference flows through the wrapper this module's project builds. Every second the user waits flows through the same wrapper. The wrapper is small — a few hundred lines of Python — but it sits on the request path of every interaction, and the engineering decisions inside it shape the system's economics for as long as the system runs. Designing the wrapper well is not a follow-up optimisation; it is part of the foundation, and the curriculum places it second in Phase 4 because the foundation has to be laid before the rest of the production concerns (observability, advanced retrieval, deployment) can settle on top.

### The takeaway

The single sentence to carry forward from this module: *the cache layer is how the system stays affordable*. The earlier modules taught how to make the model produce the right output; Module 16 taught how to keep the system from producing harmful output; this module teaches how to keep the system from producing *expensive* output that it has already produced once before. The pattern is the same wrapper around the same `completion()` call; the policy is what changes. Layered checks (exact-match first, semantic second, model last), measured tuning (threshold calibration, hit-rate vs cost-saved telemetry), and operational discipline (TTL + LRU, versioning, audit logging) — the cache is engineering work in the most traditional sense, applied to a substrate whose central component happens to be an LLM. The combination of Module 16's safety and Module 17's cost is what makes a system *deployable at scale*: safe enough to ship, cheap enough to keep running, fast enough that users don't give up. The remaining Phase 4 modules build on this foundation.

The pattern repeats across the remaining Phase 4 modules. Module 18 turns the cache's per-call telemetry into a dashboard the team actually watches. Module 19 turns the retrieval that feeds the cache into a faithfulness-checked pipeline that catches a different kind of wrong-answer collapse. Each module takes a layer of the production system the curriculum has so far treated as a black box and makes it inspectable, tunable, and ownable. Caching is the layer where the dollars happen; the rest of Phase 4 is the toolkit for keeping that layer honest. Once the cache is in place, every other Phase 4 concern can assume the per-request budget is under control — and that assumption is what turns the prototype the curriculum has built so far into something the team can actually deploy.
