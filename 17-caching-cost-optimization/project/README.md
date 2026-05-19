# Project: Tiered Cache Wrapper

A drop-in cache for `completion()`. The wrapper hashes the canonicalized prompt for an exact-match lookup, then embeds the prompt locally with sentence-transformers for a cosine-similarity lookup, then calls the LLM on miss. Every call returns the response plus a `CachedResult` that says which layer hit (exact / semantic / miss), what was saved, and how long the cache lookup took. The cache persists across runs and ships with a benchmark mode that compares cold-vs-warm cost on a small query corpus.

## What you'll build

- A `TieredCache` class that owns persistence (JSON + numpy) and eviction (TTL + LRU)
- A `_cache_key(...)` function — SHA-256 of canonicalized prompt + model + system + temperature
- A local-embedding semantic layer using `sentence-transformers` (model `all-MiniLM-L6-v2`)
- A `cached_chat(...)` orchestrator that runs exact lookup → semantic lookup → `completion()` on miss
- Pydantic models (`CacheEntry`, `CacheHit`, `CachedResult`) for auditable structured output
- A bundled 15-query benchmark corpus mixing repeats, paraphrases, and novel queries
- A CLI with five modes: single-prompt, `--benchmark`, `--stats`, `--flush`, and `--no-cache` bypass
- Persistence: `.cache/cache.json` + `.cache/embeddings.npy` (gitignored), compacted on every save

The project demonstrates:
- **Tiered lookup:** exact (cheap, deterministic) → semantic (cheap, fuzzy) → LLM (expensive) — same defense-in-depth ordering as Module 16's guardrails
- **Cache-key discipline:** every parameter that changes the response goes into the key
- **TTL + LRU eviction:** the two-policy stack production systems use
- **Real cost measurement:** the benchmark mode reports dollars and seconds saved on the warm pass

## Prerequisites

- [Module 03 (Embeddings & Vector Search)](../../03-embeddings-vector-search/), [Module 04 (AI API Layer)](../../04-ai-api-layer/), [Module 15 (Evaluation & Testing)](../../15-evaluation-testing/), and [Module 16 (AI Safety & Guardrails)](../../16-ai-safety-guardrails/) recommended — the cosine-similarity lookup comes from Module 03, the `completion()` wrapper from Module 04, the cost-and-latency accounting from Module 15, and the tiered defense-in-depth ordering from Module 16.
- Completed reading the [Module 17 README](../README.md) so the cache hierarchy (exact / semantic / model-tier) and the cost math (cold vs warm pass) are fresh.
- Python 3.11+ with the project venv already installed from the repo root. No new dependencies — `numpy`, `sentence-transformers`, `litellm`, and `pydantic` are already in `requirements.txt`.

## Setup

`.env` at the repo root supplies your API key. `LLM_MODEL` defaults to `anthropic/claude-sonnet-4-20250514` if unset; pass `--model` to override at runtime without touching `.env`. The script resolves `.env` relative to the source file, so you can run it from any cwd.

The first run downloads the sentence-transformers encoder (`all-MiniLM-L6-v2`, ~80 MB) into your local Hugging Face cache (`~/.cache/huggingface/`). This is a one-time cost — subsequent runs load the encoder from disk in under a second.

### Project layout

```text
project/
├── README.md            this file
├── solution.py          the cache + wrapper + CLI (~650 lines)
├── .gitignore           ignores .cache/
└── queries/
    └── benchmark.txt    15 newline-separated queries (cold vs warm)
```

Read `solution.py` end-to-end before you run it. The cache class, the helpers, and the orchestrator are all independently callable from a REPL — useful for poking at the persistence format or testing a single layer in isolation.

## How it works

```text
user prompt ──→ canonicalize ──→ Exact-match layer (SHA-256 hash lookup)
                                       │
                                       ↓ (hit or miss)
                                ┌──────┴──────┐
                                │             │
                              hit           miss
                                │             │
                                ↓             ↓
                          return cached   Embed prompt locally
                                          (sentence-transformers)
                                                │
                                                ↓
                                          Semantic layer
                                          (cosine vs all stored embeddings)
                                                │
                                                ↓ (above threshold or below)
                                         ┌──────┴──────┐
                                         │             │
                                       hit           miss
                                         │             │
                                         ↓             ↓
                                    return cached  Call LLM, store entry
                                                       │
                                                       ↓
                                                  return response
```

- **Exact layer — SHA-256 hash lookup.** Canonicalize the prompt (strip leading/trailing whitespace, collapse internal whitespace runs to single spaces, lowercase), concatenate with model + system + temperature, then SHA-256 the whole thing. The key is the dictionary lookup index. Exact hits are the cheapest possible cache action — no model call, no embedding, no vector math. They catch the case every production cache cares about most: the same user asking the same question twice. The verdict on hit includes `similarity=1.000` and the truncated key for the audit log.
- **Semantic layer — cosine-similarity over local embeddings.** On exact miss, embed the canonicalized prompt with `all-MiniLM-L6-v2` (384-dim, runs in ~30ms on CPU), then compute cosine similarity against every stored embedding in the cache. If the top-1 score is above the threshold (default 0.95), return that entry's response. The layer runs second because embedding costs roughly two orders of magnitude more than a hash lookup but two orders of magnitude less than an LLM call — the classic middle tier. The threshold is the knob that controls the precision-recall trade-off: too low and unrelated prompts collide, too high and obvious paraphrases miss.
- **Miss-and-store — LLM call + persistence.** On both exact and semantic miss, call the LLM via LiteLLM `completion(...)`, capture the response plus cost and latency, and store a new `CacheEntry` keyed by the exact-match hash with its embedding alongside. The store path is where TTL + LRU eviction runs: drop expired entries first (any entry older than `ttl_seconds`), then if the cache is still over its `max_entries` budget, evict the least-recently-used entries until it fits. Every store path also triggers a save — JSON for the entries, a single `.npy` file for the embeddings, atomically swapped via `.tmp` files so a crash mid-save never corrupts the cache.

The shape is workflow-first with hard short-circuit boundaries, same as Module 16. Each lookup attempt is recorded on the `CachedResult` regardless of whether later layers ran, so the audit trail is complete even when the pipeline exits early on an exact hit.

## Build it step by step

1. **Define the Pydantic models** (`CacheEntry`, `CacheHit`, `CachedResult`). `CacheEntry` carries `key`, `prompt`, `response`, `model`, `system`, `temperature`, `cost`, `created_at`, and `last_accessed_at` — everything needed to score eviction and replay a hit. `CacheHit` carries `layer` (one of `"exact"`, `"semantic"`, `"miss"`), `similarity` (1.0 for exact, top-1 cosine for semantic, 0.0 for miss), `key` (the matched entry's key, or None on miss), and `lookup_ms`. `CachedResult` is the top-level record: `prompt`, `response`, `hit: CacheHit`, `cost_paid`, `cost_saved`, and `latency_ms_total` / `latency_ms_lookup` / `latency_ms_llm`. The orchestrator returns this alongside the response so callers get the full audit trail without extra plumbing.
2. **Write the helpers `_canonicalize(text)` and `_cache_key(prompt, model, system, temperature)`.** Canonicalization is the boring-but-load-bearing step: strip outer whitespace, collapse internal whitespace runs with `re.sub(r"\s+", " ", text)`, lowercase. The cache key is `hashlib.sha256(...)` over the canonicalized prompt joined with the other parameters by a delimiter unlikely to appear in any of them (`"\x00"` works). Return the hex digest. Both helpers are pure functions — test them on a few strings in a REPL before you wire them in.
3. **Write `_embed(text, encoder)` and `_cosine_similarity_search(query_emb, all_embs, threshold)`.** `_embed` is a one-liner around `encoder.encode([text], normalize_embeddings=True)[0]` — the `normalize_embeddings=True` flag means cosine similarity reduces to a dot product, which is faster and avoids numerical drift. `_cosine_similarity_search` is `(all_embs @ query_emb).argmax()` with a threshold check: return `(best_index, best_score)` if `best_score >= threshold`, else `(None, best_score)`. Both helpers run on numpy arrays — no torch, no GPU dance, just CPU matmul.
4. **Build the `TieredCache` class skeleton.** Constructor takes `cache_dir`, `max_entries` (default 1000), `ttl_seconds` (default 7 days), `similarity_threshold` (default 0.95), and `encoder_name` (default `all-MiniLM-L6-v2`). Initialize `self.entries: dict[str, CacheEntry] = {}`, `self.embeddings: dict[str, np.ndarray] = {}`, and stub out `_load()` and `_save()`. The encoder is the heavy import — wrap it in a `@property` that lazy-loads on first access so `--stats` and `--flush` don't pay the import cost.
5. **Implement `TieredCache._load()` and `TieredCache._save()`.** On load: if `cache_dir/cache.json` exists, parse the JSON entries into `CacheEntry` objects, and if `cache_dir/embeddings.npy` exists, load it as a 2D array and zip the rows with the entry keys (order-preserved by dict insertion). On save: write JSON to `cache.json.tmp`, write `np.stack(list(self.embeddings.values()))` to `embeddings.npy.tmp`, then `os.replace` both to their final paths. The atomic swap is the whole point — a crash mid-write leaves the old files intact. Compaction is automatic: `_save` only writes the entries that survived eviction, so the file shrinks naturally.
6. **Implement `TieredCache.lookup(prompt, model, system, temperature) -> CacheHit`.** Compute the key. If `key in self.entries`, update `last_accessed_at`, return `CacheHit(layer="exact", similarity=1.0, key=key, lookup_ms=...)`. Otherwise embed the canonicalized prompt, run `_cosine_similarity_search` against the stacked embeddings, and if the score clears the threshold return `CacheHit(layer="semantic", similarity=score, key=matched_key, lookup_ms=...)` after updating that entry's `last_accessed_at`. Else return `CacheHit(layer="miss", similarity=top_score_if_any, key=None, lookup_ms=...)`. Time the whole thing with `time.perf_counter()` so `lookup_ms` is real.
7. **Implement `TieredCache.store(...)` with the TTL + LRU eviction pass.** Construct the `CacheEntry`, embed the prompt, insert into both dicts. Then run eviction in two passes: (a) drop entries whose `created_at` is older than `now - ttl_seconds`; (b) if `len(self.entries) > max_entries`, sort the remaining entries by `last_accessed_at` ascending and pop the oldest until the count fits. Finally call `_save()`. The TTL-first ordering matters: an expired entry that's also LRU is double-counted otherwise.
8. **Implement `TieredCache.flush()` and `TieredCache.stats()`.** `flush()` clears both dicts and deletes the two files on disk (best-effort — ignore `FileNotFoundError`). `stats()` returns a dict with `entry_count`, `oldest_entry_age_seconds`, `newest_entry_age_seconds`, `total_cost_saved_estimate` (sum of `entry.cost` × number-of-hits, if you track hits — otherwise just sum `entry.cost` as a lower bound), and `cache_size_bytes` (sum of the two files' sizes). Keep these as pure read methods — they're for the `--stats` CLI mode and shouldn't mutate state.
9. **Implement the `cached_chat(...)` orchestrator.** Signature: `cached_chat(prompt, *, cache, model, system=None, temperature=0.7, no_cache=False) -> tuple[str, CachedResult]`. If `no_cache` is set, skip straight to `completion()` and return a `CachedResult` with `hit.layer="miss"` (the bypass mode is useful for benchmarking and for one-off "I don't trust the cache here" calls). Otherwise: `cache.lookup(...)` → on hit return the cached response with `cost_paid=0.0`, `cost_saved=matched_entry.cost`; on miss call `completion()`, capture cost and latency, `cache.store(...)`, return the response with `cost_paid=cost`, `cost_saved=0.0`. Time every segment separately so the `CachedResult.latency_ms_*` fields are honest.
10. **Implement the benchmark runner** `run_benchmark(queries, cache, model, system_prompt) -> dict`. Flush the cache to start cold. Run the queries sequentially through `cached_chat`, collecting `CachedResult` per query — this is the cold pass, every entry should miss. Run the same queries through `cached_chat` again — this is the warm pass, most should hit (exact for exact repeats, semantic for paraphrases). Aggregate: cold-pass total cost and latency, warm-pass exact/semantic/miss counts, warm-pass total cost and latency, savings (cold − warm) in both dollars and seconds, hit rate. Write three print helpers: `_print_single_report(result)`, `_print_stats(stats_dict)`, and `_print_benchmark_report(benchmark_dict)`. Match the output previews below exactly — the visual structure is part of the audit trail.
11. **Wire up the CLI with `argparse`.** Positional `prompt` (optional — required unless one of the mode flags is set). `--benchmark PATH` (corpus mode, loads newline-separated queries from a file). `--stats` (print cache stats and exit). `--flush` (clear the cache and exit). `--no-cache` (flag, bypasses the cache for this call). `--threshold FLOAT` (default 0.95, overrides the semantic threshold). `--model NAME` (default from `LLM_MODEL` env). `--system TEXT` (optional system prompt). `--cache-dir PATH` (default `.cache/` next to `solution.py`). Parse args, dispatch to the right mode, call the matching print helper. Exit nonzero only on hard errors (model API failure, unreadable benchmark file) — a missed cache is not an error.

Each step is small and independently testable. Steps 2, 3, and 7 in particular should pass on their own before you wire up the class — canonicalize a few strings, cosine-search a tiny synthetic embedding matrix, exercise the eviction logic by stuffing the cache past its limit. If those three are solid, the orchestrator is just sequencing around them.

## Run it

```bash
python solution.py "What is the capital of France?"
python solution.py "What is the capital of France?"
python solution.py "Tell me the capital of France."
python solution.py --benchmark queries/benchmark.txt
python solution.py --stats
python solution.py --flush
python solution.py "Your prompt" --no-cache
python solution.py "Your prompt" --threshold 0.98
```

Expected single-prompt output (exact values vary):

```text
=== Cache Report ===
Prompt:        "What is the capital of France?"

Lookup:
  exact         HIT     similarity=1.000  (key=a1b2c3d4...)
  semantic      (skipped — exact hit)

Response (cached): "The capital of France is Paris."

Cost:          paid $0.000000 | saved $0.001200
Latency:       cache lookup 2ms | LLM 0ms | total 2ms
```

Expected benchmark-mode output (exact values vary):

```text
=== Benchmark: queries/benchmark.txt (15 prompts) ===
Model:         anthropic/claude-sonnet-4-20250514
Threshold:     0.95

Cold pass (cache empty):
  15/15 calls, all misses
  Total cost:    $0.018400
  Total latency: 22.1s

Warm pass (cache populated):
  exact hits:    11
  semantic hits:  3
  misses:         1
  Total cost:    $0.001100
  Total latency: 3.4s

Savings:       $0.017300 (94.0%) | 18.7s faster (84.6%)
Hit rate:      14/15 (93.3%)
```

Use `--no-cache` to measure the no-cache baseline on a single prompt, and `--threshold` to feel out how the precision-recall curve bends — push it to 0.98 and the semantic layer almost never fires; drop it to 0.85 and you'll see false-positive hits on unrelated prompts.

## Extensions

Once the base cache works, these are the natural next experiments:

1. **Add tenant isolation.** Take a `--tenant-id` flag, include it in the cache key (so the same prompt under two tenants produces two different keys), and verify cross-tenant lookups miss. The interesting design question is whether tenants share the embedding matrix (faster, but a same-prompt cross-tenant near-miss could theoretically leak a few bits of timing info) or hold separate matrices (slower, but provably isolated).
2. **Add a `cache_version` constant** and include it in the cache key. Bump it programmatically — bumping invalidates every entry without deleting the files, so the old data is still around if you need to roll back. Same trick deployment systems use for cache-busting CSS bundles; here it solves the "my prompt template changed, every cached response is now stale" problem.
3. **Wrap `cached_chat` in a model-tier router** (Haiku → Sonnet → Opus). Run a cheap classifier on the prompt (heuristic or another LLM), route easy prompts to Haiku and hard prompts to Opus, then layer this cache underneath. Chain the two wrappers — the router decides which model, the cache decides whether to call any model at all. Stack the savings.
4. **Swap the on-disk persistence for a Redis backend** behind the same `TieredCache` interface. The caller code doesn't change — only `_load` and `_save` move to Redis hashes + a separate vector index (RediSearch, or compute cosine in Python against an in-memory matrix hydrated from Redis on boot). The interesting bit is the eviction story: Redis has its own LRU policy, so you can either delegate eviction entirely or keep doing it in Python and just use Redis as the durable store.
5. **Add a `--threshold-search` mode** that runs the benchmark against a grid of thresholds (`[0.90, 0.92, 0.94, 0.95, 0.96, 0.98]`) and prints the false-positive-rate vs hit-rate trade-off curve. The output is a small table: threshold, hit rate, false-positive rate (cases where the semantic layer hit but a human would say "no, that's a different question"). Requires a labeled corpus — a few hand-graded "these should hit each other" / "these shouldn't" pairs — but it turns threshold tuning from vibes into a measurement.

## Reference

Cross-links for context:

- [Module 17 README](../README.md) — the cache hierarchy, the cost math, why semantic caching needs a threshold knob, and where this fits in the production stack.
- [Module 03 (Embeddings & Vector Search)](../../03-embeddings-vector-search/) — the cosine-similarity layer is the same primitive applied to caching instead of retrieval; `all-MiniLM-L6-v2` is the same sentence-transformers encoder.
- [Module 04 (AI API Layer)](../../04-ai-api-layer/) — `completion()` is the same LiteLLM wrapper, and the cost/latency capture follows the same pattern.
- [Module 15 (Evaluation & Testing)](../../15-evaluation-testing/) — the benchmark mode is the scorecard shape applied to caching: queries in, savings table out.
- [Module 16 (AI Safety & Guardrails)](../../16-ai-safety-guardrails/) — the cheap-then-expensive layered ordering and the structured-report return value are the same shape, reused for a different threat (cost) instead of safety.

**Next:** Phase 4 wraps with deployment monitoring — the `CachedResult` you produce here is the shape that flows into the dashboards there, with hit rate, cost-saved-per-window, and warm-vs-cold latency as the rolling-window signals.
