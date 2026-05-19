"""
Tiered Cache Wrapper — complete reference implementation.

Wraps any LLM call with two cache layers:
  - Exact-match: SHA-256 of canonicalized (prompt + model + system + temperature)
  - Semantic: local sentence-transformers embedding + cosine similarity

On miss, calls the LLM and stores the response + embedding + metadata.
Persists to ./.cache/cache.json + ./.cache/embeddings.npy (gitignored).
Eviction: TTL primary, LRU as overflow guard. Compacted on every save.

Run:
    python solution.py "your prompt here"
    python solution.py --benchmark queries/benchmark.txt
    python solution.py --stats
    python solution.py --flush
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Literal

import numpy as np
from dotenv import load_dotenv
from litellm import completion, completion_cost
from pydantic import BaseModel, Field

load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

MODEL = os.getenv("LLM_MODEL", "anthropic/claude-sonnet-4-20250514")
DEFAULT_SYSTEM_PROMPT = "You are a helpful, concise assistant. Answer directly."
DEFAULT_CACHE_DIR = Path(__file__).resolve().parent / ".cache"
DEFAULT_TTL_SECONDS = 86400          # 24h
DEFAULT_MAX_SIZE = 1000
DEFAULT_SEMANTIC_THRESHOLD = 0.95
DEFAULT_ENCODER = "all-MiniLM-L6-v2"
EMBED_DIM = 384                      # all-MiniLM-L6-v2 output dim
CACHE_VERSION = 1                    # bump to invalidate all entries on load


# ---------- Pydantic models ----------


HitLayer = Literal["exact", "semantic", "miss"]


class CacheEntry(BaseModel):
    key: str
    prompt: str
    response: str
    model: str
    system: str
    temperature: float
    embedding_idx: int
    created_at: float
    last_accessed_at: float
    hit_count: int = 0
    cost: float = 0.0


class CacheHit(BaseModel):
    layer: HitLayer
    entry: CacheEntry | None = None
    similarity: float | None = None
    lookup_latency_ms: int


class CachedResult(BaseModel):
    response: str
    hit: CacheHit
    original_call_cost: float
    saved_cost: float
    sut_latency_ms: int
    total_latency_ms: int


# `from __future__ import annotations` defers type evaluation, so Pydantic models
# with Literal aliases need explicit rebuilds before validation/construction.
CacheEntry.model_rebuild()
CacheHit.model_rebuild()
CachedResult.model_rebuild()


# ---------- Helpers ----------


def _canonicalize(text: str, lower: bool = True) -> str:
    """Strip + normalize internal whitespace + optional lowercase."""
    parts = text.split()
    out = " ".join(parts)
    return out.lower() if lower else out


def _cache_key(prompt: str, model: str, system: str, temperature: float) -> str:
    """SHA-256 hex of canonicalized (prompt + model + system + temperature)."""
    payload = (
        _canonicalize(prompt)
        + "\n---\n"
        + model
        + "\n---\n"
        + _canonicalize(system)
        + "\n---\n"
        + f"{temperature:.4f}"
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _embed(text: str, encoder) -> np.ndarray:
    """Encode one string. Returns a (EMBED_DIM,) float32 unit vector."""
    vec = encoder.encode(text, normalize_embeddings=True)
    return np.asarray(vec, dtype=np.float32)


def _cosine_similarity_search(
    query_emb: np.ndarray,
    all_embs: np.ndarray,
    threshold: float,
) -> tuple[int, float] | None:
    """Return (best_idx, score) if best score >= threshold, else None.

    Assumes both inputs are already L2-normalized so dot product == cosine sim.
    """
    if all_embs.size == 0:
        return None
    scores = all_embs @ query_emb
    best_idx = int(np.argmax(scores))
    best_score = float(scores[best_idx])
    if best_score >= threshold:
        return best_idx, best_score
    return None


def _judge_cost(response) -> float:
    """Return LiteLLM-reported cost for a completion, or 0.0 on error."""
    try:
        return completion_cost(completion_response=response) or 0.0
    except Exception:
        return 0.0


# ---------- TieredCache ----------


class TieredCache:
    """Two-layer cache with on-disk persistence.

    Layer 1: exact-match (SHA-256 hash lookup).
    Layer 2: semantic (cosine similarity over local embeddings).

    Eviction: TTL primary + LRU at max_size.
    Persistence: cache.json + embeddings.npy, compacted on every save.
    """

    def __init__(
        self,
        cache_dir: Path = DEFAULT_CACHE_DIR,
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
        max_size: int = DEFAULT_MAX_SIZE,
        semantic_threshold: float = DEFAULT_SEMANTIC_THRESHOLD,
        encoder_name: str = DEFAULT_ENCODER,
    ):
        self.cache_dir = Path(cache_dir)
        self.ttl_seconds = ttl_seconds
        self.max_size = max_size
        self.semantic_threshold = semantic_threshold
        self.encoder_name = encoder_name
        self._encoder = None
        self._entries: dict[str, CacheEntry] = {}
        self._embeddings: np.ndarray = np.zeros((0, EMBED_DIM), dtype=np.float32)
        self._lifetime_hits = 0
        self._lifetime_misses = 0
        self._lifetime_saved = 0.0
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._load()

    @property
    def encoder(self):
        if self._encoder is None:
            print("Loading sentence-transformers encoder (first call only)...")
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer(self.encoder_name)
        return self._encoder

    def _load(self) -> None:
        """Load cache.json + embeddings.npy. Reset gracefully on corruption."""
        cache_file = self.cache_dir / "cache.json"
        embeds_file = self.cache_dir / "embeddings.npy"
        if not cache_file.exists():
            return
        try:
            data = json.loads(cache_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            print(f"WARNING: cache.json unreadable ({e}); resetting cache.")
            return
        if data.get("version") != CACHE_VERSION:
            print(f"WARNING: cache version mismatch; resetting cache.")
            return
        if data.get("meta", {}).get("encoder_name") != self.encoder_name:
            print("WARNING: encoder changed since last save; resetting cache.")
            return

        try:
            entries = [CacheEntry.model_validate(e) for e in data.get("entries", [])]
        except Exception as e:
            print(f"WARNING: cache entries invalid ({e}); resetting cache.")
            return

        embeds: np.ndarray
        if embeds_file.exists():
            try:
                embeds = np.load(embeds_file)
                if embeds.dtype != np.float32 or embeds.shape[1] != EMBED_DIM:
                    raise ValueError(f"unexpected shape/dtype {embeds.shape}/{embeds.dtype}")
                if embeds.shape[0] != len(entries):
                    raise ValueError(
                        f"embedding count {embeds.shape[0]} != entry count {len(entries)}"
                    )
            except Exception as e:
                print(f"WARNING: embeddings.npy unreadable ({e}); resetting cache.")
                return
        else:
            print("WARNING: embeddings.npy missing; resetting cache.")
            return

        self._entries = {e.key: e for e in entries}
        self._embeddings = embeds

    def _save(self) -> None:
        """Compact + write both files atomically.

        Walks live entries, reassigns embedding_idx = 0..N-1, writes a compact
        embeddings.npy plus cache.json with the renumbered entries.
        """
        live = list(self._entries.values())
        new_embeds = np.zeros((len(live), EMBED_DIM), dtype=np.float32)
        for new_idx, entry in enumerate(live):
            new_embeds[new_idx] = self._embeddings[entry.embedding_idx]
            entry.embedding_idx = new_idx

        cache_file = self.cache_dir / "cache.json"
        embeds_file = self.cache_dir / "embeddings.npy"
        cache_tmp = cache_file.with_suffix(".json.tmp")
        embeds_tmp = embeds_file.with_suffix(".npy.tmp")

        with open(embeds_tmp, "wb") as f:
            np.save(f, new_embeds)
        payload = {
            "version": CACHE_VERSION,
            "meta": {
                "saved_at": time.time(),
                "encoder_name": self.encoder_name,
            },
            "entries": [e.model_dump() for e in live],
        }
        cache_tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        os.replace(cache_tmp, cache_file)
        os.replace(embeds_tmp, embeds_file)
        self._embeddings = new_embeds

    def lookup(
        self,
        prompt: str,
        model: str,
        system: str,
        temperature: float,
    ) -> CacheHit:
        """Try exact then semantic lookup. Returns CacheHit (layer='miss' if neither hit)."""
        start = time.perf_counter()
        key = _cache_key(prompt, model, system, temperature)

        # ---- Layer 1: exact ----
        if key in self._entries:
            entry = self._entries[key]
            if time.time() - entry.created_at <= self.ttl_seconds:
                entry.last_accessed_at = time.time()
                entry.hit_count += 1
                latency_ms = int((time.perf_counter() - start) * 1000)
                self._lifetime_hits += 1
                self._lifetime_saved += entry.cost
                return CacheHit(
                    layer="exact",
                    entry=entry,
                    similarity=1.0,
                    lookup_latency_ms=latency_ms,
                )
            # Expired — fall through.

        # ---- Layer 2: semantic ----
        if self._embeddings.shape[0] == 0:
            latency_ms = int((time.perf_counter() - start) * 1000)
            self._lifetime_misses += 1
            return CacheHit(layer="miss", lookup_latency_ms=latency_ms)

        query_emb = _embed(prompt, self.encoder)
        match = _cosine_similarity_search(query_emb, self._embeddings, self.semantic_threshold)
        if match is not None:
            idx, score = match
            # Find the entry with this embedding_idx.
            entry = next((e for e in self._entries.values() if e.embedding_idx == idx), None)
            if entry is not None and time.time() - entry.created_at <= self.ttl_seconds:
                # Also gate by model + system + temperature equality — semantic similarity
                # between PROMPTS isn't enough; the call params must still match exactly.
                if (
                    entry.model == model
                    and _canonicalize(entry.system) == _canonicalize(system)
                    and abs(entry.temperature - temperature) < 1e-6
                ):
                    entry.last_accessed_at = time.time()
                    entry.hit_count += 1
                    latency_ms = int((time.perf_counter() - start) * 1000)
                    self._lifetime_hits += 1
                    self._lifetime_saved += entry.cost
                    return CacheHit(
                        layer="semantic",
                        entry=entry,
                        similarity=score,
                        lookup_latency_ms=latency_ms,
                    )

        latency_ms = int((time.perf_counter() - start) * 1000)
        self._lifetime_misses += 1
        return CacheHit(layer="miss", lookup_latency_ms=latency_ms)

    def store(
        self,
        prompt: str,
        response: str,
        model: str,
        system: str,
        temperature: float,
        cost: float,
    ) -> None:
        """Add an entry, evict per TTL + LRU, persist."""
        key = _cache_key(prompt, model, system, temperature)
        embedding = _embed(prompt, self.encoder)

        # Append the embedding; we'll renumber on _save() compaction.
        new_idx = self._embeddings.shape[0]
        self._embeddings = np.vstack([self._embeddings, embedding[np.newaxis, :]])

        now = time.time()
        entry = CacheEntry(
            key=key,
            prompt=_canonicalize(prompt),
            response=response,
            model=model,
            system=system,
            temperature=temperature,
            embedding_idx=new_idx,
            created_at=now,
            last_accessed_at=now,
            hit_count=0,
            cost=cost,
        )
        self._entries[key] = entry
        self._evict()
        self._save()

    def _evict(self) -> None:
        """Apply TTL pass then LRU pass. Mutates self._entries in place.

        Note: only entries are mutated; embeddings are compacted by _save().
        """
        now = time.time()
        # TTL pass.
        expired = [k for k, e in self._entries.items() if now - e.created_at > self.ttl_seconds]
        for k in expired:
            del self._entries[k]

        # LRU pass.
        if len(self._entries) > self.max_size:
            sorted_keys = sorted(
                self._entries.keys(),
                key=lambda k: self._entries[k].last_accessed_at,
            )
            n_to_drop = len(self._entries) - self.max_size
            for k in sorted_keys[:n_to_drop]:
                del self._entries[k]

    def flush(self) -> None:
        """Wipe in-memory state and delete the cache files."""
        self._entries = {}
        self._embeddings = np.zeros((0, EMBED_DIM), dtype=np.float32)
        for name in ("cache.json", "embeddings.npy"):
            path = self.cache_dir / name
            if path.exists():
                path.unlink()
        self._lifetime_hits = 0
        self._lifetime_misses = 0
        self._lifetime_saved = 0.0

    def stats(self) -> dict:
        total_lookups = self._lifetime_hits + self._lifetime_misses
        hit_rate = (self._lifetime_hits / total_lookups) if total_lookups else 0.0
        oldest = (
            min((time.time() - e.created_at for e in self._entries.values()), default=0.0)
        )
        return {
            "cache_dir": str(self.cache_dir),
            "entries": len(self._entries),
            "max_size": self.max_size,
            "embeddings_shape": list(self._embeddings.shape),
            "lifetime_hits": self._lifetime_hits,
            "lifetime_misses": self._lifetime_misses,
            "hit_rate": hit_rate,
            "lifetime_saved": round(self._lifetime_saved, 6),
            "oldest_age_s": round(oldest, 1),
            "ttl_seconds": self.ttl_seconds,
            "semantic_threshold": self.semantic_threshold,
            "encoder_name": self.encoder_name,
        }


# ---------- Orchestrator ----------


def cached_chat(
    user_input: str,
    cache: TieredCache | None,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    model: str = MODEL,
    temperature: float = 0.0,
) -> tuple[str, CachedResult]:
    """Wrap an LLM call with the tiered cache.

    Returns (response, CachedResult). When cache is None, bypasses the cache
    entirely (always makes the LLM call, reports layer='miss').
    """
    total_start = time.perf_counter()

    # ---- Cache lookup ----
    if cache is None:
        hit = CacheHit(layer="miss", lookup_latency_ms=0)
    else:
        hit = cache.lookup(user_input, model, system_prompt, temperature)

    if hit.layer in ("exact", "semantic") and hit.entry is not None:
        total_ms = int((time.perf_counter() - total_start) * 1000)
        return hit.entry.response, CachedResult(
            response=hit.entry.response,
            hit=hit,
            original_call_cost=0.0,
            saved_cost=hit.entry.cost,
            sut_latency_ms=0,
            total_latency_ms=total_ms,
        )

    # ---- Miss: call the LLM ----
    sut_start = time.perf_counter()
    response = completion(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
        ],
        temperature=temperature,
    )
    sut_latency_ms = int((time.perf_counter() - sut_start) * 1000)
    cost = _judge_cost(response)
    raw_response = response.choices[0].message.content or ""

    # ---- Store (skip when bypassing) ----
    if cache is not None:
        cache.store(
            prompt=user_input,
            response=raw_response,
            model=model,
            system=system_prompt,
            temperature=temperature,
            cost=cost,
        )

    total_ms = int((time.perf_counter() - total_start) * 1000)
    return raw_response, CachedResult(
        response=raw_response,
        hit=hit,
        original_call_cost=cost,
        saved_cost=0.0,
        sut_latency_ms=sut_latency_ms,
        total_latency_ms=total_ms,
    )


# ---------- Benchmark ----------


def _load_queries(path: str | Path) -> list[str]:
    """Read newline-separated queries; skip blank lines and # comments."""
    p = Path(path).resolve()
    queries: list[str] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        queries.append(line)
    return queries


def run_benchmark(
    queries: list[str],
    cache: TieredCache,
    model: str,
    system_prompt: str,
) -> dict:
    """Run each query once on a cold cache, then once on the warm cache.

    Returns aggregate stats: cold cost/latency, warm cost/latency, hit counts.
    """
    # ---- Cold pass: ensure the cache is empty ----
    cache.flush()
    cold_cost = 0.0
    cold_latency_ms = 0
    print("\nCold pass (cache empty):")
    for q in queries:
        _, result = cached_chat(q, cache, system_prompt=system_prompt, model=model)
        cold_cost += result.original_call_cost
        cold_latency_ms += result.total_latency_ms
    print(f"  {len(queries)}/{len(queries)} calls, all misses")
    print(f"  Total cost:    ${cold_cost:.6f}")
    print(f"  Total latency: {cold_latency_ms/1000:.1f}s")

    # ---- Warm pass: re-run on the populated cache ----
    warm_cost = 0.0
    warm_latency_ms = 0
    exact_hits = 0
    semantic_hits = 0
    misses = 0
    print("\nWarm pass (cache populated):")
    for q in queries:
        _, result = cached_chat(q, cache, system_prompt=system_prompt, model=model)
        warm_cost += result.original_call_cost
        warm_latency_ms += result.total_latency_ms
        if result.hit.layer == "exact":
            exact_hits += 1
        elif result.hit.layer == "semantic":
            semantic_hits += 1
        else:
            misses += 1
    print(f"  exact hits:    {exact_hits}")
    print(f"  semantic hits: {semantic_hits}")
    print(f"  misses:        {misses}")
    print(f"  Total cost:    ${warm_cost:.6f}")
    print(f"  Total latency: {warm_latency_ms/1000:.1f}s")

    saved_cost = cold_cost - warm_cost
    saved_latency_ms = cold_latency_ms - warm_latency_ms
    savings_pct = (saved_cost / cold_cost * 100) if cold_cost > 0 else 0.0
    speedup_pct = (saved_latency_ms / cold_latency_ms * 100) if cold_latency_ms > 0 else 0.0
    hits = exact_hits + semantic_hits
    return {
        "n": len(queries),
        "cold_cost": cold_cost,
        "warm_cost": warm_cost,
        "saved_cost": saved_cost,
        "savings_pct": savings_pct,
        "cold_latency_ms": cold_latency_ms,
        "warm_latency_ms": warm_latency_ms,
        "saved_latency_ms": saved_latency_ms,
        "speedup_pct": speedup_pct,
        "exact_hits": exact_hits,
        "semantic_hits": semantic_hits,
        "misses": misses,
        "hit_rate": hits / len(queries) if queries else 0.0,
    }


# ---------- Printing ----------


def _print_single_report(prompt: str, result: CachedResult) -> None:
    print("\n=== Cache Report ===")
    print(f"Prompt:        {prompt!r}")
    print("\nLookup:")
    if result.hit.layer == "miss":
        print(f"  exact         MISS    (key not found)")
        print(f"  semantic      MISS    (no entry above threshold)")
    elif result.hit.layer == "exact":
        sim = f"similarity={result.hit.similarity:.3f}" if result.hit.similarity else ""
        key_short = (result.hit.entry.key[:8] + "...") if result.hit.entry else "?"
        print(f"  exact         HIT     {sim}  (key={key_short})")
        print(f"  semantic      (skipped — exact hit)")
    elif result.hit.layer == "semantic":
        sim = f"similarity={result.hit.similarity:.3f}" if result.hit.similarity is not None else ""
        print(f"  exact         MISS    (no key match)")
        print(f"  semantic      HIT     {sim}")

    is_hit = result.hit.layer in ("exact", "semantic")
    label = "Response (cached)" if is_hit else "Response (fresh)"
    snippet = (result.response[:200] + "...") if len(result.response) > 200 else result.response
    print(f"\n{label}: {snippet!r}")
    print(
        f"\nCost:          paid ${result.original_call_cost:.6f} | "
        f"saved ${result.saved_cost:.6f}"
    )
    print(
        f"Latency:       cache lookup {result.hit.lookup_latency_ms}ms | "
        f"LLM {result.sut_latency_ms}ms | total {result.total_latency_ms}ms"
    )


def _print_benchmark_report(corpus_path: str, model: str, threshold: float, agg: dict) -> None:
    print(f"\n=== Benchmark: {corpus_path} ({agg['n']} prompts) ===")
    print(f"Model:         {model}")
    print(f"Threshold:     {threshold}")
    print(
        f"\nSavings:       "
        f"${agg['saved_cost']:.6f} ({agg['savings_pct']:.1f}%) | "
        f"{agg['saved_latency_ms']/1000:.1f}s faster ({agg['speedup_pct']:.1f}%)"
    )
    print(f"Hit rate:      {agg['exact_hits'] + agg['semantic_hits']}/{agg['n']} ({agg['hit_rate']*100:.1f}%)")


def _print_stats(cache: TieredCache) -> None:
    s = cache.stats()
    print("\n=== Cache Stats ===")
    print(f"Cache dir:     {s['cache_dir']}")
    print(f"Entries:       {s['entries']} / {s['max_size']}")
    print(f"Embeddings:    {s['embeddings_shape'][0]} vectors, {s['embeddings_shape'][1]} dims")
    print(
        f"Total saved:   ${s['lifetime_saved']:.6f} across "
        f"{s['lifetime_hits']} hits"
    )
    print(f"Hit-rate:      {s['hit_rate']*100:.1f}% (lifetime, this process)")
    if s['entries'] > 0:
        print(f"Oldest entry:  {s['oldest_age_s']/3600:.1f}h ago")
    print(f"TTL:           {s['ttl_seconds']/3600:.1f}h")
    print(f"Threshold:     {s['semantic_threshold']}")
    print(f"Encoder:       {s['encoder_name']}")


# ---------- CLI ----------


def main() -> None:
    parser = argparse.ArgumentParser(description="Tiered Cache Wrapper — Module 17")
    parser.add_argument("prompt", nargs="?", default=None,
                        help="A single user prompt to run through cached_chat")
    parser.add_argument("--benchmark", default=None,
                        help="Path to a newline-separated query corpus")
    parser.add_argument("--stats", action="store_true",
                        help="Print cache stats and exit")
    parser.add_argument("--flush", action="store_true",
                        help="Delete the cache files (cache.json + embeddings.npy) and exit")
    parser.add_argument("--no-cache", action="store_true",
                        help="Bypass the cache for one call (still pays LLM cost)")
    parser.add_argument("--threshold", type=float, default=DEFAULT_SEMANTIC_THRESHOLD,
                        help=f"Semantic similarity threshold (default {DEFAULT_SEMANTIC_THRESHOLD})")
    parser.add_argument("--model", default=MODEL,
                        help=f"Model override (default: {MODEL})")
    parser.add_argument("--system", default=DEFAULT_SYSTEM_PROMPT,
                        help="Override the wrapped system prompt")
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR),
                        help=f"Cache directory (default: {DEFAULT_CACHE_DIR})")
    args = parser.parse_args()

    if not (0.0 <= args.threshold <= 1.0):
        parser.error("--threshold must be in [0.0, 1.0]")

    modes_used = sum([
        bool(args.prompt),
        bool(args.benchmark),
        bool(args.stats),
        bool(args.flush),
    ])
    if modes_used == 0:
        parser.error("provide a prompt, --benchmark, --stats, or --flush")
    if modes_used > 1:
        parser.error("provide only one of: prompt, --benchmark, --stats, --flush")

    cache = TieredCache(
        cache_dir=Path(args.cache_dir),
        semantic_threshold=args.threshold,
    )

    if args.flush:
        cache.flush()
        print(f"Flushed cache at {args.cache_dir}.")
        return

    if args.stats:
        _print_stats(cache)
        return

    if args.benchmark:
        try:
            queries = _load_queries(args.benchmark)
        except (FileNotFoundError, OSError) as e:
            parser.error(f"could not load benchmark: {e}")
        if not queries:
            parser.error(f"benchmark file is empty: {args.benchmark}")
        print(f"Running benchmark on {len(queries)} queries...")
        agg = run_benchmark(queries, cache, model=args.model, system_prompt=args.system)
        _print_benchmark_report(args.benchmark, args.model, args.threshold, agg)
        return

    # Single-prompt mode
    active_cache = None if args.no_cache else cache
    response, result = cached_chat(
        args.prompt,
        active_cache,
        system_prompt=args.system,
        model=args.model,
    )
    _print_single_report(args.prompt, result)


if __name__ == "__main__":
    main()
