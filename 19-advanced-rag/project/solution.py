"""Module 19 Project: Hybrid Retrieval + Reranker.

A 5-stage RAG pipeline:

    1. Dense retrieve  (FAISS, top-K=20)
    2. Sparse retrieve (BM25,  top-K=20)
    3. RRF fusion      (top-K=20)
    4. Cross-encoder rerank (top-K=3)
    5. LLM answer with [Source N] citations

Plus a labeled eval set (15 queries) and a CLI with 5 modes:
    --ask, --explain, --eval, --list, --flush

Run from the project dir or from the repo root.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import sys
import time
from hashlib import sha256
from pathlib import Path
from typing import Literal

import faiss
import numpy as np
from dotenv import load_dotenv
from litellm import completion, completion_cost
from pydantic import BaseModel
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

MODEL = os.getenv("LLM_MODEL", "openai/gpt-4o-mini")
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
RERANK_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

K_DENSE = 20
K_SPARSE = 20
K_FUSED = 20
K_FINAL = 3
RRF_K = 60

INDEX_DIR = Path(__file__).parent / ".rag_index"
META_PATH = INDEX_DIR / "meta.json"
CHUNKS_PATH = INDEX_DIR / "chunks.json"
EMBEDS_PATH = INDEX_DIR / "embeddings.npy"
BM25_PATH = INDEX_DIR / "bm25.pkl"


class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    section: str
    text: str


class RankedChunk(BaseModel):
    chunk: Chunk
    score: float
    rank: int  # 1-based


class EvalQuery(BaseModel):
    question: str
    relevant_chunk_ids: list[str]
    kind: Literal["keyword", "semantic", "mixed"]


Chunk.model_rebuild()
RankedChunk.model_rebuild()
EvalQuery.model_rebuild()


CORPUS: dict[str, dict] = {
    "auth": {
        "title": "Authentication & Security",
        "sections": [
            {
                "section": "Bcrypt and Password Hashing",
                "text": (
                    "Storing passwords requires a slow, computationally expensive hashing "
                    "algorithm such as bcrypt, scrypt, or Argon2. Unlike fast hashes like "
                    "SHA-256, these slow hashing schemes make brute-force attacks impractical "
                    "by design. Always combine the password with a per-user salt before "
                    "hashing so that precomputed rainbow table lookups cannot match stored "
                    "digests. Modern guidance favors Argon2id for new systems, while bcrypt "
                    "remains acceptable for legacy deployments that are already in production."
                ),
            },
            {
                "section": "OAuth 2.0 Flow",
                "text": (
                    "OAuth 2.0 is a delegated authorization framework most commonly used "
                    "through the authorization code flow. The client redirects the user to "
                    "the authorization server, receives a short-lived authorization code at "
                    "its redirect URI, and exchanges that code server-to-server for an access "
                    "token. A random state parameter must be generated per request and "
                    "verified on the callback to defend against CSRF attacks that would "
                    "otherwise trick the user into binding the attacker's account."
                ),
            },
            {
                "section": "API Keys and Service Accounts",
                "text": (
                    "An API key is a long opaque string presented by a machine client to "
                    "authenticate against a backend service. Treat each key as a credential: "
                    "issue it through a service account, store it only in a secrets manager, "
                    "and rotate it on a regular schedule or whenever a team member leaves. "
                    "Apply tight scoping so each key authorizes only the endpoints and data "
                    "the caller actually needs, and revoke immediately on any suspicion of "
                    "leakage in logs or repositories."
                ),
            },
            {
                "section": "Session Cookies and CSRF",
                "text": (
                    "Browser session cookies should be issued with the HttpOnly attribute "
                    "so JavaScript cannot read them, the Secure flag so they only travel over "
                    "HTTPS, and an appropriate SameSite policy to limit cross-origin sending. "
                    "Even with SameSite=Lax, state-changing endpoints should still validate a "
                    "CSRF token submitted alongside the form to defend against same-site "
                    "subdomain takeovers and edge cases in legacy browsers that do not honor "
                    "the SameSite attribute correctly."
                ),
            },
            {
                "section": "JWT and Bearer Tokens",
                "text": (
                    "A JWT is a signed JSON document carrying claims about the subject, such "
                    "as the user id, scopes, and an expiration timestamp in the exp claim. "
                    "Symmetric HS256 signatures are simple but require the verifier to share "
                    "the secret, while asymmetric RS256 lets services verify with a public "
                    "key without ever holding the signing key. Always validate the signature, "
                    "the issuer, and the expiration claims on every request; never trust an "
                    "unsigned bearer token at face value."
                ),
            },
            {
                "section": "Multi-Factor Authentication",
                "text": (
                    "Multi-factor authentication requires a second factor in addition to the "
                    "password so that a leaked password alone is not enough to log in. TOTP "
                    "codes from an authenticator app are widely supported but vulnerable to "
                    "phishing of the one-time code. WebAuthn and passkeys use device-bound "
                    "asymmetric keys to bind authentication to the origin, making the second "
                    "factor unphishable and substantially raising the cost of credential "
                    "theft for high-value accounts."
                ),
            },
            {
                "section": "Rate Limiting Login Endpoints",
                "text": (
                    "Login endpoints are a prime target for credential stuffing and brute "
                    "force attacks, so they should be rate limited per account and per source "
                    "address. After a handful of failed attempts, introduce exponential "
                    "backoff so each subsequent attempt takes longer than the last, and "
                    "trigger a temporary account lockout once a threshold is crossed. Pair "
                    "this with monitoring that alerts on sudden spikes of failed logins from "
                    "distributed sources, which is the signature of a credential stuffing run."
                ),
            },
        ],
    },
    "cache": {
        "title": "Caching & Performance",
        "sections": [
            {
                "section": "TTL and Expiration",
                "text": (
                    "A TTL, or time to live, defines how long a cached entry remains valid "
                    "before it is considered to expire. Once the configured duration has "
                    "elapsed, the entry is treated as stale and the next reader either "
                    "refetches from the origin or serves the stale value while a background "
                    "task refreshes it. Choosing a TTL is a tradeoff between freshness and "
                    "origin load: short TTLs reduce staleness, long TTLs absorb more traffic "
                    "at the cost of letting writes take longer to propagate to readers."
                ),
            },
            {
                "section": "Cache Invalidation Strategies",
                "text": (
                    "There are only two hard things in computer science, Phil Karlton once "
                    "said, and one of them is cache invalidation. Write-through caches "
                    "update the cache synchronously on every write so readers always see "
                    "the latest value, at the cost of write latency. Write-behind buffers "
                    "writes and flushes them asynchronously, accepting a window of "
                    "inconsistency for higher throughput. Explicit invalidation on mutation, "
                    "with versioned keys, is often the most predictable strategy in practice."
                ),
            },
            {
                "section": "LRU and LFU Eviction",
                "text": (
                    "When a cache fills up, an eviction policy decides which entry to drop "
                    "to make room. LRU, or least recently used, discards the entry that has "
                    "not been touched for the longest time and works well when the working "
                    "set fits in memory and access has good temporal locality. LFU, or least "
                    "frequently used, instead tracks how often each key is accessed and "
                    "favors keeping popular items, which can outperform LRU on long-tailed "
                    "workloads where a small set of keys absorbs most of the traffic."
                ),
            },
            {
                "section": "Cache Stampede",
                "text": (
                    "A cache stampede, sometimes called a thundering herd, happens when a "
                    "hot key expires and many concurrent readers all miss the cache at once "
                    "and pile onto the origin to recompute it. The classic defense is "
                    "request coalescing: only the first caller actually recomputes while "
                    "the rest wait on a single in-flight promise. An alternative is early "
                    "refresh, where a small probability of refresh fires before the TTL "
                    "officially expires, smoothing demand against the origin over time."
                ),
            },
            {
                "section": "Negative Caching",
                "text": (
                    "Negative caching means storing a placeholder for a known absence, such "
                    "as a 404 response, instead of repeatedly querying the origin for an "
                    "item that does not exist. By choosing to cache the miss with a short "
                    "TTL, the system absorbs floods of lookups for invalid keys without "
                    "hammering the database every time. The TTL on negative entries should "
                    "be much shorter than positive entries so that newly created resources "
                    "do not appear missing for long after they are added upstream."
                ),
            },
            {
                "section": "Multi-Tier Caching",
                "text": (
                    "A multi-tier or tiered cache combines a fast in-process L1 layer with "
                    "a larger shared L2 such as Redis. Reads check the L1 first and only "
                    "fall back to L2 on a miss, giving the local + remote topology both "
                    "sub-microsecond hits and cross-process sharing. Writes typically update "
                    "L1 and asynchronously propagate to L2, and L1 invalidations are "
                    "broadcast via pub-sub so other instances do not serve stale copies "
                    "once the source of truth in L2 has changed."
                ),
            },
        ],
    },
    "observability": {
        "title": "Observability & Monitoring",
        "sections": [
            {
                "section": "Logs vs Metrics vs Traces",
                "text": (
                    "The three pillars of observability are logs, metrics, and traces. Logs "
                    "are timestamped records of discrete events, useful for debugging exact "
                    "what-happened questions after the fact. Metrics are numeric time series "
                    "aggregated over windows, useful for dashboards and alerts on rates and "
                    "percentiles. Traces stitch together the spans of a single request as it "
                    "flows across services, useful for understanding latency contributions "
                    "and causal ordering in a distributed system end to end."
                ),
            },
            {
                "section": "Structured Logging",
                "text": (
                    "Structured logging emits JSON logs with named fields rather than free "
                    "form text, so downstream tooling can filter, aggregate, and join "
                    "records without brittle regex parsing. Standardize on a small set of "
                    "log levels such as debug, info, warn, and error, and always include a "
                    "correlation id that ties together every log line emitted while "
                    "handling a single request. Consistent field names across services pay "
                    "for themselves the first time you have to debug a multi-service outage."
                ),
            },
            {
                "section": "Prometheus and Time-Series",
                "text": (
                    "Prometheus is a pull-based monitoring system that periodically scrapes "
                    "HTTP endpoints exposing time series in a simple text format. Each "
                    "series is identified by a metric name plus a set of labels, and the "
                    "combination defines the cardinality of the data. High-cardinality "
                    "labels such as raw user ids can blow up storage and query cost, so "
                    "labels should be bounded enums like region or status code rather than "
                    "unbounded identifiers that grow without limit over time."
                ),
            },
            {
                "section": "Distributed Tracing",
                "text": (
                    "Distributed tracing represents a request as a tree of spans, each with "
                    "a start time, a duration, and a parent span pointer back to its caller. "
                    "Every span shares a common trace id so the collector can reassemble "
                    "the call graph after the fact. OpenTelemetry has emerged as the "
                    "vendor-neutral standard for instrumentation, providing SDKs in most "
                    "languages and a wire protocol that can be ingested by Jaeger, Tempo, "
                    "Honeycomb, Datadog, and other tracing backends without code changes."
                ),
            },
            {
                "section": "SLOs and Error Budgets",
                "text": (
                    "An SLO is a target for a measurable SLI such as request success rate "
                    "or latency at the 99th percentile over a rolling window. The gap "
                    "between perfect and the SLO is the error budget, which the team is "
                    "allowed to spend on planned risk like deploys and experiments. "
                    "Burn rate alerts fire when the budget is being consumed faster than "
                    "the window allows, signaling that release velocity should slow down "
                    "until the budget recovers to a healthy level."
                ),
            },
            {
                "section": "Alerting on Symptoms",
                "text": (
                    "Good alerting fires on user-visible symptoms such as elevated error "
                    "rates or slow responses, not on every internal cause that might or "
                    "might not affect customers. Cause-based alerts produce noise because "
                    "they fire during routine events like deploys and scaling, training "
                    "the on-call rotation to ignore the page. Reserve a page for things a "
                    "human must act on now; route lower-severity signals to dashboards and "
                    "tickets so the inbox does not drown the real incidents in noise."
                ),
            },
            {
                "section": "Debugging in Production",
                "text": (
                    "Debugging in production requires tools that do not require restarting "
                    "or attaching a debugger to a running process. A core dump captured at "
                    "the moment of failure preserves heap and stack for offline analysis, "
                    "and remote profiling samples a running service to find hot paths. "
                    "Wrap risky changes behind a feature flag so they can be disabled "
                    "without redeploying, and keep the rollback path warm so a bad release "
                    "can be reverted in minutes rather than hours."
                ),
            },
        ],
    },
    "network": {
        "title": "Networking & TLS",
        "sections": [
            {
                "section": "TCP Handshake",
                "text": (
                    "TCP establishes a connection using a three-way handshake. The client "
                    "sends a SYN packet with an initial sequence number, the server replies "
                    "with a SYN-ACK that acknowledges the client's number and adds its own, "
                    "and the client completes the dance with an ACK that acknowledges the "
                    "server's number. After this exchange both sides have agreed on starting "
                    "sequence numbers and can begin streaming application data with "
                    "guaranteed ordering and reliable delivery on top of IP packets."
                ),
            },
            {
                "section": "TLS 1.3 Handshake",
                "text": (
                    "TLS 1.3 reduces handshake latency to a single round trip. The "
                    "ClientHello carries the supported cipher suites and a key share, and "
                    "the ServerHello immediately responds with the chosen suite and its "
                    "matching key share so application data can flow on the next flight. "
                    "Session resumption with pre-shared keys enables zero-round-trip data "
                    "on subsequent connections. All cipher suites in TLS 1.3 provide PFS, "
                    "so past traffic stays safe even if long-term keys are later compromised."
                ),
            },
            {
                "section": "HTTP/2 and HTTP/3",
                "text": (
                    "HTTP/2 multiplexes many logical streams over a single TCP connection, "
                    "removing the per-request connection overhead of HTTP/1.1. However, "
                    "because everything still rides on one TCP socket, a single dropped "
                    "packet stalls all streams in head-of-line blocking. HTTP/3 fixes this "
                    "by running over QUIC, a UDP-based transport that multiplexes streams "
                    "independently at the transport layer, so a lost packet only blocks "
                    "the affected stream and the others continue to make forward progress."
                ),
            },
            {
                "section": "DNS Resolution",
                "text": (
                    "DNS resolves human-readable names into IP addresses through a "
                    "recursive resolver that walks the hierarchy from root to the "
                    "authoritative server for the zone. An A record maps a name directly "
                    "to an IPv4 address, while a CNAME aliases one name to another so the "
                    "resolver continues following the chain. Each record carries a TTL "
                    "that controls how long resolvers and clients may cache the answer "
                    "before they must query again for a fresh value."
                ),
            },
            {
                "section": "Load Balancing Algorithms",
                "text": (
                    "Load balancers distribute traffic across a pool of backends using "
                    "policies tuned to the workload. Round robin rotates through backends "
                    "in order and is simple but blind to load. Least connections sends each "
                    "new request to the backend currently handling the fewest, balancing "
                    "long-lived requests better. Weighted variants bias toward more capable "
                    "machines, while consistent hashing routes the same key to the same "
                    "backend so caches and session affinity survive churn in the pool."
                ),
            },
            {
                "section": "Connection Pooling",
                "text": (
                    "A connection pool keeps a set of already-open TCP and TLS connections "
                    "warm and reuses them for successive requests, avoiding the handshake "
                    "cost on the hot path. Without pooling, closed sockets pile up in "
                    "TIME_WAIT for two minutes by default and rapidly exhaust the "
                    "ephemeral port range on a busy client. Tune the pool size to match "
                    "the concurrency of the upstream service and monitor checkout latency "
                    "to detect when the pool is saturating under load."
                ),
            },
        ],
    },
    "database": {
        "title": "Database Fundamentals",
        "sections": [
            {
                "section": "B-Tree vs LSM-Tree",
                "text": (
                    "A B-tree stores data in sorted pages updated in place, giving fast "
                    "point lookups and range scans with low read amplification, at the "
                    "cost of random writes that fragment pages over time. An LSM-tree "
                    "instead buffers writes in memory and flushes them as immutable sorted "
                    "files compacted in the background, which makes writes cheap but "
                    "amplifies reads across many files until compaction catches up. "
                    "Compaction itself is a major source of write amplification in LSM "
                    "engines and must be tuned against the workload."
                ),
            },
            {
                "section": "Index Selection",
                "text": (
                    "Choosing the right index starts with understanding the query and the "
                    "data distribution. High selectivity columns, where a value matches "
                    "few rows, are good candidates for an index; low cardinality columns "
                    "like boolean flags often are not. A covering index that includes "
                    "every column the query reads lets the engine answer entirely from "
                    "the index without touching the heap. Always confirm the plan with "
                    "EXPLAIN to see whether the planner actually uses the index you built."
                ),
            },
            {
                "section": "ACID and Isolation Levels",
                "text": (
                    "ACID stands for atomicity, consistency, isolation, and durability, "
                    "the guarantees a relational engine offers around transactions. The "
                    "isolation property is parameterized by levels: read committed avoids "
                    "dirty reads but allows non-repeatable reads, repeatable read pins "
                    "row snapshots so a re-read returns the same value, and serializable "
                    "additionally rules out the phantom read where a new row appears in "
                    "a previously evaluated range condition between two queries."
                ),
            },
            {
                "section": "Connection Pooling for Databases",
                "text": (
                    "Postgres assigns a process per connection, so each open connection "
                    "consumes real memory and CPU. Applications should never open a fresh "
                    "connection per request; instead they go through a connection pooler "
                    "such as PgBouncer that multiplexes many client sessions onto a much "
                    "smaller set of backend connections. Without a pooler, busy services "
                    "quickly hit max_connections and tip into pool exhaustion, where new "
                    "requests block waiting for a backend slot to free up."
                ),
            },
            {
                "section": "Replication and Read Replicas",
                "text": (
                    "A primary database streams its write-ahead log to one or more replica "
                    "nodes that apply the changes to stay in sync. Read traffic can be "
                    "directed to a read replica to scale out capacity, but replication lag "
                    "means a replica may briefly serve a value that is older than what was "
                    "just committed on the primary. Read-after-write consistency requires "
                    "either reading the user's own writes from the primary or waiting for "
                    "the replica to catch up to the relevant log position."
                ),
            },
            {
                "section": "Sharding Strategies",
                "text": (
                    "Sharding partitions a logical dataset across multiple physical "
                    "databases so each shard holds only a slice of the rows. The choice "
                    "of shard key determines how evenly traffic distributes; a poorly "
                    "chosen key creates a hot shard that absorbs disproportionate load. "
                    "Rebalancing as the dataset grows is operationally expensive because "
                    "rows must be moved without disrupting writes, which is why range and "
                    "hash sharding schemes prefer keys with naturally uniform distribution."
                ),
            },
            {
                "section": "Backup and PITR",
                "text": (
                    "A full backup captures the state of the database at a single moment, "
                    "while point-in-time recovery layers continuous WAL archiving on top "
                    "so the database can be restored to any second between backups. RPO, "
                    "or recovery point objective, is the maximum acceptable data loss "
                    "measured in time, and RTO is the maximum tolerable downtime during "
                    "restore. Together these two numbers drive backup frequency, retention, "
                    "and the choice between cold restores and warm standby topologies."
                ),
            },
        ],
    },
}


EVAL_SET: list[EvalQuery] = [
    # --- keyword (5) ---
    EvalQuery(question="what is bcrypt", relevant_chunk_ids=["auth#bcrypt-and-password-hashing"], kind="keyword"),
    EvalQuery(question="how do I prevent cache stampede", relevant_chunk_ids=["cache#cache-stampede"], kind="keyword"),
    EvalQuery(question="what does WebAuthn do", relevant_chunk_ids=["auth#multi-factor-authentication"], kind="keyword"),
    EvalQuery(question="explain QUIC and HTTP/3", relevant_chunk_ids=["network#http-2-and-http-3"], kind="keyword"),
    EvalQuery(question="what is PgBouncer for", relevant_chunk_ids=["database#connection-pooling-for-databases"], kind="keyword"),

    # --- semantic (5) ---
    EvalQuery(question="how do I make stored sessions go away after a while", relevant_chunk_ids=["cache#ttl-and-expiration"], kind="semantic"),
    EvalQuery(question="why do logins get harder after several failed attempts", relevant_chunk_ids=["auth#rate-limiting-login-endpoints"], kind="semantic"),
    EvalQuery(question="how do I keep one slow database server from blocking all reads", relevant_chunk_ids=["database#replication-and-read-replicas"], kind="semantic"),
    EvalQuery(question="how do I avoid alerts firing every time someone deploys", relevant_chunk_ids=["observability#alerting-on-symptoms"], kind="semantic"),
    EvalQuery(question="how do I make sure two clients connecting from the same address go to the same backend", relevant_chunk_ids=["network#load-balancing-algorithms"], kind="semantic"),

    # --- mixed (5) ---
    EvalQuery(question="what does the state parameter do in OAuth", relevant_chunk_ids=["auth#oauth-2-0-flow"], kind="mixed"),
    EvalQuery(question="which eviction policy should I use for a working set that fits in memory", relevant_chunk_ids=["cache#lru-and-lfu-eviction"], kind="mixed"),
    EvalQuery(question="why does my Prometheus scrape interval matter for cardinality", relevant_chunk_ids=["observability#prometheus-and-time-series"], kind="mixed"),
    EvalQuery(question="what isolation level prevents phantom reads", relevant_chunk_ids=["database#acid-and-isolation-levels"], kind="mixed"),
    EvalQuery(question="how does TLS 1.3 reduce handshake latency", relevant_chunk_ids=["network#tls-1-3-handshake"], kind="mixed"),
]


_SLUG_RE = re.compile(r"[^a-z0-9]+")
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _slugify(text: str) -> str:
    return _SLUG_RE.sub("-", text.lower()).strip("-")


def _tokenize_bm25(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _corpus_hash(corpus: dict) -> str:
    blob = json.dumps(corpus, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return sha256(blob).hexdigest()


def _safe_cost(response) -> float:
    try:
        return float(completion_cost(completion_response=response) or 0.0)
    except Exception:
        return 0.0


def _extract_chunks(corpus: dict) -> list[Chunk]:
    chunks: list[Chunk] = []
    for doc_id, doc in corpus.items():
        for section_entry in doc["sections"]:
            section = section_entry["section"]
            chunk_id = f"{doc_id}#{_slugify(section)}"
            chunks.append(Chunk(
                chunk_id=chunk_id,
                doc_id=doc_id,
                section=section,
                text=section_entry["text"].strip(),
            ))
    return chunks


_embedder: SentenceTransformer | None = None
_cross_encoder: CrossEncoder | None = None


def _get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBED_MODEL_NAME)
    return _embedder


def _get_cross_encoder() -> CrossEncoder:
    global _cross_encoder
    if _cross_encoder is None:
        print("loading reranker (first call only, ~5s)...", file=sys.stderr)
        _cross_encoder = CrossEncoder(RERANK_MODEL_NAME)
    return _cross_encoder


def _build_index(corpus: dict) -> tuple[list[Chunk], faiss.IndexFlatIP, BM25Okapi, np.ndarray]:
    chunks = _extract_chunks(corpus)
    embedder = _get_embedder()
    embeddings = embedder.encode(
        [c.text for c in chunks],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)
    faiss_index = faiss.IndexFlatIP(embeddings.shape[1])
    faiss_index.add(embeddings)
    bm25 = BM25Okapi([_tokenize_bm25(c.text) for c in chunks])
    return chunks, faiss_index, bm25, embeddings


def _save_index(chunks: list[Chunk], embeddings: np.ndarray, bm25: BM25Okapi, corpus_hash: str) -> None:
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    meta = {
        "corpus_sha256": corpus_hash,
        "embed_model_name": EMBED_MODEL_NAME,
        "n_chunks": len(chunks),
    }

    meta_tmp = META_PATH.with_suffix(META_PATH.suffix + ".tmp")
    chunks_tmp = CHUNKS_PATH.with_suffix(CHUNKS_PATH.suffix + ".tmp")
    embeds_tmp = EMBEDS_PATH.with_suffix(EMBEDS_PATH.suffix + ".tmp")
    bm25_tmp = BM25_PATH.with_suffix(BM25_PATH.suffix + ".tmp")

    meta_tmp.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    chunks_tmp.write_text(
        json.dumps([c.model_dump() for c in chunks], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    # IMPORTANT: pass an open file handle to np.save (not a path/string).
    # np.save rewrites a path that lacks .npy by appending .npy, which would
    # turn embeddings.npy.tmp into embeddings.npy.tmp.npy and break the rename.
    # An open file handle suppresses that rewrite.
    with open(embeds_tmp, "wb") as f:
        np.save(f, embeddings)
    with open(bm25_tmp, "wb") as f:
        pickle.dump(bm25, f)

    os.replace(meta_tmp, META_PATH)
    os.replace(chunks_tmp, CHUNKS_PATH)
    os.replace(embeds_tmp, EMBEDS_PATH)
    os.replace(bm25_tmp, BM25_PATH)


def _load_index(corpus_hash: str) -> tuple[list[Chunk], faiss.IndexFlatIP, BM25Okapi, np.ndarray] | None:
    if not all(p.exists() for p in (META_PATH, CHUNKS_PATH, EMBEDS_PATH, BM25_PATH)):
        return None
    try:
        meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    except Exception:
        return None
    if meta.get("corpus_sha256") != corpus_hash:
        return None
    if meta.get("embed_model_name") != EMBED_MODEL_NAME:
        return None
    try:
        chunks_data = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
        chunks = [Chunk(**c) for c in chunks_data]
        embeddings = np.load(EMBEDS_PATH).astype(np.float32)
        with open(BM25_PATH, "rb") as f:
            bm25 = pickle.load(f)
    except Exception:
        return None
    faiss_index = faiss.IndexFlatIP(embeddings.shape[1])
    faiss_index.add(embeddings)
    return chunks, faiss_index, bm25, embeddings


def _get_or_build_index(corpus: dict) -> tuple[list[Chunk], faiss.IndexFlatIP, BM25Okapi, np.ndarray]:
    corpus_hash = _corpus_hash(corpus)
    cached = _load_index(corpus_hash)
    if cached is not None:
        return cached
    print(f"building index for {sum(len(d['sections']) for d in corpus.values())} chunks...", file=sys.stderr)
    chunks, faiss_index, bm25, embeddings = _build_index(corpus)
    _save_index(chunks, embeddings, bm25, corpus_hash)
    print(f"saved index to {INDEX_DIR}", file=sys.stderr)
    return chunks, faiss_index, bm25, embeddings


def _dense_retrieve(query: str, faiss_index: faiss.IndexFlatIP, chunks: list[Chunk], k: int = K_DENSE) -> list[RankedChunk]:
    embedder = _get_embedder()
    query_vec = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
    scores, indices = faiss_index.search(query_vec, min(k, len(chunks)))
    out: list[RankedChunk] = []
    for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
        if idx < 0:
            continue
        out.append(RankedChunk(chunk=chunks[int(idx)], score=float(score), rank=rank))
    return out


def _sparse_retrieve(query: str, bm25: BM25Okapi, chunks: list[Chunk], k: int = K_SPARSE) -> list[RankedChunk]:
    tokens = _tokenize_bm25(query)
    if not tokens:
        return []
    scores = bm25.get_scores(tokens)
    order = np.argsort(scores)[::-1][:k]
    out: list[RankedChunk] = []
    for rank, idx in enumerate(order, start=1):
        score = float(scores[int(idx)])
        if score <= 0.0:
            break
        out.append(RankedChunk(chunk=chunks[int(idx)], score=score, rank=rank))
    return out


def _rrf_fuse(rankings: list[list[RankedChunk]], k_const: int = RRF_K, top_n: int = K_FUSED) -> list[RankedChunk]:
    fused_score: dict[str, float] = {}
    chunk_by_id: dict[str, Chunk] = {}
    for ranking in rankings:
        for r in ranking:
            fused_score[r.chunk.chunk_id] = fused_score.get(r.chunk.chunk_id, 0.0) + 1.0 / (k_const + r.rank)
            chunk_by_id[r.chunk.chunk_id] = r.chunk
    ordered = sorted(fused_score.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
    return [
        RankedChunk(chunk=chunk_by_id[cid], score=score, rank=rank)
        for rank, (cid, score) in enumerate(ordered, start=1)
    ]


def _rerank(query: str, candidates: list[RankedChunk], top_n: int = K_FINAL) -> list[RankedChunk]:
    if not candidates:
        return []
    cross_encoder = _get_cross_encoder()
    pairs = [(query, c.chunk.text) for c in candidates]
    scores = cross_encoder.predict(pairs)
    scored = list(zip(candidates, [float(s) for s in scores]))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [
        RankedChunk(chunk=c.chunk, score=score, rank=rank)
        for rank, (c, score) in enumerate(scored[:top_n], start=1)
    ]


def _build_prompt(question: str, ranked: list[RankedChunk]) -> list[dict]:
    if not ranked:
        return [
            {"role": "system", "content": "You answer questions strictly from provided context. If no context is given, respond exactly with: 'No relevant context found.'"},
            {"role": "user", "content": f"Question: {question}\n\nContext: (none)"},
        ]
    sources_block = "\n\n".join(
        f"[Source {i}] ({r.chunk.chunk_id})\n{r.chunk.text}"
        for i, r in enumerate(ranked, start=1)
    )
    return [
        {"role": "system", "content": "Answer the user's question using ONLY the provided context. Cite each fact you use with [Source N] where N is the source number."},
        {"role": "user", "content": f"Question: {question}\n\nContext:\n{sources_block}"},
    ]


def _ask(question: str, faiss_index, bm25, chunks) -> dict:
    t0 = time.perf_counter()
    dense = _dense_retrieve(question, faiss_index, chunks)
    sparse = _sparse_retrieve(question, bm25, chunks)
    fused = _rrf_fuse([dense, sparse])
    reranked = _rerank(question, fused)
    t_retrieve_ms = (time.perf_counter() - t0) * 1000.0

    messages = _build_prompt(question, reranked)
    response = completion(model=MODEL, messages=messages)
    answer = response.choices[0].message.content
    usage = getattr(response, "usage", None)
    return {
        "answer": answer,
        "sources": reranked,
        "tokens_in": getattr(usage, "prompt_tokens", 0) if usage else 0,
        "tokens_out": getattr(usage, "completion_tokens", 0) if usage else 0,
        "cost_usd": _safe_cost(response),
        "n_dense": len(dense),
        "n_sparse": len(sparse),
        "n_fused": len(fused),
        "n_reranked": len(reranked),
        "retrieve_ms": t_retrieve_ms,
    }


def _recall_at_k(retrieved: list[RankedChunk], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    hits = sum(1 for r in retrieved[:k] if r.chunk.chunk_id in relevant)
    return hits / len(relevant)


def _mrr_at_k(retrieved: list[RankedChunk], relevant: set[str], k: int) -> float:
    for rank, r in enumerate(retrieved[:k], start=1):
        if r.chunk.chunk_id in relevant:
            return 1.0 / rank
    return 0.0


_STRATEGIES = ("dense-only", "bm25-only", "rrf-fused", "rrf+rerank")


def _retrievals_for_query(question: str, faiss_index, bm25, chunks) -> dict[str, list[RankedChunk]]:
    dense = _dense_retrieve(question, faiss_index, chunks)
    sparse = _sparse_retrieve(question, bm25, chunks)
    fused = _rrf_fuse([dense, sparse])
    reranked = _rerank(question, fused, top_n=K_FUSED)  # keep depth for fair recall@10 comparison
    return {
        "dense-only": dense,
        "bm25-only": sparse,
        "rrf-fused": fused,
        "rrf+rerank": reranked,
    }


def _run_eval(eval_set: list[EvalQuery], faiss_index, bm25, chunks) -> dict:
    agg: dict[str, dict[str, list[float]]] = {
        s: {"recall@3": [], "recall@10": [], "MRR@10": []} for s in _STRATEGIES
    }
    by_kind: dict[str, dict[str, list[float]]] = {
        kind: {s: [] for s in _STRATEGIES} for kind in ("keyword", "semantic", "mixed")
    }
    for q in eval_set:
        relevant = set(q.relevant_chunk_ids)
        retrievals = _retrievals_for_query(q.question, faiss_index, bm25, chunks)
        for strategy, retrieved in retrievals.items():
            agg[strategy]["recall@3"].append(_recall_at_k(retrieved, relevant, 3))
            agg[strategy]["recall@10"].append(_recall_at_k(retrieved, relevant, 10))
            agg[strategy]["MRR@10"].append(_mrr_at_k(retrieved, relevant, 10))
            by_kind[q.kind][strategy].append(_recall_at_k(retrieved, relevant, 3))

    def _mean(xs: list[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    return {
        "overall": {s: {m: _mean(v) for m, v in metrics.items()} for s, metrics in agg.items()},
        "by_kind": {k: {s: _mean(v) for s, v in d.items()} for k, d in by_kind.items()},
    }


def _print_list(corpus: dict, chunks: list[Chunk], eval_set: list[EvalQuery]) -> None:
    print(f"Corpus: {len(corpus)} docs, {len(chunks)} chunks indexed")
    counts = {doc_id: 0 for doc_id in corpus}
    for c in chunks:
        counts[c.doc_id] += 1
    for doc_id, doc in corpus.items():
        print(f"  {doc_id:<14} {counts[doc_id]} chunks  ({doc['title']})")
    print()
    by_kind = {"keyword": 0, "semantic": 0, "mixed": 0}
    for q in eval_set:
        by_kind[q.kind] += 1
    print(f"Eval set: {len(eval_set)} queries  ({by_kind['keyword']} keyword, {by_kind['semantic']} semantic, {by_kind['mixed']} mixed)")


def _print_ask(result: dict) -> None:
    print(f"[retrieval: {result['n_dense']} dense + {result['n_sparse']} sparse -> {result['n_fused']} fused -> {result['n_reranked']} reranked in {result['retrieve_ms']:.0f} ms]\n")
    print(result["answer"])
    print()
    print("Sources:")
    for i, r in enumerate(result["sources"], start=1):
        print(f"  [{i}] {r.chunk.chunk_id:<40} (rerank score: {r.score:6.2f})")
    print()
    print(f"tokens: {result['tokens_in']} in / {result['tokens_out']} out  cost: ${result['cost_usd']:.6f}")


def _format_panel(title: str, ranking: list[RankedChunk], score_fmt: str, show: int = 5) -> list[str]:
    lines = [title, f"  {'rank':<5} {'score':<8} chunk_id"]
    for r in ranking[:show]:
        lines.append(f"  {r.rank:<5} {format(r.score, score_fmt):<8} {r.chunk.chunk_id}")
    while len(lines) < show + 2:
        lines.append("")
    return lines


def _print_explain(question: str, faiss_index, bm25, chunks) -> None:
    print(f'Query: "{question}"\n')
    dense = _dense_retrieve(question, faiss_index, chunks)
    sparse = _sparse_retrieve(question, bm25, chunks)
    fused = _rrf_fuse([dense, sparse])
    reranked = _rerank(question, fused)

    left = _format_panel(f"DENSE (FAISS, top 5 of {len(dense)})", dense, ".3f")
    right = _format_panel(f"BM25 (top 5 of {len(sparse)})", sparse, ".2f")
    for l, r in zip(left, right):
        print(f"{l:<50}{r}")
    print()
    left = _format_panel(f"RRF FUSED (top 5 of {len(fused)})", fused, ".4f")
    right = _format_panel(f"RERANKED (top {min(3, len(reranked))} of {K_FUSED})", reranked, ".2f")
    for l, r in zip(left, right):
        print(f"{l:<50}{r}")


def _print_eval(report: dict) -> None:
    print("Running 15-query eval set across 4 strategies...\n")
    print(f"{'Strategy':<18} {'recall@3':<10} {'recall@10':<11} {'MRR@10':<8}")
    print("-" * 50)
    for s in _STRATEGIES:
        m = report["overall"][s]
        print(f"{s:<18} {m['recall@3']:<10.2f} {m['recall@10']:<11.2f} {m['MRR@10']:<8.2f}")
    print()
    print("By query kind (recall@3):")
    for kind, by_strat in report["by_kind"].items():
        cells = "  ".join(f"{s}:{by_strat[s]:.2f}" for s in _STRATEGIES)
        print(f"  {kind:<10} {cells}")


def _flush() -> int:
    if not INDEX_DIR.exists():
        print("nothing to flush")
        return 0
    n_files = 0
    n_bytes = 0
    for p in INDEX_DIR.iterdir():
        if p.is_file():
            n_files += 1
            n_bytes += p.stat().st_size
            p.unlink()
    INDEX_DIR.rmdir()
    print(f"Deleted {INDEX_DIR.name}/ ({n_files} files, {n_bytes / 1024 / 1024:.1f} MB)")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Module 19: Hybrid Retrieval + Reranker")
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--ask", metavar="QUESTION", help="Run the full pipeline and print the answer.")
    g.add_argument("--explain", metavar="QUESTION", help="Show all 4 rankings side-by-side. No LLM call.")
    g.add_argument("--eval", action="store_true", help="Run the labeled eval set. No LLM call.")
    g.add_argument("--list", action="store_true", help="List corpus and eval set.")
    g.add_argument("--flush", action="store_true", help="Delete .rag_index/ and exit.")
    args = parser.parse_args(argv)

    if args.flush:
        return _flush()

    chunks, faiss_index, bm25, _ = _get_or_build_index(CORPUS)

    if args.list:
        _print_list(CORPUS, chunks, EVAL_SET)
        return 0
    if args.ask:
        result = _ask(args.ask, faiss_index, bm25, chunks)
        _print_ask(result)
        return 0
    if args.explain:
        _print_explain(args.explain, faiss_index, bm25, chunks)
        return 0
    if args.eval:
        report = _run_eval(EVAL_SET, faiss_index, bm25, chunks)
        _print_eval(report)
        return 0
    return 1  # unreachable


if __name__ == "__main__":
    sys.exit(main())
