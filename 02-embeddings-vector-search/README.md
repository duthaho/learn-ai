# Module 02: Embeddings & Vector Search

A deep engineering guide for senior backend developers learning AI Engineering.

---

## Table of Contents

1. [Concept Explanation](#1-concept-explanation)
2. [Why It Matters in Real Systems](#2-why-it-matters-in-real-systems)
3. [Internal Mechanics](#3-internal-mechanics)
4. [Practical Example](#4-practical-example)
5. [Hands-on Implementation](#5-hands-on-implementation)
6. [System Design Perspective](#6-system-design-perspective)
7. [Common Pitfalls](#7-common-pitfalls)
8. [Advanced Topics](#8-advanced-topics)
9. [Exercises](#9-exercises)
10. [Interview / Architect Questions](#10-interview--architect-questions)

---

## 1. Concept Explanation

### What Are Embeddings?

An **embedding** is a dense numerical vector (array of floats) that represents the **meaning** of a piece of text. Think of it as a coordinate in a high-dimensional semantic space — texts with similar meanings land near each other.

```
"How do I reset my password?" → [0.021, -0.034, 0.118, ..., 0.045]  # 1536 floats
"I forgot my login credentials" → [0.019, -0.031, 0.121, ..., 0.042]  # very similar vector
"The weather is nice today"    → [0.872, 0.441, -0.203, ..., 0.667]  # very different vector
```

**Key properties:**
- **Fixed-size:** Every input (word, sentence, paragraph) maps to the same dimensionality (e.g., 1536 for OpenAI `text-embedding-3-small`, 1024 for Voyage AI)
- **Semantic:** "car" and "automobile" have nearby vectors; "car" and "banana" do not
- **Continuous:** Small changes in meaning produce small changes in the vector
- **Language-agnostic:** Good models place "perro" (Spanish) near "dog" (English)

### What Is Vector Search?

**Vector search** (a.k.a. semantic search) finds the most similar vectors in a collection to a given query vector. Instead of keyword matching (`WHERE title LIKE '%password%'`), you find documents by **meaning**.

```
Query:   "How do I change my login?" → vector → search
Results: "Password reset instructions" (similarity: 0.94)
         "Account credentials FAQ"     (similarity: 0.91)
         "Login troubleshooting guide" (similarity: 0.88)
```

No keyword overlap needed. The system understands intent.

### The Backend Engineer Mental Model

If you know databases, think of it this way:

| Traditional DB | Vector DB |
|---|---|
| `INSERT INTO docs (text)` | `INSERT INTO docs (text, embedding)` |
| `WHERE text LIKE '%keyword%'` | `ORDER BY cosine_similarity(embedding, query_embedding) DESC` |
| B-tree index | HNSW / IVF index |
| Exact match | Approximate nearest neighbor (ANN) |

---

## 2. Why It Matters in Real Systems

### The Core Problem Embeddings Solve

LLMs have a **knowledge cutoff** and a **context window limit**. You can't stuff your entire knowledge base into a prompt. Embeddings + vector search let you **retrieve only the relevant pieces** and inject them into the prompt. This is **Retrieval-Augmented Generation (RAG)**.

### Where Companies Use This

| Use Case | Company Example | Why Embeddings |
|---|---|---|
| **Customer support bots** | Zendesk, Intercom | Search knowledge base by user intent, not keywords |
| **Internal knowledge search** | Notion AI, Confluence AI | "Find me the design doc about auth migration" |
| **Code search** | GitHub Copilot, Sourcegraph | Find semantically similar code across repos |
| **E-commerce recommendations** | Amazon, Shopify | "Products similar to what you're looking at" |
| **Legal document discovery** | Harvey AI | Find relevant case law from millions of documents |
| **Anomaly detection** | Financial systems | Transactions that are far from normal patterns |
| **RAG pipelines** | Every production LLM app | Ground LLM responses in factual, up-to-date data |

### Why Not Just Full-Text Search?

Full-text search (Elasticsearch, PostgreSQL `tsvector`) fails when:
- User query uses different words than the document ("car" vs "vehicle")
- Intent matters more than keywords ("how to handle angry customers" should match "de-escalation techniques")
- Cross-language search is needed
- You need to combine text similarity with other modalities (images, code)

**In practice, production systems combine both:** vector search for semantic recall + keyword search for precision (hybrid search).

---

## 3. Internal Mechanics

### 3.1 How Embedding Models Work

Modern embedding models are **transformer encoders** (typically based on BERT architecture, not GPT's decoder architecture):

```
Input text → Tokenizer → Transformer Encoder → Pool hidden states → Normalize → Embedding vector
```

**Step by step:**

1. **Tokenization:** Text is split into subword tokens ("embedding" → ["em", "bed", "ding"])
2. **Encoding:** Each token passes through transformer layers (self-attention + feed-forward). Each token gets a contextualized representation.
3. **Pooling:** Token-level representations are reduced to a single vector. Common strategies:
   - **[CLS] token:** Use the special classification token's output
   - **Mean pooling:** Average all token representations (most common, generally best)
4. **Normalization:** The vector is L2-normalized to unit length, so cosine similarity = dot product

### 3.2 Similarity Metrics

Given two vectors `a` and `b`:

**Cosine Similarity** (most common for text):
```
cos(a, b) = (a · b) / (||a|| × ||b||)
Range: [-1, 1], where 1 = identical direction
```

**Euclidean Distance** (L2):
```
d(a, b) = √(Σ(ai - bi)²)
Range: [0, ∞), where 0 = identical
```

**Dot Product:**
```
dot(a, b) = Σ(ai × bi)
When vectors are normalized: dot product = cosine similarity
```

**For normalized embeddings (which most APIs return), all three give equivalent rankings.** Cosine similarity is the standard for text.

### 3.3 Vector Search Algorithms

#### Brute Force (Flat Index)
- Compare query against every vector
- **O(n × d)** where n = number of vectors, d = dimensions
- Perfect recall, but doesn't scale past ~100K vectors

#### IVF (Inverted File Index)
- Cluster vectors into `nlist` groups using k-means
- At query time, only search the `nprobe` closest clusters
- Trade-off: faster search, slightly lower recall
- Good for 100K–10M vectors

```
Training phase:    vectors → k-means → cluster assignments
Query phase:       query → find closest clusters → search only those clusters
```

#### HNSW (Hierarchical Navigable Small World)
- Build a multi-layer graph where each vector connects to its nearest neighbors
- Top layers are sparse (long-range links), bottom layers are dense (local links)
- Navigate from top to bottom, greedily following closest neighbors
- **O(log n)** query time with high recall
- The dominant algorithm in production (used by pgvector, Pinecone, Weaviate, Qdrant)

```
Layer 3:  A ---------> D                    (sparse, long jumps)
Layer 2:  A ----> C --> D ----> F            (medium density)
Layer 1:  A -> B -> C -> D -> E -> F -> G    (dense, local connections)
Layer 0:  [all vectors with local neighbors]  (full graph)
```

#### Product Quantization (PQ)
- Compress vectors by splitting into subvectors and quantizing each
- Dramatically reduces memory (e.g., 1536 floats → 48 bytes)
- Used in combination with IVF for billion-scale search

### 3.4 The RAG Pipeline Architecture

```
┌──────────────────── INDEXING (offline) ────────────────────┐
│                                                             │
│  Documents → Chunk → Embed → Store in Vector DB             │
│                                                             │
└─────────────────────────────────────────────────────────────┘

┌──────────────────── QUERY (online) ────────────────────────┐
│                                                             │
│  User Query → Embed → Vector Search → Top-K chunks          │
│                          ↓                                  │
│               Chunks + Query → LLM Prompt → Response        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 4. Practical Example

### Real-World Scenario: Internal Documentation Search

**Problem:** Your company has 5,000 internal docs (engineering runbooks, product specs, HR policies). Engineers ask questions in Slack, and the answers exist somewhere in the docs — but nobody can find them.

**Solution:** Build a RAG-powered search API.

**Architecture:**

```
┌─────────────┐     ┌──────────────┐     ┌──────────────┐
│  Docs (MD,  │────→│  Chunking +  │────→│  Vector DB   │
│  PDF, HTML) │     │  Embedding   │     │  (FAISS)     │
└─────────────┘     └──────────────┘     └──────────────┘
                                               │
┌─────────────┐     ┌──────────────┐           │
│  User Query │────→│  Embed Query │──── search ┘
│  via API    │     └──────────────┘
│             │            │
│             │     ┌──────────────┐     ┌──────────────┐
│             │     │  Top-K Chunks│────→│  Claude LLM  │
│             │     └──────────────┘     │  + Context   │
│             │                          └──────┬───────┘
│  Response   │←────────────────────────────────┘
└─────────────┘
```

**Scale considerations:**
- 5,000 docs → ~50,000 chunks → FAISS in-memory is fine (< 500MB RAM)
- 100,000+ docs → consider pgvector or a managed service (Pinecone, Weaviate)
- Need metadata filtering? → pgvector (SQL WHERE + vector search)

---

## 5. Hands-on Implementation

See the accompanying code files:

- **[app.py](app.py)** — Standalone demo: embeddings, similarity, chunking, search, RAG
- **[chunking.py](chunking.py)** — Text chunking strategies (fixed-size, sentence-aware, recursive)
- **[vector_store.py](vector_store.py)** — FAISS vector store abstraction with persistence

### Quick Start

```bash
# From the repo root
pip install -r requirements.txt
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY

# Run the demo
cd 02-embeddings-vector-search
python app.py
```

### Step-by-Step Walkthrough

#### Step 1: Generate Embeddings

We use Voyage AI's embedding model via their API (purpose-built for embeddings, higher quality than general-purpose models for retrieval tasks).

For learning/development, we also provide a local fallback using sentence-transformers so you can experiment without API costs.

```python
# The core operation — turning text into vectors
import voyageai

client = voyageai.Client()  # uses VOYAGE_API_KEY env var

response = client.embed(
    texts=["How do I reset my password?"],
    model="voyage-3-large",
)
embedding = response.embeddings[0]  # list of 1024 floats
```

#### Step 2: Chunk Your Documents

You can't embed a 50-page document as one vector — the meaning gets diluted. Split it into semantically meaningful chunks.

```python
# See chunking.py for full implementation
from chunking import RecursiveChunker

chunker = RecursiveChunker(chunk_size=512, chunk_overlap=50)
chunks = chunker.chunk(long_document_text)
# Each chunk is ~512 tokens with 50-token overlap for context continuity
```

**Chunking strategies compared:**

| Strategy | Pros | Cons | Use When |
|---|---|---|---|
| Fixed-size (by tokens) | Simple, predictable | May cut mid-sentence | Uniform content |
| Sentence-aware | Preserves meaning | Variable chunk sizes | Prose, documentation |
| Recursive (by separators) | Respects document structure | More complex | Markdown, code, structured docs |
| Semantic (by meaning shift) | Best quality | Expensive (requires embedding) | High-value content |

#### Step 3: Build the Vector Index

```python
# See vector_store.py for full implementation
from vector_store import FaissVectorStore

store = FaissVectorStore(dimension=1024)

# Index your chunks
for chunk in chunks:
    embedding = embed(chunk.text)
    store.add(embedding, metadata={"text": chunk.text, "source": chunk.source})

# Persist to disk
store.save("./index_data")
```

#### Step 4: Search by Meaning

```python
query_embedding = embed("How do I change my login credentials?")
results = store.search(query_embedding, top_k=5)

# Returns chunks about password resets, account settings, etc.
# — even though the query doesn't contain those exact words
```

#### Step 5: RAG — Augment the LLM with Retrieved Context

```python
# Build the prompt with retrieved context
context = "\n\n".join([r["text"] for r in results])
prompt = f"""Based on the following documentation, answer the user's question.

Documentation:
{context}

Question: {user_query}

Answer based only on the provided documentation. If the answer isn't in the docs, say so."""

# Send to Claude
response = anthropic_client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    messages=[{"role": "user", "content": prompt}],
)
```

---

## 6. System Design Perspective

### Where Embeddings Fit in a Production Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                        API Gateway                              │
└──────────────┬──────────────────────────────────┬──────────────┘
               │                                  │
    ┌──────────▼──────────┐           ┌───────────▼───────────┐
    │   Ingestion Service  │           │    Query Service       │
    │                      │           │                        │
    │  • Receive documents │           │  • Embed user query    │
    │  • Chunk text        │           │  • Vector search       │
    │  • Generate embeds   │           │  • Re-rank results     │
    │  • Store in vector DB│           │  • Build LLM prompt    │
    │  • Store raw in S3   │           │  • Stream response     │
    └──────────┬───────────┘           └───────────┬────────────┘
               │                                   │
    ┌──────────▼───────────────────────────────────▼────────────┐
    │                     Vector Database                        │
    │  (pgvector / Pinecone / Qdrant / Weaviate)                │
    │                                                            │
    │  Stores: embedding vector + metadata + chunk text          │
    └───────────────────────────────────────────────────────────┘
               │
    ┌──────────▼──────────┐
    │  Object Store (S3)   │
    │  Original documents  │
    └─────────────────────┘
```

### Scaling Decisions

| Scale | Vector Store Choice | Why |
|---|---|---|
| < 100K vectors | FAISS in-memory | Simple, fast, no infra |
| 100K – 10M | pgvector (PostgreSQL) | SQL filtering + vectors, one DB to manage |
| 10M – 100M | Qdrant / Weaviate | Purpose-built, better performance at scale |
| 100M+ | Pinecone / custom sharding | Managed, distributed, billion-scale |

### Key Production Concerns

1. **Embedding model versioning:** If you change models, ALL vectors must be re-embedded. Version your indexes.
2. **Chunking strategy is the #1 lever for RAG quality** — not the LLM model, not the vector DB.
3. **Hybrid search:** Combine vector search (recall) + BM25/keyword search (precision) for best results.
4. **Re-ranking:** Use a cross-encoder model to re-rank top-K results before sending to the LLM.
5. **Caching:** Cache embeddings for repeated queries. Cache LLM responses for identical query+context pairs.
6. **Monitoring:** Track retrieval quality (are the right chunks being returned?) separately from generation quality.

### pgvector — The Backend Engineer's Sweet Spot

If you're already running PostgreSQL (and as a backend dev, you probably are), pgvector gives you vector search without new infrastructure:

```sql
-- Enable the extension
CREATE EXTENSION vector;

-- Create table with vector column
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    content TEXT,
    embedding vector(1024),  -- matches your embedding model's dimensions
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create HNSW index for fast search
CREATE INDEX ON documents USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 200);

-- Semantic search with metadata filtering
SELECT content, metadata,
       1 - (embedding <=> $1::vector) AS similarity
FROM documents
WHERE metadata->>'department' = 'engineering'
ORDER BY embedding <=> $1::vector
LIMIT 10;
```

---

## 7. Common Pitfalls

### Pitfall 1: Chunks Too Large or Too Small

**Too large (> 1000 tokens):** The embedding averages over too much content — meaning gets diluted. A chunk about "authentication AND caching AND deployment" won't strongly match any of those topics.

**Too small (< 100 tokens):** Loses context. "Use the `--force` flag" means nothing without knowing which command it refers to.

**Fix:** 256–512 tokens is the sweet spot for most use cases. Always include overlap (10-15%) so context isn't lost at boundaries.

### Pitfall 2: Not Evaluating Retrieval Quality

Engineers focus on LLM output quality but ignore retrieval. If the wrong chunks are retrieved, the best LLM in the world can't give a good answer.

**Fix:** Build a retrieval evaluation set: 50+ (query, expected_document) pairs. Measure recall@5, recall@10, MRR. **This is more important than evaluating the LLM.**

### Pitfall 3: Ignoring Metadata Filtering

Pure vector search returns the "most similar" text, which might be from the wrong department, outdated, or in the wrong language.

**Fix:** Always store and filter on metadata (source, date, department, access level). Pre-filter with metadata, then vector search within that subset.

### Pitfall 4: Using the Wrong Embedding Model

General-purpose embedding models (like sentence-transformers defaults) are trained on generic data. Specialized models exist for code, legal text, medical text, etc.

**Fix:** Benchmark on YOUR data. The MTEB leaderboard shows general benchmarks, but your domain may differ. Test with 50 real queries from your users.

### Pitfall 5: Embedding Model / Query Asymmetry

Some models (like the `E5` family) require prefixes — `"query: "` for queries and `"passage: "` for documents. Missing this kills performance silently.

**Fix:** Read the model card. Always test with real queries before building the pipeline.

### Pitfall 6: Re-embedding Everything on Every Update

Adding one document shouldn't require re-processing all documents.

**Fix:** Design for incremental updates. Store document hashes. Only re-embed changed content. Use the vector store's upsert operations.

### Pitfall 7: Not Handling the "No Good Results" Case

Vector search ALWAYS returns results — even when nothing relevant exists. A cosine similarity of 0.3 looks like a "match" but it's garbage.

**Fix:** Set a similarity threshold (e.g., 0.7 for most models). Below that, return "I don't have information about this" instead of hallucinating from irrelevant context.

---

## 8. Advanced Topics

Explore these next, in recommended order:

### 8.1 Hybrid Search (BM25 + Vector)

Combine keyword search (BM25/TF-IDF) with vector search using Reciprocal Rank Fusion (RRF). This is what production RAG systems use — pure vector search misses exact matches (product IDs, error codes), pure keyword search misses semantic matches.

### 8.2 Re-ranking with Cross-Encoders

A **bi-encoder** (embedding model) encodes query and document separately — fast but less accurate. A **cross-encoder** processes query+document together — slow but much more accurate. Use bi-encoder for top-100, cross-encoder to re-rank to top-5.

### 8.3 Multi-Vector Representations (ColBERT)

Instead of one vector per chunk, use one vector per **token**. Late interaction computes fine-grained similarity. Higher quality, but more storage and compute.

### 8.4 Chunking Strategies Deep Dive

- **Semantic chunking:** Split by meaning shift (embed sentences, cluster)
- **Parent-child chunking:** Embed small chunks, retrieve parent sections
- **Sliding window with stride:** Overlapping windows for dense coverage
- **Agentic chunking:** Use an LLM to decide where to split

### 8.5 Evaluation Frameworks

- **RAGAS:** Measures faithfulness, relevance, context precision/recall
- **Custom evaluation:** Build query→expected_docs test sets from real user questions
- **A/B testing:** Compare retrieval strategies with real traffic

### 8.6 Multimodal Embeddings

Embed images, audio, and text into the same vector space. CLIP (OpenAI) maps images and text together — search images with text queries and vice versa.

### 8.7 Fine-tuning Embedding Models

When off-the-shelf models don't work well on your domain, fine-tune with contrastive learning on your (query, relevant_document) pairs. Libraries: `sentence-transformers`, `MatryoshkaLoss`.

---

## 9. Exercises

### Exercise 1: Multi-Source RAG Pipeline

Build a FastAPI service that:
1. Accepts documents from multiple sources (plain text, markdown, and JSON)
2. Uses different chunking strategies based on document type
3. Stores embeddings with source metadata in FAISS
4. Implements a `/search` endpoint that filters by source before vector search
5. Implements an `/ask` endpoint with RAG using Claude

**Success criteria:** Search for "deployment process" and get relevant results from markdown runbooks even when the query words don't appear in the docs.

### Exercise 2: Hybrid Search Implementation

Extend the base implementation to add hybrid search:
1. Add BM25 keyword search alongside vector search (use the `rank_bm25` library)
2. Implement Reciprocal Rank Fusion (RRF) to merge the two result sets
3. Add an A/B comparison endpoint that shows vector-only vs. hybrid results side by side
4. Test with queries that contain specific identifiers (error codes, function names) to demonstrate where hybrid wins

**Success criteria:** Query "ERR_CONNECTION_REFUSED troubleshooting" returns results about that specific error (keyword match) AND general network debugging guides (semantic match).

### Exercise 3: Retrieval Evaluation Harness

Build a retrieval quality evaluation system:
1. Create a test dataset: 20 queries with their expected relevant documents
2. Implement recall@k, precision@k, and MRR (Mean Reciprocal Rank) metrics
3. Build a FastAPI endpoint that runs the evaluation and returns a quality report
4. Compare at least two configurations (e.g., chunk_size=256 vs chunk_size=512) and report which performs better

**Success criteria:** Produce a JSON report showing retrieval metrics for each configuration, identifying the better chunking strategy for your specific data.

---

## 10. Interview / Architect Questions

### Q1: Embedding Drift and Model Migration

*"You're running a RAG system with 10 million embedded documents. The embedding model provider releases a significantly better model. How do you migrate without downtime?"*

**What this tests:** Understanding of the tight coupling between embedding model and vector index. Knowledge of blue-green deployment patterns for ML systems.

**Key points in a strong answer:**
- All vectors must be re-embedded — you can't mix vectors from different models
- Blue-green strategy: build a parallel index with the new model, switch traffic atomically
- Consider the compute cost: 10M documents × embedding API calls = budget + time planning
- Run evaluation on both indexes before switching to prove the new model is actually better on YOUR data
- Version your indexes (model_name + model_version as metadata)

### Q2: Chunking for Heterogeneous Data

*"Your knowledge base contains API documentation (structured), support tickets (conversational), and legal contracts (formal). How do you design your chunking strategy?"*

**What this tests:** Understanding that chunking is not one-size-fits-all and is the biggest quality lever in RAG.

**Key points in a strong answer:**
- Different content types need different chunking strategies
- API docs: chunk by endpoint/section (structural boundaries)
- Support tickets: chunk by conversation turn, keep question-answer pairs together
- Legal contracts: chunk by clause, preserve clause numbering as metadata
- Store the chunking strategy as metadata so you can re-process when strategies improve
- Evaluate each strategy independently with type-specific test queries

### Q3: When Vector Search Returns Wrong Results

*"Users are complaining that the chatbot gives confident but wrong answers. The LLM is working correctly — the problem is upstream. How do you diagnose and fix retrieval quality issues?"*

**What this tests:** Systematic debugging of RAG pipelines, understanding that generation quality depends on retrieval quality.

**Key points in a strong answer:**
- Log the retrieved chunks alongside LLM responses — you need visibility into what context the LLM received
- Build a retrieval evaluation set from the failing queries
- Check similarity scores — are bad results scoring high (bad embeddings) or are good results just not in the index (missing content)?
- Common fixes: adjust chunk size, add metadata filtering, implement re-ranking, set minimum similarity thresholds
- Implement a feedback loop: let users flag bad answers, trace back to the retrieval step

### Q4: Cost–Latency–Quality Triangle

*"Design a RAG system where P99 latency must be under 500ms, you're processing 1,000 queries/second, and the knowledge base is 50 million documents. Walk me through your architecture decisions."*

**What this tests:** System design under constraints. Understanding of the trade-offs between different vector DB choices, caching strategies, and architectural patterns.

**Key points in a strong answer:**
- At 50M docs, you need a distributed vector DB (Qdrant, Weaviate, or Pinecone)
- Embedding the query takes ~50-100ms (API call) — consider self-hosted embedding model to cut network latency
- Vector search with HNSW: ~5-10ms at 50M scale with proper tuning
- LLM generation is the bottleneck (~200-400ms) — use streaming, caching, and smaller models for common queries
- Cache strategy: embed the query, hash it, check cache before LLM call. Common queries get instant responses.
- Pre-compute embeddings for anticipated queries during off-peak
- Consider a tiered approach: cache → small model for simple queries → full LLM for complex ones

### Q5: Embeddings Beyond Text Search

*"Beyond RAG and document search, what other production systems benefit from embeddings? Describe a non-obvious use case and how you'd architect it."*

**What this tests:** Breadth of understanding. Can the candidate think beyond the obvious RAG use case?

**Strong non-obvious examples:**
- **Anomaly detection:** Embed user behavior sessions, flag sessions that are far from any cluster (fraud, account compromise)
- **Content deduplication:** Near-duplicate detection at scale — cluster by embedding similarity, flag clusters for human review
- **A/B test analysis:** Embed user feedback text, cluster to find themes, quantify sentiment shift between variants
- **Cache invalidation:** When a document changes, compare old vs new embedding — if similarity > 0.98, don't invalidate dependent caches
- **Feature engineering for ML:** Use embeddings as input features for classification/regression models instead of hand-crafted NLP features
