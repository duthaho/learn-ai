# Module 03 — Embeddings & Vector Search

Representing text as numbers and searching by meaning instead of keywords.

| Detail        | Value                                     |
|---------------|-------------------------------------------|
| Level         | Beginner                                  |
| Time          | ~2 hours                                  |
| Prerequisites | Module 01 (How LLMs Work)                 |

## What you'll build

After reading this module, head to [`project/`](project/) to build a **Semantic Search Engine** — a tool that embeds a collection of documents, indexes them, and retrieves the most relevant results for natural language queries.

---

## Table of Contents

1. [What Are Embeddings?](#1-what-are-embeddings)
2. [How Embedding Models Work](#2-how-embedding-models-work)
3. [Cosine Similarity](#3-cosine-similarity)
4. [Vector Databases and FAISS](#4-vector-databases-and-faiss)
5. [Chunking](#5-chunking)
6. [The Embedding Pipeline](#6-the-embedding-pipeline)
7. [Common Pitfalls](#7-common-pitfalls)
8. [Beyond Text Search](#8-beyond-text-search)

---

## 1. What Are Embeddings?

An embedding is a dense numerical vector that represents the semantic meaning of a piece of text. The word "dense" matters — unlike sparse representations (bag-of-words, TF-IDF) where most values are zero, every dimension in an embedding carries information. A typical embedding might have 384 to 3072 dimensions, with each dimension being a floating-point number.

The core insight: **texts with similar meanings produce vectors that are close together in vector space, even when they share no words at all.**

Consider these two sentences:

```
Sentence A: "How do I reset my password?"
Sentence B: "I forgot my login credentials"
```

A keyword search for "reset password" would never return Sentence B — the words "reset" and "password" do not appear in it. But an embedding model maps both sentences to nearby points in vector space because they express the same underlying intent: a user who cannot access their account.

This is what makes embeddings transformative. They let you search by meaning, not by string matching.

### From words to vectors

You encountered embeddings briefly in Module 01 when we discussed how LLMs convert tokens into vectors. Those internal embeddings are part of the model's learned representation — every token gets mapped to a vector that captures its relationships with other tokens.

Embedding models take this further. Instead of producing per-token vectors for internal use, they compress an entire passage — a sentence, a paragraph, a document chunk — into a single vector that represents the passage's overall meaning. This is the vector you store and search against.

### What the dimensions mean

Each dimension in an embedding vector does not have a human-interpretable label. You cannot point to dimension 147 and say "this encodes the concept of urgency." Instead, meaning is distributed across all dimensions simultaneously. The vector as a whole captures semantics — individual dimensions are not meaningful in isolation.

This is similar to how RGB values work in color. The color teal is not "red" or "green" or "blue" — it is a specific combination of all three channels. An embedding's meaning emerges from the specific combination of all its dimensions.

### Why this matters for engineering

Embeddings turn unstructured text into structured data. Once text is a vector of fixed length, you can:

- **Search** — find the most similar items to a query
- **Cluster** — group related documents automatically
- **Classify** — use vectors as features for ML models
- **Deduplicate** — find near-identical content across large datasets
- **Recommend** — surface items similar to what a user has engaged with

Every one of these operations is a mathematical operation on vectors — fast, scalable, and deterministic. No LLM inference is required at query time once embeddings are precomputed.

---

## 2. How Embedding Models Work

Embedding models share architecture with LLMs — they are transformer-based neural networks — but they serve a fundamentally different purpose.

### LLMs vs embedding models

| Aspect           | LLM (e.g., GPT-4, Claude)            | Embedding model (e.g., text-embedding-3-small) |
|------------------|---------------------------------------|-------------------------------------------------|
| Input            | Text                                  | Text                                            |
| Output           | Generated text (token by token)       | A single fixed-size vector                      |
| Purpose          | Produce language                      | Represent meaning numerically                   |
| Inference cost   | High (autoregressive, many forward passes) | Low (single forward pass)                  |
| Typical use      | Chatbots, writing, reasoning          | Search, similarity, clustering                  |

An LLM reads your input and generates a continuation, one token at a time, with each token requiring a separate forward pass. An embedding model reads your input and produces one vector in a single forward pass. This makes embedding models dramatically faster and cheaper per input.

### Training: contrastive learning

Embedding models learn through **contrastive learning** — a training approach where the model sees pairs of texts and learns to pull similar pairs together in vector space while pushing dissimilar pairs apart.

```
Training signal:

  Similar pair (pull together):
    "How do I cancel my subscription?" <--> "I want to stop my monthly plan"
    Target: vectors should be close (high similarity)

  Dissimilar pair (push apart):
    "How do I cancel my subscription?" <--> "What is the weather in Tokyo?"
    Target: vectors should be far (low similarity)
```

The training data for this process comes from multiple sources: natural question-answer pairs scraped from forums, paraphrases, titles paired with their articles, and search queries paired with clicked results. Some datasets use human annotations, while others rely on naturally occurring pair structure (a question posted on a Q&A site is naturally paired with its accepted answer).

The loss function (typically InfoNCE or multiple negatives ranking loss) rewards the model when similar pairs produce close vectors and penalizes it when dissimilar pairs end up close. Over millions of training steps, the model learns a vector space where semantic similarity corresponds to geometric proximity.

### Dimensions

The dimensionality of an embedding model is the length of the vector it produces. More dimensions give the model more capacity to encode fine-grained distinctions, but at a cost: storage, memory, and computation all scale with dimensionality.

| Dimensions | Capacity                    | Typical use case                       |
|------------|-----------------------------|----------------------------------------|
| 384        | Good for most tasks         | Lightweight, fast, edge deployment     |
| 768        | Strong general performance  | Production search and retrieval        |
| 1024       | High-fidelity               | Complex semantic tasks                 |
| 1536       | Very high capacity          | Large-scale production systems         |
| 3072       | Maximum detail              | Research, specialized domains          |

More dimensions are not always better. A 384-dimensional model trained on high-quality data often outperforms a 1536-dimensional model trained on weaker data. Dimensions are capacity — training quality determines how well that capacity is used.

### Model options

The embedding model landscape has matured significantly. Here are the most commonly used models as of early 2025:

| Model                     | Provider   | Dimensions | Context  | Speed      | Cost              | Quality notes                                     |
|---------------------------|------------|------------|----------|------------|-------------------|----------------------------------------------------|
| `voyage-3`                | Voyage AI  | 1024       | 32K      | Fast       | $0.06/1M tokens   | Top-tier retrieval quality, strong on code          |
| `text-embedding-3-small`  | OpenAI     | 1536       | 8191     | Fast       | $0.02/1M tokens   | Good balance of cost and quality                    |
| `text-embedding-3-large`  | OpenAI     | 3072       | 8191     | Moderate   | $0.13/1M tokens   | Highest quality from OpenAI, supports dimension reduction |
| `all-MiniLM-L6-v2`        | Open source| 384        | 256      | Very fast  | Free (self-host)  | Good for prototyping, small context window          |
| `nomic-embed-text`        | Nomic      | 768        | 8192     | Fast       | Free (self-host)  | Strong open-source option, long context             |
| `bge-large-en-v1.5`       | BAAI       | 1024       | 512      | Moderate   | Free (self-host)  | Strong benchmark performance                       |

**How to choose:**

- **Prototyping or cost-sensitive:** `all-MiniLM-L6-v2` (free, fast, runs locally)
- **Production with API budget:** `text-embedding-3-small` (cheap, good quality)
- **Maximum retrieval quality:** `voyage-3` or `text-embedding-3-large`
- **Self-hosted, long context:** `nomic-embed-text`

### Query vs document prefixes

Some embedding models require different prefixes for queries and documents. This is called **asymmetric embedding** — the model treats "what am I looking for?" differently from "what is this about?"

```
Document embedding:
  Input: "search_document: Python is a high-level programming language
  known for its readability and versatility."

Query embedding:
  Input: "search_query: What programming language is easy to read?"
```

The prefix tells the model whether this text is a question seeking information or a passage containing information. Mixing them up — embedding a query with the document prefix or vice versa — degrades retrieval quality significantly. Always check whether your chosen model requires prefixes and use them correctly.

---

## 3. Cosine Similarity

Once you have two embedding vectors, you need a way to measure how similar they are. The standard measure is **cosine similarity** — the cosine of the angle between two vectors.

### The formula

For two vectors **a** and **b**, each with *n* dimensions:

```
                    a . b           sum(a_i * b_i)
cosine_sim(a, b) = ------- = ----------------------------
                   ||a|| ||b||   sqrt(sum(a_i^2)) * sqrt(sum(b_i^2))
```

Where:
- `a . b` is the dot product (multiply corresponding elements and sum)
- `||a||` is the magnitude (L2 norm) of vector a
- `||b||` is the magnitude (L2 norm) of vector b

### Intuition

Forget the formula for a moment. Cosine similarity measures the **angle** between two vectors, ignoring their length.

```
    ^                     ^
    |  /                  |   /
    | /  small angle       |  /  large angle
    |/   = similar         | /   = different
    +---->                 +-------->
```

Two vectors pointing in the same direction have cosine similarity 1 (angle = 0 degrees). Two perpendicular vectors have cosine similarity 0 (angle = 90 degrees). Two vectors pointing in opposite directions have cosine similarity -1 (angle = 180 degrees).

In practice, embedding vectors almost never have negative cosine similarity. Most values fall between 0.0 and 1.0, with semantically similar texts typically scoring above 0.7 and unrelated texts scoring below 0.3.

### A concrete example

Suppose we have a simplified 3-dimensional embedding space (real embeddings have hundreds of dimensions, but the math is identical):

```
"How do I reset my password?"      -> a = [0.8, 0.5, 0.2]
"I forgot my login credentials"   -> b = [0.7, 0.6, 0.3]
"What is the weather in Tokyo?"   -> c = [0.1, 0.3, 0.9]
```

Similarity between a and b (related):

```
a . b     = (0.8*0.7) + (0.5*0.6) + (0.2*0.3) = 0.56 + 0.30 + 0.06 = 0.92
||a||     = sqrt(0.64 + 0.25 + 0.04) = sqrt(0.93) = 0.964
||b||     = sqrt(0.49 + 0.36 + 0.09) = sqrt(0.94) = 0.970

cosine_sim(a, b) = 0.92 / (0.964 * 0.970) = 0.92 / 0.935 = 0.984
```

Similarity between a and c (unrelated):

```
a . c     = (0.8*0.1) + (0.5*0.3) + (0.2*0.9) = 0.08 + 0.15 + 0.18 = 0.41
||a||     = 0.964 (same as above)
||c||     = sqrt(0.01 + 0.09 + 0.81) = sqrt(0.91) = 0.954

cosine_sim(a, c) = 0.41 / (0.964 * 0.954) = 0.41 / 0.920 = 0.446
```

The password-related sentences score 0.984 (very similar). The password question vs. weather question scores 0.446 (not similar). This matches our intuition perfectly.

### Why normalize: dot product as a shortcut

Computing cosine similarity requires a division by the magnitudes — the denominator in the formula. But if you **normalize** your vectors to unit length (magnitude = 1) before storing them, the denominator becomes 1 * 1 = 1, and cosine similarity reduces to a simple dot product:

```
If ||a|| = 1 and ||b|| = 1:
  cosine_sim(a, b) = a . b
```

The dot product is one of the fastest operations in numerical computing. Normalizing once at index time saves the division on every search query. This is why most vector databases and search libraries (including FAISS) work with normalized vectors and inner product distance.

### Other distance metrics

Cosine similarity is the default for text embeddings, but other metrics exist:

| Metric              | Formula                    | When to use                                    |
|---------------------|----------------------------|------------------------------------------------|
| Cosine similarity   | dot(a,b) / (norm(a)*norm(b)) | Text similarity (most common)                |
| Euclidean (L2)      | sqrt(sum((a_i - b_i)^2))  | When magnitude matters (e.g., anomaly scores) |
| Dot product         | sum(a_i * b_i)             | Pre-normalized vectors (equivalent to cosine) |
| Manhattan (L1)      | sum(abs(a_i - b_i))        | Sparse or high-dimensional data               |

For embedding-based search, cosine similarity (or dot product on normalized vectors) is almost always the right choice.

---

## 4. Vector Databases and FAISS

An embedding by itself is just a list of numbers. To build a search system, you need infrastructure to **store** embeddings and **search** over them efficiently. That is what a vector database (or vector index) does.

### What a vector store does

At its core, a vector store provides two operations:

1. **Index** — Accept a vector (and optional metadata) and store it for later retrieval.
2. **Search** — Given a query vector, find the *k* most similar vectors in the index.

Everything else — metadata filtering, persistence, replication, access control — is layered on top of these two primitives.

### FAISS: the foundational library

FAISS (Facebook AI Similarity Search) is an open-source library developed by Meta for efficient similarity search. It is not a database — it has no built-in persistence, no network API, no access control. It is a library you call from Python (or C++) that manages an in-memory index of vectors.

Despite its simplicity, FAISS is remarkably powerful. Many production vector databases use FAISS internally for their core indexing algorithms.

### IndexFlatIP: brute-force search

The simplest FAISS index is `IndexFlatIP` — flat index with inner product distance. "Flat" means no compression, no approximation, no special data structure. It stores every vector as-is and compares the query against every single one.

```
How IndexFlatIP works:

  Index time:
    Store vector as-is in a flat array.
    O(1) per vector. Memory = n_vectors * dimensions * 4 bytes.

  Search time:
    Compare query against EVERY vector in the index.
    Return the k vectors with highest inner product.
    O(n * d) per query, where n = number of vectors, d = dimensions.
```

With normalized vectors, inner product equals cosine similarity (as explained in Section 3), so `IndexFlatIP` gives you exact cosine similarity search.

**When brute-force is enough:** For collections up to about 100,000 vectors, brute-force search on modern hardware is fast — typically under 10ms per query for 768-dimensional vectors. Most prototypes and many production systems never need anything more complex.

```
Rough performance (768-dim vectors, single CPU core):

  10,000 vectors:   ~1ms per query
  100,000 vectors:  ~5ms per query
  1,000,000 vectors: ~50ms per query
  10,000,000 vectors: ~500ms per query (too slow for real-time)
```

### Approximate nearest neighbors: IVF and HNSW

When brute-force becomes too slow, you trade perfect accuracy for speed using approximate nearest neighbor (ANN) algorithms. These find vectors that are *probably* the closest, with a small chance of missing the true nearest neighbor.

**IVF (Inverted File Index)**

IVF partitions the vector space into clusters (using k-means). At search time, it only searches the closest clusters instead of the entire index.

```
IVF approach:

  Index time:
    1. Cluster all vectors into C clusters (e.g., C = 1024)
    2. Assign each vector to its nearest cluster centroid

  Search time:
    1. Find the P closest cluster centroids to the query (e.g., P = 10)
    2. Only search vectors within those P clusters
    3. Return the top-k results

  Speed: Instead of searching all n vectors, search only n * (P/C) vectors.
  Example: 1M vectors, 1024 clusters, probe 10 = search ~10,000 vectors.
```

**HNSW (Hierarchical Navigable Small World)**

HNSW builds a multi-layer graph where each vector is connected to its neighbors. Search navigates this graph from a random entry point, jumping through layers of decreasing granularity to find nearby vectors.

```
HNSW approach:

  Index time:
    Build a multi-layer proximity graph.
    Each layer has fewer nodes; top layer has very few.
    Each node is connected to its nearest neighbors.

  Search time:
    1. Start at a random node in the top (sparsest) layer
    2. Greedily navigate to the nearest node in that layer
    3. Drop to the next layer and repeat
    4. At the bottom layer, do a more thorough local search

  Speed: O(log n) per query. Very fast even at 100M+ vectors.
  Tradeoff: Higher memory usage (stores graph edges).
```

### Choosing the right index

| Collection size     | Recommended index    | Search time    | Recall    | Notes                         |
|---------------------|----------------------|----------------|-----------|-------------------------------|
| < 10K               | `IndexFlatIP`        | < 1ms          | 100%      | No reason for approximation   |
| 10K - 100K          | `IndexFlatIP`        | 1-10ms         | 100%      | Brute-force still fast enough |
| 100K - 1M           | `IndexIVFFlat`       | 1-5ms          | 95-99%    | Good balance of speed/recall  |
| 1M - 10M            | `IndexIVF` + PQ      | < 1ms          | 90-95%    | Add product quantization      |
| 10M+                | `IndexHNSWFlat`      | < 1ms          | 95-99%    | Graph-based, high memory      |

### Production vector databases

FAISS is a library — great for prototyping and embedding into applications, but it does not provide persistence, replication, or a network API. For production systems that need these features, consider a dedicated vector database:

| Database    | Type                  | Scale           | Complexity  | Best for                                        |
|-------------|-----------------------|-----------------|-------------|--------------------------------------------------|
| FAISS       | Library (in-process)  | Millions        | Low         | Prototypes, embedded in apps, research            |
| pgvector    | PostgreSQL extension  | Millions        | Low         | Teams already using PostgreSQL, want one DB       |
| Qdrant      | Dedicated vector DB   | Billions        | Medium      | Production search with filtering and payloads     |
| Weaviate    | Dedicated vector DB   | Billions        | Medium      | Multimodal search, GraphQL API                    |
| Pinecone    | Managed service       | Billions        | Very low    | Teams wanting zero infrastructure management      |
| ChromaDB    | Lightweight vector DB | Millions        | Low         | Local development, quick prototyping              |

**How to choose:**

- **Prototyping:** FAISS or ChromaDB. Zero infrastructure. Start in 5 lines of code.
- **Already using PostgreSQL:** pgvector. One less database to manage. Works well up to a few million vectors.
- **Production with filtering:** Qdrant or Weaviate. Both support metadata filtering (e.g., "find similar documents, but only from the last 30 days").
- **No ops team:** Pinecone. Fully managed, scales automatically, but vendor lock-in and ongoing cost.

### Metadata filtering

In practice, pure vector similarity is rarely enough. You almost always want to combine semantic search with traditional filters:

```
Query:   "deployment best practices"
Filters: language = "python", created_after = "2024-01-01", author != "bot"
```

The vector index finds semantically similar documents. The metadata filter narrows results to those matching your constraints. Most production vector databases support this natively. With raw FAISS, you need to implement filtering yourself (typically by post-filtering the results).

---

## 5. Chunking

Chunking is the process of splitting documents into smaller pieces before embedding them. It is arguably **the most impactful decision in any embedding pipeline** — more so than model choice, index type, or similarity metric.

### Why you must chunk

Two forces make chunking necessary:

**1. Embedding model context limits.** Every embedding model has a maximum input length. `all-MiniLM-L6-v2` accepts only 256 tokens. Even models with larger context windows (8K, 32K tokens) produce worse embeddings for very long inputs because the meaning gets diluted — the vector tries to represent too many concepts at once and ends up representing none of them well.

**2. Retrieval granularity.** When a user searches for "how to configure logging," you want to return the specific paragraph about logging configuration, not an entire 50-page document that happens to mention logging once. Smaller chunks mean more precise retrieval.

### The chunking tradeoff

```
Chunks too small:                     Chunks too big:
+------------------+                  +------------------+
| Lost context.    |                  | Diluted meaning. |
| "Configure the"  |                  | A 2000-word chunk|
| makes no sense   |                  | about 5 different|
| without knowing  |                  | topics produces a|
| what to configure|                  | blurry vector    |
+------------------+                  +------------------+

        |                                      |
        v                                      v
  Bad retrieval:                        Bad retrieval:
  Matches on fragments                  Returns irrelevant
  that lack meaning                     content alongside
                                        relevant content
```

### Chunking strategies

**1. Fixed-size chunking**

Split text into chunks of exactly *n* tokens (or characters), regardless of content boundaries.

```
Input: "Python is a high-level language. It supports multiple paradigms.
        The standard library is extensive. NumPy adds numerical computing."

Chunk size: 50 characters

  Chunk 1: "Python is a high-level language. It supports mu"
  Chunk 2: "ltiple paradigms. The standard library is exten"
  Chunk 3: "sive. NumPy adds numerical computing."
```

This is the simplest approach but also the worst. Chunk boundaries cut through sentences, splitting "supports multiple" across two chunks. The resulting embeddings represent broken fragments.

**2. Sentence-based chunking**

Split on sentence boundaries, grouping sentences until the chunk reaches a target size.

```
Input: (same as above)
Target: ~2 sentences per chunk

  Chunk 1: "Python is a high-level language. It supports multiple paradigms."
  Chunk 2: "The standard library is extensive. NumPy adds numerical computing."
```

Better — each chunk contains complete sentences. But this ignores document structure. A heading and its first paragraph might end up in different chunks.

**3. Recursive chunking (recommended)**

Start with the largest structural units (sections, headings) and recursively split only when a chunk exceeds the target size. This respects document hierarchy.

```
Split hierarchy:
  1. First try: split on headings / section breaks (## , \n\n\n)
  2. If still too big: split on paragraph breaks (\n\n)
  3. If still too big: split on sentence boundaries (. ! ?)
  4. Last resort: split on word boundaries
```

Recursive chunking preserves the most context because it keeps related content together. A section about "Database Configuration" stays in one chunk (or is split at paragraph boundaries within that section) rather than being arbitrarily cut at a token count.

### Chunk size sweet spot

Research and practical experience converge on a sweet spot of **256 to 512 tokens** per chunk for most retrieval tasks.

| Chunk size     | Pros                                         | Cons                                           |
|----------------|----------------------------------------------|-------------------------------------------------|
| < 128 tokens   | Very precise retrieval                       | Fragments lack context, many chunks to search   |
| 128-256 tokens | Precise, good for Q&A                        | May split some concepts across chunks           |
| 256-512 tokens | Best balance for most tasks                  | Slightly less precise but much more context      |
| 512-1024 tokens| Good context preservation                    | May include irrelevant content in results        |
| > 1024 tokens  | Maximum context per chunk                    | Diluted embeddings, poor retrieval precision     |

Start with 400 tokens and adjust based on your evaluation results. There is no universal optimum — the best chunk size depends on your documents and queries.

### Overlap

When splitting text into chunks, include **overlap** between consecutive chunks — typically 10 to 15% of the chunk size. Overlap ensures that concepts spanning a chunk boundary appear in full in at least one chunk.

```
Without overlap (boundary splits a concept):

  Chunk 1: "...configure the database connection"
  Chunk 2: "string using the DB_URL environment variable..."

  A query about "database connection string" partially matches both
  chunks but fully matches neither.

With 50-token overlap:

  Chunk 1: "...configure the database connection string using the DB_URL
            environment variable. This should be set before..."
  Chunk 2: "...connection string using the DB_URL environment variable.
            This should be set before starting the application..."

  Now "database connection string" is fully contained in Chunk 1.
```

### Concrete example: how bad chunking ruins retrieval

Consider a technical document about a REST API:

```
Original document (excerpt):

  ## Authentication

  All API requests require a Bearer token in the Authorization header.
  Tokens expire after 24 hours. To refresh a token, send a POST request
  to /auth/refresh with your refresh token in the body.

  ## Rate Limiting

  The API allows 100 requests per minute per API key. If you exceed this
  limit, you will receive a 429 status code. Implement exponential
  backoff in your client.
```

**Bad chunking** (fixed-size, 100 characters, no structural awareness):

```
  Chunk 1: "## Authentication\n\nAll API requests require a Bearer token in the Authorization header. To"
  Chunk 2: "kens expire after 24 hours. To refresh a token, send a POST request\nto /auth/refresh with yo"
  Chunk 3: "ur refresh token in the body.\n\n## Rate Limiting\n\nThe API allows 100 requests per minute per"
  Chunk 4: " API key. If you exceed this limit, you will receive a 429 status code. Implement exponential"
  Chunk 5: " backoff in your client."
```

Query: "How do I refresh my API token?"

The answer is split across Chunk 2 and Chunk 3. Neither chunk contains the complete answer. Chunk 2 has "refresh a token, send a POST request to /auth/refresh with yo" (truncated). Chunk 3 starts with "ur refresh token in the body" (fragment) then jumps to rate limiting. The search might return Chunk 2, but the user gets a broken fragment.

**Good chunking** (recursive, section-aware):

```
  Chunk 1: "## Authentication\n\nAll API requests require a Bearer token in the
            Authorization header. Tokens expire after 24 hours. To refresh a
            token, send a POST request to /auth/refresh with your refresh token
            in the body."

  Chunk 2: "## Rate Limiting\n\nThe API allows 100 requests per minute per API
            key. If you exceed this limit, you will receive a 429 status code.
            Implement exponential backoff in your client."
```

Same query: "How do I refresh my API token?"

Chunk 1 matches cleanly. It contains the complete answer with full context. The user gets exactly what they need.

**The lesson:** Chunking quality matters more than embedding model quality. A state-of-the-art embedding model cannot fix chunks that split concepts at arbitrary boundaries. Invest time in your chunking strategy before optimizing anything else.

---

## 6. The Embedding Pipeline

The individual components — embeddings, similarity, indexes, chunking — combine into a pipeline. Here is the end-to-end flow.

### The pipeline

```
                         INDEXING PHASE (offline, run once)
  +--------+     +---------+     +---------+     +---------+
  |        |     |         |     |         |     |         |
  |  Raw   |---->|  Chunk  |---->|  Embed  |---->|  Index  |
  |  Docs  |     |         |     |         |     | (FAISS) |
  |        |     |         |     |         |     |         |
  +--------+     +---------+     +---------+     +---------+
                   Split into      Convert to      Store vectors
                   256-512 token   vectors via      for fast search
                   chunks with     embedding
                   overlap         model


                          SEARCH PHASE (online, per query)
  +--------+     +---------+     +---------+     +---------+
  |        |     |         |     |         |     |         |
  | User   |---->|  Embed  |---->| Search  |---->| Return  |
  | Query  |     |  Query  |     |  Index  |     |  Top-k  |
  |        |     |         |     |         |     | Chunks  |
  +--------+     +---------+     +---------+     +---------+
                   Same model      Find k           Return chunks
                   used for        nearest          + metadata +
                   indexing         vectors          similarity scores
```

### Step by step

**Step 1: Load documents.** Read your raw text from whatever source — files, databases, APIs, web scrapes. Track metadata (source URL, title, date, author) alongside the text.

**Step 2: Chunk.** Split each document into chunks of 256-512 tokens using recursive chunking with 10-15% overlap. Attach the original document's metadata to each chunk so you can trace results back to their source.

**Step 3: Embed.** Pass each chunk through your embedding model to produce a vector. If using a model with prefix requirements (like `nomic-embed-text`), add the document prefix. This step is the most time-consuming for large corpora — embedding 1 million chunks at 768 dimensions takes roughly 30-60 minutes on a single GPU.

**Step 4: Normalize.** Normalize all vectors to unit length so that dot product equals cosine similarity. Most embedding models already output normalized vectors, but verify this for your chosen model.

**Step 5: Index.** Add all vectors to your FAISS index (or vector database). For collections under 100K vectors, `IndexFlatIP` is sufficient. For larger collections, use IVF or HNSW.

**Step 6: Search.** When a query arrives, embed it with the same model (using the query prefix if applicable), normalize it, and search the index for the top-k nearest vectors. Return the corresponding chunks with their similarity scores and metadata.

### Where this pipeline leads

This embedding pipeline is not an end in itself — it is the foundation for multiple AI engineering patterns:

| Pattern                | How it uses the pipeline                                              | Module |
|------------------------|-----------------------------------------------------------------------|--------|
| RAG                    | Retrieve relevant chunks, feed them as context to an LLM             | 07     |
| Semantic search        | Return the top-k chunks directly to the user                         | This   |
| Recommendations        | Embed items, find similar items to a user's history                   | —      |
| Deduplication          | Embed all content, cluster by similarity, flag near-duplicates        | —      |
| Classification         | Use embedding vectors as features for a classifier                   | —      |

RAG (Retrieval-Augmented Generation) is the most common application. In Module 07, you will take this pipeline and add an LLM on top: retrieve relevant chunks, insert them into a prompt, and let the LLM generate an answer grounded in your actual data. The quality of your RAG system depends directly on the quality of this embedding pipeline.

### Performance characteristics

To help with capacity planning, here are rough numbers for a typical pipeline using `text-embedding-3-small` (1536 dimensions):

```
Indexing:
  Embedding speed:  ~3,000 chunks/minute via API
  Storage:          1M chunks * 1536 dims * 4 bytes = ~5.7 GB (vectors only)
  Total index time: 1M chunks takes ~5.5 hours via API

Search:
  Embedding the query: ~50ms (API round trip)
  FAISS search (100K vectors, brute-force): ~5ms
  FAISS search (1M vectors, IVF): ~2ms
  Total query latency: ~55-60ms (dominated by embedding API call)
```

The bottleneck for query latency is almost always the API call to embed the query, not the vector search itself. If latency matters, consider self-hosting an embedding model to avoid the network round trip.

---

## 7. Common Pitfalls

These are mistakes that appear repeatedly in embedding pipelines. Each one is easy to make and can silently degrade your system's quality.

### Chunks too big

**Symptom:** Search returns documents that contain the answer somewhere, but also contain large amounts of irrelevant text.

**Cause:** Chunks of 1000+ tokens embed too many concepts into a single vector. The vector becomes a blurry average of everything in the chunk.

**Fix:** Reduce chunk size to 256-512 tokens. Use recursive chunking to preserve section structure.

### Chunks too small

**Symptom:** Search returns fragments that seem relevant but lack enough context to be useful. Results like "yes, you should configure it" with no indication of what "it" refers to.

**Cause:** Chunks under 100 tokens often lack the context needed to stand alone.

**Fix:** Increase chunk size. Ensure each chunk is self-contained — it should make sense without reading the surrounding chunks. Add overlap to capture concepts at boundaries.

### Not evaluating retrieval quality

**Symptom:** You build the pipeline, it seems to work on a few test queries, and you ship it. Users complain that search results are irrelevant.

**Cause:** You never systematically measured whether the right chunks are being retrieved.

**Fix:** Build an evaluation set — a collection of queries paired with the chunks that should be retrieved. Measure recall (what percentage of relevant chunks were found?) and precision (what percentage of returned chunks were relevant?). Run this evaluation whenever you change the pipeline.

### No similarity threshold

**Symptom:** Every query returns results, even queries about topics not in your corpus. A user asks about quantum physics in a cooking recipe database and gets back the "most similar" recipes — which are not similar at all, just the least dissimilar.

**Cause:** Vector search always returns the top-k nearest neighbors, even if the nearest neighbor has a cosine similarity of 0.15. There is no built-in concept of "no match found."

**Fix:** Set a minimum similarity threshold. If no results exceed the threshold, return an empty result set (or a "no relevant results found" message). A typical threshold is 0.3-0.5, but calibrate this on your specific data.

### Mixing embedding models

**Symptom:** Some queries work well, others return nonsensical results.

**Cause:** You indexed some documents with Model A and others with Model B. Vectors from different models live in completely different vector spaces — their dimensions mean different things. Comparing a `text-embedding-3-small` vector to an `all-MiniLM-L6-v2` vector is meaningless, even if both are 384-dimensional.

**Fix:** Use one model for your entire index. If you switch models, re-embed everything.

### Ignoring metadata filtering

**Symptom:** Semantically relevant results that are practically useless. A query returns a perfect answer from a document that was deprecated two years ago.

**Cause:** Pure vector search knows nothing about dates, permissions, document status, or any other attribute. It returns whatever is semantically closest.

**Fix:** Store metadata alongside vectors. Apply filters before or after vector search to enforce business constraints (date ranges, access permissions, document categories, language).

### Query-document asymmetry

**Symptom:** Retrieval quality is mediocre despite using a high-quality model.

**Cause:** Some embedding models (e.g., `nomic-embed-text`, BGE models) require different prefixes for queries and documents. If you embed everything the same way, the model cannot distinguish "I am looking for X" from "I contain information about X."

**Fix:** Check your model's documentation. If it requires prefixes, use `search_query:` for queries and `search_document:` for documents (or whatever the model-specific prefixes are).

### Re-embedding everything on every update

**Symptom:** Adding a single new document triggers re-embedding of the entire corpus, taking hours.

**Cause:** The pipeline was built without incremental updates in mind.

**Fix:** Design for incremental indexing from the start. When a document changes, only re-embed its chunks. When a new document is added, embed and add only its chunks. Store a mapping from document IDs to chunk IDs so you can selectively remove and re-add chunks when source documents change.

---

## 8. Beyond Text Search

Embeddings are a general-purpose representation tool. Once you can represent things as vectors, you unlock a family of techniques that go well beyond search.

### Anomaly detection

Embed all items in your dataset (user sessions, log entries, transactions). Normal items cluster together. Outliers — items far from any cluster — are potential anomalies.

```
Normal user sessions cluster here:     Anomaly:
                                             *
    * * *
   * * * * *
    * * * *
     * *
```

The advantage over rule-based anomaly detection: you do not need to define what "normal" looks like. The embedding space learns this from data. A session that does not resemble any other session stands out geometrically without you writing rules for every possible anomaly type.

### Content deduplication

Exact-match deduplication (hashing) misses near-duplicates — the same article republished with minor edits, or the same product listed with slightly different descriptions. Embedding-based deduplication catches these because near-duplicates produce nearly identical vectors.

```
Approach:
  1. Embed all content
  2. For each item, find neighbors above a high similarity threshold (e.g., 0.95)
  3. Group items into clusters of near-duplicates
  4. Keep one canonical item per cluster
```

This is particularly useful for cleaning training data, deduplicating support ticket databases, or detecting plagiarism.

### Recommendations

If a user engaged with items A, B, and C, embed those items, compute an average vector (the "user interest" vector), and search for items nearest to that average. This gives you content-based recommendations without any collaborative filtering infrastructure.

```
User read: "Python async patterns," "FastAPI tutorial," "uvicorn configuration"
Average embedding --> similar to: "ASGI server comparison," "Starlette middleware guide"
```

The recommendations are explainable — you can show why each item was recommended by showing which of the user's past engagements it is most similar to.

### Classification

Instead of training a text classifier from scratch, embed your labeled examples and use the embeddings as features for a simple classifier (logistic regression, k-nearest neighbors). This approach requires far less training data than training a classifier end-to-end because the embedding model has already learned rich text representations.

```
Approach:
  1. Embed all labeled examples
  2. Train a simple classifier on the embedding vectors
  3. At inference: embed the new text, run the classifier

  Often works with as few as 20-50 labeled examples per class,
  compared to thousands needed for training from scratch.
```

### Multimodal embeddings

Models like CLIP (Contrastive Language-Image Pre-training) embed images and text into the **same vector space**. A photo of a sunset and the text "beautiful sunset over the ocean" produce similar vectors, even though one is pixels and the other is words.

This enables:

- **Text-to-image search:** type a description, find matching images
- **Image-to-image search:** upload a photo, find visually similar photos
- **Image-to-text search:** upload a photo, find relevant text descriptions
- **Zero-shot image classification:** define categories as text, embed them, classify images by nearest text vector

The same pipeline from Section 6 applies — just swap the text embedding model for a multimodal one and feed it images (or image-text pairs) instead of text chunks.

---

## Summary

Embeddings convert text into vectors where proximity means similarity. This single idea enables search by meaning, clustering by topic, and a family of techniques built on vector geometry.

| Concept              | Key takeaway                                                                  |
|----------------------|-------------------------------------------------------------------------------|
| Embeddings           | Dense vectors capturing semantic meaning. Similar texts produce nearby vectors.|
| Embedding models     | Single forward pass, trained via contrastive learning. Much cheaper than LLMs.|
| Cosine similarity    | Measures angle between vectors. Normalize vectors so dot product = cosine sim.|
| Vector stores        | Index + search. FAISS for prototyping, dedicated DBs for production.          |
| Chunking             | The most important pipeline decision. 256-512 tokens, recursive, with overlap.|
| The pipeline         | Chunk, embed, index, search. Foundation for RAG and beyond.                   |
| Common pitfalls      | No threshold, mixed models, bad chunks, no evaluation, ignoring metadata.     |
| Beyond search        | Anomaly detection, deduplication, recommendations, classification, multimodal.|

**Next step:** Open [`project/`](project/) and build a Semantic Search Engine to put these concepts into practice.
