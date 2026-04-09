# Project: Semantic Search Engine

## What you'll build

A command-line tool that embeds documents using a local embedding model, indexes them with FAISS, and finds relevant content by meaning rather than keywords. You will implement document chunking, build a vector index, and run semantic queries — all without any LLM or API key. By the end, you will have a working search engine that understands meaning and a clear picture of why similarity thresholds matter.

## Prerequisites

- Completed the Module 01 project (Token Budget Calculator)
- Read the Module 03 README on embeddings and vector search
- Python 3.11+ with project dependencies installed (`pip install -r requirements.txt` from the repo root — this installs `sentence-transformers`, `faiss-cpu`, and `numpy`)

## How to build

Create a new file `search_engine.py` in this directory. Build it step by step following the instructions below. When you are done, compare your output with `python solution.py`.

## Steps

### Step 1: Set up the file and load the embedding model

Create `search_engine.py` with these imports and a helper function for embedding text:

```python
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

def embed(texts, model):
    """Embed a list of strings and return L2-normalized float32 vectors."""
    vectors = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    return vectors.astype("float32")
```

Test it: load the model with `SentenceTransformer("all-MiniLM-L6-v2")`, embed a sentence, and print the vector's shape and L2 norm (should be 384 dimensions, norm close to 1.0).

### Step 2: Embed sentences and compute cosine similarity

Write a function that embeds 5 sentences — include pairs that are semantically similar and at least one unrelated sentence. Compute pairwise cosine similarity by multiplying the normalized vectors by their transpose (`vecs @ vecs.T`). Print the result as a labeled matrix so you can see which pairs score highest.

Good test sentences:
- Two about password resets (similar)
- Two about OAuth / third-party login (similar)
- One about an unrelated topic like weather or cooking (dissimilar)

### Step 3: Write a chunker that splits text into sections

Create a `RecursiveChunker` class that takes a `max_chunk_size` and `overlap` parameter. It should:

1. Split the input text on double newlines (paragraphs/sections).
2. If a section is still too large, split it further by sentences.
3. Optionally add overlap between consecutive chunks (take the last few words of the previous chunk and prepend them to the next).

Hardcode a sample multi-section document (a technical guide works well — authentication, API design, etc.). Run the chunker and print each chunk with its character count.

### Step 4: Build a FAISS index from chunk embeddings

Embed all chunks from Step 3 and add them to a FAISS `IndexFlatIP` (inner product) index. Since the vectors are L2-normalized, inner product equals cosine similarity.

```python
def build_index(vectors):
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)
    return index
```

### Step 5: Implement semantic search

Write a `search()` function that takes a natural language query, embeds it, and searches the FAISS index for the top-k nearest chunks. Return the matching chunks with their similarity scores.

Test with 3 queries related to your document content and print the top 3 results for each.

### Step 6: Add similarity threshold filtering and wire everything into main()

Search for something completely unrelated to your documents (e.g., "How do I bake a cake?"). Notice that FAISS still returns results — they are just the least-bad matches. Add a threshold parameter and filter out results below it. This demonstrates why production search systems need a minimum similarity cutoff.

Wire all demos into a `main()` function with clear section headers:

```python
def main():
    print("=" * 60)
    print("  Semantic Search Engine")
    print("=" * 60)
    # load model, then call each demo
    ...

if __name__ == "__main__":
    main()
```

Run your script and compare with `python solution.py`.

## Expected output

```
============================================================
  Semantic Search Engine
============================================================

  Loading embedding model (first run downloads ~80 MB)...
  Model loaded.

--- 1. Embed and Inspect a Sentence ---

  Sentence : Authentication is the process of verifying a user's identity.
  Dimension: 384
  First 10 : [0.0432, -0.0271, ...]
  L2 norm  : 1.0000  (should be ~1.0)

--- 2. Semantic Similarity Matrix ---

              S1      S2      S3      S4      S5
  S1    1.000   0.829   0.312   0.405   0.051
  S2    0.829   1.000   0.275   0.381  -0.001
  ...

--- 3. Document Chunking ---

  Chunk 1 (42 chars):
    Getting Started with Authentication
  Chunk 2 (289 chars):
    Passwords and Hashing  Storing passwords ...
  ...
  Total chunks: 8

--- 4. Build Index and Search ---

  Query: "How should I store passwords securely?"
    #1 (score: 0.6543): Passwords and Hashing ...
    ...

--- 5. Similarity Threshold Demo ---

  Query: "How do I make chocolate chip cookies?"
  Results WITHOUT threshold: ...
  Results WITH threshold (score >= 0.35):
    (no results above threshold — correctly rejected)

============================================================
  Done!
============================================================
```

## Stretch goals

1. **Metadata filtering** — Add a title or section label to each chunk and let the user filter results by section before searching.
2. **Different embedding model** — Try `all-mpnet-base-v2` (768 dimensions, more accurate) and compare similarity scores and search quality against `all-MiniLM-L6-v2`.
3. **Hybrid search** — Combine vector similarity with keyword matching (e.g., BM25). Return results that score well on both, and see if precision improves for keyword-heavy queries.
