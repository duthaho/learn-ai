# Project: RAG Q&A

Build a CLI tool that answers questions about a technical document using Retrieval-Augmented Generation. The system retrieves relevant passages, injects them into the prompt, and generates answers with source citations.

## What you'll build

A documentation Q&A assistant that:
- Loads and chunks a technical document with metadata
- Embeds chunks and indexes them with FAISS
- Retrieves relevant chunks for each question with relevance filtering
- Constructs RAG prompts with source labels
- Generates grounded answers with [Source N] citations
- Runs an interactive Q&A loop with retrieval stats

## Prerequisites

- Completed reading the Module 07 README
- Python 3.11+ with project dependencies installed
- At least one LLM provider API key configured in `.env`

## How to build

Work through the steps below in order. Each step builds on the previous one.

## Steps

### Step 1: Load & chunk the document

Define a multi-section technical document as a string constant. Implement a chunking function that splits by sections (headers), then by sentences for oversized sections, with overlap between consecutive chunks. Attach metadata to each chunk.

Functions to implement:
- `chunk_document(text: str, max_chunk_size: int = 300, overlap: int = 50) -> list[dict]` — splits text into chunks with metadata. Each chunk is a dict with `text` and `metadata` (section title, chunk index) fields.

### Step 2: Embed & index

Embed all chunks using sentence-transformers and build a FAISS index.

Functions to implement:
- `build_rag_index(chunks: list[dict], model: SentenceTransformer) -> tuple[faiss.IndexFlatIP, np.ndarray]` — embeds chunk texts, builds and returns a FAISS inner-product index and the vectors.

### Step 3: Retrieve relevant chunks

Implement a retrieval function that searches the index and filters by relevance score.

Functions to implement:
- `retrieve(query: str, model: SentenceTransformer, index: faiss.IndexFlatIP, chunks: list[dict], top_k: int = 3, threshold: float = 0.3) -> list[dict]` — embeds query, searches index, filters by threshold, returns matching chunks with scores sorted by relevance.

### Step 4: Build the RAG prompt

Construct the augmented prompt with system instructions, formatted context chunks, and the user's question.

Functions to implement:
- `build_rag_prompt(question: str, retrieved: list[dict]) -> list[dict]` — returns a messages list with system prompt (cite sources, answer only from context) and user message containing formatted chunks and question. Handles the empty retrieval case.

### Step 5: Generate answers with citations

Send the RAG prompt to the LLM and display the result with source details.

Functions to implement:
- `ask(question: str, model: SentenceTransformer, index: faiss.IndexFlatIP, chunks: list[dict], llm_model: str) -> dict` — orchestrates retrieve → build prompt → LLM call. Returns a dict with answer, sources, token usage, and cost.
- `print_answer(result: dict)` — displays the answer, sources with scores, and token/cost info.

### Step 6: Interactive Q&A loop

Wrap everything in an interactive chat loop with session tracking.

Functions to implement:
- `main()` — loads document, chunks, embeds, indexes, then runs Q&A loop. Handles `/bye`, `quit`, `exit` commands. Prints session summary on exit (total questions, tokens, cost).

## How to run

```bash
cd 07-rag/project
python solution.py
```

## Expected output

```
============================================================
  RAG Q&A — Documentation Assistant
============================================================
  Model: anthropic/claude-sonnet-4-20250514
  Document: Authentication & Security Guide
  Chunks: 12 indexed

  Loading embedding model...
  Embedding and indexing chunks...
  Ready! Type /bye to quit.

You: How should I store passwords?

  [Retrieved 3 chunks (scores: 0.82, 0.75, 0.61)]

  Passwords should never be stored in plain text. Instead, use a modern
  hashing algorithm such as bcrypt, scrypt, or Argon2 [Source 1]. These
  algorithms are designed to be computationally expensive, which slows
  down brute-force attacks [Source 1]. You should also add a unique salt
  to each password before hashing to prevent rainbow table lookups
  [Source 2].

  Sources:
    [1] Passwords and Hashing (score: 0.82)
    [2] Salting Strategies (score: 0.75)
    [3] Common Vulnerabilities (score: 0.61)

  Tokens: 45 in + 92 out | Cost: $0.0014

You: What is the capital of France?

  [No relevant chunks found (best score: 0.12)]

  I don't have enough information in my documents to answer that
  question. My knowledge base covers authentication and security topics.

  Tokens: 32 in + 28 out | Cost: $0.0005

You: /bye

============================================================
  Session Summary
============================================================
  Questions:     2
  Total tokens:  197 (77 in + 120 out)
  Total cost:    $0.0019
============================================================
```

## Stretch goals

1. **Multi-document RAG** — load multiple documents from a directory and track which document each chunk came from in the metadata
2. **Chunk size experiment** — add a `/chunk-size N` command to re-index with different chunk sizes and compare retrieval quality
3. **Streaming RAG** — stream the LLM's response token-by-token (combining Module 05 streaming with RAG)
