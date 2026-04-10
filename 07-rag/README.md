# Module 07 — RAG (Retrieval-Augmented Generation)

Grounding LLM responses in your own documents: building a pipeline that retrieves relevant context and feeds it into the prompt.

| Detail        | Value                                                         |
|---------------|---------------------------------------------------------------|
| Level         | Intermediate                                                  |
| Time          | ~3 hours                                                      |
| Prerequisites | Module 03 (Embeddings & Vector Search), Module 04 (The AI API Layer) |

## What you'll build

After reading this module, head to [`project/`](project/) to build a **RAG Q&A** — a question-answering system over a collection of Authentication & Security documentation that retrieves the most relevant passages and generates grounded, cited answers.

---

## Table of Contents

1. [What is RAG?](#1-what-is-rag)
2. [The RAG Pipeline](#2-the-rag-pipeline)
3. [Document Loading & Chunking](#3-document-loading--chunking)
4. [Retrieval](#4-retrieval)
5. [Augmenting the Prompt](#5-augmenting-the-prompt)
6. [Source Citations](#6-source-citations)
7. [RAG Quality & Failure Modes](#7-rag-quality--failure-modes)
8. [RAG vs Alternatives](#8-rag-vs-alternatives)

---

## 1. What is RAG?

LLMs have a fundamental limitation: their knowledge is frozen at training time. Ask GPT-4 about your company's internal policy document, a paper published last week, or the ticket your colleague closed yesterday, and it cannot answer — it has never seen that information.

There are three approaches to this problem:

| Approach | How it works | Best for |
|---|---|---|
| **Fine-tuning** | Retrain (or adapt) the model weights on your data | Teaching the model a new style or domain-specific reasoning |
| **Long context** | Stuff all relevant documents into a single prompt | Small, stable corpora that fit in the context window |
| **RAG** | Retrieve only the relevant documents at query time, then inject them into the prompt | Large, dynamic, or private knowledge bases |

### RAG = Retrieval + Generation

**RAG** stands for **Retrieval-Augmented Generation**. The core idea is simple:

1. **Retrieve** — search your document corpus to find the passages most relevant to the user's question
2. **Augment** — add those passages to the prompt as context
3. **Generate** — let the LLM answer the question using that context

The LLM is not searching; it is reading. You do the searching. You give it the results. It synthesizes the answer.

### The mental model

Think of RAG like an open-book exam:

- **Without RAG** — the LLM answers from memory, limited to what it learned during training
- **With RAG** — the LLM is handed the relevant pages from the textbook before answering

The LLM's job is the same: reason, synthesize, and respond. RAG just ensures it has the right information in front of it.

### Why RAG wins

RAG is the dominant approach for knowledge-base Q&A because it:

- **Requires no retraining** — add new documents without touching model weights
- **Works with any LLM** — retrieval is model-agnostic; swap providers freely
- **Stays fresh** — update the document index and the system immediately knows
- **Costs less** — embedding and indexing documents is far cheaper than fine-tuning
- **Is auditable** — you can inspect exactly which chunks were retrieved for any answer
- **Enables citations** — the LLM can reference the source documents it was given

### When RAG is NOT the right choice

RAG is not always the answer:

- **Teaching new skills or reasoning patterns** — if you need the model to think differently, not just know more, fine-tuning may be more appropriate
- **Simple, stable facts that fit in one prompt** — if your entire knowledge base is two pages, just include it in the system prompt
- **Real-time or streaming data** — if the answer requires live API calls (stock price, current weather), use tool use (Module 06) instead
- **Complex multi-hop reasoning** — if answering requires synthesizing dozens of documents with intricate dependencies, RAG alone may not suffice without agentic orchestration

### Real-world examples

- **Customer support bots** — retrieve from product docs, FAQs, and release notes to answer user questions
- **Legal document assistants** — search contracts and statutes to answer specific legal questions
- **Internal knowledge bases** — let employees query engineering runbooks, HR policies, and design docs
- **Medical reference tools** — retrieve from clinical guidelines and literature to support diagnoses
- **Code assistants** — retrieve from a codebase's documentation and source files to answer implementation questions

---

## 2. The RAG Pipeline

RAG has two distinct phases that run at different times: **indexing** (done once, offline) and **querying** (done at runtime, per request).

```
╔══════════════════════════════════════════════════════════════╗
║  INDEXING PHASE (offline, run once or on update)            ║
║                                                              ║
║  Documents → Load → Chunk → Embed → Index (vector store)    ║
╚══════════════════════════════════════════════════════════════╝
                                 │
                                 ▼
╔══════════════════════════════════════════════════════════════╗
║  QUERY PHASE (runtime, per user request)                    ║
║                                                              ║
║  Question → Embed query → Search index → Filter by score    ║
║           → Augment prompt → Generate → Return answer       ║
╚══════════════════════════════════════════════════════════════╝
```

### Indexing phase (4 steps)

**Step 1 — Load:** Read raw documents from disk, database, URL, or API. Parse PDFs, Markdown, HTML, or plain text into clean text strings.

**Step 2 — Chunk:** Split documents into smaller passages. A full document is too large and too diluted for effective retrieval. Chunks are typically 200–500 words, with overlap between adjacent chunks. (Section 3 covers this in depth.)

**Step 3 — Embed:** Convert each chunk into a dense vector using an embedding model. Every chunk in your corpus gets its own vector. This is the representation that enables semantic search.

**Step 4 — Index:** Store the chunk vectors (and the original chunk text plus metadata) in a vector store such as FAISS, Chroma, Pinecone, or Weaviate. The store supports fast approximate nearest-neighbor search.

### Query phase (6 steps)

**Step 1 — Embed query:** Convert the user's question into a vector using the same embedding model used during indexing. Consistency here is critical.

**Step 2 — Search:** Perform a vector similarity search in the index to find the top-k chunks whose embeddings are closest to the query vector.

**Step 3 — Filter:** Optionally apply relevance score thresholds to discard chunks that are not similar enough to the query, even if they ranked in the top-k.

**Step 4 — Augment:** Format the retrieved chunks and inject them into the prompt as context, alongside the user's question.

**Step 5 — Generate:** Send the augmented prompt to the LLM. The LLM reads the provided context and generates an answer.

**Step 6 — Return:** Deliver the answer to the user, optionally including citations pointing back to the source chunks.

### Why the split matters

The indexing phase is expensive (embedding many documents) but only runs when your knowledge base changes. The query phase must be fast — it runs on every user request. By pre-computing embeddings during indexing, you ensure that query-time latency is dominated by a single embedding call and a fast vector search, not by processing your entire corpus.

### Building on Module 03

If you completed Module 03 (Embeddings & Vector Search), you already know how to embed text and search FAISS. RAG is that skill applied end-to-end: embedding becomes the indexing step, and similarity search becomes the retrieval step. This module adds the upstream (chunking) and downstream (prompt augmentation, citation, quality evaluation) pieces.

---

## 3. Document Loading & Chunking

Chunking is the step most RAG practitioners underestimate. How you split documents directly determines what the retriever finds — and therefore what the LLM sees.

### Chunk size trade-offs

| Chunk size | Range | Effect |
|---|---|---|
| **Too small** | < 100 chars | Each chunk lacks context. Retrieved passages are cryptic fragments. The LLM cannot synthesize a useful answer from them. |
| **Sweet spot** | 500–2000 chars | Enough context to be self-contained, small enough for precise retrieval. One chunk ≈ one coherent idea. |
| **Too large** | > 3000 chars | Each chunk covers many topics. Similarity is diluted. Relevant content gets buried. The context window fills quickly. |

### Overlap strategy

Adjacent chunks should share some text so that ideas at chunk boundaries are not lost. Without overlap, a sentence split across two chunks is retrievable from neither.

```
Document text:
│ ... end of chunk A ... │ [OVERLAP] │ ... start of chunk B ... │

│◄────── Chunk A ────────────────────►│
                    │◄──────────── Chunk B ────────────────────►│
                    │◄── overlap ──►│
```

A 10–20% overlap is the standard recommendation. For a 1000-character chunk, that means 100–200 characters of shared text between consecutive chunks.

### Chunking strategies

| Strategy | How it works | Best for |
|---|---|---|
| **Fixed-size** | Split every N characters (or tokens), with overlap | Simple corpora, quick prototyping |
| **Sentence-based** | Split on sentence boundaries using NLP | Prose documents where sentences are coherent units |
| **Section-based** | Split on headings, page breaks, or explicit delimiters | Structured documents: manuals, wikis, reports |
| **Recursive** | Try large splits first, then recursively split oversized chunks | General-purpose; handles mixed structure |

For the project in this module, section-based chunking works well: the Authentication & Security documents are organized into named sections, and each section is a natural retrieval unit.

### Metadata preservation

Every chunk must carry metadata so you can cite the source later. At minimum, record the source document name and the section title:

```python
chunk = {
    "text": "JWT tokens must be validated on every request. The signature...",
    "metadata": {
        "source": "authentication-guide.md",
        "section": "Token Validation",
        "chunk_index": 4,
        "char_start": 2048,
    }
}
```

When the retriever returns this chunk, you can tell the user exactly where the information came from. This is what makes RAG auditable and trustworthy.

---

## 4. Retrieval

Retrieval is the step where your vector index earns its keep. Given a user question, you embed it and search for the chunks whose meaning is most similar.

### Similarity search recap

From Module 03: you embed the query with the same model used during indexing, then search the FAISS index for the nearest vectors by cosine similarity.

```python
import numpy as np

def retrieve(query: str, index, chunks, embed_fn, k: int = 5):
    # Embed the query
    query_vector = embed_fn(query)                      # shape: (1, dim)
    query_vector = np.array([query_vector], dtype="float32")

    # Search the index
    distances, indices = index.search(query_vector, k)  # top-k results

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx == -1:
            continue
        results.append({
            "chunk": chunks[idx],
            "score": float(dist),
        })
    return results
```

### Choosing k (top-k)

| k value | Tradeoff |
|---|---|
| **1–2** | Very precise. If retrieval is wrong, there is no fallback. High risk of missing the answer. |
| **3–5** | Balanced. Enough coverage to handle near-misses while keeping the prompt focused. Recommended default. |
| **8–10** | High recall. Useful when the answer might be spread across many chunks. Risk of filling the context window and diluting relevance. |

Start with k=5 and adjust based on your failure mode analysis (Section 7).

### Relevance filtering

Not all top-k results are genuinely relevant. Apply a score threshold to discard low-similarity chunks before including them in the prompt:

```python
def filter_by_score(results, threshold: float = 0.4):
    return [r for r in results if r["score"] >= threshold]
```

Typical FAISS cosine similarity thresholds:

| Score | Interpretation |
|---|---|
| > 0.7 | **High relevance** — almost certainly on topic |
| 0.4–0.7 | **Moderate relevance** — likely useful, include with care |
| < 0.3 | **Low relevance** — discard; including this chunk adds noise |

Thresholds are corpus-specific. Calibrate yours by inspecting retrieved results on a sample of test queries.

### Debugging retrieval

When your RAG system gives wrong answers, start by examining retrieval — 80% of RAG failures originate here.

1. **Print the retrieved chunks** before generating. If the answer is not in the chunks, retrieval failed — not the LLM.
2. **Check score distributions.** If all your scores cluster around 0.3–0.4, your embedding model or chunking may be mismatched.
3. **Test the query in isolation.** Embed the question and inspect the top-10 raw results before any filtering. If the right chunk is not in the top-10, no amount of prompt engineering will fix the answer.

---

## 5. Augmenting the Prompt

Once you have retrieved relevant chunks, you inject them into the prompt as context. How you structure this prompt determines whether the LLM stays grounded in the retrieved evidence.

### The RAG prompt pattern

```python
def build_rag_prompt(question: str, chunks: list[dict]) -> list[dict]:
    # Format retrieved chunks as numbered context blocks
    context_blocks = []
    for i, result in enumerate(chunks, start=1):
        section = result["chunk"]["metadata"]["section"]
        text = result["chunk"]["text"]
        context_blocks.append(f"[Source {i}: {section}]\n{text}")

    context = "\n\n".join(context_blocks)

    system_prompt = (
        "You are a helpful assistant that answers questions about "
        "Authentication & Security documentation.\n\n"
        "Answer ONLY from the context provided below. "
        "If the context does not contain enough information to answer, "
        "say so explicitly — do not fabricate details.\n\n"
        f"Context:\n{context}"
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
```

### Why "answer ONLY from context" matters

Without this instruction, LLMs default to drawing on their training data to fill gaps. For a grounded Q&A system, that is the failure mode you are trying to prevent. An LLM trained on public data will confidently combine your retrieved context with general knowledge — producing answers that look correct but mix sources in unpredictable ways.

Explicit grounding instructions ("answer ONLY from the context provided") push the model toward honest uncertainty when the retrieved context is insufficient, rather than fabrication.

### Formatting chunks with labels

Label each chunk with a source identifier so the LLM can cite it:

```
[Source 1: Token Validation]
JWT tokens must be validated on every request. The signature...

[Source 2: Session Management]
Sessions expire after 30 minutes of inactivity. Refresh tokens...
```

The `[Source N: Section Title]` format gives the LLM structured references it can include in its answer. The section title provides human-readable context that helps the LLM reason about which source is most relevant.

### "Lost in the middle" (Liu et al. 2023)

Research has found that LLMs are better at using information at the **beginning and end** of the context than in the **middle**. For RAG, this means:

- Put your **most relevant chunk first**
- If you have a clear top result, do not bury it at position 3 or 4
- With large k values, consider placing the second-most-relevant chunk last

This is a subtle but measurable effect. For k ≤ 5, it matters less. For k ≥ 8, reorder deliberately.

### Context window budget

Every token you spend on retrieved context is a token you cannot spend on conversation history or the LLM's output. Plan the budget:

```
Total context window:  128,000 tokens  (GPT-4o)
System prompt:         ~300 tokens
RAG context (k=5):     ~2,000 tokens   (5 chunks × ~400 tokens each)
Conversation history:  ~1,000 tokens
User question:         ~50 tokens
─────────────────────────────────────────
Reserved for output:   ~124,650 tokens
```

In practice, keep your RAG context budget under 4,000 tokens for smaller models (8k–16k windows). For larger windows, you have more headroom but diminishing returns — more context means more potential for the LLM to get lost.

**Formula:** `context_budget = window_size - system_tokens - history_tokens - output_reserve`

### Handling "no relevant results"

If all retrieved chunks fall below your score threshold, do not pass empty context to the LLM and hope for the best. Handle it explicitly:

```python
if not filtered_results:
    return "I could not find relevant information in the documentation to answer this question."
```

Letting the LLM answer with no context is equivalent to letting it answer from training data — which is exactly what RAG is meant to prevent.

---

## 6. Source Citations

Citations are what separate a trustworthy RAG system from one that feels like a black box. When users can see where an answer came from, they can verify it, trust it, and escalate when it is wrong.

### Why citations matter

- **Verifiability** — users can read the source and confirm the answer
- **Trust** — grounded answers feel more reliable than assertions with no provenance
- **Debuggability** — when the answer is wrong, the citation tells you whether retrieval or generation failed
- **Compliance** — some domains (legal, medical, financial) require traceable sourcing

### Instructing the LLM to cite

Add specific citation instructions to your system prompt:

```python
system_prompt = (
    "... (grounding instructions) ...\n\n"
    "When you use information from the context, cite the source using "
    "inline references like [Source 1] or [Source 2]. "
    "At the end of your answer, list the sources you cited."
)
```

Be explicit: LLMs need clear instructions on the citation format. "Cite your sources" is too vague — specify the exact format you want.

### Citation format

**Inline citation example:**

```
JWT tokens must be validated on every request [Source 1].
Sessions should be invalidated on logout to prevent replay attacks [Source 2].
```

**Sources list at the end:**

```
Sources:
- [Source 1] Token Validation — authentication-guide.md
- [Source 2] Session Management — authentication-guide.md
```

This format is easy for downstream code to parse: extract `[Source N]` references, look up the corresponding chunk metadata, and render links or tooltips in a UI.

### Citation verification challenges

LLMs sometimes hallucinate citation numbers — claiming `[Source 3]` when only 2 sources were provided, or citing `[Source 1]` for information that came from `[Source 2]`. For high-stakes applications:

- **Verify programmatically** — check that every cited source number corresponds to a real retrieved chunk
- **Cross-reference** — search the cited chunk's text for the key claim in the answer
- **Flag mismatches** — if a cited source does not contain the relevant information, surface a warning rather than silently trusting the LLM

---

## 7. RAG Quality & Failure Modes

RAG systems fail in predictable ways. Knowing the failure modes lets you diagnose issues quickly and apply targeted fixes rather than guessing.

**80% of RAG issues are retrieval issues.** Before optimizing your prompt or switching LLMs, check whether the right chunks are being retrieved.

### Failure mode 1: Retrieval miss

| | |
|---|---|
| **Symptom** | The LLM says "I could not find information about X" — but X is clearly in your documents. |
| **Cause** | The query embedding and the chunk embedding are not close enough in vector space. Common causes: the question uses different vocabulary than the document, the chunk size is too large and dilutes the relevant sentence, or the wrong embedding model is being used. |
| **Fix** | Print the top-10 retrieved chunks and their scores. If the right chunk is absent, try rephrasing the query, reducing chunk size, or switching to a better embedding model. Consider hybrid search (keyword + vector). |

### Failure mode 2: Irrelevant retrieval

| | |
|---|---|
| **Symptom** | The LLM produces an off-topic or confused answer. The retrieved chunks are about adjacent topics, not the question. |
| **Cause** | Chunks are too large (covering many topics) or score thresholds are too low (admitting unrelated results). |
| **Fix** | Raise your score threshold. Reduce chunk size. Add metadata filters to restrict retrieval to relevant document sections or document types. |

### Failure mode 3: Hallucination despite context

| | |
|---|---|
| **Symptom** | The right chunks are retrieved, but the LLM produces an answer that contradicts or extends beyond them. |
| **Cause** | Insufficient grounding instructions, or the LLM is blending retrieved context with training knowledge. |
| **Fix** | Strengthen the grounding instruction ("answer ONLY from the context below — do not use prior knowledge"). Use a lower-temperature setting. Use a more instruction-following model. |

### Failure mode 4: Lost in the middle

| | |
|---|---|
| **Symptom** | The answer ignores the most relevant chunk even though it was retrieved. |
| **Cause** | The most relevant chunk is positioned in the middle of a large context block, where LLM attention is weakest (Liu et al. 2023). |
| **Fix** | Reorder chunks to put the most relevant first. Reduce k to keep context tight. Use a model with stronger long-context performance. |

### Debugging checklist

Follow this order. Each step rules out a layer before blaming the next:

1. **Print retrieved chunks.** Confirm the answer exists somewhere in the retrieved text. If not, the problem is retrieval — stop here and fix indexing or search.
2. **Check scores.** If scores are low across the board, your embedding model or chunking may be poorly matched to your corpus.
3. **Simplify the prompt.** Remove all bells and whistles. Does a minimal system prompt with just the context and the question produce the right answer? If yes, something in your full prompt is interfering.
4. **Try a stronger model.** If retrieval is correct and the prompt is clean, the generation model may lack the capability. Upgrade and retest.

---

## 8. RAG vs Alternatives

RAG is one of four main approaches to giving LLMs access to specialized or up-to-date knowledge. Knowing when to use each — and how to combine them — is what separates RAG practitioners from RAG enthusiasts.

### Decision matrix

| | **RAG** | **Fine-tuning** | **Long context** | **Tool use** |
|---|---|---|---|---|
| **Best for** | Large, dynamic knowledge bases | New skills or reasoning styles | Small, stable corpora | Live data, external systems |
| **Freshness** | High — update index anytime | Low — requires retraining | Medium — update prompt | High — real-time API calls |
| **Cost** | Low — embedding + inference | High — GPU training time | Low — just tokens | Low to medium — per API call |
| **Latency** | Low — fast vector search | None at query time | Low — direct prompt | Variable — depends on external API |
| **Citations** | Native — chunks carry metadata | None — baked into weights | Possible — if docs are labeled | Possible — if API returns source |
| **Complexity** | Medium — indexing pipeline required | High — training infrastructure | Low — just prompt engineering | Medium — tool definitions and loop |

### When to use each

- **RAG** — when your knowledge base is too large for the context window, changes frequently, or needs traceable sourcing
- **Fine-tuning** — when you need the model to behave differently (adopt a persona, follow a format, reason in a domain-specific way) rather than just know more
- **Long context** — when your corpus is small enough to fit in a single prompt and rarely changes; great for prototyping before committing to a RAG pipeline
- **Tool use** — when answers require real-time data, computation, or actions that no static document can provide

### Combining approaches

These are not mutually exclusive. Production systems often combine multiple approaches:

- **RAG + tool use** — retrieve from static docs for background knowledge, call live APIs for current data
- **RAG + fine-tuning** — fine-tune a model on your domain's writing style and reasoning patterns, then use RAG for factual grounding
- **RAG + long context** — retrieve the top candidates, then pass a larger window of surrounding context for each, trading precision for completeness

### What Module 19 covers

This module teaches the foundational RAG pattern. Module 19 (Advanced RAG) covers techniques for production-grade systems:

- **Hybrid search** — combining vector similarity with BM25 keyword search for higher recall
- **Re-ranking** — using a cross-encoder to re-score the top-k results after retrieval
- **HyDE (Hypothetical Document Embeddings)** — generating a hypothetical answer and embedding that instead of the raw query
- **Query expansion** — rewriting or decomposing the user query to improve retrieval coverage
- **Contextual compression** — extracting only the relevant sentence from each chunk rather than passing the full chunk
- **Agentic RAG** — letting an LLM decide when to retrieve, what to search for, and how many rounds of retrieval are needed
