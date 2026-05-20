# Module 19 — Advanced RAG

Going beyond top-k dense retrieval: hybrid lexical+semantic recall, cross-encoder precision, and how to measure that any of it actually helped.

| Detail        | Value                                                                                          |
|---------------|------------------------------------------------------------------------------------------------|
| Level         | Intermediate                                                                                   |
| Time          | ~3 hours                                                                                       |
| Prerequisites | Module 07 (RAG), Module 03 (Embeddings & Vector Search), Module 15 (Evaluation & Testing)      |

## What you'll build

After reading this module, head to [`project/`](project/) to build a CLI that runs a 5-stage hybrid retrieval pipeline (dense + BM25 → RRF fusion → cross-encoder rerank → LLM answer) and includes a 15-query labeled eval set so you can quantify the lift over basic RAG with `python solution.py --eval`.

---

## Table of Contents

1. [Why Basic RAG Hits a Wall](#1-why-basic-rag-hits-a-wall)
2. [The Two-Stage Retrieval Pattern](#2-the-two-stage-retrieval-pattern)
3. [Sparse Retrieval Is Not Dead](#3-sparse-retrieval-is-not-dead)
4. [Score Fusion: RRF vs Weighted Sum](#4-score-fusion-rrf-vs-weighted-sum)
5. [Cross-Encoder Reranking](#5-cross-encoder-reranking)
6. [Measuring "Better": Recall, MRR, NDCG](#6-measuring-better-recall-mrr-ndcg)
7. [The Wider Toolbox](#7-the-wider-toolbox)
8. [When Not to Add Complexity](#8-when-not-to-add-complexity)

---

## 1. Why Basic RAG Hits a Wall

This module assumes you've built [Module 07](../07-rag/) — a dense-only pipeline that embeds the query, runs a top-k search in a vector store, pastes the chunks into a prompt, and lets the LLM answer. That pipeline works. It also breaks in four predictable ways the moment you put it in front of real users with real queries, and each failure mode has a name and a fix.

### The recall ceiling on rare-keyword queries

Dense embeddings are excellent at synonyms and paraphrases and weak at low-frequency named entities. The reason is structural: a sentence-transformer like `all-MiniLM-L6-v2` was trained to map sentences with similar *meaning* into nearby vectors, and "similar meaning" is dominated by the high-frequency words that carry the topic. A rare token — `bcrypt`, `CVE-2024-3094`, `RFC 7519`, `argon2id` — contributes some signal to the embedding, but it competes with the broader topical signal, and the broader signal usually wins.

A concrete failure: a security-docs corpus contains one chunk that mentions `bcrypt` by name. Two user queries arrive in the same minute.

- Query A: `"what is bcrypt"` — the literal word `bcrypt` appears verbatim. Dense retrieval ranks the right chunk #1 because the topic signal (hashing) and the rare-token signal (bcrypt) both point at the same place.
- Query B: `"what slow-hash algorithm should I use"` — the literal word `bcrypt` does not appear. Dense retrieval ranks an unrelated chunk about TLS handshakes #1 because "algorithm" and "use" dominate the embedding and pull it toward general crypto discussion.

The same retriever serves both queries. One works, one fails, and the failure has nothing to do with chunking, prompting, or the LLM. It is a recall ceiling on the rare-token axis. BM25 (Section 3) would invert the failure pattern — it nails query A because `bcrypt` is rare in the corpus and therefore high-IDF, but it whiffs on query B because the literal token `bcrypt` is not in the question. Neither retriever, alone, handles both queries. Together, they do.

The size of the ceiling depends on the embedding model and the corpus, but the shape is universal: a sentence-transformer benchmarked on MS MARCO will get to ~70% recall@10 on benchmark queries and drop to 45-55% on real production queries that include error codes, version numbers, and quoted error messages. The gap is not a tuning problem. It is a representational limit of single-vector dense embeddings, and no amount of chunking strategy or threshold-tuning closes it. The fix is to add a second retriever with complementary failure modes.

### Lost in the middle

Liu et al. ("Lost in the Middle: How Language Models Use Long Contexts", 2023) showed that LLMs do not attend uniformly over the input context. Recall accuracy on a question whose answer is buried at position N in a list of retrieved passages follows a U-shape: high at the beginning of the context, high at the end, sharply lower in the middle. The effect is robust across model families and persists even for models with very long context windows.

The implication for RAG is sharper than it looks. The naive instinct — "the model can handle 128k tokens, so retrieve 50 chunks and paste them all in" — produces *worse* answers than retrieving 50 chunks and reranking down to 3. The extra 47 chunks crowd the prompt with content the model under-attends to, and they push the actually-relevant chunk into the middle of the context where the model is most likely to miss it. More context is not more signal; it is more noise plus the same signal in a worse position.

The fix is the rerank stage. Retrieve wide (top 20-50), score precisely (cross-encoder, Section 5), keep only the top 3-5, and place them at the beginning of the prompt. The model attends well to the top of the context, the rerank score guarantees that the top of the context is the most relevant chunk, and the lost-in-the-middle penalty is paid on nothing important because nothing important is in the middle.

A second-order implication: the same model performs measurably worse on retrieval Q&A with k=20 than with k=5 even when recall@5 and recall@20 are both 1.0 — that is, even when the relevant chunk is in both contexts. The extra 15 chunks in the k=20 case are pure noise that dilutes the model's attention without adding information, and the answer quality degrades. The intuition "more retrieval is more knowledge" is wrong; more *relevant* retrieval is more knowledge, and more *total* retrieval past the relevant chunks is anti-knowledge.

The lost-in-the-middle effect was first measured on multi-document question-answering benchmarks, but the same shape shows up in code generation, summarization, and any other task where the LLM is given a long context with key information embedded in it. The effect persists across Claude, GPT-4, Gemini, and open-source models; it persists across context lengths from 4k tokens to 200k tokens; it gets *worse* as the context gets longer, not better. The architecture of self-attention itself produces the U-shape, and no model in the current generation escapes it.

### Position bias

A subtler version of the same effect: even within a small set of retrieved chunks, the *order* matters. The chunk in position 1 of the prompt gets cited disproportionately in the answer, regardless of whether it is the most relevant. A model that sees `[Source 1] [Source 2] [Source 3]` and is asked to cite its sources will lean on Source 1 even when Source 2 contains the more direct answer.

This is why reranking matters not just for selection but for *ordering*. The dense retriever returns chunks in dense-similarity order, which is not the order the LLM should read them in. The cross-encoder reorders them in query-document-relevance order, which is. The same three chunks in a different order produce measurably different answers, and the order set by the cross-encoder is the one that matches what a careful human reader would prioritize.

### Semantic-vs-lexical mismatch

The deepest version of the recall problem: dense and sparse retrievers fail on *different* queries because they encode different aspects of similarity.

- **Dense retrievers** collapse synonyms ("car" and "automobile" map to nearby vectors), handle paraphrases ("how do I authenticate?" matches "what is the auth flow?"), and bridge cross-lingual gaps. They lose precision on rare named entities and exact phrases — the more specific the term, the weaker its contribution to the embedding.
- **Sparse retrievers** (BM25, TF-IDF) match exact tokens with IDF weighting, which makes them excellent at rare named entities, error codes, model identifiers, and quoted phrases. They have no concept of synonymy — "car" and "automobile" are unrelated tokens to BM25 — and paraphrase queries fail completely if no query token appears in the relevant document.

Two retrievers, two failure modes, near-zero overlap between the queries that fail. The hybrid pipeline is not "dense plus a fallback for when dense fails" — it is two retrievers with structurally complementary strengths, fused so that a query that fails in one is rescued by the other. Section 4 covers how to combine the scores; Section 2 covers the overall pipeline shape.

### The four failure modes at a glance

| Failure mode              | What breaks                                                  | Fix                                  |
|---------------------------|--------------------------------------------------------------|--------------------------------------|
| Rare-token recall ceiling | Dense misses queries with rare named entities                | Add BM25 (Section 3)                 |
| Lost in the middle        | Relevant chunk buried in the middle of a long context        | Rerank and trim (Section 5)          |
| Position bias             | Chunk at position 1 cited disproportionately                 | Rerank to put most relevant first    |
| Lexical/semantic mismatch | Single retriever covers one similarity dimension, not both   | Fuse two retrievers (Section 4)      |

All four failures have the same root cause: a single dense retriever, used end-to-end, encodes a single notion of similarity in a single ranking pass, and that single pass cannot do everything the production query distribution asks of it. The two-stage hybrid pipeline is the architectural answer because it gives each stage one job and each retriever one job — and the jobs compose into a system that handles the full query distribution where any individual piece would not.

---

## 2. The Two-Stage Retrieval Pattern

The architectural answer to all four failure modes from Section 1 is the same: split retrieval into a fast, coarse **recall stage** that casts a wide net, and a slow, accurate **precision stage** that narrows the net to what actually goes in the prompt. The two stages run different models, optimize for different metrics, and have different scaling characteristics. Mixing them up — using a precision model at recall scale or a recall model where precision matters — is the most common architectural mistake in production RAG.

### Recall stage

The job of the recall stage is to make sure the relevant chunk is *somewhere* in the top K, with K large enough that the precision stage has something to work with. K is typically 20-100 for production systems, and the metric to optimize here is recall@K (Section 6): what fraction of relevant documents did we get into the top K, regardless of where they ranked within it?

The recall stage runs a **bi-encoder** (a dense embedding model that produces a single vector per document, precomputed at index time) or a **sparse retriever** (BM25 against an inverted index). Both scale to millions of documents because the per-query cost is sublinear in corpus size: vector ANN search with FAISS or HNSW is O(log n), and BM25 with an inverted index is O(query terms × posting list length), neither of which grows with the corpus the way an exhaustive comparison would.

The defining property of a recall-stage model is that the document representation is *independent of the query*. The bi-encoder embeds documents once at index time and stores the vectors. BM25 builds the inverted index once and stores the postings. At query time, only the query is processed; the document representations are looked up, not recomputed. This is what makes recall-stage retrieval cheap enough to run on the full corpus on every query.

The cost of this scaling property is representational: a single fixed-size vector (typically 384, 768, or 1536 dimensions) cannot capture every possible aspect of a document's meaning. Two documents that are similar along one axis (topic) but different along another (sentiment, recency, level of detail) collapse to nearby vectors and become indistinguishable to the bi-encoder. The recall stage accepts this loss of resolution as the price of being able to score the full corpus in a few milliseconds; the precision stage is what restores the resolution on the small candidate set.

### Precision stage


The job of the precision stage is to take the recall stage's top K and reorder it so that the top 3-5 are the chunks most relevant to *this specific query*. The metric here is MRR@k or NDCG@k (Section 6): where does the first relevant chunk land in the reranked list, and how good is the ordering of the top few?

The precision stage runs a **cross-encoder**: a transformer that takes the (query, document) pair concatenated as a single input and produces a single relevance score. Section 5 covers the mechanics. The crucial property: a cross-encoder *cannot* precompute document representations, because the relevance score depends on the joint encoding of query and document. Every query requires a fresh forward pass through the transformer for every candidate document.

This is why the cross-encoder runs second, on a small set of candidates, instead of first on the full corpus. A cross-encoder over 20 candidates at ~50ms each is ~1 second of latency; the same cross-encoder over a million-document corpus would be ~14 hours per query. The precision stage's quality wins are real but only viable on a pre-filtered candidate set. The recall stage is what makes the precision stage affordable.

### The pipeline

```
question
   │
   ├──► [1] Dense retrieve   (FAISS, top-K_dense=20)        ──┐
   │                                                          │
   ├──► [2] Sparse retrieve  (BM25, top-K_sparse=20)         ─┤
   │                                                          ▼
   │                                              [3] RRF fusion → top-K_fused=20
   │                                                          │
   │                                                          ▼
   │                                              [4] Cross-encoder rerank → top-K_final=3
   │                                                          │
   │                                                          ▼
   │                                              [5] Build prompt + LLM call
   ▼
                                                       answer + [Source N] citations
```

Five stages, three of which are retrieval-only and incur no LLM call. The cost profile is dominated by the cross-encoder rerank and the final LLM call; the dense and BM25 stages combined add ~10ms and the fusion is microseconds. The cross-encoder rerank typically adds 100-500ms depending on candidate count and model size. The final LLM call dwarfs everything else, but it sees a 3-chunk prompt instead of a 20-chunk prompt, which saves more tokens than the rerank cost.

### Latency and quality at a glance

| Stage                  | Model type                              | Latency / 20 docs | Quality              |
|------------------------|-----------------------------------------|-------------------|----------------------|
| Recall (bi-encoder)    | dense embedding + ANN search            | ~5 ms             | coarse               |
| Recall (BM25)          | inverted index + scoring                | ~1 ms             | coarse, complementary|
| Precision (cross-encoder) | (query, doc) pair through a transformer | ~50 ms            | strong               |

The asymmetry in the latency column is the whole point. The recall stage is cheap enough to run on the full corpus; the precision stage is expensive enough that you only run it on what the recall stage hands you. Production systems that try to skip the recall stage and run a cross-encoder directly hit a wall the moment the corpus exceeds a few thousand documents; production systems that try to skip the precision stage and just paste 20 chunks into the prompt hit the lost-in-the-middle wall described in Section 1. The two-stage shape is what makes both quality and scale tractable in the same pipeline.

### Stage-by-stage latency budget

For a typical production system retrieving from a ~100k-chunk corpus, the per-stage budget looks something like this:

| Stage                       | Latency budget | What dominates                          |
|-----------------------------|----------------|------------------------------------------|
| Embed query (dense)         | 5-20ms         | One forward pass of the embedding model  |
| FAISS / HNSW search         | 1-10ms         | Index size and ef_search parameter       |
| BM25 lookup                 | <5ms           | Inverted-index posting-list intersection |
| RRF fusion                  | <1ms           | Pure Python sorted-merge                 |
| Cross-encoder rerank (20)   | 50-500ms       | Number of candidates × model size        |
| Prompt assembly             | <5ms           | String concatenation                     |
| LLM call                    | 1000-5000ms    | Provider, model, output length           |

Total retrieval cost (everything before the LLM) is typically 100-600ms, comparable to one slow database query and small relative to the LLM call. The architectural choice that matters most for latency is the cross-encoder candidate count — going from 20 to 100 candidates is the difference between ~100ms and ~500ms of rerank cost — and the right answer depends on whether the quality lift from reranking 100 vs 20 candidates is visible in your eval.

### Where the project lives in this picture

The project in this module implements all five stages on a corpus of ~33 chunks. At that scale, none of the latency concerns bite — the full pipeline runs in under a second end-to-end — and the small corpus makes it easy to inspect what each stage produced and verify by eye that the cross-encoder's reordering is sensible. The 15-query labeled eval set is what lets you measure that the hybrid+rerank pipeline beats the dense-only baseline on this corpus, not just that you have implemented more code.

### A note on naming

The literature uses three terms that all refer to roughly the same thing: "two-stage retrieval", "retrieve-and-rerank", and "candidate generation + ranking". The candidate-generation/ranking framing comes from the recommendation-systems world (where YouTube and Netflix have used this pattern for over a decade); the retrieve-and-rerank framing comes from the information-retrieval world. Both communities converged on the same architectural shape because both faced the same scaling problem: an expensive scoring function that gives high-quality results but cannot run on the full corpus. The names differ; the design is the same. This module uses "recall stage" and "precision stage" because those names are descriptive of the *job* each stage does, not of the *order* they happen to run in or the *community* they came from.

---

## 3. Sparse Retrieval Is Not Dead

The 2019-2021 wave of dense-retrieval research produced models like DPR, ANCE, ColBERT, and SBERT that beat BM25 on benchmark leaderboards by 5-15 points. The takeaway from the press releases was "dense wins, sparse is legacy." The takeaway from the practitioners was different: those models beat BM25 *on the benchmark they were trained for*, and underperform a well-tuned BM25 on out-of-domain corpora, on rare-entity queries, and on any setting where the test distribution does not match the training distribution.

Lin et al. ("Pretrained Transformers for Text Ranking: BERT and Beyond", 2021) made the comeback narrative formal: across a survey of dozens of papers, BM25 baselines often beat ill-tuned dense rankers, and the published "dense wins" results frequently turned on benchmark-specific fine-tuning that did not generalize. The practical answer, the paper argued, is not "pick one"; it is "use both, fuse the rankings, and ship a hybrid system that beats either single retriever on every query type." That conclusion has held up; every major production RAG system shipped in the last three years uses sparse and dense together.

The "dense vs sparse" debate is over. The settled answer in 2024-2026 production systems is: both, fused with RRF, with cross-encoder rerank on top. Pinecone, Weaviate, Vespa, Qdrant, OpenSearch, and Elastic all ship hybrid retrieval as a first-class feature; the major LLM-application frameworks (LangChain, LlamaIndex, Haystack) all have hybrid retrievers in their core abstractions. The argument is no longer about which retriever to use; it is about how to fuse them and which reranker to put on top. This module's project implements the consensus answer from scratch so you understand what each piece does.

### BM25 mechanics in plain English

BM25 scores a document `d` against a query `q` by summing, over each query term `qi`, three factors:

- **Term frequency (TF)**: this document mentions `bcrypt` 5 times — that is more relevant than a document that mentions it once.
- **Inverse document frequency (IDF)**: out of the 33 chunks in the corpus, only 1 mentions `bcrypt` — so `bcrypt` is highly informative when it appears. A token like `the`, which appears in every chunk, contributes near-zero IDF.
- **Length normalization**: this document is 3000 words long — do not reward it just for being long enough to mention `bcrypt` 5 times. Divide by a length factor so short, focused documents are not unfairly out-competed by long ones that happen to use the term.

The full formula is:

```
BM25(q, d) = Σ_qi IDF(qi) · TF(qi, d) · (k1 + 1)
                 ──────────────────────────────────────────
                 TF(qi, d) + k1 · (1 - b + b · |d| / avgdl)
```

with `k1 = 1.5` and `b = 0.75` as the standard defaults that no one ever needs to retune for general-purpose corpora. `|d|` is the document length in tokens; `avgdl` is the average document length across the corpus. `k1` controls how quickly term-frequency saturates (with `k1 = 1.5`, the marginal value of the 6th occurrence is roughly half the marginal value of the 1st), and `b` controls how aggressively to penalize long documents (with `b = 0.75`, a document twice as long as average has its TF scaled down by ~33%).

There are no embeddings, no neural network, no GPU. BM25 is a probabilistic ranking function over an inverted index, and a tuned BM25 implementation on a million-chunk corpus runs in under a millisecond per query on a single CPU core. It is the cheapest retriever in the toolkit and remains shockingly hard to beat on the queries it is suited for.

BM25 is the successor to TF-IDF, the previous-generation IR scoring function. The two differ in one important detail: BM25 saturates the term-frequency contribution (the 6th occurrence of `bcrypt` adds less than the 1st), while TF-IDF treats every occurrence equally. The saturation matters in practice — without it, a document that repeats a query term 100 times scores wildly higher than a document that uses it twice, regardless of whether the repetition adds information. BM25's `k1` parameter controls the saturation curve, and the standard `k1=1.5` is what makes the function robust to spammy or repetitive documents that would game pure TF-IDF.

### When BM25 wins

- **Exact terms**: model names (`claude-sonnet-4-20250514`), error codes (`ERR_CERT_AUTHORITY_INVALID`), API method names (`client.messages.create`), version numbers (`Python 3.11.4`), CVE identifiers, RFC numbers. The literal token carries information the embedding model has no way to know is important.
- **Low-frequency named entities**: rare people, rare places, rare products. The IDF term rewards exactly the tokens the embedding model under-weights.
- **Queries that quote the doc**: a user who copies an error message from a stack trace and pastes it into the search box has handed BM25 a near-perfect query. Dense retrieval will pull in semantically related stack traces; BM25 will pull in the *exact* one if it is in the corpus.

### When BM25 loses

- **Paraphrases**: "how do I authenticate?" vs "what's the login flow?" share zero meaningful tokens. BM25 sees two unrelated queries; the embedding model sees the same question.
- **Synonyms**: "car" and "automobile", "logout" and "sign out", "delete" and "remove". The user types one word; the document uses the other; BM25 finds nothing.
- **Cross-language**: a query in English against a document in Spanish. BM25 has no signal; multilingual embeddings handle this case routinely.
- **Implicit-topic queries**: "the slow-hash algorithm the OWASP guide recommends" — the document never uses the phrase "slow-hash algorithm", and BM25 has no path from the query phrasing to the actual content.

The two failure-mode lists are nearly disjoint. That disjointness is the structural reason hybrid retrieval works: a query that fails on BM25 because it is a paraphrase rarely fails on dense, and a query that fails on dense because it contains a rare named entity rarely fails on BM25. Fusion (Section 4) is the mechanism that converts the disjointness into a strict improvement over either retriever alone.

### Implementation choices

The standard Python BM25 implementations are:

- **`rank_bm25`** — pure-Python, simple, no dependencies beyond NumPy. ~10x slower than the alternatives but trivial to install and good enough for corpora up to ~100k chunks. Used in this module's project.
- **`bm25s`** — written by Xing Han Lu, sparse-matrix backed. 50-500x faster than `rank_bm25` on the same corpus. The right choice for any corpus that will not comfortably tokenize and score in `rank_bm25`'s time budget.
- **Pyserini** — Python wrapper around Lucene's BM25. The reference implementation for IR research benchmarks. Heavier dependency footprint (a JVM) but the gold-standard implementation. Use it when reproducing published results matters.
- **Elasticsearch / OpenSearch** — production search engines built around inverted indexes with BM25 as the default scoring function. The right answer when BM25 is part of a larger search infrastructure rather than a component of a single RAG pipeline.

All four produce the same algorithm with the same `k1=1.5` `b=0.75` defaults. The choice is operational, not algorithmic: pick the implementation that fits your dependency budget and corpus size.

BM25 variants exist — BM25+, BM25L, BM25F (for fielded documents with separate per-field weighting) — but none of them outperform vanilla BM25 by enough to matter for general-purpose retrieval. The BM25F variant is the one to know about: it scores a query against a document by combining BM25 scores from multiple fields (title, body, anchor text) with per-field weights. Useful when your corpus has structured fields with different relevance properties; overkill for plain-text chunks.

### Tokenization caveats

BM25 quality depends on tokenization in ways that dense retrieval does not. A BM25 index built on whitespace-split tokens treats `bcrypt`, `Bcrypt`, and `BCRYPT` as three different terms, with three different IDF values and three different posting lists. A query for one will not match documents containing the others. The standard mitigations — lowercasing, stemming (Porter, Snowball), removing stopwords — are not optional decorations; they are load-bearing for BM25 quality. Most production BM25 implementations apply them by default; if you are getting unexpectedly weak BM25 results, the tokenization pipeline is the first place to look.

The dense retriever does not have this problem because the embedding model was trained on raw text and learned its own normalization implicitly. The hybrid pipeline thus has two different tokenization pipelines running in parallel (one for the embedding model, one for BM25) and the BM25 side needs explicit attention that the dense side does not.

---

## 4. Score Fusion: RRF vs Weighted Sum

You have two ranked lists: dense's top 20 and BM25's top 20. They overlap in some chunks, disagree on others, and have score distributions that are not on the same scale. You need to produce one fused ranked list to hand to the cross-encoder. There are two standard approaches; one is the obvious choice, and it is the wrong one.

### The scoring problem

Dense cosine-similarity scores live in `[0, 1]` (or `[-1, 1]` depending on the model), are bounded, and behave roughly comparably across queries. A score of 0.7 means roughly the same thing on query A as it does on query B.

BM25 scores are unbounded positives that depend on the query's IDF mass, the document length, and the corpus statistics. A BM25 score of 8.4 on query A is not comparable to a BM25 score of 8.4 on query B — different queries have different maximum possible scores, and the scale shifts every time the corpus is updated.

You cannot add a 0.7 cosine to an 8.4 BM25 directly. The arithmetic produces a number; the number does not mean anything. Two strategies handle the mismatch.

### Weighted sum (the obvious choice, the wrong one)

The intuitive fix: normalize the BM25 scores to `[0, 1]` (typically by min-max scaling over the returned result set), then take a weighted sum:

```
score(d) = α · dense_score(d) + (1 - α) · bm25_normalized(d)
```

This works in a demo. It fails in production for three reasons.

- **Min-max normalization is fragile.** The min and max are computed over *this query's* result set, which means a chunk's normalized score depends on which other chunks happened to appear in the top 20. A relevant chunk that scored 4.0 might be normalized to 0.95 on a query where the top result scored 4.2, and to 0.30 on a query where the top result scored 12.0. The scale shifts per query in ways the weighted sum cannot compensate for.
- **`α` needs tuning per corpus.** The right balance between dense and sparse depends on the corpus's mix of synonym-heavy vs entity-heavy queries, which varies per domain and per user population. A medical-docs corpus might want `α = 0.4` (favor BM25 for drug names); a customer-support corpus might want `α = 0.7` (favor dense for paraphrased questions). Tuning `α` requires a labeled eval set, which is exactly what most teams adding hybrid retrieval do not have yet.
- **It does not survive a new query distribution.** The `α` that was optimal on last month's queries is not optimal on this month's, because the query distribution drifts. Weighted-sum fusion is fragile to drift in a way that pure-ranking fusion is not.

### Reciprocal Rank Fusion (RRF)

The published alternative, from Cormack et al. ("Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods", SIGIR 2009):

```
RRF(d) = Σ_r  1 / (k + rank_r(d))
```

Sum, over each ranker `r`, the reciprocal of the document's rank in that ranker's list, with a smoothing constant `k = 60` (the published default — almost no one retunes this). Documents that appear in only one ranker's list get a single term in the sum; documents that appear in both get two terms.

The key insight: RRF uses ranks only. The actual score values from dense and BM25 are discarded once the ranking is established. No normalization step. No `α` to tune. The scale-mismatch problem disappears because the scales are never on speaking terms; only the ordinal positions matter.

### Worked example

Consider two chunks in a fused top-20:

- **Chunk X**: ranked #3 by dense, #5 by BM25. RRF score = `1 / (60 + 3) + 1 / (60 + 5)` = `0.01587 + 0.01538` ≈ `0.03125`.
- **Chunk Y**: ranked #1 by dense, missing from BM25's top 20. RRF score = `1 / (60 + 1)` ≈ `0.01639`.

Chunk X wins, despite ranking lower than Chunk Y in dense and appearing nowhere near the top of either list. That is the RRF insight in one number: **agreement across rankers beats peak score from one**. A chunk that two independent retrievers both think is relevant is more reliably relevant than a chunk that one retriever loves and the other ignores, even if the "loving" ranker put it at the very top.

The same logic generalizes. A chunk that all three of dense, BM25, and a third retriever (say, a code-aware retriever) rank in their top 10 is probably the right chunk — three independent measures of similarity all pointing at the same place is a much stronger signal than any one of them at peak. This is why RRF is preferred in production: it converts retriever diversity into a quality signal automatically, without any per-retriever tuning, in a way that weighted sum cannot match.

The `k = 60` smoothing matters here too. Without it (i.e., using just `1 / rank`), the rank-1 chunks would dominate so heavily that nothing else would matter — the difference between rank-1 and rank-2 would be 0.5, but the difference between rank-10 and rank-11 would be only 0.009. With `k = 60`, the curve flattens: rank-1 vs rank-2 is `1/61 - 1/62 ≈ 0.0003`, and rank-10 vs rank-11 is `1/70 - 1/71 ≈ 0.0002`. Rank-1 still matters more, but not so much more that mid-rank agreement is drowned out.

### Why RRF is the production default

- **No tuning.** `k = 60` works across domains; the published value has been the default for fifteen years.
- **No normalization.** Score scales never need to be aligned, which means dense and BM25 can be replaced or upgraded independently without breaking the fusion.
- **Robust to drift.** Score-scale drift (BM25's IDF shifts when the corpus is reindexed, dense scores shift when the embedding model is changed) does not affect ranks. A retriever upgrade is a routine operation, not a fusion-recalibration project.
- **Extensible.** A third retriever (e.g., a dedicated code-aware retriever, or a graph-based retriever) is added by appending one more term to the sum. Weighted sum requires re-tuning `α` into a multi-dimensional weight vector; RRF just gets a new summand.

The project in this module uses RRF with `k = 60`. If you find yourself wanting to tune `k`, the answer is almost always "you don't" — the gains from tuning are smaller than the gains from improving the underlying retrievers, and the tuning cost is real.

### Side-by-side

| Property                       | Weighted sum                                  | RRF                                            |
|--------------------------------|-----------------------------------------------|------------------------------------------------|
| Inputs                         | Raw scores (after normalization)              | Ranks only                                     |
| Per-corpus tuning              | Yes (`α`, normalization params)               | No (`k=60` default works)                      |
| Survives score-scale drift     | No                                            | Yes                                            |
| Survives retriever swap        | Requires recalibration                        | Works unchanged                                |
| Extending to a third retriever | Requires new weight vector                    | Append one term to the sum                     |
| Implementation complexity      | ~30 lines including normalization edge cases  | ~10 lines including the sum                    |
| Production track record        | Common in pre-2015 IR systems                 | The default for modern hybrid retrievers       |

There are cases where weighted sum can outperform RRF — narrow domains with stable score distributions, single-corpus systems where the cost of recalibration is amortized over millions of queries, and research settings where the per-query cost of tuning is paid offline. Outside those cases, RRF is the production default and the default in this module's project.

---

## 5. Cross-Encoder Reranking

The fused top-20 from RRF still has the lost-in-the-middle problem if you paste all 20 into the prompt, and it still has chunks in the wrong order if you trust the fused ranking to be the final ranking. The cross-encoder fixes both: it reorders the 20 by direct query-document relevance, and it filters down to the top 3-5 that actually go in the prompt. Understanding why cross-encoders win requires understanding what makes them structurally different from bi-encoders.

### Bi-encoder vs cross-encoder

A **bi-encoder** is the standard sentence-transformer setup. The query goes through the transformer and produces a vector. Each document goes through the same transformer (at index time) and produces a vector. Relevance is the dot product (or cosine) between the two vectors. The query and the document never see each other inside the transformer; their encodings are independent, and the "relevance" signal is a coarse measure of how similar two independently-computed representations are.

```
Bi-encoder:
  query  → [transformer] → q_vec     ─┐
  doc    → [transformer] → d_vec     ─┴► dot(q_vec, d_vec) = score
```

A **cross-encoder** concatenates the query and the document into a single input — typically `[CLS] query [SEP] document [SEP]` — and runs the joint sequence through the transformer. The transformer's self-attention layers can attend across both halves: each token in the document can attend to each token in the query and vice versa, layer after layer. The output is a single relevance score, produced by a classifier head on top of the `[CLS]` token's final-layer representation.

```
Cross-encoder:
  [CLS] query [SEP] doc [SEP] → [transformer with cross-attention] → score
```

The architectural difference looks small. The behavioral difference is large.

### The cost consequence

A bi-encoder is `O(n)` to score `n` documents at query time, but with precomputed document embeddings the per-document cost is one dot product — microseconds on a CPU, nanoseconds on a GPU. The transformer forward pass over a document happens once at index time and is amortized over every future query.

A cross-encoder is `O(n)` at query time *with a full transformer forward pass per candidate*. There is no precomputation possible: the joint encoding depends on the specific query, and a different query against the same document requires a fresh forward pass. A ~25M-parameter cross-encoder runs each (query, document) pair in roughly 30-80ms on CPU, or 2-10ms on GPU. Multiply by the candidate count: 20 candidates is 600-1600ms on CPU; 100 candidates is 3-8 seconds.

This is the structural reason cross-encoders are only viable on the top 10-100 candidates from the recall stage. Running a cross-encoder over a million-document corpus would take hours per query. Running it over 20 candidates is a tractable subsecond rerank that produces measurably better rankings than the bi-encoder alone.

### Why cross-encoders win on quality

The reason is not "they're bigger" — many bi-encoders are larger than the cross-encoders that beat them. The reason is the attention pattern. A cross-encoder can model query-document *interactions* in a way a bi-encoder structurally cannot.

Concrete example. Two documents about the same topic:

- Document A: "yes, JWT tokens should be signed with HS256 for shared-secret authentication"
- Document B: "no, HS256 is not appropriate for JWT signing because it requires a shared secret"

A query: "should I use HS256 for JWT?"

A bi-encoder embeds the query and the two documents independently. Both documents discuss JWT and HS256; both vectors are close to the query vector. The bi-encoder cannot tell them apart in any meaningful way — it sees two topically-relevant chunks. The cross-encoder, with attention spanning both halves, can integrate "should I use" with "yes" or "no" and produce a stark relevance difference based on the *answer the document actually gives to the query's question*. The first document scores high; the second scores low.

This is the kind of quality improvement that recall@k cannot measure (both chunks are "retrieved") but that MRR@k captures cleanly (the user wants the chunk that answers their question first, not the chunk that mentions the topic).

The MS MARCO training data that most off-the-shelf cross-encoders use is specifically labeled for this distinction: human annotators marked passages as relevant or not relevant to specific queries, with care taken to distinguish "this passage mentions the topic" from "this passage answers the question". The cross-encoder learns the answer-vs-mention distinction directly because the training labels encode it; the bi-encoder learns generic semantic similarity because that is what its training objective rewards. Two different training objectives, two different behaviors, two different roles in the pipeline.

### Model choices

A few cross-encoder options, in order of cost-quality tradeoff:

- **`cross-encoder/ms-marco-MiniLM-L-6-v2`** — the one used in the project. ~80MB, 6 transformer layers, trained on MS MARCO passage ranking. ~50ms per (query, doc) pair on CPU; the standard production default for self-hosted reranking. Good quality on general English text.
- **`cross-encoder/ms-marco-MiniLM-L-12-v2`** — same training, 12 layers. ~120MB, ~120ms per pair on CPU. Marginally better quality than the 6-layer version; usually not worth the latency cost unless your eval shows a real lift.
- **`cross-encoder/ms-marco-electra-base`** — ~400MB ELECTRA backbone, stronger on harder queries, ~200ms per pair. Reach for this when MiniLM is not enough and you have GPU available.
- **Cohere Rerank** — proprietary API (Cohere's hosted rerank-3, currently top-of-leaderboard on most rerank benchmarks). Per-call cost, network latency, but the highest quality available off-the-shelf. The right pick when self-hosted MiniLM is the bottleneck on retrieval quality.

The project uses `ms-marco-MiniLM-L-6-v2` because it is small enough to download in the install step, runs on any laptop without a GPU, and produces a visible quality lift over no-rerank on the eval set. Production teams typically start there and migrate to a larger model or to Cohere's API only after measuring that the MiniLM rerank is the bottleneck — which is rare, because the rerank step usually adds 5-15 points of MRR@5 over no-rerank regardless of which cross-encoder you use.

### Cross-encoder model comparison

| Model                                 | Size   | Latency / pair (CPU) | Notes                                     |
|---------------------------------------|--------|----------------------|-------------------------------------------|
| `ms-marco-MiniLM-L-6-v2`              | ~80MB  | ~50ms                | The standard default; used in the project |
| `ms-marco-MiniLM-L-12-v2`             | ~120MB | ~120ms               | Slightly better, rarely worth the latency |
| `ms-marco-electra-base`               | ~400MB | ~200ms               | Reach when MiniLM caps your eval          |
| Cohere `rerank-3`                     | hosted | network-bound        | Highest quality, paid per call            |
| `bge-reranker-base`                   | ~280MB | ~150ms               | BAAI open-source competitor, strong       |
| `bge-reranker-large`                  | ~1.1GB | ~600ms               | GPU recommended                           |

### What rerank cannot fix

A reranker only reorders chunks the recall stage produced. If the relevant chunk is not in the recall stage's top K, the reranker cannot retrieve it — there is nothing to reorder. This is the structural reason recall@K (on the recall stage's output) is the upper bound on the entire pipeline's end-to-end recall: a perfect reranker on a recall stage with recall@20 of 0.6 produces an end-to-end recall@5 of at most 0.6. Improvement past that ceiling requires improving the recall stage, not the reranker.

In practice this means: when end-to-end recall is the bottleneck, invest in retrieval (better embeddings, BM25, query rewriting); when end-to-end MRR is the bottleneck and recall is already high, invest in reranking (bigger cross-encoder, Cohere Rerank). Knowing which one your system is bottlenecked on is the entire purpose of the per-stage eval breakdown described in Section 6.

---

## 6. Measuring "Better": Recall, MRR, NDCG

A hybrid pipeline with reranking is more code, more latency, more dependencies, and more operational surface area than basic RAG. The only question that matters is: does it produce measurably better retrieval on your queries? Without a labeled eval set and the metrics to read it with, that question has no answer, and adding complexity without measurement is theatre. This section covers the three metrics that read out the answer and how to read each one without misinterpreting it.

### Recall@k

**Recall@k** is the fraction of the relevant documents for a query that appear anywhere in the top K retrieved results. Range `[0, 1]`. A recall@5 of 1.0 means every relevant chunk was in the top 5; a recall@5 of 0.0 means none of them were.

```
recall@k = |relevant ∩ retrieved_top_k| / |relevant|
```

Recall@k is the **primary metric for the recall stage**. The job of the recall stage is to make sure the relevant chunks are in the top K; the precision stage can only rerank what the recall stage produced. If recall@20 is 0.6, a perfect reranker on the recall stage's output can never exceed an end-to-end recall of 0.6, because 40% of the relevant chunks were never seen.

Recall@k does not care about ordering within the top K. The chunk at position 1 and the chunk at position 20 both count equally. This is appropriate for the recall stage (the reranker handles ordering) and inappropriate as a stand-alone end-to-end metric (the user does see the ordering, and a relevant chunk at position 20 is functionally invisible to a 3-chunk-context LLM call).

### MRR@k

**MRR@k** (Mean Reciprocal Rank at K) is the average, over queries, of `1 / rank_of_first_relevant`, capped at K and 0 if no relevant chunk is in the top K.

```
RR_q = 1 / rank_of_first_relevant_in_top_k  (or 0 if not in top k)
MRR@k = mean(RR_q)  over all queries
```

A query whose first relevant chunk is at position 1 contributes 1.0; position 2 contributes 0.5; position 5 contributes 0.2; position 20 contributes 0.05.

MRR@k is the **primary metric for the precision stage and for end-to-end systems where the user only sees the top few**. It is sensitive to where the first relevant hit lands, which is what matters when the LLM is reading the top 3 chunks: a system that puts the relevant chunk at position 1 produces noticeably better answers than one that puts it at position 3, and MRR@k captures that difference where recall@k does not.

MRR@k has a limitation: it only counts the *first* relevant chunk. A query with three relevant chunks gets the same MRR whether the system retrieved one of them at position 1 or all three at positions 1-3. NDCG@k addresses that.

### NDCG@k

**NDCG@k** (Normalized Discounted Cumulative Gain at K) generalizes MRR in two ways: it accounts for *graded* relevance (a perfect chunk is worth more than a partially relevant chunk), and it rewards the correct ordering of *multiple* relevant chunks (a query with two relevant chunks ranked 1-2 scores higher than the same query ranked 1-3).

```
DCG@k = Σ (rel_i / log2(i + 1))  for i in 1..k
NDCG@k = DCG@k / IDCG@k          where IDCG is the DCG of the optimal ranking
```

Each retrieved chunk contributes its relevance grade discounted by log of its position, and the result is normalized against the maximum possible DCG (which is the DCG if the chunks were ordered optimally by relevance). Range `[0, 1]`. NDCG@k of 1.0 means the top K is in the optimal order; lower values mean the ordering is degraded.

Use NDCG@k when relevance is not binary — when the corpus has chunks that are "perfectly on-topic", "tangentially useful", and "irrelevant", and you want the metric to reward the system for putting the perfect chunks above the tangential ones. For the small binary-labeled eval in this module's project, MRR is the simpler and equally informative choice; NDCG comes into its own on larger evals with graded relevance labels.

A worked NDCG example. Suppose a query has three relevant chunks at grades 3, 2, 1, and the retriever returns them at positions 2, 1, 3 respectively (so position 1 has the grade-2 chunk, position 2 has the grade-3 chunk, position 3 has the grade-1 chunk). The DCG@3 is `2/log2(2) + 3/log2(3) + 1/log2(4)` = `2.0 + 1.893 + 0.5` = `4.393`. The ideal DCG (grades 3, 2, 1 at positions 1, 2, 3) is `3/log2(2) + 2/log2(3) + 1/log2(4)` = `3.0 + 1.262 + 0.5` = `4.762`. NDCG@3 = `4.393 / 4.762` = `0.923`. The same retrieved chunks in the optimal order would have scored 1.0; this ranking is penalized for putting the grade-2 chunk above the grade-3 chunk.

A standard relevance-grading scheme for NDCG:

- **3 — Perfect**: directly answers the query in full.
- **2 — Highly relevant**: contains the answer or most of it, possibly mixed with adjacent content.
- **1 — Marginally relevant**: discusses the topic but does not answer the specific question.
- **0 — Irrelevant**: does not address the query.

This four-level scheme is what MS MARCO uses and what most graded-relevance corpora adopt. Labeling at this granularity is roughly 2x the effort of binary labeling and produces a measurably more discriminating eval — the difference between a system that ranks 3-3-2 vs one that ranks 3-2-3 is invisible to recall@3 and MRR@3 but visible in NDCG@3.

### Worked example

| Query | Relevant chunks | Top-3 retrieved | recall@3 | MRR@3 |
|-------|-----------------|-----------------|----------|-------|
| Q1    | {A, B}          | [A, X, B]       | 1.0      | 1.0   |
| Q2    | {C}             | [X, C, Y]       | 1.0      | 0.5   |
| Q3    | {D}             | [X, Y, Z]       | 0.0      | 0.0   |

Average recall@3: `(1.0 + 1.0 + 0.0) / 3 = 0.667`. Average MRR@3: `(1.0 + 0.5 + 0.0) / 3 = 0.500`.

Q1 and Q2 both have recall@3 = 1.0 (the relevant chunks are in the top 3), but their MRR differs because Q1's relevant chunk is at position 1 and Q2's is at position 2. The recall metric cannot distinguish them; the MRR metric can. Q3 has no relevant chunk in the top 3 — both metrics correctly score it 0.

### Cross-link to the eval harness

The CLI in this module's project includes a 15-query labeled eval set and a `--eval` flag that runs the full hybrid pipeline against it and prints recall@3, MRR@3, and a per-query breakdown. The pattern is a domain-specific instance of the general eval-harness work in [Module 15 (Evaluation & Testing)](../15-evaluation-testing/): a fixed dataset, a deterministic scoring function, a printed report that an engineer can read in 30 seconds. If you want the general framework for building eval harnesses across any LLM workload, M15 is the canonical reference; this module's `--eval` is what that framework looks like applied to retrieval quality specifically.

### Where to read off the per-query result

A useful eval report does not just print aggregate metrics; it prints, for every query, which chunks were retrieved and which of the labeled relevant chunks were missed. The aggregate numbers tell you *that* something is wrong; the per-query breakdown tells you *what*. A query where the eval expected `["sec_05", "sec_12"]` and the pipeline returned `["sec_03", "sec_07", "sec_11"]` is a retrieval miss — neither expected chunk made it in, and the fix is on the retrieval side. A query where the pipeline returned `["sec_11", "sec_05", "sec_12"]` is a precision problem — both expected chunks are in the top 3 but the wrong chunk got position 1. Different diagnoses, different fixes; the per-query breakdown is what tells them apart.

### What an eval set actually looks like

A useful labeled eval row for retrieval has three fields and nothing else:

```json
{
  "query": "what slow-hash algorithm should I use for passwords",
  "relevant_chunk_ids": ["sec_05", "sec_12"],
  "notes": "tests rare-token via paraphrase; bcrypt and argon2id mentioned in different chunks"
}
```

The `relevant_chunk_ids` list is the ground truth. The eval harness runs the pipeline, gets back a ranked list of chunk IDs, and computes recall@k and MRR@k by checking which of the relevant IDs appear in the top-k and where. The `notes` field is for the human reviewer — it documents *why* the query was added, which makes it possible to maintain the eval set as the corpus changes.

The labels do not need to be exhaustive. A query where the eval lists `["sec_05", "sec_12"]` as relevant does not assert that *only* those two chunks are relevant; it asserts that those two are *known* relevant. The retriever might return a third chunk that is also relevant but was not in the labeled set; that chunk does not get credit, but it also does not get penalized. This is the standard trade-off for cheap labeling: undercount precision (some unlabeled relevant chunks exist) in exchange for fast iteration on the eval set.

### Honest caveats

15 queries is far too few to draw production conclusions from. With 15 queries, the difference between a system with recall@3 = 0.80 and a system with recall@3 = 0.87 is one query going one way or the other, which is well inside the noise floor. The 15-query eval is enough to demonstrate the pattern — to show that hybrid+rerank visibly outperforms dense-alone on the kinds of queries the corpus contains — but not enough to commit to architectural decisions on.

A real eval set is 100-1000 hand-labeled queries plus continuous evaluation against production traffic with periodic re-labeling. The hand-labeled set is the regression suite; the production-traffic eval is the drift detector. Neither replaces the other. Both belong in any RAG system you intend to maintain past the prototype stage.

### Which metric to report when

| Stage                         | Primary metric  | Why                                                                           |
|-------------------------------|-----------------|-------------------------------------------------------------------------------|
| Recall stage (dense, BM25)    | recall@20       | Did the relevant chunk make it into the rerankable candidate set?             |
| Fusion stage (RRF)            | recall@20, MRR@20 | Did fusion preserve recall and start to improve ordering?                   |
| Precision stage (reranker)    | MRR@5, NDCG@5   | Did the reranker put the relevant chunk near the top?                         |
| End-to-end (with LLM answer)  | answer accuracy on labeled QA pairs | Did the user get the right answer? (orthogonal to retrieval metrics) |

The per-stage breakdown is what lets you diagnose which stage is the bottleneck when end-to-end quality is lower than expected. A pipeline where recall@20 = 0.5 will never have end-to-end accuracy above ~0.5; the diagnosis is "improve the recall stage", not "tune the reranker". A pipeline where recall@20 = 1.0 but MRR@5 = 0.3 has the right chunks but in the wrong order; the diagnosis is "improve the reranker". A pipeline where recall@20 = 1.0, MRR@5 = 0.9, and end-to-end accuracy = 0.5 has good retrieval and bad answering; the diagnosis is "fix prompting, model, or grounding", not retrieval at all.

This per-stage decomposition is the single most useful debugging tool for production RAG. Without it, every quality issue looks like a single black-box problem and the team makes uninformed changes in random places. With it, the same quality issue is decomposed into one of three concrete subsystems with concrete fixes. The eval harness in this module's project produces the per-stage breakdown by design.

---

## 7. The Wider Toolbox

Two-stage hybrid retrieval is the highest-leverage improvement over basic RAG and the one this module focuses on. It is not the only one. The techniques below are individually narrower in scope, but each one fixes a specific failure mode the two-stage pipeline does not address. Reach for them when you have measured that the residual failures fall into one of these categories.

**HyDE (Hypothetical Document Embeddings)** — proposed by Gao et al. (2022). Instead of embedding the query, ask the LLM to generate a short hypothetical answer to the query, and embed *that* instead. The hypothetical answer is the same length and shape as the documents in the corpus, so the embedding lands in the right neighborhood of the vector space even when the original query was too short or too question-shaped to embed well. Fixes the **query-document length mismatch**: short questions ("what is X?") embed poorly compared to paragraph-length passages because the embedding model was trained on passages, not on questions. Costs one extra LLM call per retrieval, so use it where retrieval quality is the bottleneck and latency budget allows.

**Parent-document retrieval** — index small chunks (200-400 tokens) for precision, but when a small chunk is retrieved, return the larger parent chunk (1500-3000 tokens) that contains it. The retriever finds the precise sentence; the LLM gets the surrounding context. Fixes the **"tiny chunks fragment context, large chunks hurt retrieval" tension** that every chunking strategy walks into: small chunks give precise retrieval but cut off the surrounding paragraph the LLM needs; large chunks preserve context but dilute the embedding so much that retrieval misses the right one. Parent-document retrieval gives you both axes by indexing on one granularity and serving on another.

**Multi-query rewriting** — ask the LLM to generate 3-5 paraphrases of the user's query, retrieve against each, and union the results before fusion and reranking. Fixes **one-shot query phrasing brittleness**: a user's first phrasing of a question might be the wrong phrasing to match the corpus, and the user has no way to know which phrasing would have worked. Generating multiple phrasings is the system doing the rephrasing the user would otherwise have to do by hand. Costs one LLM call upfront and N times the retrieval work; the retrieval cost is usually trivial.

**Self-querying / metadata filtering** — parse the user query into a *semantic part* and a *filter expression*. `"papers about RAG from 2023"` becomes `semantic: "RAG"`, `filter: year=2023`. The vector search runs on the semantic part; the filter pre- or post-filters by document metadata. Fixes the **"the embedding doesn't know what 'recent' means"** failure: temporal, numeric, and categorical constraints are structural, not semantic, and trying to encode them into the embedding is a losing game. Run them as filters instead. Requires a small LLM call to do the parsing and a metadata-indexed vector store; LangChain's `SelfQueryRetriever` and most production vector stores have this as a first-class feature.

**Contextual compression** — after retrieval, pass each chunk through a small LLM (or a dedicated extractor model) that returns only the sentences in the chunk that are relevant to the query. The cross-encoder filters by *chunk*; contextual compression filters by *sentence*. Fixes the **lost-in-the-middle problem at sub-chunk granularity**: a 500-token chunk where the relevant sentence is in the middle of a long paragraph forces the LLM to attend across irrelevant prose to find it. Costs an extra inference per chunk (usually a cheap model — Haiku, GPT-4o-mini, or a small local model — is sufficient), and the inferences parallelize, so the wall-clock cost is one round-trip per pipeline rather than one per chunk.

**Maximal Marginal Relevance (MMR)** — a diversity-aware reranking algorithm that penalizes chunks too similar to chunks already selected. Fixes the **redundant-chunk failure mode**: the top 5 dense-retrieval results are often the same paragraph quoted in 5 places, providing no additional information. MMR trades a small amount of per-chunk relevance for diversity, which often improves answer quality on questions that need multiple angles. Cheap (a few extra dot products) and worth trying when your retrieved chunks look duplicative.

**Query routing / classifier-first retrieval** — train a small classifier (or use an LLM) to decide which retriever or which index to consult for a given query. A query about Python error messages routes to the BM25-on-codebase index; a query about high-level architecture routes to the dense-on-design-docs index. Fixes the **one-corpus-for-all-queries failure mode** in systems with multiple distinct knowledge sources. The classification step is a small fraction of overall latency and can dramatically improve quality when the underlying indexes have different strengths.

### When each technique earns its complexity

| Technique                | Failure mode it addresses                        | Added latency       | Added cost            |
|--------------------------|--------------------------------------------------|---------------------|-----------------------|
| HyDE                     | Short queries embed poorly                       | +1 LLM call         | +1 LLM call per query |
| Parent-document          | Chunk size tradeoff (precision vs context)       | Negligible          | Slightly larger prompt|
| Multi-query rewriting    | One-shot query phrasing brittleness              | +1 LLM call, N retrievals | +1 LLM call     |
| Self-querying            | Structural filters disguised as semantic search  | +1 LLM call         | +1 LLM call per query |
| Contextual compression   | Sub-chunk lost-in-the-middle                     | +1 round-trip       | +N small inferences   |

The pattern: each technique is one LLM call's worth of cost and one failure mode's worth of fix. Adopt the ones whose failure modes your eval shows; skip the ones whose failure modes you do not have. The cost of running everything regardless of need is paid on every query, not just the queries that benefit, which is why "kitchen sink" RAG pipelines are routinely slower and worse than focused pipelines that fix the specific failure modes that matter.

### Techniques deliberately not in this module

A short list of advanced techniques worth knowing exist but not covered here:

- **Dense retrieval with learned sparse representations** (SPLADE, uniCOIL) — sparse vectors learned from a transformer, combining lexical-match precision with semantic-match recall in a single representation. Strong on benchmarks, more complex to operate than dense+BM25 hybrid, and the quality lift over a tuned hybrid is small enough that it rarely justifies the operational cost for most teams.
- **Late-interaction models** (ColBERT, ColBERTv2) — produce token-level embeddings instead of sentence-level, and compute relevance via maxsim over token pairs. Higher quality than single-vector dense for the same cost class, with a higher storage footprint (one vector per token instead of one per chunk).
- **Graph-based retrieval** (GraphRAG, knowledge-graph augmented RAG) — extract entities and relationships from the corpus into a graph, retrieve via graph traversal rather than (or alongside) vector similarity. Strong for queries that require multi-hop reasoning across linked entities; complex to build and maintain, narrow set of queries that benefit.
- **Agentic RAG** — let an LLM decide what to retrieve, when to retrieve again, and when it has enough information to answer. This pattern is the [Module 11 (Building AI Agents)](../11-building-ai-agents/) loop applied to retrieval; this module covers the retrieval primitives, M11 covers the orchestration.

### Forward link to Module 20

Operating any of these techniques in production — with latency budgets, fallbacks when the rerank model is down, A/B testing of retrieval strategies behind a feature flag, blue/green deploys of new embedding models, monitoring of retrieval quality drift over time — is the deployment-patterns territory. Module 20 (Deployment Patterns) covers the operational scaffolding. This module covers the retrieval algorithms; the deployment module covers how to ship them safely.

---

## 8. When Not to Add Complexity

Every section above this one argued for more retrieval machinery. This section argues the opposite. Hybrid+rerank pipelines add latency (50-500ms per request depending on the cross-encoder), add operational surface area (a second index to maintain, a model to download and version, a fusion function to test), and add cost (the cross-encoder inference is small but non-zero). The improvement is real on the queries that need it; it is invisible on the queries that don't. Adding complexity to a system that does not benefit from it is the most expensive non-improvement in production engineering.

### Concrete decision rules

- **If basic RAG on your eval set hits >90% recall@3, stop.** The headroom for improvement is small enough that the latency and operational costs probably exceed the quality gain. Spend the engineering time on other parts of the system — better prompts, better chunking, better failure handling, faster end-to-end response. Hybrid+rerank is for systems where basic RAG is *visibly failing*, not for systems where it is already working.

- **If your corpus is < ~1k chunks, BM25 alone often beats dense+rerank.** IDF needs a corpus large enough to distinguish informative from uninformative terms, but on small corpora the BM25 signal is concentrated enough that adding dense retrieval brings in mostly noise and the cross-encoder is reordering a too-small candidate set. Try BM25-only with `bm25s` or `rank-bm25`, measure, and only add dense if BM25 fails on a measurable fraction of your eval queries.

- **If your users always type the same canonical phrasing, dense alone is fine.** Internal tools, structured search forms, anything where the query shape is constrained — paraphrase robustness is not load-bearing because there is no paraphrase variation. The two-stage pipeline's main quality gains are on the long tail of unexpected phrasings, and a system without that long tail does not need them.

- **If your queries are all rare-named-entity lookups, BM25 alone often beats dense.** Engineering-docs search where every query is an error code or an API name, codebase search by function name, log search — these are domains where lexical matching dominates and dense retrieval adds little. The hybrid pipeline does not hurt, but it costs more than it earns.

- **If you don't have an eval set, build *that* first.** Adding complexity without a way to measure it is theatre. A 15-50 query labeled set (questions paired with the chunk IDs that should answer them) is a few hours of work and the only thing that lets you tell whether a pipeline change is an improvement, a regression, or noise. Every other decision in this module — when to add reranking, which fusion to use, which cross-encoder model — depends on having an eval set to read the answer with. Build the meter before you build the engine.

### The YAGNI heuristic for retrieval

The general shape: most production RAG systems should start with the simplest thing that could possibly work (basic dense retrieval with reasonable chunking, k=5, and good prompts), build the eval set first, run it, look at the failures, and only add the specific complexity that addresses the specific failure mode the eval surfaced. Hybrid retrieval addresses recall failures on lexical queries. Reranking addresses precision failures on the ordering of retrieved chunks. HyDE addresses query-document length mismatches. Each technique has a failure mode it fixes and a cost it imposes; adopt it when the failure mode is present in your eval, not because the technique exists.

The opposite anti-pattern is the most common architectural mistake in production RAG: implementing the full hybrid+rerank+HyDE+multi-query pipeline before measuring whether basic RAG was actually broken. Teams that do this end up with a slow, expensive, hard-to-debug system whose quality is indistinguishable from what they would have had with basic RAG and better prompts. The complexity is real; the improvement is not.

### Where to spend the engineering time instead

If basic RAG works on your queries, the marginal hour is better spent on:

- **Chunking that respects document structure** — section-based chunking, semantic chunking, or hierarchical chunking will often beat fixed-size chunking by more than rerank would gain.
- **Prompt engineering for the answering LLM** — explicit grounding instructions, refusal handling, source-citation format, output-shape constraints. Most "the answer is wrong" failures in basic RAG are answering-side, not retrieval-side.
- **Embedding model upgrades** — moving from a small general embedder to a domain-specific or larger embedder is often a bigger lift than adding rerank.
- **Failure-mode debugging** — print the retrieved chunks for every failing query in your eval. The category of failure (retrieval miss, retrieval irrelevance, answering error) is what tells you which improvement to invest in next.
- **Latency optimization** — for many production systems, the LLM call dominates end-to-end latency and the retrieval improvements are invisible to users. Streaming the LLM output (Module 05), caching frequent queries (Module 17), and routing simple queries to faster models will often deliver more user-perceived improvement than a hybrid retriever would.
- **Observability** — instrumenting retrieval with the trace pattern from [Module 18 (Observability & Monitoring)](../18-observability-monitoring/) lets you see which chunks were retrieved for every production query, which is the single most useful debugging signal for any RAG system in operation. A retrieval improvement you cannot measure in production is one you cannot validate, and a system without retrieval traces is opaque exactly where the failures happen.

The two-stage hybrid pipeline is in this module because it is the single highest-leverage improvement when you need it. The point of this last section is that "when you need it" is a measurable condition, not a default assumption. Build the eval first, measure where you actually are, and add only the complexity that the eval says will help. That is the practitioner's discipline that separates a working RAG system from a complicated one.

### The cost of premature optimization

A specific anti-pattern worth naming: the team that builds a full hybrid+rerank+HyDE+multi-query pipeline as their first iteration, then spends weeks debugging why retrieval quality is *worse* than what they had with basic RAG. The most common cause is one of the components being mis-configured (a stale embedding index, a tokenization mismatch between BM25 and the dense retriever, an `α` weight that was optimal on the developer's test queries and pessimal on the production distribution), and the team has no way to isolate the issue because they cannot run the components separately. The eval set is the diagnostic tool that would let them isolate it; they did not build one because they were focused on building the pipeline.

The discipline that avoids this trap is: ship basic RAG first, build the eval set against basic RAG, then add hybrid retrieval and measure the lift, then add rerank and measure the lift again, then add HyDE or multi-query *only if the eval shows the underlying failure mode is present*. Each addition is a measurable improvement against a known baseline; each addition is reversible if it does not help. This is the same incremental-delivery discipline that produces good software in every other domain; it applies with full force to retrieval pipelines, which are software.

### Quick reference: when to add what

| Symptom in your eval                                        | First thing to try            |
|-------------------------------------------------------------|-------------------------------|
| Recall@20 < 0.7 on rare-token queries                       | Add BM25 + RRF (Sections 3-4) |
| Recall@20 high, MRR@5 low                                   | Add cross-encoder rerank (Section 5) |
| Recall@20 high, MRR@5 high, but answer accuracy low         | Fix prompts and grounding, not retrieval |
| Queries are short and embed poorly                          | HyDE (Section 7)              |
| Chunks too small for context but too large for retrieval    | Parent-document retrieval     |
| Same question phrased many ways                             | Multi-query rewriting         |
| Queries mix semantic + structural ("from 2023", "by author X") | Self-querying with metadata filters |
| Relevant sentence buried inside otherwise-irrelevant chunks | Contextual compression        |

The table is not exhaustive and the categories overlap. The underlying discipline is the same regardless of which row applies: measure, identify the dominant failure mode, apply the targeted fix, measure again. Each cycle takes hours; the gains are visible in the eval; the system improves on the metrics that matter without bloating with techniques that do not help. That is what "production-grade RAG" actually means in practice — not a maximally complex pipeline, but a pipeline whose every component earns its place in the eval.

### The module in one paragraph

Basic RAG (Module 07) is a single dense retriever followed by an LLM call. It works on the easy queries — paraphrased, semantically clean, well-aligned with the embedding model's training distribution — and fails on rare-token queries, lost-in-the-middle prompts, and queries that need both lexical and semantic matching. The fix is a two-stage pipeline: a recall stage that combines dense embeddings and BM25 via Reciprocal Rank Fusion (giving you both similarity dimensions for the price of two cheap retrievers), and a precision stage that runs a cross-encoder over the fused top-20 candidates and returns the top 3-5 in their query-relevance order. The whole pipeline is ~300 lines of Python; the quality lift over basic RAG is large enough to measure on a 15-query eval and unmistakable on a 100-query eval. The discipline that makes this work in practice is: build the eval first, add components incrementally, measure each addition, keep what helps, remove what doesn't. The project that follows is the concrete implementation. Read the project README next to see how the pieces fit together in code.
