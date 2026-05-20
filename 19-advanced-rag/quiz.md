# Module 19 Quiz: Advanced RAG

Self-assessment questions for Module 19. Test your understanding before revealing each answer.

---

### Q1: Why does dense-only retrieval miss exact-keyword queries that BM25 catches?

<details>
<summary>Answer</summary>

Sentence embeddings collapse synonyms but also smooth out rare named entities. As covered in Section 3, the embedding for "bcrypt" lives in the same neighborhood as "Argon2," "scrypt," and "password hashing" because all four co-occur constantly in the training corpus. So the query "what is bcrypt" retrieves whatever chunk most centrally discusses *hashing* — not necessarily the chunk that actually names bcrypt. BM25 scores by exact-term overlap weighted by IDF, and "bcrypt" has very high IDF because it appears in few documents. The bcrypt-specific chunk lights up immediately. This is why hybrid wins on identifier-heavy queries (API names, error codes, library names, SKUs) — exactly the cases where users type the term they already know.

</details>

---

### Q2: What does the "k" constant in RRF control, and why is the published default 60?

<details>
<summary>Answer</summary>

`k` is a smoothing constant that controls how much weight a high-rank position gets versus a low-rank one. As Section 4 walks through, with k=60 the fused contribution from rank 1 (1/61 ≈ 0.0164) versus rank 10 (1/70 ≈ 0.0143) is a small but real gap, and rank 20 (1/80 ≈ 0.0125) still contributes meaningfully. Lower k makes the top ranks dominate the fusion; higher k flattens the curve so deeper ranks matter more. The value 60 came from Cormack, Clarke, and Buettcher's 2009 TREC experiments and survived because no one has found a corpus where retuning it materially helps. Treat it as a constant, not a hyperparameter — spend your tuning budget elsewhere.

</details>

---

### Q3: Bi-encoder vs cross-encoder — what's the architectural difference and the cost consequence?

<details>
<summary>Answer</summary>

A bi-encoder encodes the query and each document independently into vectors and compares them with a dot product. Doc vectors are precomputed once at index time; the query is a single forward pass at query time; scoring is just a similarity lookup. A cross-encoder concatenates `(query, doc)` and runs the pair through the transformer together, producing one score via joint attention across both. Crucially, doc embeddings cannot be precomputed because they depend on the query. As Section 5 spells out, the consequence is brutal: bi-encoders scale to millions of docs cheaply, while cross-encoders scale only to a top-K candidate set of 10-100. That's why the cross-encoder always runs second, on the shortlist the bi-encoder already produced.

</details>

---

### Q4: Why is reranking always second, never first?

<details>
<summary>Answer</summary>

Cross-encoder reranking costs roughly 50ms per (query, doc) pair on CPU. As Section 2 makes concrete, running it on a 33-chunk corpus costs about 1.6 seconds; running it on a million docs costs 14 hours per query. The whole point of the two-stage pattern is that the recall stage uses cheap methods (BM25 + bi-encoder) to cut the candidate set from 1M to ~20, so the precision stage's per-pair expense is bounded by a small constant. Reversing the order would mean paying the expensive cost across the entire corpus on every query, which defeats the architecture entirely. Recall is cheap and wide; precision is expensive and narrow. That ordering is non-negotiable.

</details>

---

### Q5: Why is `recall@k` the right primary metric for the retrieval stage (rather than precision)?

<details>
<summary>Answer</summary>

Recall@k captures whether the relevant docs are *available* in the top-K candidate set. As Section 6 lays out, the rerank stage can only reorder what the recall stage hands it — if recall@20 is zero, the best cross-encoder in the world cannot recover a result. Precision at the recall stage is almost irrelevant because rerank will discard the irrelevant chunks downstream anyway; the recall stage's job is just to not lose the gold doc. At the precision stage the metric flips: MRR is right because the user only sees the top one or two results, and rank position is what matters. Picking the metric that matches the stage's actual job is the whole point.

</details>

---

### Q6: Your `--eval` shows hybrid beats dense on `keyword` queries but ties on `semantic`. What does that tell you?

<details>
<summary>Answer</summary>

It tells you your dense retriever is already strong on paraphrases — which is exactly what semantic queries test — so adding BM25 contributes no new signal on that slice. The hybrid gain is concentrated entirely on keyword queries, where dense was missing exact-term matches that BM25 nails via IDF. As Section 8 frames it, the right action is not to drop hybrid (it's still a clean win on the keyword slice for zero quality cost), but to ask whether your real-world traffic mix justifies the extra index complexity. If 90% of production queries are semantic paraphrases, dense-only may be the right call. If users type identifiers, error codes, or library names, hybrid earns its keep. Let traffic shape, not benchmark averages, decide.

</details>

---

### Q7: When would you *not* add hybrid + rerank to a basic RAG pipeline?

<details>
<summary>Answer</summary>

Three concrete situations from Section 8: (a) your basic-RAG eval set already hits >90% recall@3 — the headroom isn't there, and added latency buys nothing measurable; (b) your corpus is small (<1k chunks) and BM25 alone has enough IDF signal to find the right doc reliably; (c) your users phrase queries canonically — internal tools, structured forms, fixed vocabulary — where paraphrase robustness simply isn't load-bearing. There's also a fourth case worth naming: if you don't have an eval set at all, add complexity later and build the eval first. The correct test is always "does my eval set show a measurable lift?" Adding hybrid + rerank without that measurement is cargo-culting, and you'll pay the latency for nothing.

</details>

---

### Q8: Name three advanced-RAG techniques this project intentionally omits, and what each fixes.

<details>
<summary>Answer</summary>

From Section 7's wider toolbox: **HyDE** fixes the query-doc length mismatch — short queries don't embed well against long passages — by having an LLM generate a hypothetical answer first, then embedding *that* for retrieval. **Parent-document retrieval** fixes the chunking tension where small chunks fragment context and large chunks hurt retrieval precision; you index small chunks but return the larger parent on a hit. **Multi-query rewriting** fixes the brittleness of one-shot query phrasing by having an LLM generate several paraphrases and retrieving on each. Honorable mentions: self-querying parses structured filters (dates, categories) out of a natural-language query, and contextual compression LLM-summarizes each chunk down to just the query-relevant sentences to dodge the lost-in-the-middle problem.

</details>
