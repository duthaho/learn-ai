# Module 07: RAG — Quiz

Test your understanding. Try answering before revealing the answer.

---

### Q1: What problem does RAG solve that fine-tuning and long context windows don't?

<details>
<summary>Answer</summary>
RAG solves grounding LLM answers in large, changing knowledge bases with verifiable citations. Fine-tuning bakes knowledge into weights (can't update without retraining, no citations, expensive). Long context windows require entire corpus to fit in one prompt (impractical for large docs, expensive per-request). RAG retrieves only relevant passages at query time, works with any document size, updates instantly when docs change, and provides source citations because you know which chunks were used.
</details>

---

### Q2: What are the two phases of a RAG pipeline and what happens in each?

<details>
<summary>Answer</summary>
1. Indexing phase (offline, done once): load documents, split into chunks, embed each chunk into a vector, store vectors in a vector index (like FAISS). No LLM involved — pure preparation. 2. Query phase (online, per question): embed user's question with same embedding model, search vector index for top-k similar chunks, filter by relevance score, construct prompt with retrieved chunks as context, send to LLM, return answer with citations.
</details>

---

### Q3: Why does chunk size matter — what goes wrong if chunks are too small or too large?

<details>
<summary>Answer</summary>
Too small (< 100 chars): embeddings capture too little meaning, retrieval finds fragments, LLM can't form coherent answers. Too large (1000+ chars): embeddings average out multiple topics (diluted signal), irrelevant text retrieved alongside relevant text, fewer chunks fit in context window. Sweet spot is 200-500 chars where each chunk captures a focused topic with enough context for useful answers.
</details>

---

### Q4: How do you decide how many chunks (top-k) to retrieve?

<details>
<summary>Answer</summary>
Start with k=3-5 as practical default. Adjust based on: chunk size (smaller chunks → higher k), context window budget (larger windows fit more), document diversity (more topics → higher k). Too few (k=1-2) is fragile — one bad match ruins the answer. Too many (k=8-10) introduces noise, wastes tokens, triggers "lost in the middle" problem.
</details>

---

### Q5: What is "lost in the middle" and how does chunk ordering affect generation?

<details>
<summary>Answer</summary>
"Lost in the middle" (Liu et al., 2023) is the finding that LLMs pay most attention to information at the beginning and end of their context, while underusing middle information. Chunk ordering matters: most relevant chunk should go first (gets most attention). Limit total chunks to 3-5 to minimize the middle. If many chunks needed, place most important at both start and end.
</details>

---

### Q6: How should the system handle a query where no retrieved chunks meet the relevance threshold?

<details>
<summary>Answer</summary>
Should NOT force LLM to answer from irrelevant or no context. Instead respond with clear message like "I don't have enough information in my documents to answer that." Better than hallucinating from noise. System prompt should instruct this, and code should detect when no chunks pass threshold and handle explicitly.
</details>

---

### Q7: Why should RAG prompts instruct the LLM to only use provided context?

<details>
<summary>Answer</summary>
Without this instruction, LLM blends retrieved facts with training data. This defeats RAG's purpose: can't verify where answer came from, model might mix outdated training data with current documents, citations become unreliable. The explicit constraint forces grounding — answer must come from retrieved passages, making it verifiable and trustworthy.
</details>

---

### Q8: When would you choose RAG over fine-tuning, and vice versa?

<details>
<summary>Answer</summary>
Choose RAG when: large or frequently changing knowledge base, need source citations, want to work with any LLM without retraining, keep costs low. Choose fine-tuning when: need specific style or persona, domain-specific terminology, particular response format, latency matters (no retrieval step). Can combine: fine-tune for style/format, use RAG for knowledge/citations.
</details>
