# Module 03: Embeddings & Vector Search — Quiz

Test your understanding. Try answering before revealing the answer.

---

### Q1: What are embeddings, and why do they enable semantic search that keyword matching cannot?

<details>
<summary>Answer</summary>

Embeddings are dense numerical vectors (arrays of floats) that represent the meaning of text in a high-dimensional space. Words and sentences with similar meanings end up close together in this space, even if they share no keywords. For example, "forgot my login credentials" and "how do I reset my password" have zero word overlap but high vector similarity. Keyword matching (like SQL LIKE or regex) requires exact or fuzzy string matches, so it misses these semantic connections entirely. Embeddings encode meaning, not surface form, which is what makes semantic search possible.
</details>

---

### Q2: What is the relationship between cosine similarity and dot product for normalized vectors, and why does this matter for FAISS index choice?

<details>
<summary>Answer</summary>

For L2-normalized vectors (vectors with a magnitude of 1.0), cosine similarity and dot product produce identical results. Cosine similarity is defined as dot(A, B) / (||A|| * ||B||), and when both norms equal 1, the denominator is 1, so it reduces to the dot product. This matters because FAISS offers `IndexFlatIP` (inner product / dot product) and `IndexFlatL2` (Euclidean distance). If you normalize your embeddings before indexing, you can use `IndexFlatIP` and get cosine similarity rankings directly, which is both simpler and more intuitive. Without normalization, dot product favors longer vectors regardless of direction, which produces misleading similarity scores.
</details>

---

### Q3: What are the tradeoffs of chunk size — what goes wrong when chunks are too large or too small?

<details>
<summary>Answer</summary>

When chunks are too large, the embedding averages over too many concepts, diluting the meaning. A chunk covering both password hashing and OAuth will not match well against a specific query about either topic — the vector sits somewhere between both meanings. Search precision drops because the chunk contains relevant and irrelevant content mixed together. When chunks are too small (a single sentence), the embedding lacks context and may miss important nuances. A sentence like "This prevents the attack" means nothing without the surrounding explanation. Small chunks also increase the total number of vectors, which raises index size and search cost. The sweet spot depends on your content and query patterns, but 200-500 characters per chunk is a common starting range for technical documents.
</details>

---

### Q4: When would you use an approximate nearest neighbor index (like FAISS IVF or HNSW) instead of a brute-force flat index?

<details>
<summary>Answer</summary>

Brute-force flat indexes (`IndexFlatIP`, `IndexFlatL2`) compare the query against every vector in the index, guaranteeing exact results. This is fine for small datasets (under roughly 100,000 vectors) because the search is fast enough. Beyond that, brute-force becomes too slow — search time scales linearly with dataset size. Approximate indexes (IVF, HNSW, PQ) trade a small amount of recall accuracy for dramatically faster search. IVF partitions vectors into clusters and only searches relevant clusters. HNSW builds a navigable graph for logarithmic search time. Use approximate indexes when you have hundreds of thousands to billions of vectors and can tolerate occasionally missing the absolute best match (recall typically stays above 95% with good tuning).
</details>

---

### Q5: Vector search always returns results, even for completely irrelevant queries. Why is this a problem, and how do you address it?

<details>
<summary>Answer</summary>

FAISS and other vector databases return the k nearest neighbors by definition — they find the closest vectors in the index regardless of whether those vectors are actually relevant. If you search for "chocolate chip cookie recipe" in an authentication document index, you still get 3 results — they are just the least irrelevant chunks. Without a threshold, your application would present these garbage results to the user with confidence. The fix is a similarity threshold: after retrieval, filter out any result below a minimum score (e.g., 0.35 for cosine similarity). The exact threshold depends on your embedding model, content domain, and acceptable precision. You should calibrate it empirically by testing with known-relevant and known-irrelevant queries. Some systems also use a reranker as a second pass to further validate relevance.
</details>

---

### Q6: Your company decides to switch from one embedding model to another. What breaks, and what do you need to do?

<details>
<summary>Answer</summary>

Everything breaks. Different embedding models produce vectors in different spaces — even if two models both output 384-dimensional vectors, the dimensions do not correspond to the same features. A vector from Model A is meaningless when compared to a vector from Model B. You must re-embed your entire document corpus with the new model and rebuild the index from scratch. Queries must also be embedded with the same model used for the index. This means: (1) re-process all documents, which can be expensive for large corpora; (2) update any cached or stored embeddings; (3) recalibrate similarity thresholds, because different models produce different score distributions; (4) regression-test search quality to verify the new model actually performs better. For these reasons, embedding model choice is a high-commitment decision — plan for migration cost before switching.
</details>

---

### Q7: Why is metadata filtering important in a production vector search system?

<details>
<summary>Answer</summary>

Vector similarity alone cannot enforce business logic constraints. Consider a multi-tenant SaaS application: a search for "billing policy" should only return documents belonging to the querying customer, not documents from other tenants that happen to be semantically similar. Metadata filtering lets you attach structured attributes to each vector (tenant ID, document type, date, access level) and restrict the search to vectors matching specific criteria. Without metadata filtering, you would need a separate index per tenant (wasteful) or post-filter results (inefficient — you might retrieve k results and discard most of them). Production vector databases like Pinecone, Weaviate, and Qdrant support pre-filtering, which narrows the candidate set before the similarity search runs, giving you both correctness and performance.
</details>

---

### Q8: Embeddings are often associated with text search, but what other problems can they solve?

<details>
<summary>Answer</summary>

Embeddings work for any data type that can be projected into a meaningful vector space. Common applications beyond text search include: (1) Image similarity — models like CLIP embed images and text into a shared space, enabling "search images with a text description" and reverse image search. (2) Recommendation systems — embed users and items into the same space; recommend items whose vectors are close to the user's vector. (3) Anomaly detection — embed log entries or transactions; points far from any cluster may be anomalies. (4) Deduplication — embed documents or records; pairs with very high similarity are likely duplicates. (5) Classification — embed inputs and compare them to labeled examples (few-shot classification without fine-tuning). (6) Clustering — embed a corpus and run k-means or DBSCAN on the vectors to discover topic groups automatically. The core idea — representing meaning as geometry — applies wherever similarity or relatedness matters.
</details>
