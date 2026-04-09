# Module 01: How LLMs Work — Quiz

Test your understanding. Try answering before revealing the answer.

---

### Q1: What is a token, and why don't LLMs operate on words?

<details>
<summary>Answer</summary>

A token is a subword unit produced by a tokenizer (like BPE). LLMs use tokens instead of words because: (1) a fixed vocabulary of 32K-200K tokens can represent any text, including rare words, code, and multilingual input, by splitting into known pieces; (2) words alone would require an impossibly large vocabulary; (3) subword tokenization handles unseen words gracefully.
</details>

---

### Q2: Why is autoregressive generation sequential — and what does that mean for latency?

<details>
<summary>Answer</summary>

Each token's probability depends on all previous tokens. Token 5 can't be generated before token 4. This means latency = time-to-first-token + (output_tokens x time_per_token). Longer outputs take proportionally longer. This is why streaming exists — you deliver tokens as they're generated instead of waiting for the complete response.
</details>

---

### Q3: Verbose JSON costs 800 tokens. Compact format costs 200 tokens. At $3/M input tokens and 1M requests/month, what are the annual savings?

<details>
<summary>Answer</summary>

Verbose: 800 x 1M x $3/1M = $2,400/month. Compact: 200 x 1M x $3/1M = $600/month. Savings: $1,800/month = $21,600/year. Token efficiency matters at scale.
</details>

---

### Q4: What is the KV cache and why does it matter?

<details>
<summary>Answer</summary>

During generation, the model would normally recompute attention over all previous tokens for each new token — O(n^2) work. The KV cache stores Key and Value projections from previous tokens so only the new token's Query needs computing — reducing to O(n). The tradeoff: KV cache memory grows linearly with sequence length (40GB+ for long contexts on large models).
</details>

---

### Q5: When should you use temperature 0.0 vs 1.0?

<details>
<summary>Answer</summary>

T=0.0 (greedy/deterministic): structured output, classification, extraction, code generation — anything needing consistency. T=1.0 (full sampling): creative writing, brainstorming, generating diverse options. T=0.3-0.7: general Q&A, summarization — balance of coherence and variety.
</details>

---

### Q6: Your API returns truncated JSON. What happened and how do you fix it?

<details>
<summary>Answer</summary>

The max_tokens limit was reached before generation finished — stop_reason is "max_tokens" not "end_turn". Fix: always check stop_reason. For structured output, use tool_use/function calling for schema guarantees. Increase max_tokens, or use retry with continuation. Always try/catch JSON parsing.
</details>

---

### Q7: What are the three training stages of an LLM?

<details>
<summary>Answer</summary>

1. Pre-training: next-token prediction on trillions of tokens. Creates a base model that completes text but doesn't follow instructions. 2. Supervised Fine-Tuning (SFT): training on (instruction, response) pairs. Model learns to follow directions. 3. RLHF/DPO (alignment): optimizing on human preference data. Model becomes helpful, harmless, honest.
</details>

---

### Q8: Context is 200K tokens. Your document is 150K tokens and you set max_tokens to 60K. Why does the call fail?

<details>
<summary>Answer</summary>

Context window must fit both input AND output: 150K + 60K = 210K > 200K. Always pre-flight check: input_tokens + max_tokens <= context_window. Fix: reduce input (use RAG for relevant chunks only) or reduce max_tokens.
</details>
