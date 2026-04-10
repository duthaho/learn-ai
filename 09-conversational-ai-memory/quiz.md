# Module 09: Conversational AI & Memory — Quiz

Test your understanding. Try answering before revealing the answer.

---

### Q1: Why are LLMs stateless, and what does this mean for building conversational applications?

<details>
<summary>Answer</summary>
LLMs are stateless because each API call is completely independent — the model retains no memory of previous calls. When ChatGPT appears to "remember" your conversation, the application is sending the entire conversation history (the messages list) with every request. For developers, this means you are responsible for managing what the LLM "remembers." You must maintain the messages list, decide what to keep or discard, and handle the growing token cost. Without active management, conversations will eventually overflow the context window or become prohibitively expensive.
</details>

---

### Q2: How do you count tokens in a messages list, and why does this matter for context management?

<details>
<summary>Answer</summary>
Use the tiktoken library: get the encoding for your model (or fall back to cl100k_base), then encode each message's content and sum the token counts, adding ~4 tokens per message for role/content markers and ~2 tokens for assistant reply priming. This matters because the context window is measured in tokens, not characters or words. You need accurate token counts to know when you're approaching the limit, to decide when to trigger truncation or summarization, and to budget your context window across system prompt, history, and output reserve.
</details>

---

### Q3: What is the context window budget formula, and how do you allocate it across system prompt, history, and output?

<details>
<summary>Answer</summary>
The formula is: available_for_history = context_window - system_prompt_tokens - output_reserve. System prompt tokens are fixed (determined by your persona and injected memories). Output reserve is the space you set aside for the model's response (typically 512-1024 tokens). Everything remaining is available for conversation history. When history tokens exceed this budget, you must apply a memory management strategy (truncation, summarization, or both). Budget conservatively even with large context windows because cost and latency scale with token count.
</details>

---

### Q4: Describe sliding window truncation. What are its trade-offs compared to summarization?

<details>
<summary>Answer</summary>
Sliding window truncation drops the oldest message pairs (user + assistant) from the conversation while always preserving the system prompt at index 0. It continues removing pairs until the total tokens fit within the budget. Pros: simple to implement, no extra API calls, fast, predictable. Cons: loses early context completely (the user might reference something that was dropped), no gradual degradation (information is either fully present or completely gone). Compared to summarization: truncation is cheaper (no LLM call) but less intelligent — summarization preserves key information in compressed form while truncation discards it entirely.
</details>

---

### Q5: How does conversation summarization work, and what's the trade-off between rolling and threshold approaches?

<details>
<summary>Answer</summary>
Summarization sends older messages to the LLM with a prompt asking for a concise summary preserving key topics, user facts, decisions, and open questions. The summary replaces the old messages, achieving ~80-90% token compression. Rolling summaries run after every N turns, incorporating the previous summary — good for very long conversations but requires frequent API calls. Threshold summaries run only when approaching the context limit — fewer API calls but may try to summarize a large batch at once. Rolling gives smoother memory management; threshold is more cost-efficient for moderate conversations. Both cost one extra LLM call per summarization.
</details>

---

### Q6: What makes an effective system prompt persona? What are common anti-patterns?

<details>
<summary>Answer</summary>
An effective persona has three components: identity (who — name, role, expertise), style (how — tone, verbosity, formality), and boundaries (what not — scope limits, safety guardrails). Common anti-patterns: over-constraining with 50+ rules leads to contradictions and unpredictable behavior; under-specifying with just "you are helpful" gives no useful guidance; conflicting instructions like "be concise" plus "explain in detail" confuse the model; leaking implementation details like "you are an LLM" breaks the experience. The system prompt is permanent memory — it's never truncated — so invest in getting it right.
</details>

---

### Q7: What is the difference between episodic and semantic long-term memory?

<details>
<summary>Answer</summary>
Semantic memory stores user facts and attributes — things the system "knows" about the user: name, preferences, interests, background (e.g., "User prefers Python over JavaScript"). Episodic memory stores conversation summaries — records of "what happened" in past sessions (e.g., "2024-03-15: Discussed async patterns in Python"). Both are extracted from conversations using the LLM and stored persistently (e.g., in a JSON file). At session start, both types are loaded and injected into the system prompt so the assistant can reference past knowledge without the user repeating themselves.
</details>

---

### Q8: How does conversation memory relate to RAG (Module 07) and tool use (Module 06)?

<details>
<summary>Answer</summary>
RAG is conceptually external memory — it stores information in a document corpus, retrieves relevant pieces via vector search, and injects them into the prompt. Long-term memory follows the same pattern but for user-specific data: store facts, retrieve at session start, inject into the system prompt. The difference is RAG retrieves from a static knowledge base while memory retrieves from a dynamic per-user store. Tool use needs conversation memory because multi-turn tool interactions generate many messages rapidly (tool calls + results), and without context management a few tool calls can consume 1000+ tokens. The truncation and summarization strategies from this module keep tool-heavy conversations manageable.
</details>
