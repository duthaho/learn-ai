# Module 09 — Conversational AI & Memory

Building multi-turn chatbots that remember: managing conversation history, handling context limits, and persisting memory across sessions.

| Detail        | Value                                          |
|---------------|------------------------------------------------|
| Level         | Intermediate                                   |
| Time          | ~3 hours                                       |
| Prerequisites | Module 04 (The AI API Layer), Module 05 (Streaming & Real-Time AI) |

## What you'll build

After reading this module, head to [`project/`](project/) to build a **Memory Chatbot** — an interactive CLI chatbot with configurable personas, sliding window and summarization for context management, and persistent long-term memory that survives across sessions.

---

## Table of Contents

1. [Why Conversation Memory Matters](#1-why-conversation-memory-matters)
2. [The Messages List](#2-the-messages-list)
3. [Context Window Budgeting](#3-context-window-budgeting)
4. [Sliding Window Truncation](#4-sliding-window-truncation)
5. [Conversation Summarization](#5-conversation-summarization)
6. [System Prompts & Persona Design](#6-system-prompts--persona-design)
7. [Long-Term Memory](#7-long-term-memory)
8. [Memory in the AI Stack](#8-memory-in-the-ai-stack)

---

## 1. Why Conversation Memory Matters

Every time you call an LLM API, you start from a blank slate. The model has no idea who you are, what you discussed five minutes ago, or what it said in its last response. LLMs are **stateless** — each API call is a completely independent transaction.

This is a fundamental property of how LLMs work, not a bug or limitation to be patched. But it creates an obvious problem for building conversational experiences: how do you build a chatbot that *remembers*?

The answer is that you, the developer, must manage memory explicitly. You maintain the conversation history and send it with every API call. **The messages list IS the memory.**

### Traditional apps vs LLM apps

The memory model for LLM apps is different from everything you learned building traditional applications:

| Traditional apps | LLM apps |
|---|---|
| Persistent state lives in a **database** | Persistent state lives in the **messages list** |
| Users identified by **sessions** | User context lives in the **system prompt** |
| Expensive data lives in a **cache** | Long conversations compressed into **summaries** |
| User preferences live in **profiles** | User preferences injected as **long-term memory** |

The mental shift: you are not storing data in a database and querying it. You are curating a text conversation and deciding what the model should be able to read right now.

### Three problems that emerge without management

As a conversation grows, three problems escalate:

**Problem 1 — Cost growth.** Every message you send to the LLM costs tokens. If you include the full conversation history on every turn, your per-message cost grows linearly with conversation length. A 100-turn conversation costs 50× more per message than a 2-turn conversation, assuming history doubles with each turn.

**Problem 2 — Context overflow.** Every LLM has a maximum context window — a hard limit on the number of tokens it can process in a single API call. Exceed it and the API returns an error. Long conversations will eventually hit this wall unless you manage history size actively.

**Problem 3 — Quality degradation.** Counterintuitively, more context is not always better. Research shows that LLMs struggle to use information buried in the middle of very long contexts. A bloated history full of old, irrelevant exchanges can actually make the model *less* coherent, not more.

These three problems share a common solution: actively curating what goes into the messages list on each turn.

---

## 2. The Messages List

The messages list is the data structure at the center of every multi-turn LLM application. Understanding how it works — and how it grows — is the foundation for every memory management technique that follows.

### Anatomy of a multi-turn conversation

Every message in the list has a `role` and `content`. Three roles matter:

- **`system`** — instructions that shape the assistant's behavior; sent once at the beginning (or prepended to every request)
- **`user`** — what the human said
- **`assistant`** — what the LLM replied

A conversation starts with a system message and grows by appending user/assistant pairs after each turn:

```python
# Turn 0: initial state
messages = [
    {"role": "system", "content": "You are a helpful assistant."}
]

# Turn 1: user asks, assistant replies
messages.append({"role": "user", "content": "What is a context window?"})
# ... call the API ...
messages.append({"role": "assistant", "content": "A context window is the maximum number of tokens an LLM can process in a single call."})

# Turn 2: follow-up question
messages.append({"role": "user", "content": "How big is GPT-4o's context window?"})
# ... call the API ...
messages.append({"role": "assistant", "content": "GPT-4o has a context window of 128,000 tokens."})

# Turn 3: follow-up builds on prior context
messages.append({"role": "user", "content": "How many pages of text is that?"})
# ... call the API ...
messages.append({"role": "assistant", "content": "Roughly 96,000 words, or about 320 pages of a typical novel."})

# messages now has 7 entries: 1 system + 3 user + 3 assistant
```

### Token growth over time

Every message adds tokens. Here is what unmanaged growth looks like for a typical chatbot where each turn averages ~200 tokens:

| Turns | Messages | Approx tokens sent per request |
|---|---|---|
| 0 | 1 (system only) | ~50 |
| 5 | 11 | ~1,100 |
| 10 | 21 | ~2,200 |
| 20 | 41 | ~4,300 |
| 50 | 101 | ~10,600 |
| 100 | 201 | ~21,200 |

At 100 turns, you are sending over 21,000 tokens per request just for history — before the LLM generates a single token in reply. With GPT-4o pricing, that is a 10× cost multiplier compared to turn 5.

### Counting tokens with tiktoken

Before you can manage token budgets, you need to count tokens. OpenAI's `tiktoken` library provides the same tokenizer the API uses:

```python
import tiktoken

def count_tokens(messages: list[dict], model: str = "gpt-4o") -> int:
    """Count the total tokens in a messages list."""
    enc = tiktoken.encoding_for_model(model)
    total = 0
    for message in messages:
        # Every message has a 4-token overhead for formatting
        total += 4
        for key, value in message.items():
            total += len(enc.encode(str(value)))
    # 3 tokens for the reply primer (every completion starts with <|start|>assistant<|message|>)
    total += 3
    return total

# Example
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is a context window?"},
    {"role": "assistant", "content": "A context window is the maximum number of tokens an LLM can process in a single call."},
]
print(count_tokens(messages))  # ~40 tokens
```

### The messages list as a single source of truth

The messages list is not just a log — it defines everything the LLM currently "knows" about the conversation:

- **Adding a message** — the model now knows that fact, question, or reply
- **Removing a message** — the model has no memory of that exchange
- **Modifying a message** — you are rewriting history; the model will reason from the modified version
- **Reordering messages** — you change the conversation flow; the model reads messages in sequence

This means every memory management technique reduces to a single question: *which messages should be in the list for this API call?*

---

## 3. Context Window Budgeting

Before you can manage context intelligently, you need to understand how much space you have and how to allocate it.

### Model context windows

Different models have very different context limits — and the practical usable limit is always smaller than the advertised maximum:

| Model | Context window | Practical limit | Notes |
|---|---|---|---|
| GPT-3.5 Turbo | 16,384 tokens | ~12,000 tokens | Fast and cheap; limited for long conversations |
| GPT-4o | 128,000 tokens | ~100,000 tokens | Strong long-context performance |
| GPT-4o Mini | 128,000 tokens | ~100,000 tokens | Budget-friendly with large context |
| Claude 3.5 Sonnet | 200,000 tokens | ~150,000 tokens | Excellent for very long documents and histories |
| Claude 3.5 Haiku | 200,000 tokens | ~150,000 tokens | Fast and affordable with large context |
| Gemini 1.5 Pro | 1,000,000 tokens | ~700,000 tokens | Experimental at very long lengths; cost scales steeply |

"Practical limit" accounts for cost, latency, and the known quality degradation at very long contexts. Even models that technically support 1M tokens rarely perform well across the full window.

### The budget formula

Think of the context window as a fixed budget that you allocate across four buckets:

```
available_for_history = window_size - system_tokens - output_reserve - current_user_tokens
```

Where:
- `window_size` — the model's maximum context window
- `system_tokens` — tokens consumed by the system prompt (fixed per session)
- `output_reserve` — tokens you hold back for the model's reply (set this to your max expected response length)
- `current_user_tokens` — tokens in the current user message

### Worked example

```
Model:           GPT-4o         (128,000 tokens)
System prompt:   ~300 tokens    (persona + instructions)
Output reserve:  ~2,000 tokens  (typical response length)
Current message: ~50 tokens     (user's question)
─────────────────────────────────────────────────────
Available for history:  128,000 - 300 - 2,000 - 50 = 125,650 tokens
```

In this example you have over 125,000 tokens for conversation history — far more than most conversations will ever use. But on a GPT-3.5 model with a more elaborate system prompt:

```
Model:           GPT-3.5 Turbo  (16,384 tokens)
System prompt:   ~800 tokens    (detailed instructions)
Output reserve:  ~1,000 tokens
Current message: ~50 tokens
─────────────────────────────────────────────────────
Available for history:  16,384 - 800 - 1,000 - 50 = 14,534 tokens
```

Now you are working with roughly 14,500 tokens for history — enough for about 70 turns at 200 tokens each. The budget runs out faster than you might expect.

For the project in this module, we use a conservative 4,096-token budget to make memory management strategies visible and testable:

```
Budget:          4,096 tokens   (conservative demo budget)
System prompt:   ~200 tokens    (persona + memories)
Output reserve:  1,024 tokens   (room for response)
Current message: ~50 tokens
─────────────────────────────────────────────────────
Available for history:  4,096 - 200 - 1,024 - 50 = 2,822 tokens
```

At ~100 tokens per turn, that is roughly 28 turns before you need to start managing memory — perfect for demonstrating truncation and summarization.

### Why budget conservatively

Even if you have headroom, there are four reasons to cap your history budget well below the maximum:

- **Cost** — every token sent costs money; unused headroom is not free if it fills with old, irrelevant exchanges
- **Latency** — larger contexts take longer to process; first-token latency grows with context size
- **Quality** — LLMs perform best with focused, relevant context; bloated history adds noise
- **Portability** — if you switch to a smaller, cheaper model later, a well-budgeted conversation will require fewer changes

### Budget check before every API call

```python
def check_budget(
    messages: list[dict],
    model: str = "gpt-4o",
    window_size: int = 128_000,
    output_reserve: int = 2_000,
) -> dict:
    """Return token budget analysis for the current messages list."""
    used = count_tokens(messages, model)
    available = window_size - output_reserve
    remaining = available - used
    utilization = used / available

    return {
        "used": used,
        "available": available,
        "remaining": remaining,
        "utilization": utilization,
        "over_budget": remaining < 0,
    }

# Example
budget = check_budget(messages)
if budget["over_budget"]:
    raise ValueError(f"Messages exceed context budget by {-budget['remaining']} tokens")
print(f"Using {budget['utilization']:.1%} of context budget ({budget['used']} / {budget['available']} tokens)")
```

---

## 4. Sliding Window Truncation

The simplest memory management strategy is also the most widely used: keep only the most recent N messages and discard the oldest. This is called a **sliding window**.

### How it works

The sliding window strategy has two rules:

1. **Always keep the system prompt** — it defines who the assistant is; discarding it would break the persona entirely
2. **Drop the oldest user/assistant pairs first** — remove from the front of the history, never from the back

The window "slides" forward as the conversation grows: as new messages are added at the end, old ones are dropped from the front.

### Before and after

```
BEFORE (9 messages, exceeds budget):

  [system]   You are a helpful assistant.
  [user]     What is a neural network?            ← oldest, will be dropped
  [asst]     A neural network is...               ← oldest, will be dropped
  [user]     How does backpropagation work?       ← will be dropped
  [asst]     Backpropagation is...               ← will be dropped
  [user]     What is an activation function?
  [asst]     An activation function is...
  [user]     Give me an example.
  [asst]     ReLU is a common example...


AFTER (5 messages, within budget):

  [system]   You are a helpful assistant.
  [user]     What is an activation function?      ← now oldest (preserved)
  [asst]     An activation function is...
  [user]     Give me an example.
  [asst]     ReLU is a common example...
```

The two oldest user/assistant pairs were dropped. The system prompt is untouched.

### Implementation

```python
def truncate_messages(
    messages: list[dict],
    max_tokens: int,
    model: str = "gpt-4o",
) -> list[dict]:
    """
    Truncate the messages list to fit within max_tokens using a sliding window.
    Always preserves the system prompt (first message if role == 'system').
    Removes oldest user/assistant pairs until the list fits.
    """
    # Separate system prompt from conversation history
    if messages and messages[0]["role"] == "system":
        system = [messages[0]]
        history = messages[1:]
    else:
        system = []
        history = list(messages)

    # Drop oldest pairs until we fit within budget
    while history and count_tokens(system + history, model) > max_tokens:
        # Remove the oldest pair (or single message if unpaired)
        if len(history) >= 2 and history[0]["role"] == "user" and history[1]["role"] == "assistant":
            history = history[2:]  # drop a full user/assistant pair
        else:
            history = history[1:]  # drop a single message if unpaired

    return system + history
```

### Trade-offs

| Pros | Cons |
|---|---|
| Simple to implement and understand | Loses earlier context abruptly — the LLM has no memory of dropped turns |
| Fast — no additional API calls required | May lose critical information (e.g., the user's name stated in turn 1) |
| Deterministic — easy to test and reason about | Users may be confused when the assistant "forgets" something they said |
| Works with any LLM and any token counter | Offers no graceful degradation — context is present or absent, nothing in between |

### When sliding window is enough

Sliding window works well for:

- **Task-focused conversations** — if users typically accomplish a goal in 5–10 turns, old history rarely matters
- **Stateless Q&A** — if each question is largely independent, older context is not needed
- **Short-session chatbots** — if sessions are brief and users do not expect the assistant to remember across sessions
- **Prototyping** — before investing in summarization, start with a sliding window and measure how often users complain about forgetting

For longer, more complex conversations — or when critical facts appear early and are needed later — combine sliding window with summarization.

---

## 5. Conversation Summarization

Sliding window discards old context entirely. Summarization compresses it: instead of dropping old messages, you use the LLM to write a summary of what was discussed, then inject that summary as a single condensed message. The model "remembers" the gist without needing every turn verbatim.

### How it works

When the conversation exceeds a token threshold, you:

1. Take the oldest N messages (the ones that would otherwise be dropped)
2. Ask the LLM to summarize them
3. Replace those N messages with a single summary message
4. Continue the conversation with the summary in place

### Before and after

```
BEFORE (history growing too large):

  [system]    You are a helpful assistant.
  [user]      Hi, I'm Alex. I'm learning Python.
  [asst]      Nice to meet you, Alex! What would you like to learn?
  [user]      Let's start with variables.
  [asst]      Variables in Python are declared by assignment...
  [user]      Now let's talk about loops.
  [asst]      Python has two loop types: for and while...
  [user]      What about functions?        ← recent, keep verbatim
  [asst]      Functions are defined with def...  ← recent, keep verbatim


AFTER (oldest turns replaced by summary):

  [system]    You are a helpful assistant.
  [system]    Summary of earlier conversation: The user's name is Alex. They
              are learning Python and have covered variables, for loops, and
              while loops so far.
  [user]      What about functions?        ← verbatim, unchanged
  [asst]      Functions are defined with def...  ← verbatim, unchanged
```

The summary captures the key facts (Alex, is learning Python, has covered variables and loops) in ~40 tokens instead of the ~300 tokens the original four turns occupied.

### Two summarization approaches

| Approach | When it triggers | How it works | Token growth | API calls | Best for |
|---|---|---|---|---|---|
| **Rolling summary** | After every N turns | Summarizes the oldest batch and replaces it | Stays flat — summary replaces turns | One extra call per batch | Long-running conversations that grow indefinitely |
| **Threshold summary** | When token count exceeds limit | Summarize everything old enough to compress | Step-downs at each trigger | One call per threshold crossing | Conversations with unpredictable length |

### Generating the summary

Ask the LLM to be precise and factual — you want a memory aid, not a narrative:

```python
def summarize_messages(messages: list[dict], model: str = "gpt-4o") -> str:
    """Use the LLM to summarize a list of messages into a compact memory string."""
    # Build a plain-text transcript for summarization
    transcript = "\n".join(
        f"{m['role'].upper()}: {m['content']}"
        for m in messages
        if m["role"] != "system"
    )

    summary_prompt = [
        {
            "role": "system",
            "content": (
                "You are a conversation summarizer. "
                "Summarize the following conversation excerpt into a concise paragraph. "
                "Focus on: user's name and background (if mentioned), topics discussed, "
                "decisions made, open questions, and any facts the assistant should remember. "
                "Be factual and specific. Do not editorialize."
            ),
        },
        {
            "role": "user",
            "content": f"Conversation to summarize:\n\n{transcript}",
        },
    ]

    from openai import OpenAI
    client = OpenAI()
    response = client.chat.completions.create(model=model, messages=summary_prompt)
    return response.choices[0].message.content
```

### Token savings analysis

Summarization can dramatically reduce token consumption for long conversations:

| Scenario | Raw tokens | After summarization | Savings |
|---|---|---|---|
| 20 turns (~200 tokens each) | ~4,000 tokens | ~400 token summary | ~90% |
| 50 turns | ~10,000 tokens | ~600 token summary | ~94% |
| 100 turns | ~20,000 tokens | ~800 token summary | ~96% |

The summary tokens grow logarithmically while raw history grows linearly — a powerful asymmetry for long-running conversations.

### Combining summarization with truncation

In practice, use both strategies together:

1. **Summarize** the oldest batch of messages when history exceeds your soft threshold
2. **Truncate** any remaining excess with a sliding window if the conversation still exceeds budget after summarization

This gives you the graceful compression of summarization for normal growth, with a hard safety net from truncation for edge cases (very long single messages, unexpected spikes).

```python
def manage_context(
    messages: list[dict],
    soft_limit: int = 4_000,   # summarize when history exceeds this
    hard_limit: int = 6_000,   # truncate if still over after summarization
    batch_size: int = 10,      # number of old messages to summarize per pass
    model: str = "gpt-4o",
) -> list[dict]:
    """Apply summarization then truncation to keep messages within budget."""
    if messages and messages[0]["role"] == "system":
        system = [messages[0]]
        history = messages[1:]
    else:
        system = []
        history = list(messages)

    # Step 1: summarize oldest batch if over soft limit
    if count_tokens(system + history, model) > soft_limit and len(history) > batch_size:
        to_summarize = history[:batch_size]
        remaining = history[batch_size:]
        summary_text = summarize_messages(to_summarize, model)
        summary_message = {"role": "system", "content": f"Summary of earlier conversation: {summary_text}"}
        history = [summary_message] + remaining

    # Step 2: hard truncation as safety net
    result = truncate_messages(system + history, hard_limit, model)
    return result
```

---

## 6. System Prompts & Persona Design

The system prompt is the one part of the messages list that stays constant throughout a conversation. It is the LLM's permanent memory — everything it reliably knows at all times, regardless of how much history has been truncated or summarized.

### The system prompt as permanent memory

Unlike conversation history, the system prompt is never dropped by sliding window or summarized away. Whatever you put in the system prompt, the model knows for the entire session. This makes it the right place for:

- **Identity** — who is this assistant, what is its name, what role does it play
- **Behavioral style** — tone, verbosity, how to handle uncertainty
- **Hard rules** — what the assistant must never do, what it must always do
- **Session-scoped memory** — facts injected at session start that should persist (long-term memory, user preferences, current task context)

### Three components of a well-designed system prompt

**1. Identity** — tells the model who it is. Specific identities produce more consistent behavior than vague ones.

```
You are Aria, a Python programming tutor specializing in beginners.
```

**2. Style** — tells the model how to communicate. Without this, tone and verbosity are inconsistent.

```
Use simple language. Avoid jargon unless you define it first.
Keep responses under 150 words unless a code example is required.
When you write code, always explain what it does line by line.
```

**3. Boundaries** — tells the model what it must and must not do. Essential for scoped assistants.

```
Only answer questions about Python programming.
If the user asks about another topic, politely redirect them.
Never write code that could be used for malicious purposes.
```

### Example personas

```python
PERSONAS = {
    "assistant": {
        "name": "Aria",
        "role": "general-purpose assistant",
        "style": "concise, friendly, and precise",
        "boundaries": "Answer any reasonable question. Decline requests for harmful content.",
    },
    "tutor": {
        "name": "Prof. Byte",
        "role": "Python programming tutor for beginners",
        "style": "patient, encouraging, uses simple analogies and plenty of examples",
        "boundaries": "Only answer Python programming questions. Redirect off-topic requests.",
    },
    "writer": {
        "name": "Muse",
        "role": "creative writing assistant",
        "style": "expressive, imaginative, enthusiastic about craft and story",
        "boundaries": "Help with creative writing only. Decline requests for factual research.",
    },
}
```

### Building the system prompt dynamically

A good system prompt builder combines the persona definition with any session-scoped memory injected at startup:

```python
def build_system_prompt(persona_key: str, long_term_memory: list[str] = None) -> str:
    """Build a system prompt from a persona definition and optional long-term memory."""
    persona = PERSONAS[persona_key]

    prompt = (
        f"You are {persona['name']}, a {persona['role']}.\n\n"
        f"Style: {persona['style']}\n\n"
        f"Rules: {persona['boundaries']}"
    )

    if long_term_memory:
        memory_block = "\n".join(f"- {fact}" for fact in long_term_memory)
        prompt += f"\n\nThings you know about this user:\n{memory_block}"

    return prompt

# Example output:
# You are Prof. Byte, a Python programming tutor for beginners.
#
# Style: patient, encouraging, uses simple analogies and plenty of examples
#
# Rules: Only answer Python programming questions. Redirect off-topic requests.
#
# Things you know about this user:
# - Their name is Alex
# - They are a complete beginner with no prior programming experience
# - They prefer short explanations with code examples
```

### Anti-patterns in system prompt design

Avoid these common mistakes:

- **Over-constraining** — writing so many rules that the model spends more effort complying than being helpful. Start with the minimum necessary constraints.
- **Under-specifying** — a single sentence like "Be helpful" provides almost no guidance. The model's defaults may not match your intent.
- **Conflicting instructions** — "Be concise" and "Always provide comprehensive explanations" cannot both be satisfied. Prioritize and reconcile before writing.
- **Leaking implementation details** — do not describe your memory management strategy ("You have a sliding window of 10 messages"). Users can read system prompts in some interfaces, and this degrades trust and experience.
- **Prompt injection surface** — if user input is ever interpolated directly into the system prompt, you have created an injection vulnerability. Always sanitize and validate before injecting user-supplied content.

---

## 7. Long-Term Memory

Everything discussed so far is **short-term memory** — it exists within a single session and is lost when the conversation ends. Long-term memory persists across sessions: the next time the user starts a new conversation, the assistant already knows relevant facts about them.

### Short-term vs long-term memory

| | Short-term memory | Long-term memory |
|---|---|---|
| **Scope** | Current session only | Persists across sessions |
| **Implementation** | The messages list | External storage (file, database) |
| **Managed by** | Sliding window, summarization | Explicit extraction and retrieval |
| **Typical size** | Thousands of tokens | Tens to hundreds of facts |
| **Reset on** | Session end | Never (until explicitly deleted) |

### Types of long-term memory

| Type | What it stores | Examples |
|---|---|---|
| **Semantic** | Facts and knowledge about the user | Name, occupation, expertise level, preferences, goals |
| **Episodic** | Events and experiences the user has mentioned | "Started a new job last week", "Is preparing for a job interview", "Completed the beginner Python course" |

Semantic memory answers "who is this user?" Episodic memory answers "what has this user been through?"

### Extracting memories automatically

After each conversation, you can run a memory extraction step — ask the LLM to identify facts worth remembering:

```python
def extract_memories(messages: list[dict], model: str = "gpt-4o") -> list[str]:
    """
    Extract facts worth storing in long-term memory from a completed conversation.
    Returns a list of concise fact strings.
    """
    transcript = "\n".join(
        f"{m['role'].upper()}: {m['content']}"
        for m in messages
        if m["role"] != "system"
    )

    extraction_prompt = [
        {
            "role": "system",
            "content": (
                "You are a memory extractor. Given a conversation transcript, "
                "identify facts about the user that are worth remembering for future sessions. "
                "Return each fact as a single concise sentence. "
                "Only include facts that would help an assistant serve this user better later. "
                "Ignore facts specific to this conversation that won't matter next time. "
                "Return one fact per line. Return an empty response if there is nothing worth remembering."
            ),
        },
        {
            "role": "user",
            "content": f"Conversation transcript:\n\n{transcript}",
        },
    ]

    from openai import OpenAI
    client = OpenAI()
    response = client.chat.completions.create(model=model, messages=extraction_prompt)
    raw = response.choices[0].message.content.strip()
    if not raw:
        return []
    return [line.strip() for line in raw.splitlines() if line.strip()]
```

### Storing and loading memories

Store extracted memories as JSON — simple, portable, and easy to inspect:

```json
{
  "user_id": "alex_chen",
  "updated_at": "2024-03-15T14:32:00Z",
  "memories": [
    "User's name is Alex Chen.",
    "Alex is a backend developer with 3 years of experience in Java.",
    "Alex is learning Python to transition into data science.",
    "Alex prefers concise explanations with runnable code examples.",
    "Alex has completed modules on variables, loops, and functions.",
    "Alex is preparing for a Python technical interview in two weeks."
  ]
}
```

Load them at session start and inject into the system prompt via `build_system_prompt()`:

```python
import json
from pathlib import Path

def load_memories(user_id: str, memory_dir: str = "memories") -> list[str]:
    """Load long-term memories for a user from disk."""
    path = Path(memory_dir) / f"{user_id}.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    return data.get("memories", [])

def save_memories(user_id: str, memories: list[str], memory_dir: str = "memories") -> None:
    """Save long-term memories for a user to disk."""
    from datetime import datetime, timezone
    Path(memory_dir).mkdir(exist_ok=True)
    path = Path(memory_dir) / f"{user_id}.json"
    data = {
        "user_id": user_id,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "memories": memories,
    }
    path.write_text(json.dumps(data, indent=2))
```

### A note on privacy

Long-term memory raises real privacy questions. You are storing personal information about users — potentially sensitive details about their background, goals, or experiences. Before building long-term memory into a production system:

- **Tell users** — be transparent that facts are being retained across sessions
- **Give users control** — provide a way to view, edit, and delete their stored memories
- **Minimize what you store** — store only what genuinely improves the experience; resist the urge to capture everything
- **Secure the storage** — treat memory files like user data, because they are: access controls, encryption at rest, and proper key management apply
- **Consider regulations** — GDPR, CCPA, and similar laws treat stored personal information as regulated data with rights attached

---

## 8. Memory in the AI Stack

Conversation memory is not a standalone concern. It is a cross-cutting pattern that shows up in every part of the AI application stack. Understanding how memory connects to other AI techniques prepares you for the more advanced modules ahead.

### Agents need working memory

Module 11 (AI Agents) covers systems where an LLM orchestrates multi-step tasks — using tools, making decisions, running loops. Agents need **working memory**: a running record of what they have done, what they have learned, and what they still need to accomplish.

This is exactly the messages list pattern, applied to agent reasoning rather than human conversation. An agent appends its thoughts, tool calls, and tool results to the messages list just as a chatbot appends user and assistant turns. All the same management challenges apply: token budgets, sliding windows, summarization.

### RAG is external memory

Module 07 (RAG) introduced a pattern for giving LLMs access to external knowledge: store documents in a vector index, retrieve relevant ones at query time, inject them into the prompt.

That pattern is structurally identical to long-term memory: **store → retrieve → inject**. The difference is the source (a document corpus vs a user's personal fact store) and the retrieval mechanism (semantic search vs user ID lookup). The prompt augmentation step is the same.

You can combine both: use RAG to retrieve relevant knowledge from a corpus, and long-term memory to inject user-specific facts, both injected into the same system prompt before each call.

### Tool use requires conversation tracking

Module 06 (Tool Use & Function Calling) showed how LLMs invoke functions — searching the web, querying databases, running code. Each tool call and its result must be tracked in the messages list so the LLM can reason about what it found and decide what to do next.

Tool-use conversations grow faster than text-only conversations: each tool invocation adds at least two messages (the call and the result), and results can be very large (a full API response, a database query result). Aggressive context management is essential for agentic systems.

### Structured output enables memory extraction

Module 08 (Structured Output) covered techniques for getting LLMs to return JSON or other structured formats instead of free text. Long-term memory extraction is a natural application: instead of asking the LLM to return memories as a bulleted list, use structured output to get a typed, validated list of memory objects:

```python
from pydantic import BaseModel

class Memory(BaseModel):
    type: str          # "semantic" or "episodic"
    content: str       # the fact itself
    confidence: float  # how confident the model is (0.0–1.0)

class MemoryExtractionResult(BaseModel):
    memories: list[Memory]
```

Structured extraction makes memory processing more reliable — you can filter by type, sort by confidence, and validate the output before storing it.

### Production memory patterns

In production systems, the simple file-based storage shown in Section 7 is replaced by proper infrastructure:

| Pattern | Implementation | Use case |
|---|---|---|
| **User memory store** | PostgreSQL, DynamoDB, Firestore | Per-user long-term memory with access controls |
| **Session cache** | Redis, Memcached | Fast retrieval of recent conversation history |
| **Vector memory** | Pinecone, Weaviate, pgvector | Semantic search over large memory stores |
| **Audit log** | Append-only event store | Compliance, debugging, and memory replay |

The right infrastructure depends on your scale and requirements. For a personal project, JSON files work fine. For a multi-tenant SaaS product, you need proper user isolation, encryption, and access control from day one.

### Looking ahead to Module 11

Module 11 (AI Agents) builds directly on everything in this module:

- The messages list becomes the agent's **action history**
- System prompts carry the agent's **goals and capabilities**
- Long-term memory stores the agent's **learned state** across tasks
- Summarization keeps the agent's history from exceeding its context window as it works through long-running tasks

The memory management skills you have built here — budgeting, truncation, summarization, extraction, and injection — are the foundation for every agent system you will build.

---
