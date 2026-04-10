# Project: Memory Chatbot

Build an interactive CLI chatbot with configurable personas, context window management, and persistent long-term memory.

## What you'll build

A memory-aware chatbot that:
- Lets you choose from three personas (Helpful Assistant, Coding Tutor, Creative Writer)
- Counts tokens and manages context window budget
- Applies sliding window truncation when history grows too long
- Summarizes older messages to preserve key context
- Extracts and persists user facts across sessions using a JSON file
- Injects long-term memories into the system prompt

## Prerequisites

- Completed reading the Module 09 README
- Python 3.11+ with project dependencies installed
- At least one LLM provider API key configured in `.env`

## How to build

Work through the steps below in order. Each step builds on the previous one.

## Steps

### Step 1: Token counting

Implement `count_tokens(messages, model)` — counts the total tokens in a messages list using tiktoken. Falls back to a word-based estimate if tiktoken doesn't have the encoding for the model.

### Step 2: Sliding window truncation

Implement `truncate_messages(messages, max_tokens, model)` — preserves the system prompt at index 0 and drops the oldest user+assistant pairs until the messages fit within the token budget.

### Step 3: Conversation summarization

Implement `summarize_messages(messages, model)` — sends older messages to the LLM with a summarization prompt and returns a compact summary string.

### Step 4: Context management

Implement `manage_context(messages, max_tokens, model)` — orchestrates the memory strategy: checks token usage, summarizes old messages if over threshold, injects summary after system prompt, falls back to truncation if still over budget.

### Step 5: Persona and system prompt

Define three persona dicts and implement `build_system_prompt(persona, memories)` — builds a system prompt combining the persona definition with injected long-term memories (user facts and conversation summaries).

### Step 6: Long-term memory

Implement `extract_memories(messages, model)`, `load_memory(memory_file)`, and `save_memory(memory_data, memory_file)` — extract user facts from conversation using the LLM, and persist/load them as JSON.

### Step 7: Chat and main loop

Implement `chat(messages, model)` for sending messages to the LLM, and `main()` for the interactive loop with persona selection, per-turn context management, `/memory`, `/clear`, `/bye` commands, and session summary.

## How to run

```bash
cd 09-conversational-ai-memory/project
python solution.py
```

## Expected output

```
============================================================
  Memory Chatbot
============================================================
  Model: anthropic/claude-sonnet-4-20250514

  Choose a persona:
    1. Helpful Assistant
    2. Coding Tutor
    3. Creative Writer
  > 1

  Persona: Helpful Assistant
  Memory: 2 user facts loaded, 1 conversation summary loaded
  Context budget: 4096 tokens (1024 reserved for output)
  Type /memory, /clear, or /bye

You: Hi, my name is Alex and I'm learning Python

  [Tokens: 892/3072 used]

  Assistant: Hello Alex! Welcome! I'd be happy to help you with your
  Python learning journey. What topics are you most interested in?

  Tokens: 45 in + 38 out | Cost: $0.0008

You: /memory

  Long-term memories:
    User facts:
      - User's name is Alex
      - User prefers Python over JavaScript
    Conversation summaries:
      - 2024-03-15: Discussed Python async patterns...

You: /bye

  Extracting memories from this conversation...
  Saved 1 new fact: "User is learning Python"

============================================================
  Session Summary
============================================================
  Turns:        1
  Total tokens: 83 (45 in + 38 out)
  Total cost:   $0.0008
  Memory file:  memory.json
============================================================
```

## Stretch goals

1. **Streaming responses** — combine Module 05 streaming with memory management for a real-time chat experience
2. **Memory search** — add a `/search <query>` command that searches long-term memories using keyword matching
3. **Memory editing** — add `/forget <fact>` to remove specific memories from the JSON file
