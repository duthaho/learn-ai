# Module 04 — The AI API Layer

Making your first real LLM API calls: requests, responses, errors, retries, and cost tracking.

| Detail        | Value                                     |
|---------------|-------------------------------------------|
| Level         | Beginner                                  |
| Time          | ~2 hours                                  |
| Prerequisites | Module 01 (How LLMs Work)                 |

## What you'll build

After reading this module, head to [`project/`](project/) to build an **LLM API Explorer** — a CLI tool that probes your API setup, compares models side-by-side, tracks costs and latency, and handles errors with retries.

---

## Table of Contents

1. [Anatomy of an LLM API Call](#1-anatomy-of-an-llm-api-call)
2. [Authentication & API Keys](#2-authentication--api-keys)
3. [The Unified API Layer (LiteLLM)](#3-the-unified-api-layer-litellm)
4. [Token Usage & Cost](#4-token-usage--cost)
5. [Error Handling](#5-error-handling)
6. [Retries & Backoff](#6-retries--backoff)
7. [Rate Limits & Throttling](#7-rate-limits--throttling)
8. [Latency & Performance](#8-latency--performance)
9. [Best Practices for Production](#9-best-practices-for-production)

---

## 1. Anatomy of an LLM API Call

Every LLM provider exposes an HTTP API. You send a JSON request with your prompt and parameters; you get back a JSON response with the model's output, token counts, and metadata. Understanding this request-response cycle is the foundation of everything else in AI engineering.

### The chat completions pattern

All major providers have converged on the **chat completions** format. Even for single-turn questions, you send a `messages` array. This is because:

- The messages array **is** the conversation state — the model is stateless
- Each call sends the full history; the server stores nothing between calls
- Single-turn is just a messages array with one entry
- Multi-turn means appending the assistant's response and the user's next message

OpenAI originally had a legacy `/v1/completions` endpoint (raw text in, text out) that was deprecated in 2023. The chat format replaced it because the messages structure is strictly more expressive — it separates system instructions from user input and provides a natural place for conversation history.

### OpenAI request structure

```json
POST /v1/chat/completions

{
  "model": "gpt-4o",
  "messages": [
    {"role": "system", "content": "You are a helpful coding assistant."},
    {"role": "user", "content": "Explain recursion in Python"},
    {"role": "assistant", "content": "Recursion is when a function calls itself..."},
    {"role": "user", "content": "Show me an example"}
  ],
  "temperature": 0.7,
  "max_tokens": 1024,
  "top_p": 1.0,
  "stop": ["\n\n"]
}
```

**Message roles:**

| Role | Purpose |
|------|---------|
| `system` | Sets behavior and persona. Processed once at the start. |
| `developer` | Introduced with o1 models (Jan 2025). Replaces `system` for newer models — semantically marks instructions as coming from the app developer, not the end user. For GPT-4o and older, both work identically. |
| `user` | End-user input. |
| `assistant` | Model's prior responses (for multi-turn context). |
| `tool` | Return value from a tool/function call. Must include `tool_call_id`. |

**Key parameters:**

| Parameter | Type | Default | Purpose |
|-----------|------|---------|---------|
| `model` | string | required | Which model to use (e.g., `gpt-4o`, `gpt-4o-mini`) |
| `temperature` | 0-2 | 1 | Controls randomness. 0 = deterministic, 1 = balanced, 2 = very creative |
| `max_tokens` | int | model default | Maximum tokens in the response. A ceiling, not a target |
| `top_p` | 0-1 | 1 | Nucleus sampling. 0.1 = only top 10% probability mass. Alternative to temperature — set one, leave the other at default |
| `stop` | array | null | Up to 4 sequences where generation halts. The stop sequence itself is not included in the output |
| `frequency_penalty` | -2 to 2 | 0 | Penalizes tokens proportional to how often they've appeared |
| `presence_penalty` | -2 to 2 | 0 | Penalizes tokens that have appeared at all |
| `seed` | int | null | For reproducible outputs (best-effort) |

### OpenAI response structure

```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1710000000,
  "model": "gpt-4o-2024-08-06",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Here is a recursive fibonacci function..."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 42,
    "completion_tokens": 128,
    "total_tokens": 170,
    "prompt_tokens_details": {
      "cached_tokens": 0
    },
    "completion_tokens_details": {
      "reasoning_tokens": 0
    }
  },
  "system_fingerprint": "fp_a1b2c3d4"
}
```

**`finish_reason` values — always check this field:**

| Value | Meaning | What to do |
|-------|---------|------------|
| `stop` | Natural end or hit a stop sequence | Normal — use the response |
| `length` | Hit `max_tokens` limit — output is **truncated** | Increase `max_tokens` or continue with another call |
| `tool_calls` | Model wants to invoke a tool | Execute the tool and send the result back |
| `content_filter` | Content blocked by safety filter | Modify the prompt or handle gracefully |

The `choices` array usually has one element. It can have multiple if you set `n > 1` to request multiple completions (rarely used in practice due to cost).

The `usage` object is critical for cost tracking — it tells you exactly how many tokens were consumed. `prompt_tokens_details.cached_tokens` shows how many input tokens were served from OpenAI's automatic cache. `completion_tokens_details.reasoning_tokens` applies to o1/o3 reasoning models.

### How Anthropic's Messages API differs

Anthropic's API follows the same concept but with several structural differences:

```json
POST /v1/messages

{
  "model": "claude-sonnet-4-20250514",
  "max_tokens": 1024,
  "system": "You are a helpful coding assistant.",
  "messages": [
    {"role": "user", "content": "Explain recursion in Python"},
    {"role": "assistant", "content": "Recursion is when a function calls itself..."},
    {"role": "user", "content": "Show me an example"}
  ],
  "temperature": 0.7
}
```

```json
{
  "id": "msg_01XFDUDYJgAACzvnptvVoYEL",
  "type": "message",
  "role": "assistant",
  "content": [
    {
      "type": "text",
      "text": "Here is a recursive fibonacci function..."
    }
  ],
  "model": "claude-sonnet-4-20250514",
  "stop_reason": "end_turn",
  "stop_sequence": null,
  "usage": {
    "input_tokens": 42,
    "output_tokens": 128
  }
}
```

### Key differences at a glance

| Aspect | OpenAI | Anthropic |
|--------|--------|-----------|
| System prompt | `system`/`developer` role in messages array | Top-level `system` parameter |
| Response wrapper | `choices` array (supports `n` completions) | Single message object |
| Content format | `content` is a string | `content` is an array of typed blocks |
| Stop reason field | `finish_reason` | `stop_reason` |
| Stop values | `stop`, `length`, `tool_calls`, `content_filter` | `end_turn`, `max_tokens`, `stop_sequence`, `tool_use` |
| Usage fields | `prompt_tokens`, `completion_tokens`, `total_tokens` | `input_tokens`, `output_tokens` |
| `max_tokens` | Optional (model default) | **Required** |
| Message ordering | Flexible | Must alternate user/assistant, starting with user |
| Multiple completions | `n` parameter | Not supported |

### Multi-turn: the client manages state

The model is completely stateless. To have a conversation, you append each exchange and resend:

```
Turn 1 → send: [user: "Hi"]
         recv: [assistant: "Hello!"]

Turn 2 → send: [user: "Hi", assistant: "Hello!", user: "What's 2+2?"]
         recv: [assistant: "4"]

Turn 3 → send: [user: "Hi", assistant: "Hello!", user: "What's 2+2?",
                 assistant: "4", user: "Thanks!"]
         recv: [assistant: "You're welcome!"]
```

Token usage grows with every turn. Eventually you need to truncate or summarize old messages — but that's a topic for Module 09 (Conversational AI & Memory).

---

## 2. Authentication & API Keys

API keys are secret strings that identify and authorize your requests. Every provider requires one.

### Key formats

| Provider | Key format | Get yours at |
|----------|-----------|-------------|
| OpenAI | `sk-proj-...` (project key) or `sk-...` (legacy) | platform.openai.com/api-keys |
| Anthropic | `sk-ant-api03-...` | console.anthropic.com/settings/keys |
| Google Gemini | `AIza...` | aistudio.google.com/apikey |

### How keys are transmitted

Each provider uses a different HTTP header:

```
# OpenAI
Authorization: Bearer sk-proj-abc123...

# Anthropic
x-api-key: sk-ant-api03-abc123...
anthropic-version: 2023-06-01

# Google Gemini
x-goog-api-key: AIza...
```

You don't need to set these headers manually — SDKs and LiteLLM handle it. But understanding what's happening under the hood helps when debugging authentication failures.

### Environment variables

The universally adopted pattern — used by all official SDKs and by LiteLLM:

```bash
# .env file (never commit this)
OPENAI_API_KEY=sk-proj-abc123...
ANTHROPIC_API_KEY=sk-ant-api03-abc123...
GEMINI_API_KEY=AIza...
```

Load them in Python:

```python
from dotenv import load_dotenv
import os

load_dotenv()  # reads .env into os.environ
key = os.getenv("OPENAI_API_KEY")
```

Official SDKs auto-detect these variables. `openai.OpenAI()` reads `OPENAI_API_KEY` automatically — no need to pass it explicitly.

### Security essentials

**Never commit keys to git.** Add to `.gitignore`:

```
.env
.env.*
!.env.example
```

**Provide a `.env.example`** (committed, no real values) so teammates know what's needed:

```bash
# .env.example — safe to commit
OPENAI_API_KEY=your-openai-key-here
ANTHROPIC_API_KEY=your-anthropic-key-here
```

**If a key is leaked:** Revoke it immediately in the provider's dashboard. Generate a new one. Check your git history for exposure: `git log -p --all -S 'sk-proj-'`.

**In production:** Use a secrets manager (AWS Secrets Manager, GCP Secret Manager, HashiCorp Vault) rather than `.env` files. Set per-key spending limits in provider dashboards.

---

## 3. The Unified API Layer (LiteLLM)

Every provider has its own SDK, auth mechanism, message format, and response shape. Switching between or comparing models means rewriting integration code — unless you use a unified layer.

### The problem

```python
# OpenAI: content is a string
response.choices[0].message.content

# Anthropic: content is a list of blocks
response.content[0].text

# Google: completely different structure
response.candidates[0].content.parts[0].text
```

Three providers, three ways to extract the response text. This multiplies across every feature: error handling, streaming, token counting, cost tracking.

### LiteLLM: one interface, 100+ providers

LiteLLM wraps all major providers behind the OpenAI chat completions interface. You write code once and switch models by changing a string:

```python
from litellm import completion

# Same function, same format — any provider
response = completion(
    model="anthropic/claude-sonnet-4-20250514",
    messages=[{"role": "user", "content": "Hello"}],
    temperature=0.7,
    max_tokens=256
)

# Response is always in OpenAI format, regardless of provider
print(response.choices[0].message.content)
print(response.usage.prompt_tokens)
print(response.usage.completion_tokens)
```

### Model naming: `provider/model-name`

| Provider | Prefix | Example |
|----------|--------|---------|
| OpenAI | `openai/` | `openai/gpt-4o`, `openai/gpt-4o-mini` |
| Anthropic | `anthropic/` | `anthropic/claude-sonnet-4-20250514` |
| Google | `gemini/` | `gemini/gemini-2.0-flash` |
| Ollama (local) | `ollama/` | `ollama/llama3` |
| Groq | `groq/` | `groq/llama-3.1-70b-versatile` |
| Azure OpenAI | `azure/` | `azure/my-gpt4-deployment` |

OpenAI models also work without the prefix (`model="gpt-4o"`), but using the prefix is recommended for clarity.

### What LiteLLM normalizes

- **Authentication** — reads each provider's env var automatically
- **System messages** — extracts `system` role messages and passes them as Anthropic's top-level `system` parameter
- **Content format** — maps Anthropic's content blocks to plain strings
- **Response fields** — maps `stop_reason` → `finish_reason`, `input_tokens` → `prompt_tokens`, etc.
- **Streaming** — `stream=True` yields chunks in a uniform format across providers

### What LiteLLM can't abstract

- **Provider-specific features** — Anthropic's prompt caching (`cache_control`), OpenAI's structured output (`response_format` with JSON schema), Gemini's grounding — these pass through but only work for their target provider
- **Model capability differences** — switching from GPT-4o to Claude doesn't guarantee identical behavior for tool calling, multimodal inputs, or system prompt handling
- **Token counting** — uses approximations for non-OpenAI models
- **Version lag** — new provider features may take days/weeks to appear in LiteLLM releases

### Additional features

```python
from litellm import completion, completion_cost

# Cost tracking
response = completion(model="openai/gpt-4o", messages=[...])
cost = completion_cost(completion_response=response)  # USD float

# Built-in retries
response = completion(model="openai/gpt-4o", messages=[...], num_retries=3)

# Fallbacks: try models in order
response = completion(
    model="openai/gpt-4o",
    messages=[...],
    fallbacks=["anthropic/claude-sonnet-4-20250514"]
)
```

---

## 4. Token Usage & Cost

Every API response tells you exactly how many tokens were consumed. Understanding this is essential because tokens are the unit of billing, context windows, and latency.

### Usage in API responses

**OpenAI:**
```json
"usage": {
  "prompt_tokens": 125,
  "completion_tokens": 50,
  "total_tokens": 175,
  "prompt_tokens_details": {"cached_tokens": 0},
  "completion_tokens_details": {"reasoning_tokens": 0}
}
```

**Anthropic:**
```json
"usage": {
  "input_tokens": 125,
  "output_tokens": 50,
  "cache_creation_input_tokens": 0,
  "cache_read_input_tokens": 0
}
```

LiteLLM normalizes both to the OpenAI format: `usage.prompt_tokens`, `usage.completion_tokens`, `usage.total_tokens`.

### Cost formula

```
cost = (input_tokens × input_price / 1,000,000)
     + (output_tokens × output_price / 1,000,000)
```

**Worked example** — 2000 input + 500 output tokens on Claude Sonnet ($3/M in, $15/M out):

```
Input:  2,000 / 1,000,000 × $3.00  = $0.006
Output:   500 / 1,000,000 × $15.00 = $0.0075
Total: $0.0135 per request

At 10,000 requests/day: $135/day ≈ $4,000/month
```

### Pricing table (2025 — verify at provider sites)

| Model | Input (per 1M) | Output (per 1M) | Notes |
|-------|---------------|-----------------|-------|
| Claude Sonnet 4 | $3.00 | $15.00 | Anthropic mid-tier |
| Claude Haiku 3.5 | $0.80 | $4.00 | Fast and cheap |
| GPT-4o | $2.50 | $10.00 | OpenAI flagship |
| GPT-4o-mini | $0.15 | $0.60 | Budget powerhouse |
| GPT-4.1 | $2.00 | $8.00 | Coding-optimized |
| GPT-4.1-mini | $0.40 | $1.60 | Mid-range |
| GPT-4.1-nano | $0.10 | $0.40 | Cheapest OpenAI |
| Gemini 2.0 Flash | $0.10 | $0.40 | Google's fast model |

*Output tokens cost 3-5x more than input tokens.* This is because generating each output token requires a full forward pass through the model, while input tokens are processed in parallel.

### Cached token pricing

Providers offer discounts when input tokens are cached:

**Anthropic (opt-in):** Add `cache_control` breakpoints to messages. First use writes to cache (+25% cost). Subsequent reads cost 90% less. Cache TTL is 5 minutes, refreshed on each hit.

**OpenAI (automatic):** No opt-in needed. Repeated prefixes of 1024+ tokens are automatically cached at 50% discount.

### Cost optimization

1. **Use smaller models for simple tasks** — GPT-4o-mini ($0.15/M) vs GPT-4o ($2.50/M) is a 16x difference. Route classification, extraction, and simple Q&A to cheaper models.
2. **Shorter prompts** — every token costs money. Trim verbose system prompts, use concise formats.
3. **Set `max_tokens`** — prevents runaway generation. A model that goes on a tangent without a cap can generate thousands of expensive output tokens.
4. **Cache stable content** — system prompts and few-shot examples that don't change between requests are ideal for caching.

---

## 5. Error Handling

LLM APIs fail. Networks time out, rate limits trigger, models go down. Robust error handling is the difference between a demo and a production system.

### Error taxonomy

| HTTP | Error type | Cause | Retryable? |
|------|-----------|-------|------------|
| 400 | `invalid_request_error` | Malformed request, context overflow, bad params | No — fix the request |
| 401 | `authentication_error` | Invalid, expired, or missing API key | No — fix the key |
| 403 | `permission_error` | Key lacks required permissions | No — check permissions |
| 404 | `not_found_error` | Model doesn't exist | No — fix the model name |
| 429 | `rate_limit_error` | Too many requests (TPM or RPM exceeded) | **Yes** — wait and retry |
| 500 | `server_error` | Provider internal error | **Yes** — transient |
| 502 | Bad gateway | Network issue | **Yes** — transient |
| 503 | `overloaded_error` | Service overloaded | **Yes** — transient |
| 529 | `overloaded_error` | Anthropic API overloaded (Anthropic-specific) | **Yes** — transient |

**The rule:** 4xx errors (except 429) mean your request is wrong — retrying the same request will always fail. 429 and 5xx errors are transient — the same request may succeed later.

### Error response formats

**OpenAI:**
```json
{
  "error": {
    "message": "This model's maximum context length is 128000 tokens. However, your messages resulted in 130000 tokens.",
    "type": "invalid_request_error",
    "param": null,
    "code": "context_length_exceeded"
  }
}
```

**Anthropic:**
```json
{
  "type": "error",
  "error": {
    "type": "invalid_request_error",
    "message": "prompt is too long: 130000 tokens > 128000 maximum"
  }
}
```

### Context length exceeded

This is the most common production error. It happens when your input tokens + max_tokens exceeds the model's context window. Both providers include the actual token count in the error message, which is helpful for debugging.

**Detect before sending:** Count tokens client-side using `tiktoken` (for OpenAI) before making the API call. This is faster and cheaper than waiting for a 400 error.

### LiteLLM exception classes

LiteLLM maps provider-specific errors to unified exception classes:

```python
import litellm

try:
    response = litellm.completion(model="openai/gpt-4o", messages=[...])
except litellm.AuthenticationError:
    # 401 — bad API key
    print("Check your API key")
except litellm.NotFoundError:
    # 404 — model doesn't exist
    print("Check the model name")
except litellm.ContextWindowExceededError:
    # 400 — input too long (subclass of BadRequestError)
    print("Reduce input length")
except litellm.RateLimitError:
    # 429 — rate limited (retryable)
    print("Rate limited, retry after backoff")
except litellm.InternalServerError:
    # 500 — server error (retryable)
    print("Server error, retrying...")
except litellm.ServiceUnavailableError:
    # 503/529 — overloaded (retryable)
    print("Service unavailable, retrying...")
except litellm.Timeout:
    # Request timed out (retryable)
    print("Timeout, retrying...")
except litellm.APIConnectionError:
    # Network error (retryable)
    print("Connection error, retrying...")
```

`ContextWindowExceededError` is a subclass of `BadRequestError`, so you can catch it specifically or catch `BadRequestError` for all 400 errors.

---

## 6. Retries & Backoff

Transient errors (429, 500, 503, 529) are temporary — the same request will likely succeed if you wait and retry. But how you wait matters.

### Why not just retry immediately?

If the server is overloaded and you retry instantly, you add to the overload. If thousands of clients all retry instantly, the server never recovers. This is called the **thundering herd problem**.

### Exponential backoff

Instead of retrying immediately, wait longer between each attempt:

```
Attempt 1: wait ~1 second
Attempt 2: wait ~2 seconds
Attempt 3: wait ~4 seconds
Attempt 4: wait ~8 seconds
```

The formula:

```
delay = min(base_delay × 2^attempt, max_delay)
```

With `base_delay=1`, `max_delay=60`: delays are 1s, 2s, 4s, 8s, 16s, 32s, 60s, 60s...

### Why jitter is essential

Even with exponential backoff, if 1000 clients all start retrying at the same time (e.g., after an outage), they'll all retry at 1s, then 2s, then 4s — still hitting the server in synchronized waves.

**Jitter** adds randomness to break the synchronization:

```
delay = random(0, min(base_delay × 2^attempt, max_delay))
```

This is called **full jitter**. AWS's research showed it's the most effective strategy — it produces fewer total calls and faster completion times than any other approach.

Concrete example with full jitter (`base=1`, `max=60`):
```
Attempt 1: random between 0-1s    → e.g., 0.7s
Attempt 2: random between 0-2s    → e.g., 1.3s
Attempt 3: random between 0-4s    → e.g., 2.8s
Attempt 4: random between 0-8s    → e.g., 5.1s
```

### When NOT to retry

- **401 Authentication** — your key is wrong; retrying won't fix it
- **400 Bad Request** — the request itself is malformed
- **403 Permission** — access control issue, not transient
- **404 Not Found** — the model doesn't exist
- **Context overflow** — same oversized input will always fail
- **Content policy violation** — same content will always be flagged

### LiteLLM built-in retries

```python
import litellm

# Per-call
response = litellm.completion(
    model="openai/gpt-4o",
    messages=[...],
    num_retries=3  # retries only on retryable errors
)

# Global setting
litellm.num_retries = 3
```

### Practical recommendation

For production: **3-5 retries, exponential backoff with full jitter, base delay 1s, max delay 60s.** This handles transient failures while failing fast on permanent errors. Always set a total timeout so retries don't block indefinitely.

---

## 7. Rate Limits & Throttling

Providers limit how much you can use the API per minute to ensure fair access and prevent abuse. Understanding rate limits is essential for any application that makes more than a few calls.

### Two independent limits

| Limit | What it measures | Example |
|-------|-----------------|---------|
| **RPM** (Requests Per Minute) | Number of API calls | 500 RPM |
| **TPM** (Tokens Per Minute) | Total tokens (input + output) | 200,000 TPM |

**Whichever you hit first applies.** A few large requests can exhaust TPM while RPM is fine. Many tiny requests can exhaust RPM while TPM has plenty of room.

### Rate limit headers

Every response includes headers telling you your current limits and remaining budget.

**OpenAI headers:**

| Header | Example | Meaning |
|--------|---------|---------|
| `x-ratelimit-limit-requests` | `500` | Your RPM cap |
| `x-ratelimit-remaining-requests` | `499` | Requests remaining this window |
| `x-ratelimit-reset-requests` | `120ms` | Time until RPM resets |
| `x-ratelimit-limit-tokens` | `200000` | Your TPM cap |
| `x-ratelimit-remaining-tokens` | `199000` | Tokens remaining this window |
| `x-ratelimit-reset-tokens` | `5s` | Time until TPM resets |

**Anthropic headers** (same concept, different names):

| Header | Example |
|--------|---------|
| `anthropic-ratelimit-requests-limit` | `1000` |
| `anthropic-ratelimit-requests-remaining` | `999` |
| `anthropic-ratelimit-requests-reset` | `2025-01-01T00:01:00Z` |
| `anthropic-ratelimit-tokens-limit` | `80000` |
| `anthropic-ratelimit-tokens-remaining` | `79000` |
| `anthropic-ratelimit-tokens-reset` | `2025-01-01T00:01:00Z` |

Key difference: OpenAI uses relative durations (`120ms`, `5s`), Anthropic uses absolute ISO 8601 timestamps.

### Tier systems

Your rate limits depend on your account tier, which is based on how much you've paid.

**OpenAI tiers** (approximate, varies by model):

| Tier | Qualification | GPT-4o RPM | GPT-4o TPM |
|------|--------------|------------|------------|
| Free | Default | 3 | 40,000 |
| Tier 1 | $5 paid | 500 | 30,000 |
| Tier 2 | $50 paid + 7 days | 5,000 | 450,000 |
| Tier 3 | $100 paid + 7 days | 5,000 | 800,000 |
| Tier 4 | $250 paid + 14 days | 10,000 | 2,000,000 |
| Tier 5 | $1,000 paid + 30 days | 10,000 | 10,000,000 |

**Anthropic tiers:**

| Tier | Spend requirement | Claude Sonnet RPM | Claude Sonnet TPM |
|------|-------------------|-------------------|-------------------|
| Build (1) | $0 | 50 | 40,000 |
| Tier 2 | $40+ | 1,000 | 80,000 |
| Tier 3 | $200+ | 2,000 | 160,000 |
| Tier 4 | $400+ | 4,000 | 400,000 |

*Exact limits change frequently. Check the provider's rate limits documentation for current numbers.*

### When you hit a 429

1. Read the `retry-after` header — it tells you exactly how long to wait
2. If absent, use exponential backoff with jitter (Section 6)
3. Monitor `remaining-*` headers proactively — slow down before hitting the limit

### Strategies for high-throughput applications

1. **Spread requests evenly** — instead of bursting 500 requests, space them across the minute (~1 request every 120ms)
2. **Monitor remaining headers** — track `remaining-requests`/`remaining-tokens` on every response, back off as you approach zero
3. **Use smaller models** — cheaper models often have higher rate limits
4. **Batch API** — OpenAI offers a batch API for non-real-time workloads with higher limits and 50% cost discount

---

## 8. Latency & Performance

Understanding what makes LLM API calls slow helps you build responsive applications.

### Two-phase latency

LLM responses have two distinct phases:

```
|←— TTFT —→|←———— Generation ————→|
[send request]  [first token]  [last token]
```

- **TTFT (Time to First Token):** Time from sending the request to receiving the first token. Dominated by processing your input through the model.
- **Generation throughput (TPS):** Tokens generated per second after the first token. Relatively constant per model.

**Total time = TTFT + (output_tokens / TPS)**

### What affects latency

| Factor | TTFT impact | TPS impact | Notes |
|--------|------------|------------|-------|
| Input length | High — roughly linear | Minimal | 10K tokens takes ~5x longer to prefill than 2K |
| Model size | High | High | Larger models are slower at both phases |
| Output length | None | Determines total time | 1000 tokens at 80 TPS = 12.5s |
| Server load | High | Moderate | Peak hours can double TTFT |
| Geography | Moderate | Minimal | 100-300ms RTT from Asia/Europe to US servers |

### Typical ranges (approximate)

| Model | TTFT | TPS | 500-token response |
|-------|------|-----|-------------------|
| GPT-4o-mini | 0.2-0.5s | 80-120 | ~4-7s |
| GPT-4o | 0.3-1.0s | 50-80 | ~7-11s |
| Claude Haiku 3.5 | 0.2-0.5s | 80-120 | ~4-7s |
| Claude Sonnet 4 | 0.3-1.2s | 50-70 | ~8-11s |
| Gemini 2.0 Flash | 0.1-0.4s | 100-150 | ~3-5s |

### Streaming: perceived vs actual latency

Streaming does **not** make generation faster. The model produces tokens at the same rate. But it dramatically improves the user experience because they see the first token in 0.3-1s instead of waiting 7-15s for the complete response.

For chat UIs, streaming is essentially mandatory. For batch processing or when you need the full response before continuing, non-streaming is simpler.

### Connection pooling

Create client instances once and reuse them. Each new client creates a TLS handshake (~100-200ms overhead):

```python
# GOOD: create once at module level
client = openai.OpenAI()

# BAD: new client per request — TLS handshake every time
def handle_request():
    client = openai.OpenAI()
```

---

## 9. Best Practices for Production

### Timeouts

Always set explicit timeouts. Without them, a stuck request blocks forever.

```python
import httpx
import openai

client = openai.OpenAI(
    http_client=httpx.Client(
        timeout=httpx.Timeout(
            connect=5.0,     # Time to establish TCP connection
            read=120.0,      # Time to wait for response data
            write=10.0,      # Time to send request
            pool=10.0        # Time to wait for a connection from the pool
        )
    )
)
```

**Connect timeout (5-10s):** If the API server is unreachable, fail fast.
**Read timeout (30-120s):** Must be long enough for slow models and long outputs. A 2000-token response from a large model can take 30-60s.

### Fallback models

When the primary model fails with a transient error, try a secondary:

```python
models = ["anthropic/claude-sonnet-4-20250514", "openai/gpt-4o", "openai/gpt-4o-mini"]

for model in models:
    try:
        return completion(model=model, messages=messages)
    except (RateLimitError, InternalServerError, ServiceUnavailableError):
        continue  # try next model
    except (AuthenticationError, BadRequestError):
        raise  # don't fall back on permanent errors
raise RuntimeError("All models failed")
```

Only fall back on transient errors (429, 5xx). A 400 error means your request is wrong — it'll fail on every model.

### Logging

**What to log:**
```python
logger.info("llm_call", extra={
    "model": "gpt-4o",
    "input_tokens": 125,
    "output_tokens": 50,
    "cost_usd": 0.0008,
    "latency_ms": 1243,
    "status": "success",
    "request_id": response.id,
})
```

**What NOT to log:** Full prompt text (may contain PII), full responses (same concern), API keys (never). If you must log prompts for debugging, use a separate secure store with access controls and retention policies.

### Cost guardrails

1. **`max_tokens` on every request** — prevents runaway generation
2. **Pre-flight cost estimate** — before calling, calculate max cost: `(input_tokens × price + max_tokens × output_price)`. Reject if over threshold.
3. **Per-user budgets** — track cumulative cost per user/session
4. **Provider alerts** — set billing alerts at 50%, 80%, 100% of monthly budget in provider dashboards

### Circuit breaker pattern

When an API fails repeatedly, stop calling it temporarily:

```
CLOSED (normal)
  → failure count ≥ threshold →
OPEN (reject all calls immediately — don't wait for timeout)
  → wait cooldown period →
HALF_OPEN (allow one test call)
  → success → CLOSED
  → failure → OPEN
```

This prevents cascading failures and wasted timeout waits. Use a library like `pybreaker` or `tenacity` rather than building your own.

### Key management

| Environment | Method |
|-------------|--------|
| Local dev | `.env` files + `python-dotenv` |
| CI/CD | Platform secrets (GitHub Actions secrets, etc.) |
| Production | Secrets manager (AWS Secrets Manager, GCP Secret Manager, Vault) |

Never hardcode keys. Rotate on a schedule (every 30-90 days) and immediately if exposed. Use separate keys per environment (dev/staging/prod).
