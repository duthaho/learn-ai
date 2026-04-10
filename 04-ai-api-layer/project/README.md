# Project: LLM API Explorer

## What you'll build

A command-line tool that probes your API setup, sends prompts to multiple models, compares responses side-by-side with cost and latency tracking, handles errors gracefully with retries, and prints a usage summary. This is the tool you'll wish you had every time you evaluate a new model or debug an API integration.

## Prerequisites

- Completed reading the Module 04 README
- Python 3.11+ with project dependencies installed (`pip install -r requirements.txt`)
- At least one LLM provider API key configured in `.env`

## How to build

Create a new file `api_explorer.py` in this directory. Build it step by step following the instructions below. When you're done, compare your output with `python solution.py`.

## Steps

### Step 1: Setup and API health check

Set up imports, load environment variables, and define the list of models to test.

```python
import os
import time
import random
from pathlib import Path
from dotenv import load_dotenv
from litellm import completion, completion_cost
import litellm

load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")
```

Define a list of models to probe. Include models from providers whose API keys you have:

```python
MODELS = [
    "anthropic/claude-sonnet-4-20250514",
    "anthropic/claude-haiku-3-5-20241022",
    "openai/gpt-4o",
    "openai/gpt-4o-mini",
]
```

Write a `probe_model(model)` function that:
- Sends a minimal request: `messages=[{"role": "user", "content": "Say OK"}]` with `max_tokens=5`
- Measures latency using `time.monotonic()`
- Returns a dict with `model`, `status` ("ok" or "error"), `latency_ms`, and `error` (None or the error message)
- Wraps the call in try/except to catch any `Exception`

Write a `health_check(models)` function that:
- Calls `probe_model` for each model
- Prints a formatted table with model name, status (✓ OK or ✗ ERROR), and latency
- Returns only the models that passed (status "ok")

Test: run the health check and see which of your configured models respond.

### Step 2: Single model call with response parsing

Write an `ask()` function that makes an API call and returns a structured result. This is the core helper used by all subsequent steps.

```python
def ask(prompt, model, system="", temperature=0.0, max_tokens=1024):
```

The function should:
- Build the messages list (add system message if provided, then user message)
- Record start time with `time.monotonic()`
- Call `litellm.completion()` with the given parameters
- Calculate latency in milliseconds
- Extract from the response: content text, input tokens, output tokens, total tokens, finish reason
- Calculate cost using `litellm.completion_cost(completion_response=response)`
- Return a dict with: `model`, `content`, `input_tokens`, `output_tokens`, `total_tokens`, `cost`, `latency_ms`, `finish_reason`

Demo it: call `ask()` with the prompt `"What is Python? Answer in one sentence."` using your first healthy model. Print all fields formatted nicely.

### Step 3: Multi-model comparison

Write a `compare_models()` function that sends the same prompt to multiple models and displays results side-by-side.

```python
def compare_models(prompt, models, system=""):
```

The function should:
- Call `ask()` for each model in the list, collecting results
- If a call fails, record the error instead of crashing
- Sort results by latency
- Print a comparison table showing: model name, first 80 chars of response, input tokens, output tokens, cost, latency
- Highlight the fastest and cheapest model

Test with the prompt: `"Explain Python list comprehensions in 2 sentences."` across all healthy models.

### Step 4: Error handling

Write a `classify_error()` function that takes an exception and returns a structured error report:

```python
def classify_error(error):
```

It should return a dict with: `error_type` (class name), `retryable` (bool), `message` (error message), `action` (suggested fix).

Classification rules:
- `AuthenticationError` → not retryable → "Check your API key in .env"
- `NotFoundError` → not retryable → "Verify the model name"
- `ContextWindowExceededError` → not retryable → "Reduce input length or use a larger context model"
- `RateLimitError` → retryable → "Wait and retry with backoff"
- `InternalServerError`, `ServiceUnavailableError` → retryable → "Transient server error, retry"
- `Timeout`, `APIConnectionError` → retryable → "Network issue, retry"
- Anything else → not retryable → "Unexpected error"

Write a `demo_error_handling()` function that deliberately triggers errors and handles them:
- Bad API key: call with `api_key="sk-invalid-key-for-testing"`
- Nonexistent model: call with `model="openai/gpt-nonexistent"`
- For each, catch the exception, classify it, and print: test name, error type, retryable status, suggested action

### Step 5: Retry with exponential backoff

Write a `retry_with_backoff()` function:

```python
def retry_with_backoff(fn, max_retries=3, base_delay=1.0, max_delay=60.0):
```

The function should:
- Call `fn()` in a loop up to `max_retries + 1` times
- On retryable errors (`RateLimitError`, `InternalServerError`, `ServiceUnavailableError`, `Timeout`, `APIConnectionError`), calculate delay using full jitter: `random.uniform(0, min(base_delay * 2 ** attempt, max_delay))`
- Print each retry attempt and the delay
- On non-retryable errors, raise immediately
- On success, return the result

Demo it: wrap a normal `ask()` call in `retry_with_backoff` and show it succeeding on the first try (no retry needed).

### Step 6: Usage summary

Create a `UsageTracker` class that records all API calls and produces a summary.

```python
class UsageTracker:
    def __init__(self):
        self.calls = []
```

Methods:
- `record(result)` — append a result dict to the calls list
- `record_error(model, error_type)` — record a failed call
- `summary()` — return a dict with: total_calls, total_input_tokens, total_output_tokens, total_tokens, total_cost, avg_latency_ms, error_count, error_rate, fastest_model, cheapest_model

Wire the tracker into all the previous steps — after each `ask()` call, record the result. After the error demos, record those too.

Write a `print_summary(tracker)` function that formats and prints the report.

## How to run

```bash
python api_explorer.py
```

Or compare with the reference:

```bash
python solution.py
```

## Expected output

```
============================================================
  LLM API Explorer
============================================================

--- 1. API Health Check ---

  Model                                    Status    Latency
  anthropic/claude-sonnet-4-20250514       ✓ OK      823ms
  anthropic/claude-haiku-3-5-20241022      ✓ OK      412ms
  openai/gpt-4o                            ✓ OK      587ms
  openai/gpt-4o-mini                       ✓ OK      245ms

--- 2. Single Call Demo ---

  Model:         openai/gpt-4o-mini
  Response:      Python is a high-level, interpreted programming...
  Tokens:        18 in / 32 out / 50 total
  Cost:          $0.0000
  Latency:       487ms
  Finish reason: stop

--- 3. Multi-Model Comparison ---

  Prompt: "Explain Python list comprehensions in 2 sentences"

  Model                                  Tokens (in/out)  Cost       Latency
  openai/gpt-4o-mini                     14 / 42          $0.0000    312ms
  openai/gpt-4o                          14 / 48          $0.0005    587ms
  anthropic/claude-haiku-3-5-20241022    14 / 45          $0.0002    612ms
  anthropic/claude-sonnet-4-20250514     14 / 51          $0.0008    923ms

  Fastest: openai/gpt-4o-mini (312ms)
  Cheapest: openai/gpt-4o-mini ($0.0000)

--- 4. Error Handling ---

  Test: bad API key
    Error:     AuthenticationError
    Retryable: NO
    Action:    Check your API key in .env

  Test: nonexistent model
    Error:     NotFoundError
    Retryable: NO
    Action:    Verify the model name

--- 5. Retry Demo ---

  Attempt 1: success (no retry needed)

--- 6. Usage Summary ---

  Total calls:     8
  Total tokens:    542 (112 in + 430 out)
  Total cost:      $0.0024
  Avg latency:     587ms
  Errors:          2 (25.0%)
  Fastest model:   openai/gpt-4o-mini (312ms avg)
  Cheapest model:  openai/gpt-4o-mini ($0.0000 avg)

============================================================
  Done!
============================================================
```

## Stretch goals

1. **Add streaming** — modify `ask()` to support `stream=True`, print tokens as they arrive, measure TTFT vs total time. (Teaser for Module 05.)
2. **Export results** — save the comparison and summary to a JSON file for later analysis.
3. **Add a model** — add a Gemini or Groq model to the probe list, get a key, and see how it compares.
