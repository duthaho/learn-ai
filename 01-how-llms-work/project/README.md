# Project: Token Budget Calculator

## What you'll build

A command-line tool that analyzes prompts before you send them to an LLM. It tokenizes text, estimates API costs across different models, checks if prompts fit within context windows, and compares format efficiency. This is a tool you'll actually use when building AI applications — every production system needs token budget awareness.

## Prerequisites

- Completed reading the Module 01 README
- Python 3.11+ with the project dependencies installed (`pip install -r requirements.txt`)

## How to build

Create a new file `token_budget.py` in this directory. Build it step by step following the instructions below. When you're done, compare your output with `python solution.py`.

## Steps

### Step 1: Set up the file and model data

Create `token_budget.py`. Add imports and a dictionary of model pricing:

```python
import tiktoken

MODELS = {
    "claude-sonnet": {"context_window": 200_000, "input_cost_per_million": 3.00, "output_cost_per_million": 15.00},
    "claude-haiku":  {"context_window": 200_000, "input_cost_per_million": 0.25, "output_cost_per_million": 1.25},
    "gpt-4o":        {"context_window": 128_000, "input_cost_per_million": 2.50, "output_cost_per_million": 10.00},
    "gpt-4o-mini":   {"context_window": 128_000, "input_cost_per_million": 0.15, "output_cost_per_million": 0.60},
}
```

### Step 2: Tokenize text

Write a function that takes a string and returns a breakdown of its tokens:
- Token count
- The actual token strings (decoded from IDs)
- Characters-per-token ratio

Use `tiktoken.get_encoding("cl100k_base")` to get an encoder. Use `encoder.encode(text)` to get token IDs, and `encoder.decode([single_id])` to get each token's string.

Test: `"Hello, world!"` should produce 4 tokens. `"unhappiness"` should produce 3.

### Step 3: Estimate API costs

Write a function that takes input token count, output token count, and a model name, then calculates the cost:
- `input_cost = input_tokens * price_per_million / 1_000_000`
- `output_cost = output_tokens * price_per_million / 1_000_000`

Test: 1000 input + 500 output on `claude-sonnet` should cost $0.0105.

### Step 4: Check context budget

Write a function that takes a system prompt, user prompt, max output tokens, and model name, then checks whether everything fits in the context window:
- Count tokens in each prompt
- Total needed = input tokens + max output tokens
- Compare against the model's context window
- Return whether it fits, utilization %, and remaining tokens

Test: a short system prompt + long user prompt + 2000 max output on claude-sonnet (200K context) should fit easily.

### Step 5: Compare format efficiency

Write a function that takes a dict of format names to text strings, counts tokens for each, and returns them sorted by efficiency (fewest tokens first).

Test with:
- Verbose JSON: `'{"user_full_name": "John Doe", "user_email_address": "john@example.com", "user_age_years": 30}'`
- Compact JSON: `'{"name":"John Doe","email":"john@example.com","age":30}'`
- Key-Value: `"name:John Doe|email:john@example.com|age:30"`

The key-value format should be most efficient.

### Step 6: Wire it all together

Write a `main()` function that demonstrates all four features: tokenization breakdown on several example texts, cost estimation across all models, a context budget check, and a format comparison. Run it and compare with `python solution.py`.

## Expected output

```
============================================================
  Token Budget Calculator
============================================================

--- 1. Tokenization Breakdown ---

  "Hello, world!"
    Tokens: 4 | ['Hello', ',', ' world', '!']
    Ratio: 3.2 chars/token

  "unhappiness"
    Tokens: 3 | [...]
    ...

--- 2. Cost Estimation ---

  claude-sonnet: $0.0105 (1000 in + 500 out)
  claude-haiku: $0.0009 (1000 in + 500 out)
  ...

--- 3. Context Budget Check ---

  System prompt:   12 tokens
  User prompt:     ... tokens
  ...
  Fits: YES
  ...

--- 4. Format Comparison ---

  ... tokens | Key-Value
  ... tokens | Compact JSON
  ... tokens | Verbose JSON
  ...
```

## Stretch goals

1. **Add a new model** — add Gemini or Mistral to the MODELS dict with their real pricing. Re-run to see how costs compare.
2. **Read from file** — accept a file path argument and analyze a prompt stored in a `.txt` file.
3. **Budget warning** — add a function that takes a monthly budget (e.g., $100) and a requests-per-day estimate, and warns if the projected cost exceeds the budget.
