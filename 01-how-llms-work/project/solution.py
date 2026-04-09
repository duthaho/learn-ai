"""
Token Budget Calculator — Module 01 Project (Solution)

A CLI tool that analyzes prompts before you send them to an LLM.
Tokenizes text, estimates costs, checks context window budgets,
and compares prompt format efficiency.

Run: python solution.py
"""

import os
import tiktoken
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")


# ---------------------------------------------------------------------------
# Model pricing and context windows
# ---------------------------------------------------------------------------

MODELS = {
    "claude-sonnet": {
        "context_window": 200_000,
        "input_cost_per_million": 3.00,
        "output_cost_per_million": 15.00,
    },
    "claude-haiku": {
        "context_window": 200_000,
        "input_cost_per_million": 0.25,
        "output_cost_per_million": 1.25,
    },
    "gpt-4o": {
        "context_window": 128_000,
        "input_cost_per_million": 2.50,
        "output_cost_per_million": 10.00,
    },
    "gpt-4o-mini": {
        "context_window": 128_000,
        "input_cost_per_million": 0.15,
        "output_cost_per_million": 0.60,
    },
}


def get_encoder() -> tiktoken.Encoding:
    """Get a tiktoken encoder for token counting."""
    return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str, encoder: tiktoken.Encoding) -> int:
    """Count the number of tokens in a text string."""
    return len(encoder.encode(text))


def tokenize_and_display(text: str, encoder: tiktoken.Encoding) -> dict:
    """Tokenize text and return a detailed breakdown."""
    token_ids = encoder.encode(text)
    tokens = [encoder.decode([tid]) for tid in token_ids]
    return {
        "token_count": len(token_ids),
        "tokens": tokens,
        "token_ids": token_ids,
        "chars_per_token": len(text) / len(token_ids) if token_ids else 0,
    }


def estimate_cost(input_tokens: int, output_tokens: int, model_name: str) -> dict:
    """Estimate the cost of an LLM API call."""
    model = MODELS[model_name]
    input_cost = input_tokens * model["input_cost_per_million"] / 1_000_000
    output_cost = output_tokens * model["output_cost_per_million"] / 1_000_000
    return {
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": input_cost + output_cost,
        "model": model_name,
    }


def check_context_budget(
    system_prompt: str,
    user_prompt: str,
    max_output_tokens: int,
    model_name: str,
    encoder: tiktoken.Encoding,
) -> dict:
    """Check if a prompt fits within a model's context window."""
    system_tokens = count_tokens(system_prompt, encoder)
    user_tokens = count_tokens(user_prompt, encoder)
    total_input = system_tokens + user_tokens
    total_needed = total_input + max_output_tokens
    context_window = MODELS[model_name]["context_window"]

    return {
        "system_tokens": system_tokens,
        "user_tokens": user_tokens,
        "total_input": total_input,
        "max_output": max_output_tokens,
        "total_needed": total_needed,
        "context_window": context_window,
        "fits": total_needed <= context_window,
        "utilization": total_input / context_window * 100,
        "remaining": context_window - total_input,
    }


def compare_formats(texts: dict[str, str], encoder: tiktoken.Encoding) -> list[dict]:
    """Compare token counts across different text formats."""
    results = []
    for name, text in texts.items():
        token_count = count_tokens(text, encoder)
        results.append({
            "format": name,
            "tokens": token_count,
            "chars": len(text),
            "chars_per_token": len(text) / token_count if token_count else 0,
        })
    results.sort(key=lambda x: x["tokens"])
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    encoder = get_encoder()

    print("=" * 60)
    print("  Token Budget Calculator")
    print("=" * 60)

    print("\n--- 1. Tokenization Breakdown ---\n")
    examples = [
        "Hello, world!",
        "unhappiness",
        '{"user_name": "John", "age": 30}',
        "def fibonacci(n):\n    return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)",
    ]
    for text in examples:
        result = tokenize_and_display(text, encoder)
        print(f'  "{text[:50]}"')
        print(f'    Tokens: {result["token_count"]} | {result["tokens"][:6]}{"..." if len(result["tokens"]) > 6 else ""}')
        print(f'    Ratio: {result["chars_per_token"]:.1f} chars/token')
        print()

    print("--- 2. Cost Estimation ---\n")
    input_tokens = 1000
    output_tokens = 500
    for model_name in MODELS:
        cost = estimate_cost(input_tokens, output_tokens, model_name)
        print(f'  {model_name}: ${cost["total_cost"]:.4f} ({input_tokens} in + {output_tokens} out)')
    print()

    print("--- 3. Context Budget Check ---\n")
    system = "You are a helpful assistant that answers questions about Python programming."
    user = "Explain decorators in Python with examples. " * 50
    budget = check_context_budget(system, user, max_output_tokens=2000, model_name="claude-sonnet", encoder=encoder)
    print(f'  System prompt:   {budget["system_tokens"]:,} tokens')
    print(f'  User prompt:     {budget["user_tokens"]:,} tokens')
    print(f'  Total input:     {budget["total_input"]:,} tokens')
    print(f'  Output reserved: {budget["max_output"]:,} tokens')
    print(f'  Context window:  {budget["context_window"]:,} tokens')
    print(f'  Fits: {"YES" if budget["fits"] else "NO — OVER BUDGET!"}')
    print(f'  Utilization:     {budget["utilization"]:.1f}%')
    print(f'  Remaining:       {budget["remaining"]:,} tokens')
    print()

    print("--- 4. Format Comparison ---\n")
    formats = {
        "Verbose JSON": '{"user_full_name": "John Doe", "user_email_address": "john@example.com", "user_age_years": 30}',
        "Compact JSON": '{"name":"John Doe","email":"john@example.com","age":30}',
        "Key-Value": "name:John Doe|email:john@example.com|age:30",
    }
    results = compare_formats(formats, encoder)
    for r in results:
        print(f'  {r["tokens"]:3d} tokens | {r["format"]}')
    if len(results) >= 2:
        ratio = results[-1]["tokens"] / results[0]["tokens"]
        print(f'\n  Most verbose is {ratio:.1f}x more expensive than most compact.')

    print("\n" + "=" * 60)
    print("  Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
