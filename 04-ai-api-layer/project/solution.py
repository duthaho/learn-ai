"""
LLM API Explorer — Module 04 Project (Solution)

A CLI tool that probes your API setup, compares models side-by-side,
tracks costs and latency, and handles errors with retries.

Run: python solution.py
"""

import os
import time
import random
from pathlib import Path
from dotenv import load_dotenv
from litellm import completion, completion_cost
import litellm

load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")


# ---------------------------------------------------------------------------
# Models to test
# ---------------------------------------------------------------------------

MODELS = [
    "anthropic/claude-sonnet-4-20250514",
    "anthropic/claude-haiku-3-5-20241022",
    "openai/gpt-4o",
    "openai/gpt-4o-mini",
]


# ---------------------------------------------------------------------------
# Step 1: API Health Check
# ---------------------------------------------------------------------------

def probe_model(model: str) -> dict:
    """Send a tiny probe to a model and return status info."""
    try:
        start = time.monotonic()
        completion(
            model=model,
            messages=[{"role": "user", "content": "Say OK"}],
            max_tokens=5,
        )
        latency_ms = int((time.monotonic() - start) * 1000)
        return {"model": model, "status": "ok", "latency_ms": latency_ms, "error": None}
    except Exception as e:
        return {"model": model, "status": "error", "latency_ms": None, "error": str(e)[:80]}


def health_check(models: list[str]) -> list[str]:
    """Probe all models and return the list of healthy ones."""
    print("--- 1. API Health Check ---\n")
    print(f"  {'Model':<45} {'Status':<10} {'Latency'}")
    healthy = []
    for model in models:
        result = probe_model(model)
        if result["status"] == "ok":
            print(f"  {model:<45} OK        {result['latency_ms']}ms")
            healthy.append(model)
        else:
            print(f"  {model:<45} ERROR     ---")
    print()
    return healthy


# ---------------------------------------------------------------------------
# Step 2: Single Model Call
# ---------------------------------------------------------------------------

def ask(
    prompt: str,
    model: str,
    system: str = "",
    temperature: float = 0.0,
    max_tokens: int = 1024,
) -> dict:
    """Make an LLM API call and return a structured result."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    start = time.monotonic()
    response = completion(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    latency_ms = int((time.monotonic() - start) * 1000)

    choice = response.choices[0]
    usage = response.usage

    return {
        "model": model,
        "content": choice.message.content,
        "input_tokens": usage.prompt_tokens,
        "output_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens,
        "cost": completion_cost(completion_response=response),
        "latency_ms": latency_ms,
        "finish_reason": choice.finish_reason,
    }


def demo_single_call(model: str, tracker: "UsageTracker") -> None:
    """Demonstrate a single API call with full response parsing."""
    print("--- 2. Single Call Demo ---\n")
    result = ask("What is Python? Answer in one sentence.", model)
    tracker.record(result)
    print(f"  Model:         {result['model']}")
    print(f"  Response:      {result['content'][:70]}...")
    print(f"  Tokens:        {result['input_tokens']} in / {result['output_tokens']} out / {result['total_tokens']} total")
    print(f"  Cost:          ${result['cost']:.4f}")
    print(f"  Latency:       {result['latency_ms']:,}ms")
    print(f"  Finish reason: {result['finish_reason']}")
    print()


# ---------------------------------------------------------------------------
# Step 3: Multi-Model Comparison
# ---------------------------------------------------------------------------

def compare_models(
    prompt: str,
    models: list[str],
    system: str = "",
    tracker: "UsageTracker | None" = None,
) -> list[dict]:
    """Send the same prompt to multiple models and compare results."""
    results = []
    for model in models:
        try:
            result = ask(prompt, model, system=system)
            results.append(result)
            if tracker:
                tracker.record(result)
        except Exception as e:
            results.append({"model": model, "error": str(e)[:80]})
            if tracker:
                tracker.record_error(model, type(e).__name__)

    # Sort by latency (errors at the end)
    results.sort(key=lambda r: r.get("latency_ms", float("inf")))
    return results


def print_comparison(prompt: str, results: list[dict]) -> None:
    """Print a formatted comparison table."""
    print("--- 3. Multi-Model Comparison ---\n")
    print(f'  Prompt: "{prompt}"\n')
    print(f"  {'Model':<45} {'Tokens (in/out)':<18} {'Cost':<11} {'Latency'}")

    successful = []
    for r in results:
        if "error" in r:
            print(f"  {r['model']:<45} ERROR: {r['error'][:40]}")
        else:
            tokens = f"{r['input_tokens']} / {r['output_tokens']}"
            print(f"  {r['model']:<45} {tokens:<18} ${r['cost']:<10.4f} {r['latency_ms']}ms")
            successful.append(r)

    if successful:
        fastest = min(successful, key=lambda r: r["latency_ms"])
        cheapest = min(successful, key=lambda r: r["cost"])
        print(f"\n  Fastest:  {fastest['model']} ({fastest['latency_ms']}ms)")
        print(f"  Cheapest: {cheapest['model']} (${cheapest['cost']:.4f})")
    print()


# ---------------------------------------------------------------------------
# Step 4: Error Handling
# ---------------------------------------------------------------------------

def classify_error(error: Exception) -> dict:
    """Classify an LLM error as retryable or fatal with a suggested action."""
    error_map = {
        "AuthenticationError": (False, "Check your API key in .env"),
        "NotFoundError": (False, "Verify the model name"),
        "ContextWindowExceededError": (False, "Reduce input length or use a larger context model"),
        "BadRequestError": (False, "Fix the request parameters"),
        "ContentPolicyViolationError": (False, "Modify the prompt content"),
        "RateLimitError": (True, "Wait and retry with backoff"),
        "InternalServerError": (True, "Transient server error, retry"),
        "ServiceUnavailableError": (True, "Service overloaded, retry"),
        "Timeout": (True, "Request timed out, retry"),
        "APIConnectionError": (True, "Network issue, check connection and retry"),
    }

    error_type = type(error).__name__
    retryable, action = error_map.get(error_type, (False, "Unexpected error"))

    return {
        "error_type": error_type,
        "retryable": retryable,
        "message": str(error)[:120],
        "action": action,
    }


def demo_error_handling(tracker: "UsageTracker") -> None:
    """Deliberately trigger common errors and handle them."""
    print("--- 4. Error Handling ---\n")

    tests = [
        ("bad API key", {"model": "openai/gpt-4o", "messages": [{"role": "user", "content": "test"}], "max_tokens": 5, "api_key": "sk-invalid-key-for-testing"}),
        ("nonexistent model", {"model": "openai/gpt-nonexistent", "messages": [{"role": "user", "content": "test"}], "max_tokens": 5}),
    ]

    for test_name, kwargs in tests:
        try:
            completion(**kwargs)
            print(f"  Test: {test_name} — unexpectedly succeeded")
        except Exception as e:
            info = classify_error(e)
            tracker.record_error(kwargs["model"], info["error_type"])
            print(f"  Test: {test_name}")
            print(f"    Error:     {info['error_type']}")
            print(f"    Retryable: {'YES' if info['retryable'] else 'NO'}")
            print(f"    Action:    {info['action']}")
            print()
    print()


# ---------------------------------------------------------------------------
# Step 5: Retry with Exponential Backoff
# ---------------------------------------------------------------------------

RETRYABLE_ERRORS = (
    litellm.RateLimitError,
    litellm.InternalServerError,
    litellm.ServiceUnavailableError,
    litellm.Timeout,
    litellm.APIConnectionError,
)


def retry_with_backoff(
    fn,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
):
    """Retry a function with exponential backoff and full jitter."""
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except RETRYABLE_ERRORS as e:
            if attempt == max_retries:
                raise
            delay = random.uniform(0, min(base_delay * 2**attempt, max_delay))
            print(f"    Attempt {attempt + 1} failed: {type(e).__name__}. Retrying in {delay:.1f}s...")
            time.sleep(delay)
        except Exception:
            raise  # Non-retryable errors: fail immediately


def demo_retry(model: str, tracker: "UsageTracker") -> None:
    """Demonstrate the retry mechanism."""
    print("--- 5. Retry Demo ---\n")

    def make_call():
        return ask("Say hello in one word.", model, max_tokens=10)

    result = retry_with_backoff(make_call)
    tracker.record(result)
    print(f"  Attempt 1: success (no retry needed)")
    print()


# ---------------------------------------------------------------------------
# Step 6: Usage Summary
# ---------------------------------------------------------------------------

class UsageTracker:
    """Track all API calls and produce a summary."""

    def __init__(self):
        self.calls = []
        self.errors = []

    def record(self, result: dict) -> None:
        """Record a successful API call."""
        self.calls.append(result)

    def record_error(self, model: str, error_type: str) -> None:
        """Record a failed API call."""
        self.errors.append({"model": model, "error_type": error_type})

    def summary(self) -> dict:
        """Produce a summary of all recorded calls."""
        total_calls = len(self.calls) + len(self.errors)
        if not self.calls:
            return {
                "total_calls": total_calls,
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "total_tokens": 0,
                "total_cost": 0,
                "avg_latency_ms": 0,
                "error_count": len(self.errors),
                "error_rate": 100.0 if total_calls > 0 else 0,
                "fastest_model": None,
                "cheapest_model": None,
            }

        total_input = sum(c["input_tokens"] for c in self.calls)
        total_output = sum(c["output_tokens"] for c in self.calls)
        total_cost = sum(c["cost"] for c in self.calls)
        avg_latency = sum(c["latency_ms"] for c in self.calls) / len(self.calls)

        # Find fastest and cheapest by average per model
        model_stats = {}
        for c in self.calls:
            m = c["model"]
            if m not in model_stats:
                model_stats[m] = {"latencies": [], "costs": []}
            model_stats[m]["latencies"].append(c["latency_ms"])
            model_stats[m]["costs"].append(c["cost"])

        fastest = min(model_stats, key=lambda m: sum(model_stats[m]["latencies"]) / len(model_stats[m]["latencies"]))
        cheapest = min(model_stats, key=lambda m: sum(model_stats[m]["costs"]) / len(model_stats[m]["costs"]))
        fastest_avg = int(sum(model_stats[fastest]["latencies"]) / len(model_stats[fastest]["latencies"]))
        cheapest_avg = sum(model_stats[cheapest]["costs"]) / len(model_stats[cheapest]["costs"])

        return {
            "total_calls": total_calls,
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "total_tokens": total_input + total_output,
            "total_cost": total_cost,
            "avg_latency_ms": int(avg_latency),
            "error_count": len(self.errors),
            "error_rate": len(self.errors) / total_calls * 100 if total_calls > 0 else 0,
            "fastest_model": fastest,
            "fastest_avg_ms": fastest_avg,
            "cheapest_model": cheapest,
            "cheapest_avg_cost": cheapest_avg,
        }


def print_summary(tracker: UsageTracker) -> None:
    """Print a formatted usage summary."""
    print("--- 6. Usage Summary ---\n")
    s = tracker.summary()
    print(f"  Total calls:     {s['total_calls']}")
    print(f"  Total tokens:    {s['total_tokens']:,} ({s['total_input_tokens']} in + {s['total_output_tokens']} out)")
    print(f"  Total cost:      ${s['total_cost']:.4f}")
    print(f"  Avg latency:     {s['avg_latency_ms']:,}ms")
    print(f"  Errors:          {s['error_count']} ({s['error_rate']:.1f}%)")
    if s.get("fastest_model"):
        print(f"  Fastest model:   {s['fastest_model']} ({s['fastest_avg_ms']}ms avg)")
    if s.get("cheapest_model"):
        print(f"  Cheapest model:  {s['cheapest_model']} (${s.get('cheapest_avg_cost', 0):.4f} avg)")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  LLM API Explorer")
    print("=" * 60)
    print()

    tracker = UsageTracker()

    # Step 1: Health check
    healthy_models = health_check(MODELS)
    if not healthy_models:
        print("  No models available. Check your API keys in .env")
        return

    # Step 2: Single call demo
    demo_single_call(healthy_models[0], tracker)

    # Step 3: Multi-model comparison
    prompt = "Explain Python list comprehensions in 2 sentences"
    results = compare_models(prompt, healthy_models, tracker=tracker)
    print_comparison(prompt, results)

    # Step 4: Error handling
    demo_error_handling(tracker)

    # Step 5: Retry demo
    demo_retry(healthy_models[0], tracker)

    # Step 6: Usage summary
    print_summary(tracker)

    print("=" * 60)
    print("  Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
