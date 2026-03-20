"""
Test script to exercise each endpoint and observe LLM mechanics.

Run the server first:  uvicorn app:app --reload
Then run this:         python test_concepts.py
"""

import httpx

BASE = "http://localhost:8000"


def section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def main():
    client = httpx.Client(base_url=BASE, timeout=60)

    # --- 1. Tokenization ---
    section("1. TOKENIZATION — How text becomes tokens")

    examples = [
        "Hello, world!",
        "unhappiness",
        '{"user_name": "John", "user_age": 30}',  # verbose JSON
        '{"n":"John","a":30}',  # compact JSON
        "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
    ]

    for text in examples:
        resp = client.post("/tokenize", json={"text": text}).json()
        print(f"\nText: {text[:60]}...")
        print(f"  Tokens ({resp['token_count']}): {resp['tokens']}")
        print(f"  {resp['cost_insight']}")

    # --- 2. Token cost comparison ---
    section("2. TOKEN COST COMPARISON — Why format matters")

    resp = client.post("/compare-tokens", json={
        "texts": [
            '{"user_full_name": "John Doe", "user_email_address": "john@example.com", "user_age_years": 30}',
            '{"name":"John Doe","email":"john@example.com","age":30}',
            "name:John Doe|email:john@example.com|age:30",
        ]
    }).json()
    for item in resp["comparisons"]:
        print(f"  {item['token_count']:3d} tokens: {item['text_preview']}")

    # --- 3. Context window check ---
    section("3. CONTEXT WINDOW — Pre-flight check before API call")

    resp = client.post("/context-check", json={
        "prompt": "Explain transformers." * 100,
        "system": "You are an AI tutor.",
        "max_tokens": 500,
    }).json()
    for k, v in resp.items():
        print(f"  {k}: {v}")

    # --- 4. Temperature effects ---
    section("4. TEMPERATURE — Same prompt, different randomness")

    resp = client.post("/temperature-demo", json={
        "prompt": "Complete this sentence in exactly 10 words: 'The robot walked into the bar and'",
        "temperatures": [0.0, 0.5, 1.0],
        "max_tokens": 50,
    }).json()
    for r in resp["results"]:
        print(f"  T={r['temperature']}: {r['output'][:100]}")

    # --- 5. Streaming vs non-streaming ---
    section("5. STREAMING — Token-by-token autoregressive output")

    print("\n  Non-streaming (full response at once):")
    resp = client.post("/generate", json={
        "prompt": "Count from 1 to 10, one number per line.",
        "max_tokens": 100,
        "temperature": 0.0,
    }).json()
    print(f"  {resp['content'][:200]}")
    print(f"  Usage: {resp['usage']}")

    print("\n  Streaming (token by token):")
    print("  ", end="")
    with client.stream("POST", "/generate/stream", json={
        "prompt": "Count from 1 to 5, one number per line.",
        "max_tokens": 100,
        "temperature": 0.0,
    }) as stream:
        for chunk in stream.iter_text():
            print(chunk, end="", flush=True)
    print()

    # --- 6. Structured output ---
    section("6. STRUCTURED OUTPUT — Constraining the distribution")

    diff = """\
- def get_user(id):
-     query = f"SELECT * FROM users WHERE id = {id}"
-     return db.execute(query)
+ def get_user(user_id):
+     query = "SELECT * FROM users WHERE id = %s"
+     return db.execute(query, (user_id,))
"""
    resp = client.post("/review", json={"code_diff": diff}).json()
    print(f"  Parse success: {resp['parse_success']}")
    print(f"  Usage: {resp['usage']}")
    if resp["parsed"]:
        for item in resp["parsed"]:
            print(f"  [{item.get('severity', '?')}] Line {item.get('line', '?')}: {item.get('issue', '?')}")

    section("DONE — All concepts demonstrated!")


if __name__ == "__main__":
    main()
