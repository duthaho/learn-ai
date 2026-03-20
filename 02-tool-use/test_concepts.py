"""
Test script to exercise each endpoint and observe tool use mechanics.

Run the server first:  uvicorn app:app --reload --port 8001
Then run this:         python test_concepts.py
"""

import httpx

BASE = "http://localhost:8001"


def section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def main():
    client = httpx.Client(base_url=BASE, timeout=120)

    # --- 1. Inspect tool definitions ---
    section("1. TOOL DEFINITIONS — What the model sees")

    resp = client.get("/tools").json()
    print(f"\n  {resp['count']} tools registered:")
    for tool in resp["tools"]:
        required = tool["input_schema"].get("required", [])
        optional = [
            k for k in tool["input_schema"]["properties"]
            if k not in required
        ]
        print(f"\n  Tool: {tool['name']}")
        print(f"    Description: {tool['description'][:80]}...")
        print(f"    Required params: {required}")
        print(f"    Optional params: {optional}")
    print(f"\n  Tip: {resp['tip']}")

    # --- 2. Single-turn: weather query (triggers get_weather) ---
    section("2. WEATHER QUERY — Model calls get_weather tool")

    resp = client.post("/chat", json={
        "message": "What's the weather like in Tokyo right now?",
    }).json()
    print(f"\n  Response: {resp['response'][:200]}")
    print(f"  Iterations: {resp['iterations']}")
    print(f"  Tool calls:")
    for tc in resp["tool_calls"]:
        print(f"    [{tc['iteration']}] {tc['tool']}({tc['input']}) → {tc['result']}")

    # --- 3. Single-turn: order lookup (triggers search_orders) ---
    section("3. ORDER LOOKUP — Model calls search_orders tool")

    resp = client.post("/chat", json={
        "message": "Can you check what orders bob@example.com has? Any pending ones?",
    }).json()
    print(f"\n  Response: {resp['response'][:300]}")
    print(f"  Iterations: {resp['iterations']}")
    print(f"  Tool calls:")
    for tc in resp["tool_calls"]:
        print(f"    [{tc['iteration']}] {tc['tool']}({tc['input']})")
        print(f"      → {tc['result'][:100]}")

    # --- 4. Single-turn: calculation (triggers calculate) ---
    section("4. CALCULATION — Model uses calculate tool (not mental math)")

    resp = client.post("/chat", json={
        "message": "What is 1847 multiplied by 293, plus 15?",
    }).json()
    print(f"\n  Response: {resp['response'][:200]}")
    print(f"  Tool calls:")
    for tc in resp["tool_calls"]:
        print(f"    {tc['tool']}({tc['input']}) → {tc['result']}")

    # --- 5. Multi-tool: model calls multiple tools ---
    section("5. MULTI-TOOL — Model chains multiple tools in one request")

    resp = client.post("/chat", json={
        "message": (
            "I need three things: "
            "1) Weather in Tokyo and London, "
            "2) All orders for alice@example.com, "
            "3) Calculate the total of 149.99 + 49.99"
        ),
    }).json()
    print(f"\n  Response: {resp['response'][:400]}")
    print(f"  Iterations: {resp['iterations']}")
    print(f"  Tool calls ({len(resp['tool_calls'])} total):")
    for tc in resp["tool_calls"]:
        print(f"    [{tc['iteration']}] {tc['tool']}({tc['input']})")

    # --- 6. No tool needed: model responds directly ---
    section("6. NO TOOL — Model decides no tool is needed")

    resp = client.post("/chat", json={
        "message": "Hello! How are you today?",
    }).json()
    print(f"\n  Response: {resp['response'][:200]}")
    print(f"  Tool calls: {len(resp['tool_calls'])} (should be 0)")
    print(f"  Iterations: {resp['iterations']} (should be 1)")

    # --- 7. Force tool use ---
    section("7. FORCE TOOL — Require a specific tool to be called")

    resp = client.post("/force-tool", json={
        "message": "Tell me about Paris.",
        "tool_name": "get_weather",
    }).json()
    print(f"\n  Response: {resp['response'][:200]}")
    print(f"  Forced tool call:")
    if resp["forced_tool_call"]:
        tc = resp["forced_tool_call"]
        print(f"    {tc['tool']}({tc['input']}) → {tc['result']}")

    # --- 8. Multi-turn conversation ---
    section("8. MULTI-TURN — Conversation with tool history")

    # Turn 1: ask about weather
    resp1 = client.post("/conversation", json={
        "messages": [
            {"role": "user", "content": "What's the weather in Sydney?"},
        ],
    }).json()
    print(f"\n  Turn 1: {resp1['response'][:150]}")
    print(f"  Tools used: {[tc['tool'] for tc in resp1['tool_calls']]}")

    # Turn 2: follow-up referencing previous context
    resp2 = client.post("/conversation", json={
        "messages": [
            {"role": "user", "content": "What's the weather in Sydney?"},
            {"role": "assistant", "content": resp1["response"]},
            {"role": "user", "content": "How about in London? Is it warmer or colder than Sydney?"},
        ],
    }).json()
    print(f"\n  Turn 2: {resp2['response'][:200]}")
    print(f"  Tools used: {[tc['tool'] for tc in resp2['tool_calls']]}")

    section("DONE — All tool use concepts demonstrated!")


if __name__ == "__main__":
    main()
