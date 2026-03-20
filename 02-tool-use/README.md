# Module 02: Tool Use / Function Calling

> A deep engineering-level guide for backend developers building production AI systems.

---

## Table of Contents

1. [Concept Explanation](#1-concept-explanation)
2. [Why It Matters in Real Systems](#2-why-it-matters-in-real-systems)
3. [Internal Mechanics](#3-internal-mechanics)
4. [Practical Example](#4-practical-example)
5. [Hands-on Implementation](#5-hands-on-implementation)
6. [System Design Perspective](#6-system-design-perspective)
7. [Common Pitfalls](#7-common-pitfalls)
8. [Advanced Topics](#8-advanced-topics)
9. [Exercises](#9-exercises)
10. [Interview / Architect Questions](#10-interview--architect-questions)

---

## 1. Concept Explanation

LLMs generate text — that's their only primitive operation. They cannot query databases, call APIs, read files, or perform any real-world action. **Tool use** (also called function calling) bridges this gap by letting the model *declare intent* to call a function, which your code then executes.

### The Core Mental Model

Think of the LLM as a very smart dispatcher in a microservices architecture. It reads the user's request, decides which backend service to call, and formats the request payload — but **your code actually executes the call and returns the result**.

```
User prompt → LLM reasons → outputs structured tool call (name + args)
    → Your code executes the function → result fed back to LLM
    → LLM generates final response incorporating the result
```

The model does NOT run code. It produces a JSON object like:

```json
{
  "type": "tool_use",
  "name": "get_weather",
  "input": {"city": "Tokyo", "unit": "celsius"}
}
```

Your application parses this, calls the real `get_weather()` function, and sends the result back. The model then incorporates that result into its natural language answer.

### Key Distinction: Tool Use vs. RAG vs. Prompting

| Approach | When data is injected | Who decides what data | LLM's role |
|---|---|---|---|
| **Prompting** | Before generation, statically | Developer hardcodes it | Reason over fixed context |
| **RAG** | Before generation, dynamically | Retrieval system (embeddings) | Reason over retrieved context |
| **Tool Use** | During generation, on demand | The LLM itself decides | Decide what to fetch, then reason |

RAG retrieves context *before* the model runs. Tool use happens *during* generation — the model decides mid-response that it needs external data and explicitly requests it. This is the fundamental difference: **tool use gives the model agency**.

### The Analogy for Backend Developers

If you've built REST APIs, you already understand tool use:

```
Tool Definition    ≈  OpenAPI/Swagger spec     (what the endpoint does)
Tool Implementation ≈  Route handler            (the actual business logic)
Tool Call          ≈  HTTP request              (model calls the endpoint)
Tool Result        ≈  HTTP response             (data sent back to model)
Agentic Loop       ≈  Saga/Workflow orchestrator (multi-step coordination)
```

The LLM reads the "API specs" (tool definitions), decides which "endpoint" to call based on the user's request, and formats the "request body" (tool input). Your code is the server that processes the request and returns the response.

---

## 2. Why It Matters in Real Systems

### 2.1 From Text Generator to Action Taker

Without tool use, an LLM is a fancy autocomplete — it can only generate text based on its training data. With tool use, it becomes an **agent** that can observe and act on the real world.

```
Without tools:
  User: "What's my order status?"
  LLM:  "I don't have access to your order information." ← useless

With tools:
  User: "What's my order status?"
  LLM:  [calls search_orders(email="user@example.com")]
  LLM:  "Your keyboard (ORD-1001) was delivered, and your USB hub (ORD-1042) is shipped." ← useful
```

### 2.2 Why Not Just Use RAG?

RAG is passive — it retrieves context before generation. Tool use is active — the model decides what it needs. They solve different problems:

| Scenario | RAG | Tool Use | Why |
|---|---|---|---|
| "What's our refund policy?" | ✅ | ❌ | Static knowledge, retrieve from docs |
| "Process a refund for order #1234" | ❌ | ✅ | Requires action + real-time data |
| "What's the weather in Tokyo?" | ❌ | ✅ | Real-time data, not in training set |
| "Summarize our API documentation" | ✅ | ❌ | Existing documents, no action needed |
| "Find similar support tickets and then create a Jira issue" | ✅ + ✅ | ✅ | Needs both retrieval AND action |

### 2.3 Where Companies Use Tool Use

| Company Type | Tool Use Application |
|---|---|
| **Customer support** | Look up order status, initiate refunds, update accounts — real actions, not just chat |
| **Code assistants** | Read files, run tests, search codebases, apply edits (exactly what Claude Code does) |
| **Data analysts** | Query SQL databases, run calculations, generate charts from live data |
| **Booking platforms** | Search flights, check availability, make reservations, process payments |
| **DevOps copilots** | Check deployment status, roll back releases, scale infrastructure, query metrics |
| **Financial services** | Retrieve account balances, execute trades, generate compliance reports |

### 2.4 Business Impact

Companies use tool use because:

- **Accuracy**: the model doesn't hallucinate facts it can look up via tools
- **Freshness**: real-time data from APIs instead of stale training data
- **Action**: the system can *do things*, not just *say things*
- **Composability**: small, well-tested functions combine into complex workflows
- **Cost efficiency**: one LLM + tools replaces building custom UIs for every operation

---

## 3. Internal Mechanics

### 3.1 How the Model "Learns" to Call Tools

During fine-tuning (SFT + RLHF stage, covered in Module 01), models are trained on thousands of examples where the correct response is a structured tool call rather than plain text. The training data looks like:

```
System: You have access to get_weather(city, unit)
User:   What's the temperature in Tokyo?
Model:  [tool_use: get_weather, input: {city: "Tokyo", unit: "celsius"}]
```

The model learns a decision boundary: "When the user asks about weather, emit a tool_use block instead of guessing the temperature." This is essentially classification — the model classifies each turn as either "answer directly" or "call tool X with arguments Y."

### 3.2 What the Model Sees (Context Injection)

Your tool definitions are serialized into the model's context window. When you send:

```python
tools = [{
    "name": "get_weather",
    "description": "Get current weather for a city...",
    "input_schema": {
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "City name"},
            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
        },
        "required": ["city"]
    }
}]
```

The API serializes these into the system portion of the prompt. The model sees something like:

```
You have access to the following tools:

Tool: get_weather
Description: Get current weather for a city...
Parameters:
  - city (string, required): City name
  - unit (string, optional): celsius or fahrenheit

When you need to use a tool, respond with a tool_use block.
```

This means **tool definitions consume input tokens on every API call**. Ten tools with verbose descriptions can easily add 2,000-3,000 tokens to every request.

### 3.3 The API Protocol (Claude)

The full message flow for a tool use interaction:

```
┌─────────────────────────────────────────────────────────────┐
│ Step 1: Define tools as JSON schemas                        │
│   (name, description, input_schema with JSON Schema)        │
├─────────────────────────────────────────────────────────────┤
│ Step 2: Send user message + tool definitions to API         │
│   POST /messages                                            │
│   {                                                         │
│     model: "claude-sonnet-4-...",                            │
│     tools: [...],                                           │
│     messages: [{role: "user", content: "Weather in Tokyo?"}]│
│   }                                                         │
├─────────────────────────────────────────────────────────────┤
│ Step 3: Model returns stop_reason: "tool_use"               │
│   {                                                         │
│     stop_reason: "tool_use",                                │
│     content: [                                              │
│       {type: "text", text: "I'll check the weather..."},    │
│       {type: "tool_use", id: "toolu_abc123",                │
│        name: "get_weather",                                 │
│        input: {city: "Tokyo", unit: "celsius"}}             │
│     ]                                                       │
│   }                                                         │
├─────────────────────────────────────────────────────────────┤
│ Step 4: YOUR CODE executes the function                     │
│   result = get_weather("Tokyo", "celsius")                  │
│   → "22°C, Partly Cloudy, Humidity: 65%"                    │
├─────────────────────────────────────────────────────────────┤
│ Step 5: Send tool_result back to the model                  │
│   messages: [                                               │
│     {role: "user", content: "Weather in Tokyo?"},           │
│     {role: "assistant", content: [<text + tool_use blocks>]}│
│     {role: "user", content: [{                              │
│       type: "tool_result",                                  │
│       tool_use_id: "toolu_abc123",                          │
│       content: "22°C, Partly Cloudy, Humidity: 65%"         │
│     }]}                                                     │
│   ]                                                         │
├─────────────────────────────────────────────────────────────┤
│ Step 6: Model generates final natural language response     │
│   "The weather in Tokyo is currently 22°C and partly        │
│    cloudy with 65% humidity."                               │
│   stop_reason: "end_turn"                                   │
└─────────────────────────────────────────────────────────────┘
```

### 3.4 The Agentic Loop — The Most Important Pattern

The model may need multiple rounds of tool calls to answer a single question. This is the **agentic loop** — the core pattern behind every tool-use application:

```python
while iteration < max_iterations:
    response = client.messages.create(
        model=MODEL,
        tools=TOOL_DEFINITIONS,
        messages=messages,
    )

    # Exit: model produced a final text response
    if response.stop_reason == "end_turn":
        return extract_text(response)

    # Continue: model wants to use tools
    if response.stop_reason == "tool_use":
        # 1. Append the FULL assistant message (text + tool_use blocks)
        messages.append({"role": "assistant", "content": response.content})

        # 2. Execute each tool call and collect results
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                result = execute_tool(block.name, block.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,   # MUST match
                    "content": result,
                })

        # 3. Send all results back in one user message
        messages.append({"role": "user", "content": tool_results})

        # 4. Loop — model sees results and decides: answer or call more tools
```

**Why it's a loop, not a single call:**

Consider: "What's the weather in Tokyo, and what are Alice's orders?" The model might:
- Iteration 1: Call `get_weather("Tokyo")` AND `search_orders("alice@example.com")` (parallel)
- Iteration 2: See both results → generate final answer with both pieces of data

Or for a more complex scenario: "Is it good weather for Alice to pick up her delivered orders?"
- Iteration 1: Call `search_orders("alice@example.com")`
- Iteration 2: See that Alice has a delivered order → call `get_weather("Alice's city")`
- Iteration 3: Combine weather + order info → generate recommendation

Each iteration is an API call → tool execution → API call cycle. The model decides when it has enough information to answer.

### 3.5 Parallel Tool Calls

Claude can emit **multiple tool_use blocks in a single response**. This is the model saying "I need data from multiple tools simultaneously":

```json
{
  "content": [
    {"type": "text", "text": "Let me check both for you."},
    {"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {"city": "Tokyo"}},
    {"type": "tool_use", "id": "toolu_2", "name": "get_weather", "input": {"city": "London"}}
  ]
}
```

You must execute ALL tool calls and send ALL results back in a single user message:

```json
{
  "role": "user",
  "content": [
    {"type": "tool_result", "tool_use_id": "toolu_1", "content": "22°C, Partly Cloudy"},
    {"type": "tool_result", "tool_use_id": "toolu_2", "content": "14°C, Rainy"}
  ]
}
```

This is a significant performance optimization — instead of two separate loop iterations, the model gets both results in one round trip.

### 3.6 The `tool_choice` Parameter

You can control whether and how the model uses tools:

```python
# Auto (default): model decides whether to call a tool
tool_choice={"type": "auto"}

# Any: model MUST call some tool (but it picks which one)
tool_choice={"type": "any"}

# Specific: model MUST call THIS tool
tool_choice={"type": "tool", "name": "get_weather"}
```

Use `"any"` or specific tool when you **know** a tool call is needed and don't want the model to skip it. This is useful in pipelines where the LLM's job is specifically to extract structured data for a known tool.

### 3.7 Message Structure Rules

The Claude API enforces strict message structure rules for tool use:

1. Messages must alternate: `user → assistant → user → assistant → ...`
2. A `tool_use` block must appear in an `assistant` message
3. A `tool_result` block must appear in the `user` message that immediately follows
4. Every `tool_result` must reference a `tool_use_id` from the preceding assistant message
5. Every `tool_use` block must have a corresponding `tool_result`

Violating any of these rules produces an API error. This is the most common source of bugs when implementing tool use.

---

## 4. Practical Example

### E-Commerce Customer Support Agent

**Problem:** Build a support agent that can look up orders, check weather for delivery estimates, and perform calculations — handling multi-step requests autonomously.

**User says:** *"I'm alice@example.com. Can you check my orders and calculate the total cost of everything?"*

**What happens internally:**

```
Iteration 1:
  Model thinks: "I need to look up Alice's orders first"
  Model emits:  tool_use → search_orders(email="alice@example.com")

Your code:
  Executes search_orders → returns order list with prices

Iteration 2:
  Model sees: ORD-1001 ($149.99), ORD-1042 ($49.99)
  Model thinks: "Now I need to calculate the total"
  Model emits:  tool_use → calculate(expression="149.99 + 49.99")

Your code:
  Executes calculate → returns "199.98"

Iteration 3:
  Model sees: calculation result
  Model generates: "Hi Alice! You have 2 orders:
    1. Mechanical Keyboard (ORD-1001) — $149.99, delivered
    2. USB-C Hub (ORD-1042) — $49.99, shipped
    Your total is $199.98."
  stop_reason: "end_turn"
```

**Architecture:**

```
User Message ────▶ FastAPI /chat endpoint
                         │
                    ┌─────▼──────┐
                    │  Agentic   │◀─── tool definitions
                    │  Loop      │     (JSON schemas)
                    └─────┬──────┘
                          │
         ┌────────────────┼────────────────┐
         ▼                ▼                ▼
   ┌──────────┐    ┌──────────┐    ┌──────────┐
   │  Weather  │    │  Orders  │    │  Calc    │
   │  Service  │    │  Service │    │  Service │
   └──────────┘    └──────────┘    └──────────┘
   (ext. API)      (database)      (math eval)
```

**Key engineering decisions driven by tool use understanding:**

| Decision | Reasoning |
|---|---|
| Separate tool definitions from implementations | Mirrors OpenAPI spec vs handler — testable independently |
| Max iterations guard on the loop | Prevents runaway costs if the model loops endlessly |
| Log every tool call with input/output | Essential for debugging when the model calls the wrong tool |
| Return tool call history in the response | Client-side observability — users can see what happened |
| Execute tools sequentially, not concurrently | Simpler error handling; parallelize only when needed |

---

## 5. Hands-on Implementation

### Project Structure

```
02-tool-use/
├── .env.example        # API key template
├── requirements.txt    # Dependencies
├── tools.py            # Tool definitions (JSON schemas) + implementations
├── app.py              # FastAPI service with agentic loop endpoints
├── test_tool_use.py    # Unit tests + integration tests
└── test_concepts.py    # Interactive script to exercise all endpoints
```

### Setup

```bash
cd 02-tool-use

# Create virtual environment
python -m venv venv
source venv/bin/activate        # Linux/Mac
# or: venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Configure API key
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY

# Start the server
uvicorn app:app --reload --port 8001

# In another terminal, run the test script
python test_concepts.py
```

### File-by-File Walkthrough

#### File 1: `tools.py` — Tool Definitions and Implementations

**Why it's a separate file:** This mirrors the backend pattern of separating API specifications from handlers. The tool definitions (JSON schemas) are what the model sees. The implementations are what your code executes. You can test them independently.

##### Tool Definitions (what the model sees)

**File:** `tools.py:17-77`

```python
TOOL_DEFINITIONS = [
    {
        "name": "get_weather",
        "description": (
            "Get current weather for a specific city. Returns temperature, "
            "condition, and humidity. Use this when the user asks about "
            "weather, temperature, or outdoor conditions in a location."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "City name, e.g. 'Tokyo', 'New York'",
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "Temperature unit. Default: celsius.",
                },
            },
            "required": ["city"],
        },
    },
    # ... more tools
]
```

**Critical details:**
- **`description`** is the most important field. The model reads this to decide WHEN to call the tool. Vague descriptions → wrong tool selection. Include "Use this when..." clauses.
- **`input_schema`** uses standard JSON Schema. The model generates arguments that conform to this schema.
- **`required`** tells the model which parameters it must provide. Optional parameters get sensible defaults in your implementation.
- **`enum`** constrains the model to specific values — prevents it from inventing invalid options.
- **Parameter `description`** helps the model fill in the right values. "City name" is vague; "City name, e.g. 'Tokyo', 'New York'" gives concrete examples.

##### Tool Implementations (what your code executes)

**File:** `tools.py:94-136`

```python
def execute_tool(tool_name: str, tool_input: dict) -> str:
    """Dispatch tool calls to their implementations."""
    if tool_name == "get_weather":
        return _get_weather(tool_input)
    elif tool_name == "search_orders":
        return _search_orders(tool_input)
    elif tool_name == "calculate":
        return _calculate(tool_input)
    else:
        return f"Error: Unknown tool '{tool_name}'"
```

**Production concerns this dispatcher would handle:**
- Authentication/authorization checks (does this user have access to this tool?)
- Input validation (the model can send malformed arguments)
- Rate limiting (prevent excessive tool calls)
- Timeout enforcement (don't let a slow API hang the loop)
- Audit logging (who called what tool with what arguments)
- Error handling with user-friendly messages (don't leak stack traces)

##### Security: The Calculator Tool

**File:** `tools.py:131-156`

```python
def _calculate(params: dict) -> str:
    expression = params["expression"]

    # Security: NEVER use raw eval() with untrusted input
    allowed_chars = set("0123456789+-*/.(,) ")
    allowed_words = {"sqrt", "abs", "round", "min", "max"}

    words = set(re.findall(r"[a-zA-Z_]+", expression))
    if not words.issubset(allowed_words):
        unsafe = words - allowed_words
        return f"Error: Unsafe operations not allowed: {unsafe}"

    result = eval(expression, {"__builtins__": {}}, SAFE_MATH)
    return f"Result: {result}"
```

**Critical lesson:** The model's output is **untrusted input** from a security perspective. If the model generates `"__import__('os').system('rm -rf /')"` as a calculation expression and you pass it to `eval()`, you've given the LLM code execution. Always sanitize, whitelist, and sandbox.

#### File 2: `app.py` — The FastAPI Service

##### Endpoint 1: `POST /chat` — The Complete Agentic Loop

**File:** `app.py:47-99`

This is the most important endpoint. It implements the full agentic loop:

```bash
curl -X POST http://localhost:8001/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is the weather in Tokyo?"}'
```

**Expected response:**

```json
{
  "response": "The weather in Tokyo is currently 22°C and partly cloudy with 65% humidity.",
  "tool_calls": [
    {
      "iteration": 1,
      "tool": "get_weather",
      "input": {"city": "Tokyo"},
      "result": "Weather in Tokyo: 22°C, Partly Cloudy, Humidity: 65%"
    }
  ],
  "iterations": 2,
  "usage": {"input_tokens": 512, "output_tokens": 45}
}
```

**What to observe:**
- `iterations: 2` — iteration 1 was the tool call, iteration 2 was the final response
- The `tool_calls` log shows exactly what the model requested and what it received
- The model's `response` incorporates the tool result naturally into conversation

**Try these variations to see different behaviors:**

```bash
# Multi-tool: model calls multiple tools
curl -X POST http://localhost:8001/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Weather in Tokyo and London, and calculate 15*23"}'

# No tool needed: model responds directly
curl -X POST http://localhost:8001/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello, how are you?"}'

# The model decides which tool based on the question
curl -X POST http://localhost:8001/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Can you check orders for alice@example.com?"}'
```

**How the agentic loop works internally:**

```python
# The while loop is the agentic loop
while iteration < req.max_iterations:
    iteration += 1
    response = client.messages.create(
        model=MODEL,
        tools=TOOL_DEFINITIONS,
        messages=messages,
    )

    # EXIT: model produced final text
    if response.stop_reason == "end_turn":
        return {"response": extract_text(response), ...}

    # CONTINUE: model wants to use tools
    if response.stop_reason == "tool_use":
        # Append the FULL assistant message (critical!)
        messages.append({"role": "assistant", "content": response.content})

        # Execute tools and collect results
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                result = execute_tool(block.name, block.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,  # must match
                    "content": result,
                })

        # Send results back → model sees them on next iteration
        messages.append({"role": "user", "content": tool_results})
```

Key points:
1. **`response.content` is a list** — it can contain both `text` and `tool_use` blocks. You must append the entire list, not just the tool blocks.
2. **`tool_use_id` must match** — every `tool_result` references the `id` from its corresponding `tool_use`. A mismatch causes API errors.
3. **Multiple tools per iteration** — the model can request multiple tools at once (parallel calls). Your code must process all of them and send all results back in one message.
4. **`max_iterations` is a safety guard** — without it, a confused model could loop forever, burning tokens.

##### Endpoint 2: `POST /conversation` — Multi-Turn with Tool History

**File:** `app.py:106-157`

Multi-turn conversation where the model retains context from previous turns, including previous tool calls.

```bash
# Turn 1
curl -X POST http://localhost:8001/conversation \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "What is the weather in Sydney?"}
    ]
  }'

# Turn 2 (include previous context)
curl -X POST http://localhost:8001/conversation \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "What is the weather in Sydney?"},
      {"role": "assistant", "content": "The weather in Sydney is 26°C and clear."},
      {"role": "user", "content": "How about London? Warmer or colder?"}
    ]
  }'
```

**What to observe:**
- In turn 2, the model remembers Sydney's weather from turn 1 (it's in the message history)
- The model only calls `get_weather("London")` — it doesn't re-fetch Sydney because it already has that data
- If you strip tool_use/tool_result messages from history, the model loses context and will re-call tools

**Production note:** In a real system, you'd store the full message history (including tool_use and tool_result messages) in a database. The simplified ConversationMessage schema here only handles text messages — a production version would need to handle the full content block format.

##### Endpoint 3: `GET /tools` — Inspect What the Model Sees

**File:** `app.py:163-177`

```bash
curl http://localhost:8001/tools | python -m json.tool
```

Returns the exact tool definitions sent to the model. Use this to:
- Debug tool selection issues ("why did it call the wrong tool?")
- Verify your descriptions are clear and unambiguous
- Count how many tokens your tool definitions consume

##### Endpoint 4: `POST /force-tool` — Controlling Tool Selection

**File:** `app.py:183-231`

```bash
# Force the model to check weather even for a non-weather question
curl -X POST http://localhost:8001/force-tool \
  -H "Content-Type: application/json" \
  -d '{"message": "Tell me about Paris", "tool_name": "get_weather"}'
```

**What to observe:**
- Even though "Tell me about Paris" doesn't explicitly ask for weather, the model is forced to call `get_weather`
- The model still generates sensible arguments (city: "Paris")
- After receiving the result, it incorporates the weather data into its response
- This demonstrates `tool_choice: {"type": "tool", "name": "..."}` — overriding the model's decision

**When to use forced tool selection:**
- In pipelines where you KNOW a tool call is needed (e.g., data extraction)
- When the model keeps skipping a tool that should be called
- In A/B testing tool selection accuracy

### Running the Interactive Test Script

```bash
# Start the server in one terminal
uvicorn app:app --reload --port 8001

# Run the interactive demo in another terminal
python test_concepts.py
```

The test script exercises all 8 scenarios sequentially:
1. Inspect tool definitions
2. Weather query (single tool)
3. Order lookup (single tool)
4. Calculation (single tool)
5. Multi-tool request (multiple tools in one query)
6. No tool needed (direct response)
7. Forced tool use
8. Multi-turn conversation

Each scenario prints the model's response, which tools were called, how many iterations the loop took, and the raw tool inputs/outputs.

### Running Unit and Integration Tests

```bash
# Unit tests only (no API key needed — tests tool implementations)
pytest test_tool_use.py -k "not TestAgenticLoop" -v

# All tests including API integration (requires ANTHROPIC_API_KEY)
pytest test_tool_use.py -v
```

Unit tests validate:
- Tool implementations return correct results for known inputs
- Security: calculator rejects unsafe expressions
- Tool definitions have all required fields
- Descriptions are long enough for accurate selection

Integration tests validate:
- Weather questions trigger `get_weather`
- Order questions trigger `search_orders`
- Math questions trigger `calculate`
- Simple greetings trigger NO tools
- Multi-tool queries trigger multiple tools

---

## 6. System Design Perspective

### Where Tool Use Fits in a Production AI Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                         API Gateway                              │
│              (Auth, Rate Limiting, Request Routing)               │
├──────────────────────────────────────────────────────────────────┤
│                      Orchestration Layer                         │
│                                                                  │
│  ┌──────────┐    ┌──────────────┐    ┌────────────────────┐     │
│  │ Message   │───▶│  Agentic     │───▶│  Response Builder  │     │
│  │ Router    │    │  Loop        │    │  (stream / batch)  │     │
│  └──────────┘    └──────┬───────┘    └────────────────────┘     │
│                         │                                        │
│              ┌──────────▼──────────┐                             │
│              │   Tool Dispatcher   │                             │
│              │                     │                             │
│              │ • Auth check        │                             │
│              │ • Input validation  │                             │
│              │ • Rate limiting     │                             │
│              │ • Timeout enforce   │                             │
│              │ • Audit logging     │                             │
│              │ • Error handling    │                             │
│              └──────────┬──────────┘                             │
│                         │                                        │
├─────────────────────────┼────────────────────────────────────────┤
│           Tool Implementations (your microservices)              │
│                         │                                        │
│  ┌──────────┐  ┌───────┴──┐  ┌──────────┐  ┌──────────┐       │
│  │ Database │  │ REST API │  │ Search   │  │ Internal │       │
│  │ Query    │  │ Client   │  │ Engine   │  │ Service  │       │
│  │ Tool     │  │ Tool     │  │ Tool     │  │ Tool     │       │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘       │
│       │              │              │              │             │
├───────┼──────────────┼──────────────┼──────────────┼─────────────┤
│       ▼              ▼              ▼              ▼             │
│   PostgreSQL    External API    Elasticsearch   gRPC Service    │
└──────────────────────────────────────────────────────────────────┘
```

### Production Components You Must Build

#### 1. Tool Dispatcher — The Critical Middleware

This sits between the model and your tools. It's the most important component:

```python
class ToolDispatcher:
    def __init__(self, tools: dict, auth_service, rate_limiter):
        self.tools = tools
        self.auth = auth_service
        self.limiter = rate_limiter

    async def execute(
        self, tool_name: str, tool_input: dict, user_context: UserContext
    ) -> str:
        # 1. Authorization: can this user call this tool?
        if not self.auth.can_access(user_context, tool_name):
            return "Error: You don't have permission to use this tool."

        # 2. Rate limiting: has this user exceeded their tool call budget?
        if not self.limiter.allow(user_context.user_id, tool_name):
            return "Error: Rate limit exceeded. Please try again later."

        # 3. Input validation: does the input match the schema?
        tool = self.tools[tool_name]
        validation_error = validate_input(tool.schema, tool_input)
        if validation_error:
            return f"Error: Invalid input — {validation_error}"

        # 4. Execute with timeout
        try:
            result = await asyncio.wait_for(
                tool.execute(tool_input), timeout=10.0
            )
        except asyncio.TimeoutError:
            return "Error: Tool execution timed out."

        # 5. Audit log
        await self.audit_log.record(
            user=user_context.user_id,
            tool=tool_name,
            input=tool_input,
            result=result,
            latency_ms=elapsed,
        )

        return result
```

#### 2. Idempotency for Mutating Tools

The model might call the same tool twice due to retries or loop errors. Write operations must be idempotent:

```python
async def initiate_refund(params: dict) -> str:
    idempotency_key = f"refund-{params['order_id']}-{params['amount']}"

    # Check if this refund was already processed
    existing = await refund_store.get(idempotency_key)
    if existing:
        return f"Refund already processed: {existing.refund_id}"

    # Process the refund
    refund = await payment_service.refund(
        order_id=params["order_id"],
        amount=params["amount"],
        idempotency_key=idempotency_key,
    )

    return f"Refund {refund.id} processed: ${refund.amount}"
```

#### 3. Timeout Budgets

Each agentic loop iteration is an API call (~1-3s) plus tool execution time. You need a global deadline:

```python
async def chat_with_deadline(message: str, deadline_seconds: float = 30):
    deadline = time.time() + deadline_seconds

    while iteration < max_iterations:
        remaining = deadline - time.time()
        if remaining <= 0:
            return "I'm taking too long. Let me give you what I have so far..."

        response = await client.messages.create(
            model=MODEL,
            tools=TOOL_DEFINITIONS,
            messages=messages,
            # Don't ask for too many tokens if we're running low on time
            max_tokens=min(1024, int(remaining * 50)),
        )
        # ... rest of agentic loop
```

#### 4. Observability — Your Debugging Lifeline

Log every tool call. When the model calls the wrong tool, these logs are the only way to debug:

```python
# Structured log for every tool call
{
    "event": "tool_call",
    "request_id": "req_abc123",
    "user_id": "user_456",
    "iteration": 2,
    "tool_name": "search_orders",
    "tool_input": {"email": "alice@example.com"},
    "tool_result_length": 245,
    "tool_latency_ms": 34,
    "total_elapsed_ms": 1520,
    "model": "claude-sonnet-4-20250514",
    "cumulative_input_tokens": 1847,
    "cumulative_output_tokens": 312,
}
```

**Dashboard metrics to track:**
- Tool call rate per tool (which tools are used most?)
- Tool selection accuracy (manual review sample)
- Average iterations per request (trending up = prompt problem)
- Tool execution latency per tool (which tools are slow?)
- Loop timeout rate (how often do you hit max_iterations?)

#### 5. Graceful Degradation

If a tool fails, send a clear error message back to the model. It can often recover:

```python
try:
    result = await execute_tool(block.name, block.input)
except DatabaseConnectionError:
    result = (
        "Error: The order database is temporarily unavailable. "
        "Please let the user know and suggest they try again in a few minutes."
    )
except ExternalAPIError as e:
    result = f"Error: The weather service returned an error: {e.message}"
```

The model will incorporate the error into its response naturally: *"I'm sorry, I wasn't able to check your orders right now because our order system is temporarily unavailable. Please try again in a few minutes."*

---

## 7. Common Pitfalls

### Pitfall 1: Vague Tool Descriptions

**The mistake:**

```python
{
    "name": "get_data",
    "description": "Gets data from the system",  # useless
}
```

**Why it fails:** The model selects tools based on descriptions. "Gets data" matches everything and nothing. The model will call it for the wrong queries or never call it at all.

**The fix:**

```python
{
    "name": "search_orders",
    "description": (
        "Search for customer orders by email address. Returns a list of "
        "recent orders with their status, items, and totals. Use this when "
        "the user asks about their orders, deliveries, or purchase history. "
        "Do NOT use this for product searches or inventory questions."
    ),
}
```

Include: what it does, what it returns, when to use it, when NOT to use it.

### Pitfall 2: Dropping Content from the Assistant Message

**The mistake:**

```python
# WRONG — only keeps tool_use blocks, drops text blocks
tool_blocks = [b for b in response.content if b.type == "tool_use"]
messages.append({"role": "assistant", "content": tool_blocks})
```

**Why it fails:** The assistant message may contain BOTH text and tool_use blocks. The text often contains the model's reasoning ("Let me check that for you..."). Dropping it corrupts the conversation context and can cause API errors.

**The fix:**

```python
# RIGHT — preserve the entire content list
messages.append({"role": "assistant", "content": response.content})
```

### Pitfall 3: Mismatching tool_use_id

**The mistake:**

```python
tool_results.append({
    "type": "tool_result",
    "tool_use_id": "some_hardcoded_id",  # wrong!
    "content": result,
})
```

**Why it fails:** Every `tool_result` must reference the exact `id` from its corresponding `tool_use` block. The API rejects mismatches.

**The fix:**

```python
tool_results.append({
    "type": "tool_result",
    "tool_use_id": block.id,  # from the tool_use block
    "content": result,
})
```

### Pitfall 4: No Loop Guard

**The mistake:**

```python
while True:  # infinite loop!
    response = client.messages.create(...)
    if response.stop_reason == "tool_use":
        # process tools...
        # but what if the model NEVER emits end_turn?
```

**Why it fails:** A confused model can loop indefinitely — calling the same tool with the same arguments, or cycling between tools without making progress. Each iteration costs tokens.

**The fix:**

```python
MAX_ITERATIONS = 10

while iteration < MAX_ITERATIONS:
    iteration += 1
    # ... agentic loop ...

if iteration >= MAX_ITERATIONS:
    return "I wasn't able to complete your request within the allowed steps."
```

### Pitfall 5: Using eval() for Model Output

**The mistake:**

```python
def calculate(expression: str) -> str:
    return str(eval(expression))  # DANGEROUS!
```

**Why it fails:** The model's output is **untrusted input**. It could generate `__import__('os').system('rm -rf /')`. Even with a "well-behaved" model, adversarial user prompts can manipulate the model into generating malicious tool arguments.

**The fix:** Whitelist allowed operations, use a sandboxed math parser, or use a dedicated library like `asteval` or `sympy`.

### Pitfall 6: Too Many Tools

**The mistake:** Registering 50+ tools in every request.

**Why it fails:**
1. Each tool definition consumes input tokens (description + schema). 50 tools can easily add 5,000+ tokens per API call.
2. Model accuracy in tool selection degrades with more options.
3. The context window fills up faster, leaving less room for conversation history.

**The fix:**
- Group related tools into categories. Use a two-stage approach: first route to a category, then load only those tools.
- Use embeddings to dynamically select the most relevant 5-10 tools per query.
- Split into specialized agents, each with a focused tool set.

### Pitfall 7: Stripping Tool Messages from History

**The mistake:** Cleaning conversation history by removing `tool_use` and `tool_result` messages to "save tokens."

**Why it fails:** The model loses track of what data it already has. It will re-call tools unnecessarily, wasting tokens and latency. Worse, if it references a previous tool result that's been removed, it will hallucinate the data.

**The fix:** Keep tool messages in history. If you need to reduce context size, summarize the tool results rather than removing them.

### Pitfall 8: Not Handling Parallel Tool Calls

**The mistake:**

```python
for block in response.content:
    if block.type == "tool_use":
        result = execute_tool(block.name, block.input)
        # Process only the first tool call and break
        break  # WRONG — misses remaining tool calls
```

**Why it fails:** Claude can emit multiple `tool_use` blocks in one response. If you only process the first one, the model gets partial results and may behave unpredictably.

**The fix:** Process ALL tool_use blocks and send ALL results back:

```python
tool_results = []
for block in response.content:
    if block.type == "tool_use":
        result = execute_tool(block.name, block.input)
        tool_results.append({
            "type": "tool_result",
            "tool_use_id": block.id,
            "content": result,
        })
messages.append({"role": "user", "content": tool_results})
```

---

## 8. Advanced Topics

Explore these next, in recommended order:

### 8.1 Streaming with Tool Use

In production, you stream text to the user but must buffer `tool_use` blocks until they're complete before executing.

- **Challenge:** The model streams text tokens, then a `tool_use` block (which arrives as partial JSON), then more text after the tool result.
- **Pattern:** Stream text blocks to the user immediately. Buffer `tool_use` blocks until the `input_json` delta is complete. Execute the tool. Then continue streaming.
- **Why important:** Users expect real-time responses, but tool execution must happen between streaming segments.

### 8.2 MCP (Model Context Protocol)

Anthropic's open standard for connecting LLMs to external tools and data sources.

- **Key concepts:** MCP servers expose tools via a standardized protocol. Clients (like Claude Code) discover and call tools dynamically.
- **Why important:** Instead of hardcoding tool definitions in your app, MCP servers advertise their capabilities. This enables tool interoperability across different LLM applications.
- **When to use:** When building tools that should work with multiple LLM clients, not just your application.

### 8.3 Dynamic Tool Loading

Load different tools per user, per context, or per query instead of static definitions.

- **Pattern:** Embed tool descriptions, match against the user's query with cosine similarity, load the top-K most relevant tools.
- **Why important:** Reduces context usage, improves selection accuracy, and enables per-user tool access control.

### 8.4 Human-in-the-Loop for Dangerous Tools

For tools with side effects (refunds, deletes, deployments), pause and get human approval before executing.

- **Pattern:** Tool dispatcher returns a "pending approval" status. The frontend shows the proposed action. A human clicks "approve" or "reject." The loop resumes.
- **Why important:** Balances automation speed with safety. The model decides what to do; a human decides whether to let it.

### 8.5 Multi-Agent Tool Delegation

Different agents with different tool sets collaborate on complex tasks.

- **Pattern:** Supervisor agent receives the request. Routes to specialist agents: Agent A (database tools), Agent B (API tools), Agent C (calculation tools). Supervisor synthesizes results.
- **Why important:** Reduces per-agent tool count, improves specialization, and enables parallel execution.

### 8.6 Tool Result Caching

Cache deterministic tool results to save latency and cost.

- **Pattern:** Hash the tool name + input. Check cache before executing. Store results with TTL.
- **Subtlety:** The model may ask the same question differently ("weather in Tokyo" vs "Tokyo weather"). You may need semantic caching with embeddings.

### 8.7 Retry and Fallback Strategies

What happens when a tool call fails?

- **Retry:** For transient errors (timeouts, rate limits), retry with exponential backoff.
- **Fallback:** If the primary data source is down, try a secondary source.
- **Graceful error:** Send a clear error message back to the model. It can often work around it.
- **Circuit breaker:** If a tool fails repeatedly, stop calling it and inform the model it's unavailable.

### 8.8 Tool Use Evaluation and Testing

How do you measure tool use quality?

- **Tool selection accuracy:** Given a test set of queries, does the model call the right tool?
- **Argument accuracy:** Does the model extract the right parameters from the user's message?
- **End-to-end evaluation:** Does the final response correctly use the tool results?
- **Regression testing:** When you change tool descriptions, do previously working queries still work?

---

## 9. Exercises

### Exercise 1: Add a New Tool — Product Search

**Objective:** Add a `search_products` tool that searches a simulated product catalog by category and price range.

**Requirements:**
1. Define the tool schema with parameters: `category` (string, required), `max_price` (number, optional), `min_price` (number, optional)
2. Implement the tool with a simulated catalog of 10+ products across 3 categories
3. Add clear description with "Use this when..." and "Do NOT use this for..." clauses
4. Add unit tests that verify correct filtering by category and price range
5. Test via `/chat` with queries like: "What laptops do you have under $1000?"

**What you'll learn:** Writing effective tool definitions, how description quality affects selection accuracy.

**Stretch goals:**
- Add a `sort_by` parameter (price, name, rating)
- Test with ambiguous queries ("I need something for gaming") — does the model pick the right category?
- Measure token cost: how much does adding this tool increase input tokens per request?

---

### Exercise 2: Human-in-the-Loop Approval Gate

**Objective:** Build an approval workflow for "dangerous" tools.

**Requirements:**
1. Add an `initiate_refund` tool (simulated) that requires approval before execution
2. Tag tools as `"requires_approval": true` in their definitions
3. When the model calls a dangerous tool, pause the loop and return the pending action:
   ```json
   {"status": "pending_approval", "tool": "initiate_refund", "input": {"order_id": "ORD-1001"}, "approval_id": "apr_xyz"}
   ```
4. Add `POST /approve/{approval_id}` endpoint that resumes execution
5. Add `POST /reject/{approval_id}` endpoint that sends a rejection message back to the model
6. Store pending actions in memory (dict) with a 5-minute expiry

**What you'll learn:** Stateful agentic loops, human-in-the-loop patterns, handling side effects safely.

**Stretch goals:**
- Add a WebSocket endpoint that streams the conversation, pausing when approval is needed
- Implement approval history/audit log
- Add automatic approval for low-value refunds (< $10) and manual for high-value

---

### Exercise 3: Tool Result Caching with Observability

**Objective:** Add a caching layer to the tool dispatcher and measure its impact.

**Requirements:**
1. Implement a cache (dict with TTL) in `execute_tool` that stores results keyed by `tool_name + sorted(input)`
2. Set different TTLs per tool: weather = 5 min, orders = 30 sec, calculate = infinite
3. Return cache hit/miss status in the tool call log
4. Add a `GET /cache/stats` endpoint showing: total hits, total misses, hit rate per tool, cache size
5. Test by sending the same query twice — verify the second call uses cached results

**What you'll learn:** Caching strategies for AI systems, how caching affects model behavior, observability.

**Stretch goals:**
- Implement cache invalidation when the underlying data changes
- Add semantic caching: "weather in Tokyo" and "Tokyo weather" should share a cache entry (use string normalization or embeddings)
- Measure: how much does caching reduce total API costs over 100 requests?

---

## 10. Interview / Architect Questions

### Q1: Wrong Tool Selection

> "The model keeps calling the wrong tool. It uses `get_weather` when the user asks about orders. How do you debug and fix this?"

**Strong answer covers:**

1. **Check tool descriptions** — they're likely ambiguous. Add explicit "use this when..." and "do NOT use this for..." clauses. The description is the #1 factor in tool selection.
2. **Check for description overlap** — if two tools have similar descriptions, the model can't distinguish them. Make each tool's purpose unique and specific.
3. **Log the full message history** — see what context the model had when it made the wrong choice. Was the user's message ambiguous?
4. **Isolate the problem** — use `tool_choice: {"type": "tool", "name": "search_orders"}` to confirm the tool itself works. If it does, the problem is selection, not execution.
5. **Reduce tool count** — if you have 20+ tools, selection accuracy drops. Group into categories or use dynamic tool loading.
6. **Test with the system prompt** — the system prompt may contain conflicting instructions that bias the model toward the wrong tool.

---

### Q2: Scaling to 40+ Tools

> "You're building a customer support agent with 40 tools. Response quality drops as you add more tools. What's your architecture?"

**Strong answer covers:**

Don't send all 40 tools in every request. Use a **two-tier approach**:

**Tier 1 — Category Router (5-6 tools):**
```python
router_tools = [
    {"name": "billing_tools", "description": "For billing, payments, refunds..."},
    {"name": "shipping_tools", "description": "For delivery, tracking, returns..."},
    {"name": "account_tools", "description": "For profile, settings, password..."},
]
```

**Tier 2 — Category-Specific Tools (6-8 per category):**
```python
billing_tools = [
    {"name": "check_balance", ...},
    {"name": "process_refund", ...},
    {"name": "update_payment_method", ...},
]
```

Alternative approaches:
- **Embedding-based tool selection:** Embed all 40 tool descriptions. For each user query, compute similarity and load the top-K most relevant tools.
- **Multi-agent:** Split into specialized agents (billing agent, shipping agent), each with their own tool set. A supervisor routes to the right agent.
- **Dynamic loading with MCP:** Tools register themselves dynamically. The orchestrator discovers and loads only what's needed.

---

### Q3: Side Effects in a Retry Loop

> "A tool call mutates state (processes a refund), but the model's final response fails to generate due to a timeout. The refund went through but the user sees an error. How do you handle this?"

**Strong answer covers:**

This is the classic "side-effect in a retry loop" problem from distributed systems:

1. **Idempotency keys** — Make all mutating tools idempotent. The refund API accepts an idempotency key derived from `(order_id, amount, timestamp)`. If the same refund is requested twice, the second call returns the existing refund instead of creating a new one.

2. **Two-phase execution** — First create a "pending refund" (reversible). Only confirm it after the full response succeeds. If the response fails, roll back the pending refund.

3. **Tool execution log** — Store every tool execution result in a durable log keyed by request ID. If the final response times out, the next attempt can check the log: "refund already processed for this request → skip re-execution."

4. **Compensating actions** — If the refund went through but the user saw an error, detect this on retry and inform the user: "Your refund was actually processed. Here's the confirmation."

5. **Never retry mutating tool calls blindly** — always check if the previous attempt succeeded before re-executing.

---

### Q4: Token Cost of Tool Use

> "How does tool use affect token costs, and what strategies reduce spend in a high-volume system?"

**Strong answer covers:**

**The cost multiplier:**
- Tool definitions are injected as input tokens on EVERY API call in the loop
- 10 tools with detailed schemas ≈ 2,000-3,000 extra input tokens per call
- 5 loop iterations = definitions counted 5 times = 10,000-15,000 tokens just for tools
- Plus: tool results add to the message history, growing input size each iteration

**Cost reduction strategies:**

| Strategy | Savings | Trade-off |
|---|---|---|
| Minimize description length | 20-40% tool token reduction | Risk: lower selection accuracy |
| Dynamic tool loading (top-K) | 60-80% fewer tool tokens | Complexity: need embedding-based selection |
| Use `tool_choice` when known | Skip selection reasoning | Only works when you know the tool in advance |
| Cache tool results | Fewer loop iterations | Stale results if TTL too long |
| Better system prompts | Fewer iterations (model gets it right faster) | Prompt engineering effort |
| Model routing | Use Haiku for simple tool-selection routing | Added latency from two-stage call |

**Back-of-envelope calculation:**

```
Without optimization:
  10 tools × 300 tokens/tool = 3,000 tokens/call
  × 3 iterations avg = 9,000 extra input tokens
  × 100K requests/day = 900M extra tokens/day
  × $3/M tokens = $2,700/day = $81,000/month just for tool definitions

With dynamic loading (top-3 tools):
  3 tools × 300 tokens = 900 tokens/call
  × 2 iterations avg = 1,800 extra input tokens
  × 100K requests/day = 180M extra tokens/day
  × $3/M tokens = $540/day = $16,200/month
  Savings: ~80%
```

---

### Q5: Tool Use vs. RAG — When to Use Each, and When to Combine

> "Compare tool use with RAG. When would you use each, and when would you combine them?"

**Strong answer covers:**

| Dimension | RAG | Tool Use |
|---|---|---|
| **Timing** | Before generation (retrieval phase) | During generation (model requests it) |
| **Agency** | Passive — retrieval system decides what's relevant | Active — the model decides what it needs |
| **Data type** | Unstructured documents (docs, articles, policies) | Structured operations (APIs, databases, calculations) |
| **Latency** | Added upfront (retrieval before first token) | Added mid-response (pauses generation) |
| **Best for** | Knowledge questions ("What's our refund policy?") | Action questions ("Process a refund") |
| **Freshness** | As fresh as your index (hours-days lag) | Real-time (queries live systems) |
| **Cost** | Embedding + retrieval cost per query | API call cost per tool invocation |

**When to use RAG:** You have a knowledge base (documentation, policies, FAQs) that the model should reference for grounding. The data is relatively static and can be pre-indexed.

**When to use tool use:** The model needs to take actions, query live data, or perform computations. The data is dynamic or user-specific.

**When to combine both:**

```
User: "Am I eligible for a refund on order #1234?"

Step 1 (RAG): Retrieve refund policy from documentation
  → "Refunds allowed within 30 days of delivery for non-sale items"

Step 2 (Tool Use): Model calls search_orders("order #1234")
  → "Delivered 5 days ago, item: laptop (non-sale), $1,499"

Step 3 (Model): Combines policy (from RAG) with order data (from tool)
  → "Yes, you're eligible! Your laptop was delivered 5 days ago, well within
     our 30-day refund window, and it's not a sale item."
```

RAG provides the rules; tool use provides the data to apply them to.

---

## Quick Reference Card

```
┌──────────────────────────────────────────────────────────────┐
│               TOOL USE ENGINEERING CHEATSHEET                │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  TOOL DEFINITION:                                           │
│    name:         unique identifier                          │
│    description:  MOST IMPORTANT — drives selection accuracy  │
│    input_schema: JSON Schema for parameters                 │
│                                                              │
│  AGENTIC LOOP:                                              │
│    while not done and under budget:                         │
│      response = llm(messages, tools)                        │
│      if end_turn → return response                          │
│      if tool_use → execute → append results → loop          │
│                                                              │
│  MESSAGE RULES:                                             │
│    • Alternate user ↔ assistant                             │
│    • Append FULL assistant content (text + tool_use)        │
│    • Match tool_use_id in every tool_result                 │
│    • Send ALL tool_results in one user message              │
│                                                              │
│  tool_choice OPTIONS:                                       │
│    auto  → model decides (default)                          │
│    any   → must use some tool                               │
│    tool  → must use specific tool                           │
│                                                              │
│  SECURITY:                                                  │
│    • Model output is UNTRUSTED INPUT                        │
│    • Never eval() model-generated code                      │
│    • Validate tool arguments before executing               │
│    • Idempotency keys for mutating operations               │
│                                                              │
│  COST FORMULA:                                              │
│    tool_tokens = num_tools × avg_tokens_per_tool            │
│    loop_cost = iterations × (tool_tokens + history_tokens)  │
│    Minimize: fewer tools, fewer iterations, cache results   │
│                                                              │
│  GOLDEN RULES:                                              │
│    1. Description quality = selection accuracy              │
│    2. Always guard the loop with max_iterations             │
│    3. Append FULL assistant message, not just tool blocks   │
│    4. Handle parallel tool calls (multiple tool_use blocks) │
│    5. Log every tool call — it's your debugging lifeline    │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

---

**Previous Module:** [01 — How LLMs Work](../01-how-llms-work/)

**Next Module:** Embeddings & Vector Search — how to convert text to vectors and build RAG systems.
