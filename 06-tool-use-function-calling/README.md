# Module 06 — Tool Use & Function Calling

Giving LLMs the ability to call external functions: tool definitions, the execution loop, and building reliable tool-using systems.

| Detail        | Value                                     |
|---------------|-------------------------------------------|
| Level         | Intermediate                              |
| Time          | ~2 hours                                  |
| Prerequisites | Module 04 (The AI API Layer)              |

## What you'll build

After reading this module, head to [`project/`](project/) to build an **AI Assistant with Tools** — a CLI assistant with real tools (time, math, weather, file reading) that the LLM decides when and how to use.

---

## Table of Contents

1. [What is Tool Use?](#1-what-is-tool-use)
2. [Defining Tools](#2-defining-tools)
3. [The Tool Use Loop](#3-the-tool-use-loop)
4. [Tool Calls in the API](#4-tool-calls-in-the-api)
5. [Tool Choice](#5-tool-choice)
6. [Parallel Tool Calls](#6-parallel-tool-calls)
7. [Error Handling](#7-error-handling)
8. [Designing Good Tools](#8-designing-good-tools)

---

## 1. What is Tool Use?

LLMs are powerful text generators, but they have fundamental limitations: they can't check the current time, perform reliable calculations, access live data, or take actions in the world. Their knowledge is frozen at training time.

**Tool use** solves this by letting you define functions the LLM can request to call. The LLM doesn't execute the functions itself — it tells you which function to call and with what arguments, you execute it, and you send the result back. The LLM then uses that result to formulate its response.

### The mental model

Think of it as a collaboration:
- **The LLM** is the brain — it decides what needs to happen and interprets results
- **Tools** are the hands — they interact with the real world
- **Your code** is the nervous system — it connects the brain to the hands

### The tool use loop (overview)

```
1. You send: messages + tool definitions
2. LLM returns: "I want to call get_weather(location='London')"
3. You execute: get_weather("London") → "72°F, sunny"
4. You send: the result back to the LLM
5. LLM returns: "It's currently 72°F and sunny in London."
```

The LLM never sees your function code. It only sees the tool's name, description, and parameter schema — and decides based on that whether and how to call it.

### Tools vs agents

Tools are the primitive. An agent (Module 11) is a loop that uses tools autonomously to accomplish goals. This module teaches the foundation — how tools work at the API level.

---

## 2. Defining Tools

Tools are defined as JSON schemas that describe what each function does and what parameters it accepts. You send these definitions alongside your messages, and the LLM reads them to decide which tool to call.

### The tool definition structure

```json
{
  "type": "function",
  "function": {
    "name": "get_weather",
    "description": "Get the current weather for a city. Returns temperature and conditions.",
    "parameters": {
      "type": "object",
      "properties": {
        "location": {
          "type": "string",
          "description": "The city name, e.g., 'London' or 'Tokyo'"
        },
        "unit": {
          "type": "string",
          "enum": ["celsius", "fahrenheit"],
          "description": "Temperature unit. Defaults to celsius if not specified."
        }
      },
      "required": ["location"]
    }
  }
}
```

### Breaking it down

| Field | Purpose |
|-------|---------|
| `name` | The function identifier. Use `snake_case`. The LLM uses this to request the call. |
| `description` | What the tool does and when to use it. **This is what the LLM reads to decide.** |
| `parameters` | JSON Schema defining accepted arguments. |
| `required` | Which parameters must be provided. Optional parameters get defaults. |

### Parameter types

| Type | Example | Use for |
|------|---------|---------|
| `string` | `"location": {"type": "string"}` | Text inputs, names, queries |
| `number` | `"amount": {"type": "number"}` | Decimal values, prices |
| `integer` | `"count": {"type": "integer"}` | Whole numbers |
| `boolean` | `"verbose": {"type": "boolean"}` | Flags, on/off switches |
| `array` | `"tags": {"type": "array", "items": {"type": "string"}}` | Lists of values |
| `enum` | `"unit": {"type": "string", "enum": ["c", "f"]}` | Fixed set of options |

### Common tool examples

**No parameters:**
```json
{
  "type": "function",
  "function": {
    "name": "get_current_time",
    "description": "Get the current date and time. Use when the user asks about the current time or date.",
    "parameters": {"type": "object", "properties": {}}
  }
}
```

**Simple parameter:**
```json
{
  "type": "function",
  "function": {
    "name": "calculate",
    "description": "Evaluate a mathematical expression. Use for any math calculations. Supports +, -, *, /, **, sqrt(), etc.",
    "parameters": {
      "type": "object",
      "properties": {
        "expression": {
          "type": "string",
          "description": "The math expression to evaluate, e.g., '2 + 3 * 4' or 'sqrt(16)'"
        }
      },
      "required": ["expression"]
    }
  }
}
```

### Descriptions matter

The description is the most important field. It's what the LLM reads to decide whether to call the tool. Compare:

- **Bad:** `"description": "Weather function"` — the LLM doesn't know when to use it or what it returns
- **Good:** `"description": "Get the current weather for a city. Returns temperature and conditions. Use when the user asks about weather, temperature, or outdoor conditions."` — clear purpose, clear trigger, clear output

### Provider format differences

**OpenAI** wraps tools in `{"type": "function", "function": {...}}`:
```json
{"type": "function", "function": {"name": "get_weather", "description": "...", "parameters": {...}}}
```

**Anthropic** uses `input_schema` instead of `parameters` and no wrapper:
```json
{"name": "get_weather", "description": "...", "input_schema": {...}}
```

**LiteLLM** accepts the OpenAI format and translates automatically for all providers. Always use the OpenAI format.

---

## 3. The Tool Use Loop

Tool use is not a single API call — it's a loop. The LLM requests tool calls, you execute them, send results back, and the LLM may request more calls or give a final answer.

### The full flow

```
┌─────────────────────────────────────────────────────┐
│  1. Send messages + tools to LLM                    │
│  2. LLM responds                                    │
│     ├── finish_reason == "stop"  → return text      │
│     └── finish_reason == "tool_calls" → continue    │
│  3. Extract tool_calls from response                │
│  4. Execute each function locally                   │
│  5. Append assistant message (with tool_calls)      │
│  6. Append tool result messages                     │
│  7. Go to step 1                                    │
└─────────────────────────────────────────────────────┘
```

### The critical pattern

You must append **two things** to the conversation history:
1. The assistant's message containing `tool_calls` — this is the LLM saying "I want to call these functions"
2. The tool result messages — this is you saying "here are the results"

If you skip the assistant message, the API returns an error because tool results reference orphaned `tool_call_id`s. If you skip the tool results, the LLM doesn't know what happened.

### Code implementation

```python
import json
from litellm import completion

def run_tool_loop(messages, tools, available_functions, model):
    """Run the tool use loop until the LLM gives a final text response."""
    while True:
        response = completion(model=model, messages=messages, tools=tools)
        message = response.choices[0].message
        finish_reason = response.choices[0].finish_reason

        # If no tool calls, we're done
        if finish_reason != "tool_calls":
            return message.content

        # Append the assistant's tool_calls message
        messages.append(message)

        # Execute each tool call
        for tool_call in message.tool_calls:
            name = tool_call.function.name
            args = json.loads(tool_call.function.arguments)

            # Call the actual function
            result = available_functions[name](**args)

            # Append the tool result
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": str(result),
            })

        # Loop back — LLM will see the results and either
        # call more tools or give a final answer
```

### Multi-round tool calls

The LLM might need multiple rounds. Example:

```
User: "What's 15% tip on my dinner and is the weather good for walking?"

Round 1: LLM calls calculate("85 * 0.15")
         You return: "12.75"

Round 2: LLM calls get_weather("New York")
         You return: "72°F, sunny"

Round 3: LLM gives final answer:
         "The tip is $12.75. The weather is 72°F and sunny — perfect for a walk!"
```

The loop handles this naturally — it keeps going until `finish_reason` is `"stop"`.

---

## 4. Tool Calls in the API

Understanding the exact response format helps you parse tool calls correctly.

### OpenAI/LiteLLM response format

When the LLM wants to call tools, the response looks like:

```json
{
  "choices": [{
    "message": {
      "role": "assistant",
      "content": null,
      "tool_calls": [
        {
          "id": "call_abc123",
          "type": "function",
          "function": {
            "name": "get_weather",
            "arguments": "{\"location\": \"London\", \"unit\": \"celsius\"}"
          }
        }
      ]
    },
    "finish_reason": "tool_calls"
  }]
}
```

### Key details

**`content` is `null`** — when the LLM calls tools, it typically doesn't include text content (though it can sometimes include both text and tool calls).

**`arguments` is a JSON string** — not a dict. You must parse it:
```python
args = json.loads(tool_call.function.arguments)
# args is now {"location": "London", "unit": "celsius"}
```

**`id` is unique** — each tool call gets a unique ID. Your tool result must reference this exact ID.

**`finish_reason` is `"tool_calls"`** — this is how you detect that the LLM wants tools instead of giving a text response. Always check this field.

### Tool result message format

After executing the function, send the result back:
```python
{
    "role": "tool",
    "tool_call_id": "call_abc123",  # must match the tool_call's id
    "content": "72°F, sunny"        # always a string
}
```

The `content` must be a string. If your function returns a dict or complex object, use `json.dumps()`:
```python
result = {"temp": 72, "conditions": "sunny", "humidity": 45}
messages.append({
    "role": "tool",
    "tool_call_id": tool_call.id,
    "content": json.dumps(result),
})
```

### How Anthropic differs (under the hood)

Anthropic uses content blocks instead of a `tool_calls` array:

```json
{
  "content": [
    {"type": "text", "text": "Let me check the weather."},
    {"type": "tool_use", "id": "toolu_abc", "name": "get_weather", "input": {"location": "London"}}
  ],
  "stop_reason": "tool_use"
}
```

And tool results are sent as `tool_result` content blocks in a `user` message, not as a `tool` role message.

**You don't need to handle this.** LiteLLM normalizes everything to the OpenAI format — `tool_calls` array, `finish_reason: "tool_calls"`, and `tool` role messages. Write your code once using the OpenAI format.

---

## 5. Tool Choice

The `tool_choice` parameter controls whether and how the LLM uses tools.

### Options

| Value | Behavior |
|-------|----------|
| `"auto"` (default) | LLM decides — may use tools or respond directly |
| `"required"` | Must call at least one tool (will never respond with text only) |
| `"none"` | Cannot use tools (even if tools are provided) |
| `{"type": "function", "function": {"name": "get_weather"}}` | Must call this specific tool |

### When to use each

**`auto`** — the default, and usually the right choice. The LLM decides when tools are useful:
```python
# "What's 2+2?" → LLM might use calculate() or just answer directly
# "Tell me a joke" → LLM responds without tools
response = completion(model=model, messages=messages, tools=tools, tool_choice="auto")
```

**`required`** — when you know the request needs a tool:
```python
# User asked "what time is it?" — we know this needs get_current_time
response = completion(model=model, messages=messages, tools=tools, tool_choice="required")
```

**`none`** — when you want the LLM to think without acting:
```python
# "Plan what tools you'd need" — reason about tools without calling them
response = completion(model=model, messages=messages, tools=tools, tool_choice="none")
```

**Specific tool** — when you need a particular function called:
```python
# Force weather check regardless of the prompt
response = completion(
    model=model, messages=messages, tools=tools,
    tool_choice={"type": "function", "function": {"name": "get_weather"}}
)
```

---

## 6. Parallel Tool Calls

The LLM can request multiple tool calls in a single response. This happens when the prompt requires independent pieces of information.

### Example

Prompt: "What's the weather in London and Tokyo?"

Response:
```json
{
  "tool_calls": [
    {"id": "call_1", "function": {"name": "get_weather", "arguments": "{\"location\": \"London\"}"}},
    {"id": "call_2", "function": {"name": "get_weather", "arguments": "{\"location\": \"Tokyo\"}"}}
  ]
}
```

### Handling parallel calls

Execute all tool calls and send all results back before the next LLM call. Each result must reference the correct `id`:

```python
for tool_call in message.tool_calls:
    name = tool_call.function.name
    args = json.loads(tool_call.function.arguments)
    result = available_functions[name](**args)
    messages.append({
        "role": "tool",
        "tool_call_id": tool_call.id,
        "content": str(result),
    })
```

The loop code from Section 3 already handles this — it iterates over all `tool_calls`.

### Controlling parallel behavior

OpenAI supports `parallel_tool_calls=False` to force one tool call per response. This is useful when tool calls have dependencies (tool B needs tool A's output):

```python
response = completion(
    model=model, messages=messages, tools=tools,
    parallel_tool_calls=False  # one tool at a time
)
```

By default, the LLM may call multiple tools at once when it determines they're independent.

---

## 7. Error Handling

Tool execution can fail — networks time out, files don't exist, expressions are invalid. The key principle: **send the error back to the LLM as a tool result.** The LLM can often recover.

### Sending errors as results

```python
for tool_call in message.tool_calls:
    name = tool_call.function.name
    try:
        args = json.loads(tool_call.function.arguments)
        result = available_functions[name](**args)
    except json.JSONDecodeError:
        result = "Error: malformed arguments. Please provide valid input."
    except KeyError:
        result = f"Error: unknown tool '{name}'."
    except Exception as e:
        result = f"Error: {e}"

    messages.append({
        "role": "tool",
        "tool_call_id": tool_call.id,
        "content": str(result),
    })
```

### The LLM self-corrects

When the LLM receives an error, it typically:
1. **Retries with corrected arguments** — "Error: invalid expression 'sqrt(banana)'" → tries "sqrt(16)" instead
2. **Asks the user for clarification** — "I couldn't find that city. Could you provide the full city name?"
3. **Falls back to a different approach** — tries another tool or answers without tools

This is why you send errors as results instead of raising exceptions — the loop continues, and the LLM adapts.

### Malformed JSON arguments

The LLM sometimes generates invalid JSON in `function.arguments`. Always wrap `json.loads()` in a try/except:

```python
try:
    args = json.loads(tool_call.function.arguments)
except json.JSONDecodeError:
    # Tell the LLM its arguments were invalid
    messages.append({
        "role": "tool",
        "tool_call_id": tool_call.id,
        "content": "Error: could not parse arguments. Please provide valid JSON.",
    })
    continue
```

### Security: validate before executing

The LLM can hallucinate arguments. Never blindly execute:
- **File operations** — validate paths, check for path traversal (`../../../etc/passwd`)
- **Shell commands** — never pass LLM arguments to `os.system()` or `subprocess`
- **Database queries** — use parameterized queries, not string concatenation
- **Calculations** — use a safe evaluator, not `eval()`

---

## 8. Designing Good Tools

Well-designed tools make the LLM more reliable. Poorly designed tools lead to wrong tool selection, bad arguments, and frustrating debugging.

### Naming

Use `verb_noun` format. Be specific:
- **Good:** `get_weather`, `search_files`, `calculate_expression`, `send_email`
- **Bad:** `tool1`, `helper`, `do_stuff`, `process`

### Descriptions

Write for the LLM, not for humans. Include:
- **What** it does — "Get the current weather for a city"
- **When** to use it — "Use when the user asks about weather, temperature, or outdoor conditions"
- **What** it returns — "Returns temperature, conditions, and humidity"
- **Limitations** — "Only works for cities, not addresses or coordinates"

### Parameters

- **Prefer flat schemas** — `{"city": "string", "unit": "string"}` over `{"location": {"city": "string", "country": "string", "coordinates": {...}}}`
- **Use enums** when options are known — `"enum": ["celsius", "fahrenheit"]` instead of free text
- **Include parameter descriptions** — helps the LLM provide correct values
- **Mark required fields** — the LLM respects `required` and fills in optional fields only when needed

### Granularity

- **One tool per action** — `get_weather` and `get_forecast` instead of `weather_operations(action="get"|"forecast")`
- **Don't make a "do_everything" tool** — it confuses the LLM's decision-making
- **Don't split too fine** — `get_temperature`, `get_humidity`, `get_wind_speed` → just `get_weather`
- **Rule of thumb:** if you'd explain it as one sentence ("get the weather for a city"), it's one tool

### Tool count

More tools means more tokens in every request (tool definitions count as input tokens) and more confusion for the LLM. Practical limits:
- **5-10 tools** — works well, LLM reliably picks the right one
- **10-20 tools** — usually fine with good descriptions
- **20+ tools** — consider grouping or dynamically loading only relevant tools per request

### Testing

Test your tools with ambiguous prompts:
- "What's the weather?" — does the LLM ask for a city or guess?
- "Calculate 2+2 and tell me the time" — does it call both tools?
- "Tell me about London" — does it call get_weather or just answer from knowledge?

If the LLM picks the wrong tool, improve the description first. Description quality is the number one factor in tool selection accuracy.
