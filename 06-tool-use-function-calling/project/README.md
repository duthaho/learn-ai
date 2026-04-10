# Project: AI Assistant with Tools

## What you'll build

A CLI assistant that has access to real tools — getting the current time, doing math, checking the weather (simulated), and reading files. You build the tool definitions, the execution loop, and watch the LLM decide which tools to call and when. This is the foundation pattern for building AI agents.

## Prerequisites

- Completed reading the Module 06 README
- Python 3.11+ with project dependencies installed (`pip install -r requirements.txt`)
- At least one LLM provider API key configured in `.env`

## How to build

Create a new file `assistant.py` in this directory. Build it step by step following the instructions below. When you're done, compare your output with `python solution.py`.

## Steps

### Step 1: Define tools and implement functions

Set up imports and define your tools:

```python
import os
import json
import math
import datetime
from pathlib import Path
from dotenv import load_dotenv
from litellm import completion
import litellm

load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

MODEL = os.getenv("LLM_MODEL", "anthropic/claude-sonnet-4-20250514")
```

Define 4 tool functions:
- `get_current_time()` — returns the current date and time as a formatted string
- `calculate(expression)` — safely evaluates a math expression (support basic operators, `sqrt`, `pow`, etc. but NOT `eval()` on raw input — use a safe approach)
- `get_weather(location, unit="celsius")` — returns simulated weather data (pick a random-ish temperature and condition based on the city name — this is a demo, not a real API)
- `read_file(path)` — reads and returns the contents of a file (with error handling for missing files)

Then define the tools list as JSON schemas in OpenAI format. Each tool needs: name, description (detailed enough for the LLM to know when to use it), and parameters.

Create a dict mapping function names to their Python functions:
```python
AVAILABLE_FUNCTIONS = {
    "get_current_time": get_current_time,
    "calculate": calculate,
    "get_weather": get_weather,
    "read_file": read_file,
}
```

Test: verify your functions work by calling them directly.

### Step 2: Single tool call

Send a prompt that triggers a tool call and handle the response:
- Build messages: `[{"role": "user", "content": "What time is it?"}]`
- Call `completion()` with `tools=TOOLS`
- Check `finish_reason` — it should be `"tool_calls"`
- Extract the tool call: `message.tool_calls[0]`
- Parse arguments: `json.loads(tool_call.function.arguments)`
- Execute the function
- Build the tool result message: `{"role": "tool", "tool_call_id": tool_call.id, "content": result}`
- Append both the assistant message and tool result to messages
- Call `completion()` again to get the final text response

Print each step: the tool call name and arguments, the result, and the final response.

### Step 3: The tool loop

Write a `run_tool_loop()` function:

```python
def run_tool_loop(messages, model=MODEL):
```

The function should:
- Call `completion()` with messages and tools
- Loop while `finish_reason == "tool_calls"`:
  - Append the assistant message to history
  - For each tool call: parse args, execute, append result
  - Call `completion()` again
- Return the final text response when `finish_reason == "stop"`

Test with a multi-step prompt: "What's 15% tip on an $85 dinner? Also, what time is it?" — the LLM should call both `calculate` and `get_current_time`.

### Step 4: Parallel tool calls

Test with prompts that trigger multiple tool calls at once:
- "What's the weather in London and Tokyo?"
- Verify `message.tool_calls` has 2 entries
- Both are executed and results sent back

Print each tool call and result to show the parallel execution.

### Step 5: Tool choice control

Demonstrate the `tool_choice` parameter by calling the same prompt with different settings:
- `tool_choice="auto"` — LLM decides (default behavior)
- `tool_choice="required"` — LLM must use a tool
- `tool_choice="none"` — LLM cannot use tools

Use a prompt like "What is 2 + 2?" and show how the LLM behaves differently with each setting.

### Step 6: Error handling

Handle tool execution failures gracefully:
- Wrap tool execution in try/except
- Send error messages back as tool results
- Watch the LLM recover

Test with deliberate errors:
- Invalid math: "Calculate the square root of banana"
- Missing file: "Read the file /nonexistent/path.txt"
- For each, show the error sent to the LLM and how it responds

## How to run

```bash
python assistant.py
```

Or compare with the reference:

```bash
python solution.py
```

## Expected output

```
============================================================
  AI Assistant with Tools
============================================================
  Model: anthropic/claude-sonnet-4-20250514
  Tools: get_current_time, calculate, get_weather, read_file

--- 1. Single Tool Call ---

  You: What time is it?

  [Tool call] get_current_time()
  [Result]    2025-04-10 14:30:00

  Assistant: It's currently 2:30 PM on April 10, 2025.

--- 2. Tool Loop (multi-step) ---

  You: What's 15% tip on an $85 dinner? Also, what time is it?

  [Tool call] calculate(expression="85 * 0.15")
  [Result]    12.75
  [Tool call] get_current_time()
  [Result]    2025-04-10 14:30:05

  Assistant: A 15% tip on $85 is $12.75, making the total $97.75.
  The current time is 2:30 PM on April 10, 2025.

--- 3. Parallel Tool Calls ---

  You: Compare the weather in London and Tokyo.

  [Tool call] get_weather(location="London", unit="celsius")
  [Tool call] get_weather(location="Tokyo", unit="celsius")
  [Result]    London: 15°C, rainy
  [Result]    Tokyo: 24°C, sunny

  Assistant: London is 15°C and rainy, while Tokyo is warmer at 24°C
  and sunny.

--- 4. Tool Choice ---

  Prompt: "What is 2 + 2?"

  tool_choice="auto":     (LLM may use calculate or answer directly)
  tool_choice="required": [Tool call] calculate(expression="2 + 2")
  tool_choice="none":     The answer is 4. (no tool used)

--- 5. Error Handling ---

  You: Calculate the square root of banana

  [Tool call] calculate(expression="sqrt(banana)")
  [Error]     invalid expression: name 'banana' is not defined

  Assistant: I couldn't calculate that — "banana" isn't a valid
  mathematical expression. Could you provide a numerical value?

  You: Read the file /nonexistent/path.txt

  [Tool call] read_file(path="/nonexistent/path.txt")
  [Error]     File not found: /nonexistent/path.txt

  Assistant: The file /nonexistent/path.txt doesn't exist.
  Could you check the path?

============================================================
  Done!
============================================================
```

## Stretch goals

1. **Add a new tool** — implement a `search_web(query)` tool (simulated — return hardcoded results) and see the LLM choose between it and existing tools for knowledge questions.
2. **Conversation mode** — wrap the tool loop in an interactive chat loop with `/bye` to quit, so you can have an ongoing conversation with the assistant.
3. **Tool call logging** — track all tool calls in the session and print a summary at the end: which tools were called, how many times, success/error rate.
