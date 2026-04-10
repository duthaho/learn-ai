"""
AI Assistant with Tools — Module 06 Project (Solution)

A CLI assistant with real tools: time, math, weather, and file reading.
The LLM decides which tools to call and when.

Run: python solution.py
"""

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


# ---------------------------------------------------------------------------
# Step 1: Tool functions
# ---------------------------------------------------------------------------

def get_current_time() -> str:
    """Return the current date and time."""
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def calculate(expression: str) -> str:
    """Safely evaluate a mathematical expression."""
    # Allow only safe math operations
    allowed_names = {
        "sqrt": math.sqrt,
        "pow": math.pow,
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
        "pi": math.pi,
        "e": math.e,
    }
    try:
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return str(result)
    except Exception as e:
        raise ValueError(f"invalid expression: {e}")


def get_weather(location: str, unit: str = "celsius") -> str:
    """Return simulated weather data for a city."""
    # Simulated weather — deterministic based on city name length
    seed = sum(ord(c) for c in location.lower())
    temp_c = 10 + (seed % 25)  # 10-34°C range
    conditions = ["sunny", "cloudy", "rainy", "partly cloudy", "windy"]
    condition = conditions[seed % len(conditions)]

    if unit == "fahrenheit":
        temp = round(temp_c * 9 / 5 + 32)
        return f"{location}: {temp}°F, {condition}"
    return f"{location}: {temp_c}°C, {condition}"


def read_file(path: str) -> str:
    """Read and return the contents of a file."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    content = file_path.read_text(encoding="utf-8")
    if len(content) > 2000:
        return content[:2000] + f"\n... (truncated, {len(content)} total chars)"
    return content


# ---------------------------------------------------------------------------
# Tool definitions (OpenAI format)
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Get the current date and time. Use when the user asks about the current time or date.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate a mathematical expression. Use for any math calculations. Supports +, -, *, /, **, sqrt(), pow(), abs(), round(), min(), max(), pi, e.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The math expression to evaluate, e.g., '2 + 3 * 4' or 'sqrt(16)'",
                    },
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city. Returns temperature and conditions. Use when the user asks about weather, temperature, or outdoor conditions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city name, e.g., 'London' or 'Tokyo'",
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature unit. Defaults to celsius.",
                    },
                },
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file. Use when the user asks to read, view, or show a file's contents.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "The file path to read",
                    },
                },
                "required": ["path"],
            },
        },
    },
]

AVAILABLE_FUNCTIONS = {
    "get_current_time": get_current_time,
    "calculate": calculate,
    "get_weather": get_weather,
    "read_file": read_file,
}


# ---------------------------------------------------------------------------
# Step 2 & 3: Tool execution and loop
# ---------------------------------------------------------------------------

def execute_tool_call(tool_call) -> str:
    """Execute a single tool call and return the result string."""
    name = tool_call.function.name
    try:
        args = json.loads(tool_call.function.arguments)
    except json.JSONDecodeError:
        return "Error: could not parse arguments. Please provide valid JSON."

    func = AVAILABLE_FUNCTIONS.get(name)
    if not func:
        return f"Error: unknown tool '{name}'."

    try:
        result = func(**args)
        return str(result)
    except Exception as e:
        return f"Error: {e}"


def run_tool_loop(messages: list[dict], model: str = MODEL) -> str:
    """Run the tool use loop until the LLM gives a final text response."""
    while True:
        response = completion(model=model, messages=messages, tools=TOOLS)
        message = response.choices[0].message
        finish_reason = response.choices[0].finish_reason

        if finish_reason != "tool_calls":
            return message.content

        # Append assistant message with tool_calls
        messages.append(message)

        # Execute each tool call
        for tool_call in message.tool_calls:
            result = execute_tool_call(tool_call)
            is_error = result.startswith("Error:")
            label = "[Error]    " if is_error else "[Result]   "
            print(f"  {label} {result}")

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
            })


def ask_with_tools(prompt: str, system: str = "", model: str = MODEL) -> str:
    """Send a prompt with tools and return the final response."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    print(f"\n  You: {prompt}\n")

    # First call
    response = completion(model=model, messages=messages, tools=TOOLS)
    message = response.choices[0].message
    finish_reason = response.choices[0].finish_reason

    if finish_reason != "tool_calls":
        return message.content

    # Print tool calls
    messages.append(message)
    for tool_call in message.tool_calls:
        name = tool_call.function.name
        args = tool_call.function.arguments
        print(f"  [Tool call] {name}({args})")

        result = execute_tool_call(tool_call)
        is_error = result.startswith("Error:")
        label = "[Error]    " if is_error else "[Result]   "
        print(f"  {label} {result}")

        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": result,
        })

    # Continue the loop if needed
    while True:
        response = completion(model=model, messages=messages, tools=TOOLS)
        message = response.choices[0].message
        finish_reason = response.choices[0].finish_reason

        if finish_reason != "tool_calls":
            return message.content

        messages.append(message)
        for tool_call in message.tool_calls:
            name = tool_call.function.name
            args = tool_call.function.arguments
            print(f"  [Tool call] {name}({args})")

            result = execute_tool_call(tool_call)
            is_error = result.startswith("Error:")
            label = "[Error]    " if is_error else "[Result]   "
            print(f"  {label} {result}")

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
            })


# ---------------------------------------------------------------------------
# Step 5: Tool choice demo
# ---------------------------------------------------------------------------

def demo_tool_choice(prompt: str, model: str = MODEL) -> None:
    """Demonstrate different tool_choice settings."""
    print(f'  Prompt: "{prompt}"\n')

    for choice_name, choice_value in [
        ("auto", "auto"),
        ("required", "required"),
        ("none", "none"),
    ]:
        messages = [{"role": "user", "content": prompt}]
        response = completion(
            model=model,
            messages=messages,
            tools=TOOLS,
            tool_choice=choice_value,
        )
        message = response.choices[0].message
        finish_reason = response.choices[0].finish_reason

        if finish_reason == "tool_calls":
            tool_call = message.tool_calls[0]
            # Execute and get final response
            messages.append(message)
            result = execute_tool_call(tool_call)
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
            })
            final = completion(model=model, messages=messages, tools=TOOLS)
            text = final.choices[0].message.content
            print(f'  tool_choice="{choice_name}": [Tool: {tool_call.function.name}] → {text[:80]}')
        else:
            text = message.content
            print(f'  tool_choice="{choice_name}": {text[:80]}')

    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    tool_names = ", ".join(f["function"]["name"] for f in TOOLS)

    print("=" * 60)
    print("  AI Assistant with Tools")
    print("=" * 60)
    print(f"  Model: {MODEL}")
    print(f"  Tools: {tool_names}")

    # --- 1. Single Tool Call ---
    print("\n--- 1. Single Tool Call ---")
    result = ask_with_tools("What time is it?")
    print(f"\n  Assistant: {result}")

    # --- 2. Tool Loop (multi-step) ---
    print("\n--- 2. Tool Loop (multi-step) ---")
    result = ask_with_tools(
        "What's 15% tip on an $85 dinner? Also, what time is it?"
    )
    print(f"\n  Assistant: {result}")

    # --- 3. Parallel Tool Calls ---
    print("\n--- 3. Parallel Tool Calls ---")
    result = ask_with_tools("Compare the weather in London and Tokyo.")
    print(f"\n  Assistant: {result}")

    # --- 4. Tool Choice ---
    print("\n--- 4. Tool Choice ---\n")
    demo_tool_choice("What is 2 + 2?")

    # --- 5. Error Handling ---
    print("--- 5. Error Handling ---")
    result = ask_with_tools("Calculate the square root of banana")
    print(f"\n  Assistant: {result}")

    result = ask_with_tools("Read the file /nonexistent/path.txt")
    print(f"\n  Assistant: {result}")

    print("\n" + "=" * 60)
    print("  Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
