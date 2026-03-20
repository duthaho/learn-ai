"""
Tool definitions and implementations for the Tool Use demo.

This module separates tool DEFINITIONS (what the model sees) from
tool IMPLEMENTATIONS (what your code executes). This separation is
critical in production — it's the same pattern as OpenAPI specs vs handlers.
"""

import random
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Tool definitions — these are sent to the Claude API as JSON schemas.
# The model reads these to decide WHEN and HOW to call each tool.
#
# Key insight: the quality of descriptions directly affects tool selection
# accuracy. Vague descriptions = wrong tool calls.
# ---------------------------------------------------------------------------

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
    {
        "name": "search_orders",
        "description": (
            "Search for customer orders by email address. Returns a list "
            "of recent orders with their status. Use this when the user "
            "asks about their orders, deliveries, or purchase history."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "email": {
                    "type": "string",
                    "description": "Customer email address",
                },
                "status_filter": {
                    "type": "string",
                    "enum": ["all", "pending", "shipped", "delivered"],
                    "description": "Filter orders by status. Default: all.",
                },
            },
            "required": ["email"],
        },
    },
    {
        "name": "calculate",
        "description": (
            "Perform a mathematical calculation. Supports basic arithmetic "
            "and common functions. Use this instead of doing math yourself "
            "— LLMs are unreliable at arithmetic."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": (
                        "Math expression to evaluate, e.g. '(15 * 3) + 7.5'. "
                        "Supports: +, -, *, /, **, sqrt(), abs(), round()"
                    ),
                },
            },
            "required": ["expression"],
        },
    },
]


# ---------------------------------------------------------------------------
# Tool implementations — the actual business logic.
# In production, these would call real databases, APIs, microservices.
# Here we use fake data to focus on the tool use mechanics.
# ---------------------------------------------------------------------------

# Simulated weather data
WEATHER_DATA = {
    "tokyo": {"temp_c": 22, "condition": "Partly Cloudy", "humidity": 65},
    "new york": {"temp_c": 18, "condition": "Sunny", "humidity": 45},
    "london": {"temp_c": 14, "condition": "Rainy", "humidity": 80},
    "sydney": {"temp_c": 26, "condition": "Clear", "humidity": 55},
    "paris": {"temp_c": 16, "condition": "Overcast", "humidity": 70},
}

# Simulated order data
ORDER_DB = {
    "alice@example.com": [
        {"order_id": "ORD-1001", "item": "Mechanical Keyboard", "status": "delivered", "total": 149.99},
        {"order_id": "ORD-1042", "item": "USB-C Hub", "status": "shipped", "total": 49.99},
    ],
    "bob@example.com": [
        {"order_id": "ORD-1015", "item": "Standing Desk", "status": "pending", "total": 599.00},
        {"order_id": "ORD-1033", "item": "Monitor Arm", "status": "delivered", "total": 89.99},
        {"order_id": "ORD-1050", "item": "Webcam HD", "status": "shipped", "total": 79.99},
    ],
}

# Safe math operations whitelist — NEVER use eval() with untrusted input
import math

SAFE_MATH = {
    "sqrt": math.sqrt,
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
}


def execute_tool(tool_name: str, tool_input: dict) -> str:
    """
    Dispatch tool calls to their implementations.

    Returns a string result that gets sent back to the model.
    In production, this is where you'd add:
    - Authentication/authorization checks
    - Rate limiting
    - Input validation
    - Audit logging
    - Error handling with user-friendly messages
    """
    if tool_name == "get_weather":
        return _get_weather(tool_input)
    elif tool_name == "search_orders":
        return _search_orders(tool_input)
    elif tool_name == "calculate":
        return _calculate(tool_input)
    else:
        return f"Error: Unknown tool '{tool_name}'"


def _get_weather(params: dict) -> str:
    city = params["city"].lower().strip()
    unit = params.get("unit", "celsius")

    data = WEATHER_DATA.get(city)
    if not data:
        # Simulate unknown city with random data
        data = {
            "temp_c": random.randint(5, 35),
            "condition": random.choice(["Sunny", "Cloudy", "Rainy"]),
            "humidity": random.randint(30, 90),
        }

    temp = data["temp_c"]
    if unit == "fahrenheit":
        temp = round(temp * 9 / 5 + 32, 1)

    return (
        f"Weather in {params['city']}: {temp}°{'F' if unit == 'fahrenheit' else 'C'}, "
        f"{data['condition']}, Humidity: {data['humidity']}%"
    )


def _search_orders(params: dict) -> str:
    email = params["email"].lower().strip()
    status_filter = params.get("status_filter", "all")

    orders = ORDER_DB.get(email, [])
    if not orders:
        return f"No orders found for {email}"

    if status_filter != "all":
        orders = [o for o in orders if o["status"] == status_filter]

    if not orders:
        return f"No {status_filter} orders found for {email}"

    lines = [f"Orders for {email}:"]
    for o in orders:
        lines.append(
            f"  - {o['order_id']}: {o['item']} | ${o['total']:.2f} | {o['status']}"
        )
    return "\n".join(lines)


def _calculate(params: dict) -> str:
    expression = params["expression"]

    # Security: parse and evaluate safely — NEVER use raw eval()
    # In production, use a proper math parser like `asteval` or `sympy`
    try:
        # Only allow safe characters and functions
        allowed_chars = set("0123456789+-*/.(,) ")
        allowed_words = {"sqrt", "abs", "round", "min", "max"}

        # Extract words from expression
        import re
        words = set(re.findall(r"[a-zA-Z_]+", expression))
        if not words.issubset(allowed_words):
            unsafe = words - allowed_words
            return f"Error: Unsafe operations not allowed: {unsafe}"

        # Evaluate with restricted builtins
        result = eval(expression, {"__builtins__": {}}, SAFE_MATH)  # noqa: S307
        return f"Result: {result}"
    except Exception as e:
        return f"Calculation error: {e}"
