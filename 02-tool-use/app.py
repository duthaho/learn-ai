"""
Tool Use / Function Calling — Hands-on FastAPI Service

Demonstrates the complete tool use lifecycle:
1. Defining tools as JSON schemas
2. Sending tools to the Claude API
3. The agentic loop — detecting and executing tool calls
4. Feeding results back for final response generation
5. Multi-turn conversations with tool history
6. Parallel tool calls (model requests multiple tools at once)
"""

import os
from contextlib import asynccontextmanager

import anthropic
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel

from tools import TOOL_DEFINITIONS, execute_tool

load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize the Anthropic client on startup."""
    app.state.client = anthropic.Anthropic(
        api_key=os.environ["ANTHROPIC_API_KEY"],
    )
    yield


app = FastAPI(title="Tool Use / Function Calling", lifespan=lifespan)

MODEL = "claude-sonnet-4-20250514"

SYSTEM_PROMPT = """\
You are a helpful customer support assistant. You have access to tools for:
- Checking weather in any city
- Looking up customer orders by email
- Performing calculations

Always use tools when you need factual data — never guess or make up information.
If you need multiple pieces of information, request all relevant tools."""


# ---------------------------------------------------------------------------
# 1. Single-turn tool use — the fundamental building block
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    message: str
    max_iterations: int = 10  # safety limit for the agentic loop


@app.post("/chat")
async def chat(req: ChatRequest):
    """
    Complete tool use implementation with the agentic loop.

    This is the core pattern you'll use in every tool-use application:
    1. Send user message + tool definitions to the model
    2. If model wants to use tools → execute them → send results back
    3. Repeat until model produces a final text response

    The max_iterations guard prevents infinite loops (a real production concern).
    """
    client: anthropic.Anthropic = app.state.client

    messages = [{"role": "user", "content": req.message}]

    # Track the full interaction for observability
    tool_calls_log = []
    iteration = 0

    # === THE AGENTIC LOOP ===
    # This is the most important pattern in tool use.
    # The model may need multiple rounds of tool calls to answer one question.
    while iteration < req.max_iterations:
        iteration += 1

        # Call the model with our tools
        response = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            tools=TOOL_DEFINITIONS,
            messages=messages,
        )

        # Case 1: Model is done — produced a final text response
        if response.stop_reason == "end_turn":
            final_text = ""
            for block in response.content:
                if block.type == "text":
                    final_text += block.text
            return {
                "response": final_text,
                "tool_calls": tool_calls_log,
                "iterations": iteration,
                "usage": {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                },
            }

        # Case 2: Model wants to use tools
        if response.stop_reason == "tool_use":
            # The assistant message may contain BOTH text and tool_use blocks.
            # We must append the ENTIRE assistant message to maintain conversation flow.
            messages.append({"role": "assistant", "content": response.content})

            # Process each tool call in this response
            # (Claude can request multiple tools in parallel!)
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    # Execute the tool
                    result = execute_tool(block.name, block.input)

                    tool_calls_log.append({
                        "iteration": iteration,
                        "tool": block.name,
                        "input": block.input,
                        "result": result,
                    })

                    # Format as tool_result for the API
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,  # MUST match the tool_use block
                        "content": result,
                    })

            # Send ALL tool results back in a single user message
            messages.append({"role": "user", "content": tool_results})

            # Loop continues — model will see results and decide what to do next

    # Safety: hit max iterations
    return {
        "response": "I wasn't able to complete your request within the allowed steps.",
        "tool_calls": tool_calls_log,
        "iterations": iteration,
        "error": "max_iterations_reached",
    }


# ---------------------------------------------------------------------------
# 2. Multi-turn conversation — tool use with conversation history
# ---------------------------------------------------------------------------
class ConversationMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str


class ConversationRequest(BaseModel):
    messages: list[ConversationMessage]
    max_iterations: int = 10


@app.post("/conversation")
async def conversation(req: ConversationRequest):
    """
    Multi-turn conversation with tool use.

    In production, you'd store conversation history in a database.
    The model sees the FULL history, including previous tool calls,
    which lets it reference earlier results without re-calling tools.

    Key insight: tool_use and tool_result messages are part of the
    conversation history. If you strip them out, the model loses
    context about what data it already has.
    """
    client: anthropic.Anthropic = app.state.client

    # Convert simple messages to API format
    messages = [{"role": m.role, "content": m.content} for m in req.messages]

    tool_calls_log = []
    iteration = 0

    while iteration < req.max_iterations:
        iteration += 1

        response = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            tools=TOOL_DEFINITIONS,
            messages=messages,
        )

        if response.stop_reason == "end_turn":
            final_text = ""
            for block in response.content:
                if block.type == "text":
                    final_text += block.text
            return {
                "response": final_text,
                "tool_calls": tool_calls_log,
                "iterations": iteration,
            }

        if response.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": response.content})

            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    result = execute_tool(block.name, block.input)
                    tool_calls_log.append({
                        "iteration": iteration,
                        "tool": block.name,
                        "input": block.input,
                        "result": result,
                    })
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    })

            messages.append({"role": "user", "content": tool_results})

    return {"response": "Max iterations reached.", "tool_calls": tool_calls_log}


# ---------------------------------------------------------------------------
# 3. Inspect tool definitions — see exactly what the model sees
# ---------------------------------------------------------------------------
@app.get("/tools")
async def list_tools():
    """
    Returns the tool definitions sent to the model.

    Use this to understand what the model "sees" when deciding
    which tool to call. The description quality is the #1 factor
    in tool selection accuracy.
    """
    return {
        "tools": TOOL_DEFINITIONS,
        "count": len(TOOL_DEFINITIONS),
        "tip": (
            "These exact schemas are injected into the model's context. "
            "Better descriptions = better tool selection."
        ),
    }


# ---------------------------------------------------------------------------
# 4. Force tool use — require the model to use a specific tool
# ---------------------------------------------------------------------------
class ForceToolRequest(BaseModel):
    message: str
    tool_name: str  # must match a tool in TOOL_DEFINITIONS


@app.post("/force-tool")
async def force_tool(req: ForceToolRequest):
    """
    Demonstrates tool_choice parameter to FORCE the model to use a tool.

    Options:
    - {"type": "auto"}  → model decides (default)
    - {"type": "any"}   → model must use SOME tool
    - {"type": "tool", "name": "get_weather"} → must use THIS tool

    Use 'any' or specific tool when you KNOW a tool call is needed
    and don't want the model to skip it.
    """
    client: anthropic.Anthropic = app.state.client
    messages = [{"role": "user", "content": req.message}]

    # Force the model to use the specified tool
    response = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        tools=TOOL_DEFINITIONS,
        tool_choice={"type": "tool", "name": req.tool_name},
        messages=messages,
    )

    # Execute the forced tool call
    messages.append({"role": "assistant", "content": response.content})
    tool_results = []
    tool_info = None

    for block in response.content:
        if block.type == "tool_use":
            result = execute_tool(block.name, block.input)
            tool_info = {"tool": block.name, "input": block.input, "result": result}
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": result,
            })

    messages.append({"role": "user", "content": tool_results})

    # Let the model generate a final response using the tool result
    final = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        tools=TOOL_DEFINITIONS,
        messages=messages,
    )

    final_text = ""
    for block in final.content:
        if block.type == "text":
            final_text += block.text

    return {
        "response": final_text,
        "forced_tool_call": tool_info,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
