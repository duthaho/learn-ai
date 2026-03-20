"""
How LLMs Work — Hands-on FastAPI Service

Demonstrates core LLM concepts:
1. Tokenization & token counting
2. Streaming generation (autoregressive decoding)
3. Temperature / sampling effects
4. Structured output with system prompts
5. Context window management
"""

import os
from contextlib import asynccontextmanager

import anthropic
import tiktoken
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize the Anthropic client on startup."""
    app.state.client = anthropic.Anthropic(
        api_key=os.environ["ANTHROPIC_API_KEY"],
    )
    # tiktoken encoder for token counting demonstrations
    # (OpenAI's cl100k_base — used here to illustrate tokenization concepts;
    #  Anthropic uses its own tokenizer, but the principles are identical)
    app.state.encoder = tiktoken.get_encoding("cl100k_base")
    yield


app = FastAPI(title="How LLMs Work", lifespan=lifespan)

MODEL = "claude-sonnet-4-20250514"


# ---------------------------------------------------------------------------
# 1. Tokenization — see how text becomes tokens
# ---------------------------------------------------------------------------
class TokenizeRequest(BaseModel):
    text: str


@app.post("/tokenize")
async def tokenize(req: TokenizeRequest):
    """
    Shows how text is split into tokens.

    This makes the abstract concept concrete:
    - "unhappiness" → ["un", "happiness"]
    - Whitespace, punctuation, and casing all affect tokenization
    - JSON/code is token-expensive compared to prose
    """
    enc = app.state.encoder
    token_ids = enc.encode(req.text)
    tokens = [enc.decode([tid]) for tid in token_ids]

    return {
        "text": req.text,
        "token_count": len(token_ids),
        "token_ids": token_ids,
        "tokens": tokens,
        "cost_insight": (
            f"{len(token_ids)} tokens. "
            f"At ~$3/M input tokens, this costs ~${len(token_ids) * 3 / 1_000_000:.6f}"
        ),
    }


# ---------------------------------------------------------------------------
# 2. Compare token costs — why prompt engineering matters financially
# ---------------------------------------------------------------------------
class CompareRequest(BaseModel):
    texts: list[str]


@app.post("/compare-tokens")
async def compare_tokens(req: CompareRequest):
    """
    Compare token counts across different text representations.

    Try sending the same data as:
    - Verbose JSON with long key names
    - Compact JSON with short keys
    - Plain text / CSV

    This demonstrates why token-efficient formats save real money at scale.
    """
    enc = app.state.encoder
    results = []
    for text in req.texts:
        count = len(enc.encode(text))
        results.append({"text_preview": text[:80], "token_count": count})
    return {"comparisons": sorted(results, key=lambda x: x["token_count"])}


# ---------------------------------------------------------------------------
# 3. Streaming generation — see autoregressive decoding in action
# ---------------------------------------------------------------------------
class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 512
    temperature: float = 1.0
    system: str = "You are a helpful assistant."


@app.post("/generate")
async def generate(req: GenerateRequest):
    """
    Non-streaming generation. Returns the full response at once.
    Useful for comparing with the streaming endpoint.
    """
    client: anthropic.Anthropic = app.state.client
    message = client.messages.create(
        model=MODEL,
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        system=req.system,
        messages=[{"role": "user", "content": req.prompt}],
    )
    return {
        "content": message.content[0].text,
        "model": message.model,
        "usage": {
            "input_tokens": message.usage.input_tokens,
            "output_tokens": message.usage.output_tokens,
        },
        "stop_reason": message.stop_reason,
    }


@app.post("/generate/stream")
async def generate_stream(req: GenerateRequest):
    """
    Streaming generation — demonstrates autoregressive token-by-token output.

    Each chunk is one or more tokens. The model generates left-to-right,
    each token conditioned on all previous tokens. This is WHY streaming
    works — the model doesn't need to "think about the whole response"
    before starting to output.
    """
    client: anthropic.Anthropic = app.state.client

    def event_stream():
        with client.messages.stream(
            model=MODEL,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            system=req.system,
            messages=[{"role": "user", "content": req.prompt}],
        ) as stream:
            for text in stream.text_stream:
                yield text

    return StreamingResponse(event_stream(), media_type="text/plain")


# ---------------------------------------------------------------------------
# 4. Temperature demo — same prompt, different temperatures
# ---------------------------------------------------------------------------
class TemperatureRequest(BaseModel):
    prompt: str
    temperatures: list[float] = [0.0, 0.5, 1.0]
    max_tokens: int = 150


@app.post("/temperature-demo")
async def temperature_demo(req: TemperatureRequest):
    """
    Shows how temperature affects the output distribution.

    Temperature scales the logits before softmax:
    - T=0: Always picks the highest-probability token (deterministic)
    - T=0.5: Moderate randomness, usually coherent
    - T=1.0: Standard sampling, more creative/varied

    Run this multiple times with the same prompt to see variance at each T.
    """
    client: anthropic.Anthropic = app.state.client
    results = []

    for temp in req.temperatures:
        message = client.messages.create(
            model=MODEL,
            max_tokens=req.max_tokens,
            temperature=temp,
            messages=[{"role": "user", "content": req.prompt}],
        )
        results.append({
            "temperature": temp,
            "output": message.content[0].text,
            "output_tokens": message.usage.output_tokens,
        })

    return {"prompt": req.prompt, "results": results}


# ---------------------------------------------------------------------------
# 5. Structured output — constraining the probability distribution
# ---------------------------------------------------------------------------
class ReviewRequest(BaseModel):
    code_diff: str
    language: str = "python"


REVIEW_SYSTEM_PROMPT = """\
You are a senior code reviewer. Analyze the given diff and return your review
as a JSON array. Each element must have exactly these fields:
- "line": the line number or range (string)
- "severity": "critical" | "warning" | "suggestion"
- "issue": one-sentence description of the problem
- "fix": one-sentence suggested fix

Return ONLY the JSON array, no markdown fences, no explanation."""


@app.post("/review")
async def review_code(req: ReviewRequest):
    """
    Demonstrates structured output via strong system prompts.

    The system prompt constrains the model's output distribution toward
    valid JSON. This is how the model's probabilistic nature is channeled
    into deterministic-feeling structured responses.

    In production, you'd use tool_use / function calling for guaranteed
    schema compliance. This shows the raw prompt-engineering approach.
    """
    import json

    client: anthropic.Anthropic = app.state.client
    message = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        temperature=0.0,  # deterministic for structured output
        system=REVIEW_SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": f"Language: {req.language}\n\nDiff:\n{req.code_diff}",
        }],
    )

    raw = message.content[0].text
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = None

    return {
        "raw_response": raw,
        "parsed": parsed,
        "parse_success": parsed is not None,
        "usage": {
            "input_tokens": message.usage.input_tokens,
            "output_tokens": message.usage.output_tokens,
        },
    }


# ---------------------------------------------------------------------------
# 6. Context window awareness
# ---------------------------------------------------------------------------
@app.post("/context-check")
async def context_check(req: GenerateRequest):
    """
    Estimates token usage BEFORE sending to the API.

    This is critical in production:
    - You must know if your prompt fits in the context window
    - You need to reserve space for the response (max_tokens)
    - If input is too large, you must truncate or chunk

    Claude's context windows: 200K tokens.
    """
    enc = app.state.encoder
    system_tokens = len(enc.encode(req.system))
    prompt_tokens = len(enc.encode(req.prompt))
    total_input = system_tokens + prompt_tokens

    context_limit = 200_000  # Claude's context window
    available_for_output = context_limit - total_input
    fits = total_input + req.max_tokens <= context_limit

    return {
        "system_tokens": system_tokens,
        "prompt_tokens": prompt_tokens,
        "total_input_tokens": total_input,
        "requested_max_output": req.max_tokens,
        "context_limit": context_limit,
        "available_for_output": available_for_output,
        "fits_in_context": fits,
        "utilization_pct": round(total_input / context_limit * 100, 2),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
