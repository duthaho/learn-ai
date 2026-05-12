"""
Support Ticket Triage Pipeline — complete reference implementation.

A deterministic workflow that:
  1. classify       — categorize the ticket (refund / technical / general)
  2. route          — pick the matching handler based on category
  3. handler        — extract entities AND draft a response in parallel
  4. assemble       — combine into a TriagedTicket record

Demonstrates three workflow patterns:
- Sequential chain (classify -> handle -> assemble)
- Branching / router (category -> handler)
- Parallel fan-out + fan-in (extract ‖ draft, joined back into one record)

Run:
    python solution.py "my package never arrived order #1234"
    python solution.py --file samples/batch.json
    python solution.py --file samples/batch.json --model anthropic/claude-haiku-4-5-20251001
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Literal

from dotenv import load_dotenv
from litellm import completion, completion_cost
from pydantic import BaseModel, Field, ValidationError

load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

MODEL = os.getenv("LLM_MODEL", "anthropic/claude-sonnet-4-20250514")


# ---------- Pydantic models (the contract between steps) ----------

TicketCategory = Literal["refund", "technical", "general"]
Urgency = Literal["low", "medium", "high"]
Sentiment = Literal["negative", "neutral", "positive"]


class TicketClassification(BaseModel):
    category: TicketCategory
    urgency: Urgency
    sentiment: Sentiment
    rationale: str = Field(..., description="One-line reason for this classification.")


class RefundEntities(BaseModel):
    order_id: str | None = None
    amount: str | None = None
    reason: str | None = None


class TechnicalEntities(BaseModel):
    product: str | None = None
    version: str | None = None
    error_message: str | None = None


class GeneralEntities(BaseModel):
    topic: str
    key_question: str


class DraftedResponse(BaseModel):
    subject: str
    body: str


class StepUsage(BaseModel):
    step: str
    input_tokens: int
    output_tokens: int
    cost: float
    latency_ms: int


class TriagedTicket(BaseModel):
    input_text: str
    classification: TicketClassification
    entities: RefundEntities | TechnicalEntities | GeneralEntities
    response: DraftedResponse
    per_step: list[StepUsage]
    total_cost: float
    total_latency_ms: int


# ---------- System prompts ----------

CLASSIFIER_PROMPT = """You are a support-ticket classifier. Given a customer message, return JSON with:
- category: "refund", "technical", or "general"
- urgency: "low", "medium", or "high"
- sentiment: "negative", "neutral", or "positive"
- rationale: one short sentence explaining the classification

Do not write a customer response. Only classify."""

REFUND_EXTRACT_PROMPT = """Extract the following fields from this refund-related support message:
- order_id (string or null)
- amount (string or null, include currency symbol if present)
- reason (string or null, the customer's stated reason)

Use null for any field that is not present. Return JSON."""

TECHNICAL_EXTRACT_PROMPT = """Extract the following fields from this technical support message:
- product (string or null)
- version (string or null)
- error_message (string or null, the core error)

Use null for any field that is not present. Return JSON."""

GENERAL_EXTRACT_PROMPT = """Identify:
- topic: a short noun phrase summarizing what the message is about
- key_question: the customer's main question, phrased as one sentence

Return JSON."""

REFUND_DRAFT_PROMPT = """You are a customer support agent handling a refund request. Write a brief, empathetic response that:
- acknowledges the request
- explains the next step (e.g., review timeline)
- avoids promising a specific outcome

Return JSON with `subject` and `body`. Keep the body under 120 words."""

TECHNICAL_DRAFT_PROMPT = """You are a technical support agent. Write a brief response that:
- acknowledges the issue
- suggests one concrete next step (a check, a setting, or a request for more information)

Return JSON with `subject` and `body`. Keep the body under 120 words."""

GENERAL_DRAFT_PROMPT = """You are a friendly support agent handling a general inquiry. Write a brief, helpful response.

Return JSON with `subject` and `body`. Keep the body under 120 words."""


# ---------- Helpers ----------


def _strip_code_fence(text: str) -> str:
    """Strip a ```json ... ``` (or plain ```) fence if the model wrapped its output."""
    s = text.strip()
    if not s.startswith("```"):
        return s
    s = s[3:]
    if s.lower().startswith("json"):
        s = s[4:]
    s = s.lstrip("\r\n")
    if s.endswith("```"):
        s = s[:-3]
    return s.strip()


def _usage_from_response(response) -> tuple[int, int, float]:
    """Return (input_tokens, output_tokens, cost) from a LiteLLM response."""
    usage = getattr(response, "usage", None)
    input_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
    output_tokens = getattr(usage, "completion_tokens", 0) if usage else 0
    try:
        cost = completion_cost(completion_response=response) or 0.0
    except Exception:
        cost = 0.0
    return input_tokens, output_tokens, cost


def _call_json(step: str, system_prompt: str, user_content: str, model: str) -> tuple[dict, StepUsage]:
    """Call the LLM asking for JSON; return (parsed_dict, StepUsage)."""
    start = time.perf_counter()
    response = completion(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        response_format={"type": "json_object"},
    )
    latency_ms = int((time.perf_counter() - start) * 1000)
    raw = response.choices[0].message.content
    cleaned = _strip_code_fence(raw)
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(f"Step '{step}' returned invalid JSON: {e}\nRaw:\n{raw}") from e
    inp, out, cost = _usage_from_response(response)
    usage = StepUsage(
        step=step,
        input_tokens=inp,
        output_tokens=out,
        cost=cost,
        latency_ms=latency_ms,
    )
    return parsed, usage


# ---------- Step: classify ----------


def classify(text: str, model: str = MODEL) -> tuple[TicketClassification, StepUsage]:
    parsed, usage = _call_json("classify", CLASSIFIER_PROMPT, text, model)
    try:
        return TicketClassification.model_validate(parsed), usage
    except ValidationError as e:
        raise ValueError(f"Classifier output failed schema validation:\n{e}") from e


# ---------- Step: extract (category-specific) ----------


def extract_refund(text: str, model: str = MODEL) -> tuple[RefundEntities, StepUsage]:
    parsed, usage = _call_json("extract", REFUND_EXTRACT_PROMPT, text, model)
    try:
        return RefundEntities.model_validate(parsed), usage
    except ValidationError as e:
        raise ValueError(f"Refund extractor output failed schema validation:\n{e}") from e


def extract_technical(text: str, model: str = MODEL) -> tuple[TechnicalEntities, StepUsage]:
    parsed, usage = _call_json("extract", TECHNICAL_EXTRACT_PROMPT, text, model)
    try:
        return TechnicalEntities.model_validate(parsed), usage
    except ValidationError as e:
        raise ValueError(f"Technical extractor output failed schema validation:\n{e}") from e


def extract_general(text: str, model: str = MODEL) -> tuple[GeneralEntities, StepUsage]:
    parsed, usage = _call_json("extract", GENERAL_EXTRACT_PROMPT, text, model)
    try:
        return GeneralEntities.model_validate(parsed), usage
    except ValidationError as e:
        raise ValueError(f"General extractor output failed schema validation:\n{e}") from e


# ---------- Step: draft (category-specific) ----------


def draft_refund(text: str, model: str = MODEL) -> tuple[DraftedResponse, StepUsage]:
    parsed, usage = _call_json("draft", REFUND_DRAFT_PROMPT, text, model)
    try:
        return DraftedResponse.model_validate(parsed), usage
    except ValidationError as e:
        raise ValueError(f"Refund drafter output failed schema validation:\n{e}") from e


def draft_technical(text: str, model: str = MODEL) -> tuple[DraftedResponse, StepUsage]:
    parsed, usage = _call_json("draft", TECHNICAL_DRAFT_PROMPT, text, model)
    try:
        return DraftedResponse.model_validate(parsed), usage
    except ValidationError as e:
        raise ValueError(f"Technical drafter output failed schema validation:\n{e}") from e


def draft_general(text: str, model: str = MODEL) -> tuple[DraftedResponse, StepUsage]:
    parsed, usage = _call_json("draft", GENERAL_DRAFT_PROMPT, text, model)
    try:
        return DraftedResponse.model_validate(parsed), usage
    except ValidationError as e:
        raise ValueError(f"General drafter output failed schema validation:\n{e}") from e


# ---------- Router (branching) ----------


def route(category: TicketCategory) -> tuple[Callable, Callable]:
    """Return (extract_fn, draft_fn) for the given category."""
    match category:
        case "refund":
            return extract_refund, draft_refund
        case "technical":
            return extract_technical, draft_technical
        case "general":
            return extract_general, draft_general


# ---------- Handler (parallel fan-out + fan-in) ----------


def run_handler(
    text: str,
    extract_fn: Callable,
    draft_fn: Callable,
    model: str = MODEL,
) -> tuple[BaseModel, DraftedResponse, list[StepUsage]]:
    """Run extract_fn and draft_fn concurrently on the same input."""
    with ThreadPoolExecutor(max_workers=2) as pool:
        ex_future = pool.submit(extract_fn, text, model)
        dr_future = pool.submit(draft_fn, text, model)
        entities, ex_usage = ex_future.result()
        response, dr_usage = dr_future.result()
    return entities, response, [ex_usage, dr_usage]


# ---------- Orchestrator ----------


def triage(text: str, model: str = MODEL) -> TriagedTicket:
    """Run the full workflow on a single ticket."""
    start = time.perf_counter()

    # 1. Classify
    classification, classify_usage = classify(text, model=model)

    # 2. Route (no LLM call — pure Python branch)
    extract_fn, draft_fn = route(classification.category)

    # 3. Handler (parallel extract + draft)
    entities, response, handler_usage = run_handler(text, extract_fn, draft_fn, model=model)

    # 4. Assemble
    per_step = [classify_usage, *handler_usage]
    total_cost = sum(s.cost for s in per_step)
    total_latency_ms = int((time.perf_counter() - start) * 1000)

    return TriagedTicket(
        input_text=text,
        classification=classification,
        entities=entities,
        response=response,
        per_step=per_step,
        total_cost=round(total_cost, 6),
        total_latency_ms=total_latency_ms,
    )


def triage_batch(tickets: list[str], model: str = MODEL) -> list[TriagedTicket | dict]:
    """Process a batch of tickets. On per-ticket failure, append an error dict and continue."""
    results: list[TriagedTicket | dict] = []
    for i, text in enumerate(tickets, start=1):
        try:
            results.append(triage(text, model=model))
        except Exception as e:
            results.append({
                "error": f"{type(e).__name__}: {e}",
                "input_text": text,
                "index": i,
            })
    return results


# ---------- CLI / printing ----------


def _print_triaged(result: TriagedTicket) -> None:
    cls = result.classification
    print(f"\n=== TriagedTicket ===")
    print(f"Category:   {cls.category} ({cls.urgency} urgency, {cls.sentiment} sentiment)")
    print(f"Rationale:  {cls.rationale}")
    print(f"Entities:   {result.entities.model_dump_json()}")
    print(f"Response:")
    print(f"  Subject: {result.response.subject}")
    print(f"  Body:    {result.response.body}")
    print(f"\n=== Step Usage ===")
    print(f"{'step':10s} {'in':>5s} {'out':>5s} {'cost':>10s} {'latency':>10s}")
    for s in result.per_step:
        print(f"{s.step:10s} {s.input_tokens:>5d} {s.output_tokens:>5d} ${s.cost:>9.6f} {s.latency_ms:>8d}ms")
    sum_step_latency = sum(s.latency_ms for s in result.per_step)
    print(
        f"{'TOTAL':10s} {'':>5s} {'':>5s} ${result.total_cost:>9.6f} "
        f"{result.total_latency_ms:>8d}ms  (wall-clock; sum of step latencies = {sum_step_latency}ms)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Support Ticket Triage Pipeline (Module 13)")
    parser.add_argument("text", nargs="?", help="A single ticket as a positional string")
    parser.add_argument("--file", dest="file", help="Path to a JSON file containing a top-level list of ticket strings")
    parser.add_argument("--model", default=MODEL, help=f"Model override (default: {MODEL})")
    args = parser.parse_args()

    if bool(args.text) == bool(args.file):
        parser.error("provide exactly one of: positional `text` OR `--file PATH`")

    if args.text:
        result = triage(args.text, model=args.model)
        print(f"\n=== Triaging ticket ===\nInput: {args.text!r}")
        _print_triaged(result)
        return

    # Batch mode
    path = Path(args.file)
    if not path.exists():
        parser.error(f"file not found: {path}")
    try:
        tickets = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        parser.error(f"could not parse JSON file: {e}")
    if not isinstance(tickets, list) or not all(isinstance(t, str) for t in tickets):
        parser.error("JSON file must contain a top-level list of strings")

    results = triage_batch(tickets, model=args.model)
    print(f"\n=== Batch triage ({len(results)} tickets) ===")
    total_cost = 0.0
    ok = 0
    for i, r in enumerate(results, start=1):
        if isinstance(r, TriagedTicket):
            ok += 1
            total_cost += r.total_cost
            print(
                f"[{i}] {r.classification.category:>9s} "
                f"u={r.classification.urgency:<6s} "
                f"s={r.classification.sentiment:<8s} "
                f"cost=${r.total_cost:.6f} "
                f"t={r.total_latency_ms}ms  | {r.input_text[:60]!r}"
            )
        else:
            print(f"[{i}] ERROR: {r['error']}  | {r['input_text'][:60]!r}")
    print(f"\nProcessed: {ok}/{len(results)} | Total batch cost: ${total_cost:.6f}")


if __name__ == "__main__":
    main()
