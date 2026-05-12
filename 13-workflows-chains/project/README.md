# Project: Support Ticket Triage Pipeline

A deterministic workflow that takes a customer support ticket, classifies it, routes it to a category-specific handler, runs entity extraction and response drafting in parallel, and returns a fully triaged ticket record with per-step usage and total cost.

## What you'll build

- A `triage(text)` function that runs a 3-stage workflow on one ticket
- A `triage_batch(tickets)` function that processes a JSON file of tickets
- A CLI that accepts either a single positional string or a `--file` flag
- A `--model` flag to override the default model

The project exercises all three workflow patterns from the module:

- **Sequential chain:** classify → handle → assemble
- **Branching (router):** category routes to one of three handlers
- **Parallel fan-out + fan-in:** within each handler, extract and draft run concurrently via `ThreadPoolExecutor`

## Prerequisites

- Completed reading the [Module 13 README](../README.md)
- [Module 08 (Structured Output)](../../08-structured-output/) and [Module 12 (Multi-Agent Systems)](../../12-multi-agent-systems/) are the most directly relevant prior modules — review them if Pydantic contracts or per-step usage tracking feel unfamiliar
- Python 3.11+ with project dependencies installed (`pip install -r requirements.txt` from the repo root — no new dependencies beyond what Module 12 already required)
- An OpenAI-compatible API key set in `.env` at the repo root (`OPENAI_API_KEY` or whichever provider you use)

## Setup

```bash
cd 13-workflows-chains/project
```

Confirm that `.env` at the repo root contains your API key. The script loads it automatically via `python-dotenv` using a path resolved relative to the script file, so you do not need to be in any particular directory when you run it. `LLM_MODEL` defaults to `anthropic/claude-sonnet-4-20250514` if unset; pass `--model` to override at runtime without touching `.env`.

### Project layout

```text
project/
├── README.md          this file
├── solution.py        the full pipeline (~400 lines)
└── samples/
    ├── refund.txt
    ├── technical.txt
    ├── general.txt
    └── batch.json     4 example tickets across all three categories
```

Read `solution.py` end-to-end before you run it; each step is independently callable, so you can drop into a REPL and exercise any function in isolation.

## How it works

```text
                  ┌─ refund handler ────┐
                  │   (extract ‖ draft) │
ticket → classify ┼─ technical handler ─┼→ assemble → TriagedTicket
                  │   (extract ‖ draft) │
                  └─ general handler ───┘
                       (extract ‖ draft)
```

The pipeline has four stages:

- **Classify** sends the raw ticket text to the LLM and parses a `TicketClassification` (category, urgency, sentiment). This is the only step whose output decides where the rest of the run goes — everything downstream is deterministic given the classifier's choice.
- **Route** is a pure-Python dispatch on `category` that returns the `(extract_fn, draft_fn)` pair for the chosen handler. No LLM call is made here; the routing decision was already made upstream by the classifier.
- **Handler** runs the chosen `extract_fn` and `draft_fn` concurrently against the same ticket text, blocking until both complete. This is the fan-out + fan-in — the parallel section is where latency savings come from.
- **Assemble** packages the classification, entities, drafted response, and accumulated `StepUsage` records into a single `TriagedTicket` for the caller. No LLM call here either; just dataclass construction.

## Build it step by step

1. **Define the Pydantic models** (`TicketClassification`, `RefundEntities`, `TechnicalEntities`, `GeneralEntities`, `DraftedResponse`, `StepUsage`, `TriagedTicket`). Use `Literal` types for closed enums like `category` and `urgency` so a misclassification surfaces as a `ValidationError` rather than silently propagating downstream.
2. **Write the seven system prompts** — one classifier, three extractors, three drafters. Each prompt should include the literal JSON shape it must return; LLMs follow schema examples more reliably than prose. Keep the three extractor prompts in three separate constants rather than templating them — the per-category fields differ enough that abstraction hurts more than it helps.
3. **Implement `_call_json` and `_strip_code_fence` helpers** (same pattern as Module 12). `_call_json` wraps `litellm.completion` with `response_format={"type": "json_object"}`, calls `_strip_code_fence` on the raw string, and returns `(parsed_dict, StepUsage)` so every step records its own latency, tokens, and cost.
4. **Implement `classify(text)`** — returns `(TicketClassification, StepUsage)`. The classifier decides three fields at once (`category`, `urgency`, `sentiment`); doing them in a single call is cheaper than three separate calls and gives the model the full context for each judgement.
5. **Implement the three extract functions** (`extract_refund`, `extract_technical`, `extract_general`) — each returns `(EntitiesModel, StepUsage)`. Each function picks the relevant entities for its category: order IDs and amounts for refunds, error codes and product versions for technical, free-form `topic` and `keywords` for general.
6. **Implement the three draft functions** (`draft_refund`, `draft_technical`, `draft_general`) — each returns `(DraftedResponse, StepUsage)`. Drafters take the raw ticket text (not the extracted entities) so they can run concurrently with the extractor; the assemble step is where extracted entities get woven back in if needed.
7. **Implement the router `route(category)`** returning the `(extract_fn, draft_fn)` pair for that category. This is a pure-Python dict lookup, not an LLM call — the routing decision was already made by the classifier.
8. **Implement `run_handler(text, extract_fn, draft_fn)`** using `ThreadPoolExecutor` to fan out and fan in. Submit both calls, then `.result()` on both futures so the function blocks until both complete and propagates exceptions naturally.
9. **Implement `triage(text)`** — orchestrates classify → route → run_handler → assemble. Collect every `StepUsage` into a list and pass them to the final `TriagedTicket` so the caller can inspect per-step cost.
10. **Implement `triage_batch(tickets)`** — iterates `triage()` over a list of strings and aggregates totals. Print one summary line per ticket as it completes so long batches show progress instead of going silent.
11. **Wire up the CLI with `argparse`** — positional `text` (optional), `--file PATH` (mutually exclusive with `text`), and `--model NAME` (overrides `LLM_MODEL`). If `--file` is given, parse it as JSON and call `triage_batch`; otherwise call `triage` on the positional argument.

Each step should be small and independently testable. Steps 5 and 6 produce three near-identical functions each — resist the urge to extract a generic `extract(category, text)`. The duplication makes per-category prompt tuning trivial, and the extractor/drafter for a new category becomes a copy-paste-edit task rather than a refactor.

## Run it

```bash
python solution.py "my package never arrived order #1234"
python solution.py --file samples/batch.json
python solution.py --file samples/batch.json --model anthropic/claude-haiku-4-5-20251001
```

Expected console output (exact values vary):

```text
=== Triaging ticket ===
Input: "my package never arrived order #1234..."

[classify] category=refund urgency=high sentiment=negative (1.2s, $0.0012)
[route]    -> refund handler
[extract ‖ draft] running in parallel...
[extract]  order_id=1234 amount=null reason="package not delivered" (1.4s, $0.0009)
[draft]    subject="Re: Your missing package" (2.1s, $0.0021)

=== TriagedTicket ===
Category:        refund (high urgency, negative sentiment)
Entities:        order_id=1234, amount=null, reason="package not delivered"
Response:
  Subject: Re: Your missing package
  Body:    Thank you for reaching out about order #1234...

=== Step Usage ===
step       in     out    cost      latency
classify     85    62  $0.001200    1234ms
extract      90    48  $0.000900    1421ms
draft        92   140  $0.002100    2102ms
TOTAL                  $0.004200    3336ms  (parallel: extract+draft overlap)
```

The TOTAL latency is less than the sum of step latencies because `extract` and `draft` run concurrently — the parallel section's wall time is `max(extract_latency, draft_latency)`, not their sum. Cost still adds linearly: parallelism saves time, not tokens.

For batch mode, the script prints a one-line summary per ticket and a grand-total block at the end with combined token counts and cost. A useful experiment: run the same batch twice, once with the default model and once with `--model anthropic/claude-haiku-4-5-20251001`, and compare both cost and classification agreement between the two runs. The workflow shape makes that comparison trivial; an agent would make it nearly impossible because every run takes a different path.

### Inspecting one stage at a time

Because every step is a plain function with typed input and output, you can drop into a REPL and call `classify("...")` or `extract_refund("...")` directly without running the whole pipeline. This is the most underrated property of workflows: each step is its own debug target. When the drafter starts producing weird subject lines, you do not need to re-run classification and extraction to reproduce — you call the drafter alone, on the same input, until the prompt is fixed.

### Common pitfalls

- **Treating the classifier as infallible.** It will occasionally pick the wrong category. The right move is not "make the classifier perfect" but "make every handler robust to inputs that don't quite fit." A refund handler asked to process a technical ticket should return empty entities and a polite generic response, not crash.
- **Sequencing extract before draft.** Both functions take the original ticket text — they do not depend on each other. Running them sequentially throws away the entire parallel-section latency win. If you find yourself wanting the drafter to use extracted entities, that's a sign the assemble step should do the weaving, not the drafter.
- **Logging cost as a flat total.** A flat total hides which step is expensive. Keep the per-`StepUsage` breakdown visible in output — when you swap models, the breakdown shows you which step's cost changed and which didn't.

## Extensions

Once the base pipeline works, these are the natural next experiments — each one isolates a single property of the workflow shape:

1. **Add a fourth category and handler.** Pick something like `account_access` and add the category to the classifier prompt, a fourth entities model, a fourth `extract_*` / `draft_*` pair, and a fourth branch in `route()`. Notice how localised the change is — that is a property of the workflow shape, not a coincidence. Compare this against the same change in an agent system, where adding a category usually means re-tuning the system prompt and retesting paths that should have been untouched.
2. **Replace one LLM extractor with a regex-based extractor.** The refund handler's `extract_refund` only needs to pull an order ID and amount; a regex does this in microseconds for zero cost. Keep the LLM drafter and compare combined cost and accuracy across a batch. You will probably find the regex wins on cost and loses on robustness when ticket text is messy — which is the entire point of measuring it.
3. **Add retries with exponential backoff on transient API errors.** Wrap `_call_json` with a `tenacity` decorator that retries on `litellm.exceptions.RateLimitError` and `litellm.exceptions.APIConnectionError`. The workflow shape makes retries safe — each step is idempotent given the same input, so retrying any one step never corrupts the run.
4. **Add a Streamlit or Gradio frontend.** Wrap `triage()` in a single-textarea web UI that streams the per-step output as it arrives. The function already returns a structured `TriagedTicket`, so the frontend is mostly formatting.
5. **Swap `ThreadPoolExecutor` for `asyncio.gather` with LiteLLM's `acompletion`.** Same fan-out + fan-in shape, lower overhead per call. Useful if you ever scale `triage_batch` to thousands of tickets where thread-per-call gets expensive.

## Reference

Cross-links for context:

- [Module 13 README](../README.md) — the three workflow patterns, when to use a workflow over an agent, observability and testing.
- [Module 08 (Structured Output)](../../08-structured-output/) — the Pydantic + JSON-mode pattern used in every step here.
- [Module 11 (Building AI Agents)](../../11-building-ai-agents/) — the contrast case. Reread the agent loop after building this workflow and notice how much more code state the agent has to track at runtime.
- [Module 12 (Multi-Agent Systems)](../../12-multi-agent-systems/) — the `_call_json` helper, per-step usage tracking, and orchestrator shape carry over directly.

**Next:** Module 14 builds on this foundation by adding evaluation and observability around workflows so you can measure quality regressions as you tune prompts or swap models. The per-`StepUsage` records you produced here become the raw material for that module's eval harness.
