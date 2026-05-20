# Project: LLM Call Tracer

Build a tracer that records every LLM call your app makes, groups them into nested runs, and lets you ask "what happened in this run, why did it cost what it cost, and which call failed?" from the CLI.

## What you'll build

- A `Span` Pydantic model -- the on-disk shape: `span_id`, `parent_id`, `run_id`, `kind` (`"run"` or `"llm"`), `name`, `started_at`, `ended_at`, `duration_ms`, `status` (`"ok"` or `"error"`), `error`, and `attributes` (a free-form dict that carries `model`, `tokens_in`, `tokens_out`, and `cost_usd` on LLM spans).
- A `Tracer` class -- owns the JSONL writer, a thread-local span stack, a `run(name=...)` context manager that emits the parent span, and a `wrap_llm_call(**kwargs)` helper that calls LiteLLM `completion()` and emits a child LLM span, capturing cost and status automatically.
- Six CLI modes -- `--demo`, `--list`, `--show <run_id>`, `--stats`, `--tail`, and `--flush`.
- A 3-run demo workload -- `research_agent` (happy path, 3 calls), `failure_demo` (one deliberately broken call to a fake model), and `cost_regression` (two calls with very different system-prompt sizes) -- that produces the sample data you need for both worked exercises below.

## Prerequisites

- [Module 04 (AI API Layer)](../../04-ai-api-layer/) -- `completion()` is the LiteLLM call that `wrap_llm_call` wraps; know what `usage` and `_hidden_params` carry.
- [Module 11 (Building AI Agents)](../../11-building-ai-agents/) -- the nested-span shape comes from agents that make multiple downstream calls per logical request; reading that module explains why `parent_id` is load-bearing.
- [Module 15 (Evaluation & Testing)](../../15-evaluation-testing/) -- eval rows are the offline sibling of trace rows; the same fields (`model`, `cost`, `latency`) appear in both, for different reasons.
- [Module 17 (Caching & Cost Optimization)](../../17-caching-cost-optimization/) -- caching is the lever you pull to lower cost; this module is the meter that tells you what cost looks like before and after.

## Setup

`.env` at the repo root supplies your API key. The script resolves it three levels up (`parent.parent.parent / ".env"`), so you can run it from any working directory.

Set `LLM_MODEL` in `.env` (or your shell environment) to pick the model. Default is `openai/gpt-4o-mini` if unset.

Install the three dependencies if they are not already in your environment:

```
pip install litellm pydantic python-dotenv
```

The tracer writes to `.traces/traces.jsonl` next to `solution.py`. That directory and file are created on first use. Add `.traces/` to your `.gitignore` -- trace files are local runtime state, not source.

### Project layout

```text
project/
├── README.md        this file
├── solution.py      Tracer class + CLI (~350 lines)
└── .traces/         created at runtime, gitignored
    └── traces.jsonl one JSON object per line, one per span
```

Read `solution.py` end-to-end before running it. The `Span` model, the `Tracer` class, and the CLI are independently usable -- you can import `Tracer` into any script without touching `argparse`.

## Walkthrough

Run these four steps in order. Each one builds on the previous.

### Step 1 -- Look at the schema

Generate the demo workload, then inspect the raw trace file:

```
python solution.py --demo
python -c "import itertools; print(*itertools.islice(open('.traces/traces.jsonl'), 2), sep='')"
```

Each line is a complete JSON object. Look for these relationships:

- `kind` is either `"run"` (the parent span) or `"llm"` (a child LLM call).
- The first `"run"` span has `"parent_id": null`. Every `"llm"` span under it has `"parent_id"` set to that run's `span_id`.
- Every `"llm"` span has `attributes.model`, `attributes.tokens_in`, `attributes.tokens_out`, and `attributes.cost_usd`.

Pick any `"llm"` span and trace it upward: its `parent_id` should match the `span_id` of the `"run"` span with the same `run_id`. That parent-child link is the whole schema -- everything else is decoration.

### Step 2 -- Instrument one call yourself

Paste this into a scratch file and run it:

```python
from solution import Tracer

t = Tracer()
with t.run("my_first_run"):
    t.wrap_llm_call(
        model="openai/gpt-4o-mini",
        messages=[{"role": "user", "content": "Hi."}],
    )
```

Then:

```
python solution.py --list
```

You should see one new row in the list -- `my_first_run` -- alongside any runs already in the file. The row shows `run_id`, `name`, `status`, span count, total cost, and wall time. If the row is there, your `Tracer` is writing spans correctly and `--list` is reading them correctly.

### Step 3 -- Instrument a multi-call run

Add two more `wrap_llm_call` calls inside the same `run()` block:

```python
from solution import Tracer

t = Tracer()
with t.run("my_multi_run"):
    t.wrap_llm_call(
        model="openai/gpt-4o-mini",
        messages=[{"role": "user", "content": "Step 1."}],
    )
    t.wrap_llm_call(
        model="openai/gpt-4o-mini",
        messages=[{"role": "user", "content": "Step 2."}],
    )
    t.wrap_llm_call(
        model="openai/gpt-4o-mini",
        messages=[{"role": "user", "content": "Step 3."}],
    )
```

Then show the tree:

```
python solution.py --show <run_id>
```

Replace `<run_id>` with the `run_id` printed by `--list`. The output should show one parent span and three child spans indented under it, each with its own cost and token counts. The parent's `duration_ms` should be at least the sum of the children's -- it wraps the whole block.

### Step 4 -- Generate the full demo workload

Flush existing traces and regenerate:

```
python solution.py --flush
python solution.py --demo
python solution.py --list
```

Three runs appear: `research_agent`, `failure_demo`, and `cost_regression`. These are the runs the worked exercises below use. Re-run `--demo` any time you want a clean slate for the exercises.

## Worked exercise A: Debug the failure

After `--demo`, run:

```
python solution.py --list
```

Find the row for `failure_demo`. Copy its `run_id`. Then:

```
python solution.py --show <run_id>
```

The tree shows one parent run span and one child LLM span. The child is marked `ERROR` and its `error` field contains the exception message from LiteLLM (something like `BadRequestError` or `InvalidRequestError` -- the model name `"invalid/does-not-exist-zzz"` is not a real provider, so LiteLLM rejects it). The parent run's `status` is also `error` because the `run()` context manager catches the exception, records it on the parent span, and re-raises.

Without tracing, diagnosing this requires reading a raw stack trace from stderr and mentally correlating it to the code path that produced it. With tracing, the failed call is one command away and the error message is recorded on the span alongside the model name, the token counts (zero on failure), and the cost (zero on failure). You know exactly which call failed, which model was targeted, and what went wrong -- before you open any source file.

The implementation detail to notice: `wrap_llm_call` must catch the exception, write the span with `status="error"` and `error=str(exception)`, then re-raise. If it swallowed the exception, the parent run would complete normally and you would never know the call failed. If it did not catch the exception, the span would never be written. Catch, record, re-raise is the only shape that gives you both the trace and the error propagation.

## Worked exercise B: Spot the cost regression

After `--demo`, run:

```
python solution.py --stats
```

The output includes a per-model breakdown (token counts and total cost for each model used) and a "Top 5 runs by cost" table. Find `cost_regression` in the top-runs table. It should be noticeably more expensive than `research_agent` despite making only two calls to `research_agent`'s three.

Then show the tree:

```
python solution.py --show <run_id>
```

The two child spans reveal the cause: the first call has a short system prompt (a sentence or two), and the second call has a 4 KB system prompt -- roughly 1,000 tokens of extra input. At approximately $0.00015 per 1,000 input tokens for `gpt-4o-mini`, that extra system prompt adds around $0.0001-0.0002 per call. Small in absolute dollars. Large as a relative delta between the two calls in the same run.

This is the lesson: cost regressions from prompt changes are invisible without per-run tracking. The typical discovery path without tracing is noticing a spike in the AWS bill at the end of the month and having no way to attribute it to a specific run, call, or prompt change. With `--stats` and `--show`, the regression is visible immediately -- the expensive run is in the top-runs table, and the expensive call is the one with the large `attributes.tokens_in` value.

## Live-test commands

Run these from the repo root:

```
python 18-observability-monitoring/project/solution.py --flush
python 18-observability-monitoring/project/solution.py --demo
python 18-observability-monitoring/project/solution.py --list
python 18-observability-monitoring/project/solution.py --show <run_id>
python 18-observability-monitoring/project/solution.py --stats
python 18-observability-monitoring/project/solution.py --tail
```

`--tail` streams new spans to stdout as they are written to the JSONL file. Run `--demo` in a second terminal while `--tail` is running to see spans arrive in real time. Press `Ctrl-C` to exit.

## Where to go next

- `../19-advanced-rag/` -- applying tracing to a real RAG pipeline, where the span tree has retrieval spans alongside LLM spans and the interesting cost question is whether retrieval or generation dominates.
- `../20-deployment-patterns/` -- async span export, sampling decisions (trace every call vs. every Nth call), OTel-compatible export format, and retention policies for keeping trace storage costs bounded.
