# Project: Eval Harness ("Scorecard")

A reusable evaluation runner. Load a JSONL dataset, run a pluggable system-under-test on every row in parallel, score each row with mechanical and LLM-based evaluators, and produce a console + JSON scorecard you can compare across runs.

## What you'll build

- A `load_dataset(path)` function that reads a JSONL file of `{"input": ..., "expected": ..., "metadata": {...}}` rows
- A `load_sut(spec)` function that resolves `module.path:callable` strings via `importlib`
- Three evaluator classes: `ExactMatchEvaluator`, `SchemaEvaluator`, `LLMJudgeEvaluator`
- A `run_eval(...)` orchestrator that fans out SUT calls via `ThreadPoolExecutor`, runs evaluators per row, and aggregates a `Scorecard`
- A bundled system-under-test (`tasks/sentiment.py`) — a small LiteLLM-backed sentiment classifier
- A bundled 20-row eval dataset of movie reviews
- A CLI with `--dataset`, `--sut`, `--judge`, `--concurrency`, `--model`, `--save`, `--no-save` flags

The project demonstrates:

- **Eval loop:** dataset → parallel SUT → per-row evaluators → scorecard (Module 13 workflow shape)
- **Three evaluator types in one project:** exact-match (mechanical), schema (Module 08 cross-link), LLM-as-judge (Module 12 critic pattern)
- **Per-row failure visibility:** the scorecard shows which rows flipped and why
- **JSON persistence:** every run is a `results-{run_id}.json` you can diff later

## Prerequisites

- [Module 02 (Prompt Engineering)](../../02-prompt-engineering/), [Module 08 (Structured Output)](../../08-structured-output/), [Module 13 (Workflows & Chains)](../../13-workflows-chains/), and [Module 14 (AI Code Generation)](../../14-ai-code-generation/) recommended — schema validation, the deterministic outer pipeline, and the `_strip_code_fence` helper all come from there.
- Completed reading the [Module 15 README](../README.md) so the offline/online split and the "what makes an eval load-bearing" framing are fresh.
- Python 3.11+ with the project venv already installed from the repo root. No new dependencies beyond what Module 13 required.

## Setup

`.env` at the repo root supplies your API key. `LLM_MODEL` defaults to `anthropic/claude-sonnet-4-20250514` if unset; pass `--model` to override at runtime without touching `.env`. The script resolves `.env` relative to the source file, so you can run it from any cwd.

### Project layout

```text
project/
├── README.md          this file
├── solution.py        the harness (~550 lines)
├── tasks/
│   ├── __init__.py
│   └── sentiment.py   bundled system-under-test
└── datasets/
    └── sentiment.jsonl  20 labeled movie reviews
```

Read `solution.py` end-to-end before you run it. Every step is independently callable from a REPL.

## How it works

```text
JSONL dataset ──┐
                ├──→ Run SUT on each row in parallel (ThreadPool) ──→ For each (row, actual):
SUT (callable) ─┘                                                     run each evaluator sequentially
                                                                          │
                                                                          ↓
                                                                      aggregate per evaluator
                                                                          │
                                                                          ↓
                                                                   Scorecard (console + JSON)
```

- **Load** parses the JSONL dataset and resolves the SUT callable via `importlib`. No model calls, no I/O beyond the file read. The dataset shape is fixed (`input`, `expected`, optional `metadata`) so any SUT that accepts the same `input` type can be swapped in without touching the harness.
- **Fan out** submits one `run_sut_on_row(row, sut, model)` call per row to a `ThreadPoolExecutor`. Each call catches the SUT's exceptions, captures wall-clock latency, and returns `(actual, latency_ms, cost)`. The pool sizes via `--concurrency` so you can dial throughput vs. rate-limit headroom per run.
- **Evaluate** runs each evaluator sequentially against the `(input, expected, actual)` triple for every row. If the SUT errored, every evaluator auto-fails for that row — no point asking a judge to grade a missing answer. Each evaluator returns an `EvalResult` with `passed`, `score`, and an optional `reason`.
- **Aggregate + render** rolls per-row `EvalResult`s up into an `EvaluatorAggregate` per evaluator (pass rate, mean score, fail count), computes the overall "all evaluators passed" rate, and prints the scorecard plus the per-row failures. The full `Scorecard` is also serialized to `results-{run_id}.json` unless `--no-save` is passed.

The shape is workflow-first with a hard fan-out boundary. The dataset load, evaluator pass, and aggregate steps are deterministic. The only nondeterminism lives inside the SUT call and the optional LLM judge. That separation is what makes scorecards diffable across runs: when the overall pass rate moves, you can tell which evaluator drove the change and which rows flipped, because every layer above the model call is reproducible.

## Build it step by step

1. **Define the Pydantic models** (`EvalRow`, `EvalResult`, `RowOutcome`, `EvaluatorAggregate`, `Scorecard`). `EvalRow` is the dataset shape (`input`, `expected`, `metadata`). `EvalResult` is one evaluator's verdict on one row (`evaluator_name`, `passed`, `score`, `reason`). `RowOutcome` bundles `(row, actual, latency_ms, cost, sut_error, results: list[EvalResult])`. `EvaluatorAggregate` carries the per-evaluator stats. `Scorecard` is the top-level run record with `run_id`, `dataset`, `sut`, `model`, `concurrency`, aggregates, row outcomes, totals.
2. **Write the three evaluator classes** (`ExactMatchEvaluator`, `SchemaEvaluator`, `LLMJudgeEvaluator`) — each with a `.evaluate(input, expected, actual) -> EvalResult` method and a `.name` attribute. `ExactMatchEvaluator` is `actual == expected` with a `reason` on mismatch. `SchemaEvaluator` takes a Pydantic model class in its constructor and validates `actual` against it. `LLMJudgeEvaluator` takes a rubric string, calls the model, parses the response, and maps the raw 0-10 score into `passed = score >= threshold`.
3. **Implement the `LLMJudgeEvaluator` JSON parsing** using the `_strip_code_fence` helper from Module 14. The judge prompt asks for `{"score": int, "reason": str}` in a fenced block; the helper strips the fence, `json.loads` produces the dict, Pydantic validates it. On parse failure, return a failed `EvalResult` with the raw response in `reason` rather than letting the exception propagate — one bad judge response shouldn't tank the whole run.
4. **Implement `load_dataset(path)`** — open the file, iterate line-by-line, skip blank lines, `json.loads` each line into an `EvalRow`. Raise a domain-specific error on the first malformed row with the line number, so dataset edits fail loudly at load time rather than midway through a run.
5. **Implement `load_sut(spec)`** — split `spec` on `:`, `importlib.import_module(module_path)`, `getattr(module, callable_name)`. Return the callable. Wrap `ImportError` and `AttributeError` in a clear "couldn't resolve SUT" message that names the spec string.
6. **Implement `run_sut_on_row(row, sut, model)`** — call the SUT with `(row.input, model)`, time it with `time.perf_counter`, catch any exception and return it in the outcome rather than re-raising. Return `(actual, latency_ms, cost, error)` where `error` is `None` on success. The orchestrator decides what to do with the error; this function just captures it.
7. **Implement `run_evaluators(input, expected, actual, evaluators, sut_errored)`** — if `sut_errored` is true, return one auto-fail `EvalResult` per evaluator with `reason="SUT errored, skipping"`. Otherwise call `.evaluate(input, expected, actual)` on each evaluator sequentially and collect the results. Catch exceptions from individual evaluators too — a misbehaving judge shouldn't break a run any more than a SUT crash should.
8. **Implement `aggregate(row_outcomes, evaluator_names)`** — for each evaluator name, walk every row, count passes/fails, compute the mean score across rows that produced a result. Compute the overall "all evaluators passed" pass rate. Compute totals (rows, cost, latency). Return a `dict[str, EvaluatorAggregate]` plus the totals dict.
9. **Implement the `run_eval(...)` orchestrator** — load the dataset and SUT, build the evaluator list (the judge is opt-in via the flag), submit `run_sut_on_row` for every row to a `ThreadPoolExecutor(max_workers=concurrency)`, gather results in submission order, run the evaluator pass per row, aggregate, build the `Scorecard` with a timestamped `run_id`, optionally persist to JSON. Return the `Scorecard`.
10. **Implement the print helpers** (`_print_scorecard`). Print the header (dataset, SUT, model, concurrency, run ID), the SUT phase summary (rows, total time, avg/max per row), the evaluator phase summary (one line per evaluator with pass/fail counts and mean score), the per-row failures section (only rows where any evaluator failed), the aggregates table, and the totals. Keep pure-print — formatting only, no logic.
11. **Wire up the CLI with `argparse`.** `--dataset PATH` (required). `--sut SPEC` (default `tasks.sentiment:classify_sentiment`). `--judge` (flag, opt-in LLM judge). `--concurrency INT` (default 4). `--model NAME` (default from `LLM_MODEL` env). `--save` / `--no-save` (mutually exclusive; default save). Parse args, call `run_eval`, call `_print_scorecard`, exit nonzero if overall pass rate is below a configurable threshold (default 80%).

Each step is small and independently testable. Steps 2, 4, and 5 in particular should pass on their own before you wire up the orchestrator — instantiate an evaluator and call `.evaluate(...)` with hand-crafted args, `load_dataset` against the bundled JSONL, `load_sut` against the bundled `tasks.sentiment:classify_sentiment`. If those three are solid, the orchestrator is just plumbing around a thread pool.

## Run it

```bash
python solution.py --dataset datasets/sentiment.jsonl
python solution.py --dataset datasets/sentiment.jsonl --judge
python solution.py --dataset datasets/sentiment.jsonl --judge --concurrency 8
python solution.py --dataset datasets/sentiment.jsonl --no-save
python solution.py --dataset datasets/sentiment.jsonl --sut tasks.sentiment:classify_sentiment --model anthropic/claude-haiku-4-5-20251001
```

Expected console output (exact values vary):

```text
=== Scorecard ===
Dataset:        datasets/sentiment.jsonl (20 rows)
SUT:            tasks.sentiment:classify_sentiment
Model:          anthropic/claude-sonnet-4-20250514
Concurrency:    4
Run ID:         scorecard_20260413_142103

Running SUT on 20 rows...
  -> done in 4.2s (avg 1.1s/row, max 2.3s)

Running evaluators...
  -> exact_match     on 20 rows: 17 pass / 3 fail
  -> schema          on 20 rows: 20 pass / 0 fail
  -> llm_judge       on 20 rows: 19 pass / 1 fail (mean score 0.84)

=== Per-row failures ===
[r03] expected={'sentiment': 'positive'} actual={'sentiment': 'neutral'} | exact_match FAIL | "It was fine, I guess..."

=== Aggregates ===
evaluator         pass_rate   mean_score
exact_match            85.0%       0.85
schema                100.0%       1.00
llm_judge              95.0%       0.84  (raw 8.4/10)

Overall pass rate (all evaluators pass): 85.0%
Total cost:    $0.018400
Total latency: 8.7s

Saved: results-scorecard_20260413_142103.json
```

Use `--no-save` while iterating on the harness itself — once the layout stabilizes, drop the flag and start accumulating run JSONs so you have a diffable history.

## Extensions

Once the base harness works, these are the natural next experiments:

1. **Add a `--compare A.json B.json` flag** that diffs two saved scorecards and prints flipped rows (row IDs where overall pass changed between A and B), per-evaluator delta in pass rate, and any new fail reasons. This is the actual workflow eval harnesses unlock — runs are only useful if you can diff them.
2. **Parallelize the evaluators per row.** A second `ThreadPoolExecutor` inside `run_evaluators` so the LLM judge runs concurrently with `ExactMatchEvaluator` and `SchemaEvaluator` per row. Mechanical evaluators finish in microseconds; the judge is the long pole. Worth the extra complexity only once your dataset is large enough that the savings are visible.
3. **Write a synthetic-dataset generator** that asks an LLM to produce 50 more sentiment reviews with labels, dedupes them against the existing dataset, and writes a new `sentiment_synthetic.jsonl`. The interesting part: how do you validate the synthetic labels without making the dataset circular?
4. **Add a fourth evaluator: cosine similarity** between `actual` and `expected` using embeddings (Module 03 cross-link). Useful for free-text outputs where exact-match is too strict and the judge is too expensive. Threshold becomes a knob worth its own ablation run.
5. **Build a regression CI:** run the harness in GitHub Actions on PR, compare against the main-branch scorecard, fail the build on pass-rate regression. The plumbing is straightforward; the interesting design question is what counts as "regression" — total pass rate dropping, any single row flipping from pass to fail, or a specific evaluator dropping below threshold.

## Reference

Cross-links for context:

- [Module 15 README](../README.md) — eval taxonomy (offline vs. online, mechanical vs. judge), what makes a load-bearing eval, why JSON persistence matters.
- [Module 08 (Structured Output)](../../08-structured-output/) — the `SchemaEvaluator` is a thin wrapper around the same Pydantic-validation idea, applied as an evaluator instead of as a parser.
- [Module 12 (Multi-Agent Systems)](../../12-multi-agent-systems/) — the `LLMJudgeEvaluator` is the critic-pattern reused; the judge prompt borrows the same rubric structure.
- [Module 13 (Workflows & Chains)](../../13-workflows-chains/) — the deterministic outer pipeline shape (load → fan-out → aggregate → render) is the workflow primitive applied to eval.
- [Module 14 (AI Code Generation)](../../14-ai-code-generation/) — `_strip_code_fence` is the same helper, and the per-attempt accounting pattern carries over to per-row accounting here.

**Next:** Phase 4 lifts these scorecards into deployment monitoring — same evaluator contract, but rows come from live traffic instead of a JSONL file, and the aggregate is a rolling window over production calls. The `Scorecard` you produce here is the shape that flows into the dashboards there.
