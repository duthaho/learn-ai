"""
Scorecard — complete reference implementation of an eval harness.

Loads a JSONL eval dataset, runs a pluggable system-under-test in parallel,
scores each row with mechanical + LLM-based evaluators, and produces a
console + JSON scorecard you can compare across runs.

Three evaluators ship in the box:
  - ExactMatchEvaluator  : case-insensitive exact match on a chosen field
  - SchemaEvaluator      : Pydantic model_validate (Module 08 pattern)
  - LLMJudgeEvaluator    : rubric scoring via LLM, 0-10 normalized

Run:
    python solution.py --dataset datasets/sentiment.jsonl
    python solution.py --dataset datasets/sentiment.jsonl --judge
    python solution.py --dataset datasets/sentiment.jsonl --judge --concurrency 8
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from dotenv import load_dotenv
from litellm import completion, completion_cost
from pydantic import BaseModel, Field, ValidationError

load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

MODEL = os.getenv("LLM_MODEL", "anthropic/claude-sonnet-4-20250514")
DEFAULT_CONCURRENCY = 4
JUDGE_PASS_THRESHOLD = 7    # raw 0-10 score >= this -> evaluator pass


# ---------- Pydantic models ----------


class EvalRow(BaseModel):
    input: str | dict
    expected: dict
    metadata: dict = Field(default_factory=dict)


class EvalResult(BaseModel):
    evaluator_name: str
    passed: bool
    score: float           # 0.0 - 1.0 normalized
    reason: str
    latency_ms: int
    cost: float


class RowOutcome(BaseModel):
    row: EvalRow
    actual: dict
    sut_latency_ms: int
    sut_cost: float
    evaluations: list[EvalResult]
    row_passed: bool


class EvaluatorAggregate(BaseModel):
    name: str
    pass_count: int
    total: int
    pass_rate: float
    mean_score: float


class Scorecard(BaseModel):
    run_id: str
    timestamp: str
    dataset_path: str
    sut: str
    model: str
    concurrency: int
    rows: list[RowOutcome]
    evaluator_aggregates: list[EvaluatorAggregate]
    overall_pass_rate: float
    total_cost: float
    total_latency_ms: int


# ---------- Helpers ----------


def _strip_code_fence(text: str) -> str:
    """Strip a ```<lang> ... ``` fence if the model wrapped its output."""
    s = text.strip()
    if not s.startswith("```"):
        return s
    s = s[3:]
    i = 0
    while i < len(s) and s[i].isalpha():
        i += 1
    s = s[i:]
    s = s.lstrip("\r\n ")
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


# ---------- Evaluators ----------


class ExactMatchEvaluator:
    """Case-insensitive exact match between expected[field] and actual[field]."""

    name = "exact_match"

    def __init__(self, field: str):
        self.field = field

    def evaluate(self, input_value, expected: dict, actual: dict) -> EvalResult:
        start = time.perf_counter()
        exp = str(expected.get(self.field, "")).strip().lower()
        act = str(actual.get(self.field, "")).strip().lower()
        passed = exp == act and exp != ""
        latency_ms = int((time.perf_counter() - start) * 1000)
        return EvalResult(
            evaluator_name=self.name,
            passed=passed,
            score=1.0 if passed else 0.0,
            reason=(
                f"match on '{self.field}': '{exp}'"
                if passed
                else f"mismatch on '{self.field}': expected '{exp}', got '{act}'"
            ),
            latency_ms=latency_ms,
            cost=0.0,
        )


class SchemaEvaluator:
    """Validate that actual parses cleanly into the given Pydantic schema."""

    name = "schema"

    def __init__(self, schema: type[BaseModel]):
        self.schema = schema

    def evaluate(self, input_value, expected: dict, actual: dict) -> EvalResult:
        start = time.perf_counter()
        try:
            self.schema.model_validate(actual)
            passed = True
            reason = f"valid {self.schema.__name__}"
        except ValidationError as e:
            passed = False
            reason = f"schema validation failed: {e.errors()[0].get('msg', str(e))[:120]}"
        latency_ms = int((time.perf_counter() - start) * 1000)
        return EvalResult(
            evaluator_name=self.name,
            passed=passed,
            score=1.0 if passed else 0.0,
            reason=reason,
            latency_ms=latency_ms,
            cost=0.0,
        )


JUDGE_PROMPT = """You are an impartial evaluator. Given an input, an expected output, and an actual output from a system, score the actual output on the rubric below.

Rubric:
{rubric}

Return ONLY JSON in this shape:
{{"score": <integer 0-10>, "reason": "<one-sentence justification>"}}

Do not include markdown fences. Do not include any other text."""


class LLMJudgeEvaluator:
    """Rubric scoring via LLM. Pass if raw score >= pass_threshold (default 7)."""

    name = "llm_judge"

    def __init__(
        self,
        rubric: str,
        model: str = MODEL,
        pass_threshold: int = JUDGE_PASS_THRESHOLD,
    ):
        self.rubric = rubric
        self.model = model
        self.pass_threshold = pass_threshold

    def evaluate(self, input_value, expected: dict, actual: dict) -> EvalResult:
        start = time.perf_counter()
        system_prompt = JUDGE_PROMPT.format(rubric=self.rubric)
        user_content = (
            f"Input:\n{input_value}\n\n"
            f"Expected output:\n{json.dumps(expected)}\n\n"
            f"Actual output:\n{json.dumps(actual)}\n\n"
            "Return the JSON judgment."
        )
        response = completion(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            response_format={"type": "json_object"},
        )
        latency_ms = int((time.perf_counter() - start) * 1000)
        _, _, cost = _usage_from_response(response)
        raw = response.choices[0].message.content or ""
        cleaned = _strip_code_fence(raw)
        try:
            parsed = json.loads(cleaned)
            raw_score = int(parsed.get("score", 0))
            reason_text = str(parsed.get("reason", ""))
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            return EvalResult(
                evaluator_name=self.name,
                passed=False,
                score=0.0,
                reason=f"judge parse error: {e}",
                latency_ms=latency_ms,
                cost=cost,
            )
        raw_score = max(0, min(10, raw_score))
        normalized = raw_score / 10.0
        passed = raw_score >= self.pass_threshold
        return EvalResult(
            evaluator_name=self.name,
            passed=passed,
            score=normalized,
            reason=f"score={raw_score}/10: {reason_text}",
            latency_ms=latency_ms,
            cost=cost,
        )


# ---------- Dataset and SUT loaders ----------


def load_dataset(path: str | Path) -> list[EvalRow]:
    """Read a JSONL file and return a list of EvalRow."""
    path = Path(path).resolve()
    rows: list[EvalRow] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Line {i} of {path} is not valid JSON: {e}") from e
            try:
                rows.append(EvalRow.model_validate(data))
            except ValidationError as e:
                raise ValueError(f"Line {i} of {path} does not match EvalRow: {e}") from e
    return rows


def load_sut(sut_spec: str) -> Callable:
    """Resolve a 'module.path:callable' string to a callable."""
    if ":" not in sut_spec:
        raise ValueError(f"--sut must be 'module.path:callable', got '{sut_spec}'")
    module_path, callable_name = sut_spec.split(":", 1)
    # Make the project directory importable for `tasks.sentiment` etc.
    project_dir = Path(__file__).resolve().parent
    if str(project_dir) not in sys.path:
        sys.path.insert(0, str(project_dir))
    try:
        module = importlib.import_module(module_path)
    except ImportError as e:
        raise ValueError(f"could not import module '{module_path}': {e}") from e
    sut = getattr(module, callable_name, None)
    if sut is None or not callable(sut):
        raise ValueError(f"module '{module_path}' has no callable '{callable_name}'")
    return sut


# ---------- Per-row execution ----------


def run_sut_on_row(row: EvalRow, sut: Callable, model: str) -> tuple[dict, int, float]:
    """Call the SUT on one row. Catch exceptions and report them as actual={'error': ...}."""
    start = time.perf_counter()
    cost = 0.0
    try:
        result = sut(row.input, model=model) if _accepts_model(sut) else sut(row.input)
    except Exception as e:
        latency_ms = int((time.perf_counter() - start) * 1000)
        return {"error": f"{type(e).__name__}: {e}"}, latency_ms, 0.0

    latency_ms = int((time.perf_counter() - start) * 1000)

    # Some SUTs may return (output, usage_or_cost); we tolerate either shape.
    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], (int, float)):
        actual, cost = result
    else:
        actual = result

    if not isinstance(actual, dict):
        actual = {"value": actual}

    return actual, latency_ms, float(cost)


def _accepts_model(fn: Callable) -> bool:
    """Cheap reflection: does this callable accept a 'model' kwarg?"""
    import inspect
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return False
    return "model" in sig.parameters


def run_evaluators(
    input_value,
    expected: dict,
    actual: dict,
    evaluators: list,
    sut_errored: bool,
) -> list[EvalResult]:
    """Run each evaluator sequentially. If the SUT errored, auto-fail all evaluators."""
    results: list[EvalResult] = []
    for ev in evaluators:
        if sut_errored:
            results.append(EvalResult(
                evaluator_name=ev.name,
                passed=False,
                score=0.0,
                reason="sut error — evaluator auto-failed",
                latency_ms=0,
                cost=0.0,
            ))
            continue
        try:
            results.append(ev.evaluate(input_value, expected, actual))
        except Exception as e:
            results.append(EvalResult(
                evaluator_name=ev.name,
                passed=False,
                score=0.0,
                reason=f"evaluator crashed: {type(e).__name__}: {e}",
                latency_ms=0,
                cost=0.0,
            ))
    return results


# ---------- Aggregate ----------


def aggregate(
    row_outcomes: list[RowOutcome],
    evaluator_names: list[str],
) -> tuple[list[EvaluatorAggregate], float]:
    aggregates: list[EvaluatorAggregate] = []
    for name in evaluator_names:
        results = [ev for ro in row_outcomes for ev in ro.evaluations if ev.evaluator_name == name]
        total = len(results)
        pass_count = sum(1 for r in results if r.passed)
        mean_score = (sum(r.score for r in results) / total) if total else 0.0
        pass_rate = (pass_count / total) if total else 0.0
        aggregates.append(EvaluatorAggregate(
            name=name,
            pass_count=pass_count,
            total=total,
            pass_rate=pass_rate,
            mean_score=mean_score,
        ))
    total_rows = len(row_outcomes)
    overall = (sum(1 for ro in row_outcomes if ro.row_passed) / total_rows) if total_rows else 0.0
    return aggregates, overall


# ---------- Orchestrator ----------


def run_eval(
    dataset: list[EvalRow],
    sut: Callable,
    evaluators: list,
    model: str = MODEL,
    concurrency: int = DEFAULT_CONCURRENCY,
    sut_spec: str = "",
    dataset_path: str = "",
    verbose: bool = True,
) -> Scorecard:
    """Run the SUT on every row in parallel, then evaluators, then aggregate."""
    run_id = "scorecard_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    timestamp = datetime.now(timezone.utc).isoformat()
    evaluator_names = [ev.name for ev in evaluators]
    start = time.perf_counter()

    # ---- Phase 1: parallel SUT calls ----
    if verbose:
        print(f"\nRunning SUT on {len(dataset)} rows...")
    sut_phase_start = time.perf_counter()
    sut_results: dict[int, tuple[dict, int, float]] = {}
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(run_sut_on_row, row, sut, model): i for i, row in enumerate(dataset)}
        for fut in as_completed(futures):
            idx = futures[fut]
            sut_results[idx] = fut.result()
    sut_phase_ms = int((time.perf_counter() - sut_phase_start) * 1000)
    if verbose:
        latencies = [v[1] for v in sut_results.values()]
        avg = sum(latencies) / len(latencies) / 1000 if latencies else 0.0
        mx = max(latencies) / 1000 if latencies else 0.0
        print(f"  -> done in {sut_phase_ms/1000:.1f}s (avg {avg:.1f}s/row, max {mx:.1f}s)")

    # ---- Phase 2: per-row evaluator pass ----
    if verbose:
        print(f"\nRunning evaluators...")
    row_outcomes: list[RowOutcome] = []
    for i, row in enumerate(dataset):
        actual, sut_latency_ms, sut_cost = sut_results[i]
        sut_errored = "error" in actual
        evaluations = run_evaluators(row.input, row.expected, actual, evaluators, sut_errored)
        row_passed = (not sut_errored) and all(ev.passed for ev in evaluations)
        row_outcomes.append(RowOutcome(
            row=row,
            actual=actual,
            sut_latency_ms=sut_latency_ms,
            sut_cost=sut_cost,
            evaluations=evaluations,
            row_passed=row_passed,
        ))

    aggregates, overall_pass_rate = aggregate(row_outcomes, evaluator_names)

    if verbose:
        for agg in aggregates:
            extra = f" (mean score {agg.mean_score:.2f})" if agg.name == "llm_judge" else ""
            print(f"  -> {agg.name:<14} on {agg.total} rows: {agg.pass_count} pass / {agg.total - agg.pass_count} fail{extra}")

    total_cost = sum(ro.sut_cost for ro in row_outcomes) + sum(
        ev.cost for ro in row_outcomes for ev in ro.evaluations
    )
    total_latency_ms = int((time.perf_counter() - start) * 1000)

    return Scorecard(
        run_id=run_id,
        timestamp=timestamp,
        dataset_path=dataset_path,
        sut=sut_spec,
        model=model,
        concurrency=concurrency,
        rows=row_outcomes,
        evaluator_aggregates=aggregates,
        overall_pass_rate=overall_pass_rate,
        total_cost=round(total_cost, 6),
        total_latency_ms=total_latency_ms,
    )


# ---------- CLI / printing ----------


def _print_scorecard(card: Scorecard) -> None:
    print(f"\n=== Per-row failures ===")
    any_failure = False
    for ro in card.rows:
        if ro.row_passed:
            continue
        any_failure = True
        row_id = ro.row.metadata.get("id", "?")
        exp = ro.row.expected
        act = ro.actual
        failed_evs = [ev for ev in ro.evaluations if not ev.passed]
        ev_summary = ", ".join(
            f"{ev.evaluator_name} FAIL"
            + (f" (score {int(ev.score * 10)}/10)" if ev.evaluator_name == "llm_judge" else "")
            for ev in failed_evs
        ) or "no evaluators ran"
        snippet = (str(ro.row.input)[:60] + "...") if len(str(ro.row.input)) > 60 else str(ro.row.input)
        print(f"[{row_id}] expected={exp} actual={act} | {ev_summary} | {snippet!r}")
    if not any_failure:
        print("(none)")

    print(f"\n=== Aggregates ===")
    print(f"{'evaluator':<14} {'pass_rate':>12} {'mean_score':>12}")
    for agg in card.evaluator_aggregates:
        suffix = f"  (raw {agg.mean_score * 10:.1f}/10)" if agg.name == "llm_judge" else ""
        print(f"{agg.name:<14} {agg.pass_rate * 100:>11.1f}% {agg.mean_score:>12.2f}{suffix}")
    print(f"\nOverall pass rate (all evaluators pass): {card.overall_pass_rate * 100:.1f}%")
    print(f"Total cost:    ${card.total_cost:.6f}")
    print(f"Total latency: {card.total_latency_ms/1000:.1f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Scorecard — eval harness (Module 15)")
    parser.add_argument("--dataset", required=True, help="Path to a JSONL eval dataset")
    parser.add_argument(
        "--sut",
        default="tasks.sentiment:classify_sentiment",
        help="System-under-test as 'module.path:callable' (default: tasks.sentiment:classify_sentiment)",
    )
    parser.add_argument("--judge", action="store_true", help="Add an LLM-as-judge evaluator")
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY,
                        help=f"ThreadPool workers for the SUT phase (default {DEFAULT_CONCURRENCY})")
    parser.add_argument("--save", default=None, help="Path to write the results JSON (default: results-{run_id}.json)")
    parser.add_argument("--no-save", action="store_true", help="Skip writing the results JSON")
    parser.add_argument("--model", default=MODEL, help=f"Model override (default: {MODEL})")
    args = parser.parse_args()

    # Load dataset
    try:
        dataset = load_dataset(args.dataset)
    except (FileNotFoundError, ValueError) as e:
        parser.error(f"could not load dataset: {e}")

    # Load SUT
    try:
        sut = load_sut(args.sut)
    except ValueError as e:
        parser.error(str(e))

    # Build evaluator list
    evaluators: list = [ExactMatchEvaluator(field="sentiment")]

    # Try to import the schema from the SUT module (best-effort; skip if not present)
    try:
        module_path = args.sut.split(":", 1)[0]
        sut_module = importlib.import_module(module_path)
        schema_cls = getattr(sut_module, "SentimentLabel", None)
        if schema_cls and isinstance(schema_cls, type) and issubclass(schema_cls, BaseModel):
            evaluators.append(SchemaEvaluator(schema=schema_cls))
    except Exception:
        pass

    if args.judge:
        rubric = (
            "Score the actual output on whether its sentiment label matches "
            "the genuine sentiment of the input review. A correct label with "
            "weak confidence is fine. An incorrect label is a low score. "
            "Borderline reviews (mixed sentiment) deserve credit if the "
            "actual output is defensible."
        )
        evaluators.append(LLMJudgeEvaluator(rubric=rubric, model=args.model))

    # Print header
    print(f"=== Scorecard ===")
    print(f"Dataset:        {args.dataset} ({len(dataset)} rows)")
    print(f"SUT:            {args.sut}")
    print(f"Model:          {args.model}")
    print(f"Concurrency:    {args.concurrency}")

    # Run
    card = run_eval(
        dataset=dataset,
        sut=sut,
        evaluators=evaluators,
        model=args.model,
        concurrency=args.concurrency,
        sut_spec=args.sut,
        dataset_path=args.dataset,
    )
    print(f"Run ID:         {card.run_id}")

    _print_scorecard(card)

    # Save
    if not args.no_save:
        save_path = Path(args.save) if args.save else Path(f"results-{card.run_id}.json")
        save_path.write_text(card.model_dump_json(indent=2), encoding="utf-8")
        print(f"\nSaved: {save_path}")


if __name__ == "__main__":
    main()
