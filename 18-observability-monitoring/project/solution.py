"""Module 18 - Observability & Monitoring: Tracer + CLI.

Records LLM calls as nested spans to JSONL, then queries them from a CLI.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Literal, Optional

from dotenv import load_dotenv
from litellm import completion, completion_cost
from pydantic import BaseModel, Field

# Load .env from repo root (parent.parent.parent of this file).
_ENV_PATH = Path(__file__).resolve().parent.parent.parent / ".env"
if _ENV_PATH.exists():
    load_dotenv(_ENV_PATH)

MODEL = os.environ.get("LLM_MODEL", "openai/gpt-4o-mini")
DEFAULT_TRACE_DIR = Path(__file__).resolve().parent / ".traces"
DEFAULT_TRACE_FILE = "traces.jsonl"
PREVIEW_HEAD = 80
PREVIEW_TAIL = 80
PREVIEW_JOIN = "  ...  "
PREVIEW_FULL_THRESHOLD = PREVIEW_HEAD + PREVIEW_TAIL  # 160


class Span(BaseModel):
    """One traced unit of work - either a logical run (kind='run') or an LLM call (kind='llm')."""
    span_id: str
    run_id: str
    parent_id: Optional[str] = None
    kind: Literal["run", "llm"]
    name: str
    started_at: float
    ended_at: float
    duration_ms: int
    status: Literal["ok", "error"]
    error: Optional[str] = None
    attributes: dict[str, Any] = Field(default_factory=dict)


Span.model_rebuild()


def _new_id() -> str:
    """16-hex-char random ID."""
    return uuid.uuid4().hex[:16]


def _hash_prompt(messages: list[dict[str, Any]]) -> str:
    """Stable SHA-256 of the messages list, sorted-key JSON."""
    return hashlib.sha256(
        json.dumps(messages, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _preview(text: str) -> str:
    """First + last chars joined by ellipsis; whole string if short."""
    if len(text) <= PREVIEW_FULL_THRESHOLD:
        return text
    return text[:PREVIEW_HEAD] + PREVIEW_JOIN + text[-PREVIEW_TAIL:]


def _safe_cost(response: Any) -> float:
    """Best-effort cost extraction. Returns 0.0 on any failure (e.g., model not in price map)."""
    try:
        return float(completion_cost(response) or 0.0)
    except Exception:
        return 0.0


def _messages_to_text(messages: list[dict[str, Any]]) -> str:
    """Flatten messages into a single string for preview/length purposes."""
    return "\n".join(f"{m.get('role','')}: {m.get('content','')}" for m in messages)


def _response_text(response: Any) -> str:
    """Pull the assistant text out of a LiteLLM response object."""
    try:
        return response.choices[0].message.content or ""
    except Exception:
        return ""


class _SpanFrame:
    """Per-stack-entry record. Held in thread-local; not serialized."""

    def __init__(
        self,
        span_id: str,
        run_id: str,
        parent_id: Optional[str],
        kind: Literal["run", "llm"],
        name: str,
        started_at: float,
        attributes: dict[str, Any],
    ):
        self.span_id = span_id
        self.run_id = run_id
        self.parent_id = parent_id
        self.kind = kind
        self.name = name
        self.started_at = started_at
        self.attributes = attributes
        # Aggregates rolled up from children at close time.
        self.child_span_count = 0
        self.total_cost_usd = 0.0
        self.total_tokens_in = 0
        self.total_tokens_out = 0
        self.had_error = False


class Tracer:
    """Records spans to a JSONL file. Thread-safe within one process."""

    def __init__(self, path: Optional[Path] = None, redact: bool = False) -> None:
        self.path = Path(path) if path else (DEFAULT_TRACE_DIR / DEFAULT_TRACE_FILE)
        self.redact = redact
        self._lock = threading.Lock()
        self._local = threading.local()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    # ---- Stack management ----

    def _stack(self) -> list[_SpanFrame]:
        if not hasattr(self._local, "stack"):
            self._local.stack = []
        return self._local.stack

    def _current(self) -> Optional[_SpanFrame]:
        stack = self._stack()
        return stack[-1] if stack else None

    # ---- Span open/close ----

    def _open(
        self,
        kind: Literal["run", "llm"],
        name: str,
        attributes: Optional[dict[str, Any]] = None,
    ) -> _SpanFrame:
        parent = self._current()
        span_id = _new_id()
        run_id = parent.run_id if parent else span_id
        parent_id = parent.span_id if parent else None
        frame = _SpanFrame(
            span_id=span_id,
            run_id=run_id,
            parent_id=parent_id,
            kind=kind,
            name=name,
            started_at=time.time(),
            attributes=dict(attributes or {}),
        )
        self._stack().append(frame)
        return frame

    def _close(self, frame: _SpanFrame, status: Literal["ok", "error"], error: Optional[str]) -> None:
        ended_at = time.time()
        duration_ms = int((ended_at - frame.started_at) * 1000)
        attributes = dict(frame.attributes)
        if frame.kind == "run":
            attributes["child_span_count"] = frame.child_span_count
            attributes["total_cost_usd"] = round(frame.total_cost_usd, 6)
            attributes["total_tokens_in"] = frame.total_tokens_in
            attributes["total_tokens_out"] = frame.total_tokens_out
        # Propagate aggregates up if there's a parent.
        parent = self._stack()[-2] if len(self._stack()) >= 2 else None
        if parent is not None:
            parent.child_span_count += 1
            if frame.kind == "llm":
                parent.total_cost_usd += float(frame.attributes.get("cost_usd", 0.0))
                parent.total_tokens_in += int(frame.attributes.get("tokens_in", 0))
                parent.total_tokens_out += int(frame.attributes.get("tokens_out", 0))
            elif frame.kind == "run":
                parent.total_cost_usd += float(attributes.get("total_cost_usd", 0.0))
                parent.total_tokens_in += int(attributes.get("total_tokens_in", 0))
                parent.total_tokens_out += int(attributes.get("total_tokens_out", 0))
            if status == "error":
                parent.had_error = True
        # Pop and write.
        self._stack().pop()
        span = Span(
            span_id=frame.span_id,
            run_id=frame.run_id,
            parent_id=frame.parent_id,
            kind=frame.kind,
            name=frame.name,
            started_at=frame.started_at,
            ended_at=ended_at,
            duration_ms=duration_ms,
            status=status,
            error=error,
            attributes=attributes,
        )
        self._write(span)

    def _write(self, span: Span) -> None:
        line = json.dumps(span.model_dump(), ensure_ascii=False) + "\n"
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line)

    # ---- Public API ----

    @contextmanager
    def run(self, name: str, attributes: Optional[dict[str, Any]] = None) -> Iterator[_SpanFrame]:
        """Open a logical run. Wrap an agent loop, a chain, or a request handler."""
        frame = self._open("run", name, attributes)
        try:
            yield frame
            status: Literal["ok", "error"] = "error" if frame.had_error else "ok"
            self._close(frame, status, None)
        except Exception as e:
            self._close(frame, "error", f"{type(e).__name__}: {str(e).splitlines()[0]}")
            raise

    def wrap_llm_call(self, **litellm_kwargs: Any) -> Any:
        """Call litellm.completion(**kwargs), capturing a span. Auto-opens an 'anonymous' run if needed."""
        implicit_run: Optional[_SpanFrame] = None
        if self._current() is None:
            implicit_run = self._open("run", "anonymous", None)

        messages = litellm_kwargs.get("messages") or []
        model = litellm_kwargs.get("model", MODEL)
        prompt_text = _messages_to_text(messages)

        attrs: dict[str, Any] = {
            "model": model,
            "prompt_hash": _hash_prompt(messages),
            "prompt_chars": len(prompt_text),
        }
        if not self.redact:
            attrs["prompt_preview"] = _preview(prompt_text)

        frame = self._open("llm", "completion", attrs)
        try:
            response = completion(**litellm_kwargs)
            # Enrich attributes from response.
            response_text = _response_text(response)
            frame.attributes["response_chars"] = len(response_text)
            if not self.redact:
                frame.attributes["response_preview"] = _preview(response_text)
            usage = getattr(response, "usage", None)
            if usage is not None:
                frame.attributes["tokens_in"] = int(getattr(usage, "prompt_tokens", 0) or 0)
                frame.attributes["tokens_out"] = int(getattr(usage, "completion_tokens", 0) or 0)
            else:
                frame.attributes["tokens_in"] = 0
                frame.attributes["tokens_out"] = 0
            frame.attributes["cost_usd"] = _safe_cost(response)
            try:
                frame.attributes["finish_reason"] = response.choices[0].finish_reason
            except Exception:
                frame.attributes["finish_reason"] = None
            self._close(frame, "ok", None)
            return response
        except Exception as e:
            err = f"{type(e).__name__}: {str(e).splitlines()[0]}"
            self._close(frame, "error", err)
            raise
        finally:
            if implicit_run is not None:
                status_: Literal["ok", "error"] = "error" if implicit_run.had_error else "ok"
                self._close(implicit_run, status_, None)

    # ---- File-side helpers ----

    def flush(self) -> bool:
        """Delete the trace file. Returns True if a file was deleted."""
        if self.path.exists():
            self.path.unlink()
            return True
        return False

    @staticmethod
    def read_spans(path: Path) -> list[Span]:
        """Load every span from a JSONL file. Skips malformed lines with a warning."""
        if not path.exists():
            return []
        out: list[Span] = []
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(Span.model_validate_json(line))
                except Exception as e:
                    print(f"WARNING: skipping malformed line {i}: {e}", file=sys.stderr)
        return out


# ============================================================================
# Demo workload
# ============================================================================

DEMO_SYSTEM_SHORT = "You are a terse assistant. Answer in one sentence."
DEMO_SYSTEM_LONG = (
    "You are a meticulous research assistant. " + ("Background context. " * 200)
)


def _demo_workload(tracer: Tracer) -> list[str]:
    """Generate three runs illustrating happy-path, failure, and cost-regression scenarios.

    Returns the run_ids of the three runs, in order.
    """
    run_ids: list[str] = []

    # --- Run 1: research_agent (happy path, 3 calls) ---
    with tracer.run("research_agent") as frame:
        run_ids.append(frame.run_id)
        for question in [
            "Name one famous Stoic philosopher.",
            "What is the capital of Brazil?",
            "What does TCP stand for?",
        ]:
            tracer.wrap_llm_call(
                model=MODEL,
                messages=[
                    {"role": "system", "content": DEMO_SYSTEM_SHORT},
                    {"role": "user", "content": question},
                ],
            )

    # --- Run 2: failure_demo (2 ok, 1 error) ---
    with tracer.run("failure_demo") as frame:
        run_ids.append(frame.run_id)
        tracer.wrap_llm_call(
            model=MODEL,
            messages=[{"role": "user", "content": "Say hi."}],
        )
        try:
            tracer.wrap_llm_call(
                model="invalid/does-not-exist-zzz",
                messages=[{"role": "user", "content": "This will fail."}],
            )
        except Exception:
            pass  # Swallow so the run completes; the span captured the error.
        tracer.wrap_llm_call(
            model=MODEL,
            messages=[{"role": "user", "content": "Say bye."}],
        )

    # --- Run 3: cost_regression (2 calls, second uses 4KB system prompt) ---
    with tracer.run("cost_regression") as frame:
        run_ids.append(frame.run_id)
        tracer.wrap_llm_call(
            model=MODEL,
            messages=[
                {"role": "system", "content": DEMO_SYSTEM_SHORT},
                {"role": "user", "content": "Define entropy briefly."},
            ],
        )
        tracer.wrap_llm_call(
            model=MODEL,
            messages=[
                {"role": "system", "content": DEMO_SYSTEM_LONG},
                {"role": "user", "content": "Define entropy briefly."},
            ],
        )

    return run_ids


# ============================================================================
# Aggregator
# ============================================================================


def _group_by_run(spans: list[Span]) -> dict[str, list[Span]]:
    """Group spans by run_id, preserving order of first appearance."""
    groups: dict[str, list[Span]] = {}
    for s in spans:
        groups.setdefault(s.run_id, []).append(s)
    return groups


def _root_of(group: list[Span]) -> Optional[Span]:
    for s in group:
        if s.parent_id is None and s.kind == "run":
            return s
    return None


def _percentile(values: list[float], p: float) -> float:
    """Linear-interp percentile. Returns 0.0 for empty input."""
    if not values:
        return 0.0
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * (p / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def _per_model_stats(spans: list[Span]) -> list[dict[str, Any]]:
    """For each model seen in llm-kind spans, compute count + cost + p50/p95/p99 latency."""
    by_model: dict[str, list[Span]] = {}
    for s in spans:
        if s.kind != "llm":
            continue
        model = str(s.attributes.get("model", "unknown"))
        by_model.setdefault(model, []).append(s)
    out: list[dict[str, Any]] = []
    for model, group in sorted(by_model.items()):
        latencies = [float(s.duration_ms) for s in group]
        out.append({
            "model": model,
            "calls": len(group),
            "cost_usd": round(sum(float(s.attributes.get("cost_usd", 0.0)) for s in group), 6),
            "p50_ms": int(_percentile(latencies, 50)),
            "p95_ms": int(_percentile(latencies, 95)),
            "p99_ms": int(_percentile(latencies, 99)),
        })
    return out


def _top_runs_by_cost(spans: list[Span], n: int = 5) -> list[Span]:
    """Top N root run-spans by total_cost_usd."""
    roots = [s for s in spans if s.kind == "run" and s.parent_id is None]
    return sorted(roots, key=lambda s: float(s.attributes.get("total_cost_usd", 0.0)), reverse=True)[:n]


# ============================================================================
# Print helpers (plain ASCII)
# ============================================================================


def _short(run_id: str) -> str:
    return f"{run_id[:8]}..{run_id[-2:]}"


def _fmt_cost(c: float) -> str:
    return f"${c:.4f}"


def _print_list(spans: list[Span], limit: int) -> None:
    groups = _group_by_run(spans)
    rows: list[tuple[float, Span]] = []
    for rid, group in groups.items():
        root = _root_of(group)
        if root is None:
            continue
        rows.append((root.started_at, root))
    rows.sort(key=lambda r: r[0], reverse=True)
    rows = rows[:limit]

    if not rows:
        print("(no runs)")
        return

    header = f"{'run_id':<14} {'name':<20} {'spans':>5} {'cost':>10} {'latency':>10} {'status':>7}"
    print(header)
    print("-" * len(header))
    for _, root in rows:
        spans_count = int(root.attributes.get("child_span_count", 0)) + 1
        cost = float(root.attributes.get("total_cost_usd", 0.0))
        print(
            f"{_short(root.run_id):<14} "
            f"{root.name[:20]:<20} "
            f"{spans_count:>5} "
            f"{_fmt_cost(cost):>10} "
            f"{root.duration_ms:>7} ms "
            f"{root.status:>7}"
        )


def _print_show(spans: list[Span], run_id_prefix: str) -> int:
    matches = [s for s in spans if s.run_id.startswith(run_id_prefix)]
    if not matches:
        print(f"No run found matching '{run_id_prefix}'.")
        return 1
    run_ids = {s.run_id for s in matches}
    if len(run_ids) > 1:
        print(f"Prefix '{run_id_prefix}' matches multiple runs:")
        for rid in sorted(run_ids):
            print(f"  {rid}")
        return 1
    run_id = run_ids.pop()
    group = [s for s in spans if s.run_id == run_id]
    root = _root_of(group)
    if root is None:
        print(f"Run {run_id} has no root span; data may be truncated.")
        return 1
    cost = float(root.attributes.get("total_cost_usd", 0.0))
    spans_count = int(root.attributes.get("child_span_count", 0)) + 1
    print(
        f"{root.name}  [{spans_count} spans, {root.duration_ms} ms, "
        f"{_fmt_cost(cost)}, {root.status}]  run_id={run_id}"
    )

    by_parent: dict[Optional[str], list[Span]] = {}
    for s in group:
        by_parent.setdefault(s.parent_id, []).append(s)

    def _render(span: Span, depth: int) -> None:
        children = by_parent.get(span.span_id, [])
        for i, child in enumerate(children):
            is_last = (i == len(children) - 1)
            connector = "`-- " if is_last else "|-- "
            indent = "  " * depth
            extra = ""
            if child.kind == "llm":
                model = child.attributes.get("model", "")
                tokens_in = child.attributes.get("tokens_in", 0)
                tokens_out = child.attributes.get("tokens_out", 0)
                cost_c = float(child.attributes.get("cost_usd", 0.0))
                extra = f" {model}  {_fmt_cost(cost_c)}  [tokens {tokens_in}/{tokens_out}]"
            line = (
                f"{indent}{connector}{child.kind}  {child.name}  "
                f"{child.duration_ms} ms{extra}"
            )
            if child.status == "error":
                line += f"  ERROR: {child.error}"
            print(line)
            _render(child, depth + 1)

    _render(root, 0)
    return 0


def _print_stats(spans: list[Span]) -> None:
    runs = [s for s in spans if s.kind == "run" and s.parent_id is None]
    llm_calls = [s for s in spans if s.kind == "llm"]
    total_cost = sum(float(s.attributes.get("cost_usd", 0.0)) for s in llm_calls)
    total_in = sum(int(s.attributes.get("tokens_in", 0)) for s in llm_calls)
    total_out = sum(int(s.attributes.get("tokens_out", 0)) for s in llm_calls)
    err_runs = [s for s in runs if s.status == "error"]
    err_rate = (len(err_runs) / len(runs) * 100.0) if runs else 0.0

    print(f"Total runs:      {len(runs)}")
    print(f"Total LLM calls: {len(llm_calls)}")
    print(f"Total cost:      {_fmt_cost(total_cost)}")
    print(f"Total tokens:    {total_in} in / {total_out} out")
    print(f"Error runs:      {len(err_runs)} ({err_rate:.1f}% of runs)")
    print()

    per_model = _per_model_stats(spans)
    if per_model:
        print("Per-model:")
        header = f"  {'model':<35} {'calls':>5} {'cost':>10} {'p50':>7} {'p95':>7} {'p99':>7}"
        print(header)
        for row in per_model:
            print(
                f"  {row['model'][:35]:<35} "
                f"{row['calls']:>5} "
                f"{_fmt_cost(row['cost_usd']):>10} "
                f"{row['p50_ms']:>4} ms "
                f"{row['p95_ms']:>4} ms "
                f"{row['p99_ms']:>4} ms"
            )
        print()

    top = _top_runs_by_cost(spans, 5)
    if top:
        print("Top 5 runs by cost:")
        for s in top:
            cost = float(s.attributes.get("total_cost_usd", 0.0))
            print(f"  {_short(s.run_id):<14} {s.name[:25]:<25} {_fmt_cost(cost):>10}  ({s.duration_ms} ms, {s.status})")


def _tail_loop(path: Path, poll_interval: float = 0.25) -> None:
    """Follow the file. Prints one line per new span. Ctrl-C to exit."""
    if not path.exists():
        path.touch()
    with open(path, "r", encoding="utf-8") as f:
        f.seek(0, os.SEEK_END)
        try:
            while True:
                line = f.readline()
                if not line:
                    time.sleep(poll_interval)
                    continue
                line = line.strip()
                if not line:
                    continue
                try:
                    s = Span.model_validate_json(line)
                except Exception as e:
                    print(f"WARNING: malformed line: {e}", file=sys.stderr)
                    continue
                extra = ""
                if s.kind == "llm":
                    extra = (
                        f" cost={_fmt_cost(float(s.attributes.get('cost_usd', 0.0)))}"
                        f" model={s.attributes.get('model','?')}"
                    )
                err = f"  ERROR: {s.error}" if s.status == "error" else ""
                print(f"[{_short(s.run_id)}] {s.kind:<3} {s.name:<22} {s.duration_ms:>5} ms{extra}{err}")
        except KeyboardInterrupt:
            print()


# ============================================================================
# CLI
# ============================================================================


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="solution.py",
        description="Module 18 - LLM trace recorder + query CLI.",
    )
    p.add_argument("--trace-file", type=Path, default=None,
                   help=f"Path to the JSONL trace file (default: {DEFAULT_TRACE_DIR / DEFAULT_TRACE_FILE}).")
    p.add_argument("--redact", action="store_true",
                   help="Suppress prompt/response previews (only hash + char counts stored).")

    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--demo", action="store_true",
                      help="Run the demo workload (3 runs: research_agent, failure_demo, cost_regression).")
    mode.add_argument("--list", dest="list_mode", action="store_true",
                      help="List recent runs.")
    mode.add_argument("--show", metavar="RUN_ID",
                      help="Show the full nested span tree for one run (prefix-match OK).")
    mode.add_argument("--stats", action="store_true",
                      help="Print aggregate stats across all spans.")
    mode.add_argument("--tail", action="store_true",
                      help="Follow the trace file and print each new span as it lands.")
    mode.add_argument("--flush", action="store_true",
                      help="Delete the trace file and exit.")

    p.add_argument("--limit", type=int, default=10,
                   help="(--list only) Max number of runs to show. Default 10.")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    trace_path = args.trace_file or (DEFAULT_TRACE_DIR / DEFAULT_TRACE_FILE)

    if args.demo:
        tracer = Tracer(path=trace_path, redact=args.redact)
        run_ids = _demo_workload(tracer)
        print(f"Demo generated {len(run_ids)} runs:")
        for rid in run_ids:
            print(f"  {rid}")
        print(f"\nTraces written to: {trace_path}")
        print("Try:  python solution.py --list")
        return 0

    if args.flush:
        tracer = Tracer(path=trace_path)
        if tracer.flush():
            print(f"Deleted {trace_path}.")
        else:
            print(f"No trace file at {trace_path}.")
        return 0

    if args.tail:
        _tail_loop(trace_path)
        return 0

    spans = Tracer.read_spans(trace_path)
    if not spans and not args.show:
        print("No traces found. Run with --demo to generate sample data.")
        return 0

    if args.list_mode:
        _print_list(spans, args.limit)
        return 0
    if args.show:
        return _print_show(spans, args.show)
    if args.stats:
        _print_stats(spans)
        return 0

    return 1  # Unreachable - argparse enforces a mode.


if __name__ == "__main__":
    sys.exit(main())
