"""
Function Forge — complete reference implementation.

A test-driven code generator that:
  1. parse_spec       — read a .py stub file: signature + docstring + def test_* functions
  2. generate         — LLM produces an implementation matching the signature
  3. extract_code     — strip markdown fences, AST-validate
  4. run_tests        — write temp file, run via subprocess + timeout, capture stderr
  5. iterate          — on failure, feed stderr back and revise; up to max_attempts

Demonstrates:
- Workflow with revision sub-loop (Module 13 + Module 12)
- Code-output prompting (format constraints, role framing, examples)
- Sandboxed execution via subprocess + timeout
- Per-attempt observability with StepUsage

Run:
    python solution.py specs/fizzbuzz.py
    python solution.py specs/fibonacci.py --max-attempts 6
    python solution.py specs/parse_log_line.py --keep-temp
    python solution.py specs/fizzbuzz.py --model anthropic/claude-haiku-4-5-20251001
"""

from __future__ import annotations

import argparse
import ast
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from dotenv import load_dotenv
from litellm import completion, completion_cost
from pydantic import BaseModel

load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

MODEL = os.getenv("LLM_MODEL", "anthropic/claude-sonnet-4-20250514")
DEFAULT_MAX_ATTEMPTS = 4
SUBPROCESS_TIMEOUT_S = 10
STDERR_CAP_CHARS = 2000


# ---------- Pydantic models ----------


class CodeSpec(BaseModel):
    name: str
    signature: str
    docstring: str
    source_path: str
    raw_source: str
    imports: list[str]
    test_source: str


class Implementation(BaseModel):
    attempt_number: int
    code: str
    raw_response: str


class TestRun(BaseModel):
    passed: bool
    exit_code: int           # -1 = timeout, -2 = extract/AST failure
    stdout: str
    stderr: str
    duration_ms: int


class StepUsage(BaseModel):
    step: str
    input_tokens: int
    output_tokens: int
    cost: float
    latency_ms: int


class Attempt(BaseModel):
    implementation: Implementation
    test_run: TestRun
    usage: StepUsage


class GenerationResult(BaseModel):
    spec_name: str
    attempts: list[Attempt]
    final_code: str | None = None
    success: bool
    stop_reason: str         # "passed" | "max_attempts" | "no_progress"
    total_input_tokens: int
    total_output_tokens: int
    total_cost: float
    total_latency_ms: int


# ---------- System prompts ----------

GENERATOR_PROMPT = """You are a Python code generator.

You are given a function signature, a docstring, and the test functions that will be used to verify your implementation. The test functions use plain `assert` statements.

Return ONLY the function implementation as Python code:
- Do not include explanations.
- Do not include markdown fences.
- Do not include the test functions.
- Match the signature exactly (same name, same parameters, same return type if specified).

Your implementation will be appended to a file that already contains the necessary imports and the test functions, then executed."""

REVISER_PROMPT = """You are a Python code generator revising a prior implementation.

Your previous code failed the tests. Below you will see:
- the function signature and docstring,
- the test source,
- your prior implementation,
- the stderr from the test runner (this is the error you must fix).

Return ONLY the revised function implementation as Python code:
- Do not include explanations.
- Do not include markdown fences.
- Do not include the test functions.
- Match the signature exactly.
- Fix the failure described in the stderr."""


# ---------- Helpers ----------


def _strip_code_fence(text: str) -> str:
    """Strip a ```<lang> ... ``` fence if the model wrapped its output.

    Handles `python`, `py`, `json`, or no language tag.
    """
    s = text.strip()
    if not s.startswith("```"):
        return s
    s = s[3:]
    # strip an optional leading language tag (alpha chars up to newline/space)
    i = 0
    while i < len(s) and s[i].isalpha():
        i += 1
    s = s[i:]
    s = s.lstrip("\r\n ")
    if s.endswith("```"):
        s = s[:-3]
    return s.strip()


def _usage_from_response(response, step: str, latency_ms: int) -> StepUsage:
    """Build a StepUsage record from a LiteLLM response."""
    usage = getattr(response, "usage", None)
    input_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
    output_tokens = getattr(usage, "completion_tokens", 0) if usage else 0
    try:
        cost = completion_cost(completion_response=response) or 0.0
    except Exception:
        cost = 0.0
    return StepUsage(
        step=step,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost=cost,
        latency_ms=latency_ms,
    )


# ---------- Parse spec ----------


def parse_spec(path: str | Path) -> CodeSpec:
    """Parse a Python stub file into a CodeSpec.

    The stub file must contain:
      - top-level imports (any)
      - exactly one function definition with an empty body (the function under test)
      - one or more def test_* functions using plain `assert`
    """
    path = Path(path).resolve()
    raw_source = path.read_text(encoding="utf-8")
    tree = ast.parse(raw_source)

    imports: list[str] = []
    target_func: ast.FunctionDef | None = None
    test_funcs: list[ast.FunctionDef] = []

    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            imports.append(ast.unparse(node))
        elif isinstance(node, ast.FunctionDef):
            if node.name.startswith("test_"):
                test_funcs.append(node)
            else:
                if target_func is not None:
                    raise ValueError(
                        f"Spec file must contain exactly one non-test function; found "
                        f"both '{target_func.name}' and '{node.name}'"
                    )
                target_func = node

    if target_func is None:
        raise ValueError("Spec file must contain one non-test function definition.")
    if not test_funcs:
        raise ValueError("Spec file must contain at least one def test_* function.")

    # Signature: the def line up to and including the colon.
    sig_lines = raw_source.splitlines()
    # ast.FunctionDef.lineno is 1-based.
    signature = sig_lines[target_func.lineno - 1].rstrip()
    # Handle multi-line signatures by extending until a line ending with ':'
    i = target_func.lineno
    while not signature.rstrip().endswith(":") and i < len(sig_lines):
        signature = signature + "\n" + sig_lines[i].rstrip()
        i += 1

    docstring = ast.get_docstring(target_func) or ""

    test_source = "\n\n".join(ast.unparse(tf) for tf in test_funcs)

    return CodeSpec(
        name=target_func.name,
        signature=signature,
        docstring=docstring,
        source_path=str(path),
        raw_source=raw_source,
        imports=imports,
        test_source=test_source,
    )


# ---------- Extract code ----------


def extract_code(raw_response: str) -> str:
    """Strip fences and validate that the result is syntactically valid Python.

    Raises ValueError if the result does not AST-parse.
    """
    candidate = _strip_code_fence(raw_response)
    try:
        ast.parse(candidate)
    except SyntaxError as e:
        raise ValueError(f"Extracted code is not valid Python: {e}") from e
    return candidate


# ---------- Inline test runner (appended to every temp file) ----------

INLINE_RUNNER = '''
if __name__ == "__main__":
    import sys
    tests = [name for name, val in dict(globals()).items()
             if name.startswith("test_") and callable(val)]
    failures = []
    for tname in tests:
        try:
            globals()[tname]()
        except Exception as e:
            failures.append((tname, type(e).__name__, str(e)))
    for tname, exc_name, msg in failures:
        print(f"FAIL: {tname}: {exc_name}: {msg}", file=sys.stderr)
    sys.exit(1 if failures else 0)
'''


# ---------- Generate ----------


def generate_implementation(
    spec: CodeSpec,
    attempt_number: int,
    prior_code: str | None,
    prior_error: str | None,
    model: str = MODEL,
) -> tuple[Implementation, StepUsage]:
    """Call the LLM to produce a function body for the given spec.

    First attempt (attempt_number == 1) uses GENERATOR_PROMPT.
    Subsequent attempts use REVISER_PROMPT and include the prior code + error.
    """
    is_first = attempt_number == 1 or prior_code is None

    if is_first:
        system_prompt = GENERATOR_PROMPT
        user_content = (
            f"Function signature:\n{spec.signature}\n\n"
            f"Docstring:\n{spec.docstring or '(none)'}\n\n"
            f"Test source:\n{spec.test_source}\n\n"
            "Return the implementation."
        )
    else:
        capped_error = (prior_error or "")[:STDERR_CAP_CHARS]
        system_prompt = REVISER_PROMPT
        user_content = (
            f"Function signature:\n{spec.signature}\n\n"
            f"Docstring:\n{spec.docstring or '(none)'}\n\n"
            f"Test source:\n{spec.test_source}\n\n"
            f"Prior implementation:\n{prior_code}\n\n"
            f"Test runner stderr:\n{capped_error}\n\n"
            "Return the revised implementation."
        )

    start = time.perf_counter()
    response = completion(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
    )
    latency_ms = int((time.perf_counter() - start) * 1000)

    raw_response = response.choices[0].message.content or ""
    usage = _usage_from_response(response, step="generate", latency_ms=latency_ms)

    try:
        code = extract_code(raw_response)
    except ValueError:
        # Let the orchestrator handle this — it will produce a TestRun with exit_code=-2.
        code = raw_response  # store the raw response so the user can see what came back

    implementation = Implementation(
        attempt_number=attempt_number,
        code=code,
        raw_response=raw_response,
    )
    return implementation, usage


# ---------- Run tests ----------


def run_tests(
    implementation_code: str,
    spec: CodeSpec,
    work_dir: Path,
    attempt_number: int,
    timeout: int = SUBPROCESS_TIMEOUT_S,
) -> TestRun:
    """Write the temp file and run it via subprocess. Return a TestRun."""
    # AST-validate the implementation up front.
    try:
        ast.parse(implementation_code)
    except SyntaxError as e:
        return TestRun(
            passed=False,
            exit_code=-2,
            stdout="",
            stderr=f"AST parse failure: {e}",
            duration_ms=0,
        )

    imports_block = "\n".join(spec.imports)
    contents = (
        f"{imports_block}\n\n"
        f"{implementation_code}\n\n"
        f"{spec.test_source}\n"
        f"{INLINE_RUNNER}\n"
    )

    temp_file = work_dir / f"attempt_{attempt_number}.py"
    temp_file.write_text(contents, encoding="utf-8")

    start = time.perf_counter()
    try:
        completed = subprocess.run(
            [sys.executable, str(temp_file)],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        duration_ms = int((time.perf_counter() - start) * 1000)
        return TestRun(
            passed=completed.returncode == 0,
            exit_code=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
            duration_ms=duration_ms,
        )
    except subprocess.TimeoutExpired as e:
        duration_ms = int((time.perf_counter() - start) * 1000)
        stdout_val = e.stdout
        if isinstance(stdout_val, (bytes, bytearray)):
            stdout_val = stdout_val.decode("utf-8", errors="replace")
        return TestRun(
            passed=False,
            exit_code=-1,
            stdout=stdout_val or "",
            stderr=f"TIMEOUT after {timeout}s",
            duration_ms=duration_ms,
        )


# ---------- Orchestrator ----------


def generate_with_retries(
    spec: CodeSpec,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    model: str = MODEL,
    keep_temp: bool = False,
    verbose: bool = True,
) -> GenerationResult:
    """Run the iterate-on-failure loop."""
    work_dir = Path(tempfile.mkdtemp(prefix="function_forge_"))
    if verbose:
        print(f"  (temp dir: {work_dir})")

    attempts: list[Attempt] = []
    prior_code: str | None = None
    prior_error: str | None = None
    stop_reason = "max_attempts"
    success = False
    final_code: str | None = None
    start = time.perf_counter()

    try:
        for n in range(1, max_attempts + 1):
            if verbose:
                print(f"\n[Attempt {n}] generating...")

            implementation, usage = generate_implementation(
                spec=spec,
                attempt_number=n,
                prior_code=prior_code,
                prior_error=prior_error,
                model=model,
            )

            if verbose:
                line_count = len(implementation.code.splitlines())
                print(f"  -> extracted {line_count} lines")
                print(f"  -> running tests (timeout {SUBPROCESS_TIMEOUT_S}s)...")

            test_run = run_tests(
                implementation_code=implementation.code,
                spec=spec,
                work_dir=work_dir,
                attempt_number=n,
            )

            if verbose:
                status = "PASS" if test_run.passed else "FAIL"
                print(f"  -> {status} (exit {test_run.exit_code}, {test_run.duration_ms/1000:.1f}s)")
                if not test_run.passed:
                    for line in test_run.stderr.splitlines():
                        if line.startswith("FAIL:") or line.startswith("TIMEOUT") or line.startswith("AST parse"):
                            print(f"     {line}")
                            break

            attempts.append(Attempt(
                implementation=implementation,
                test_run=test_run,
                usage=usage,
            ))

            if test_run.passed:
                success = True
                stop_reason = "passed"
                final_code = implementation.code
                break

            # No-progress check: same code three times in a row.
            if len(attempts) >= 3:
                last_three = [a.implementation.code for a in attempts[-3:]]
                if last_three[0] == last_three[1] == last_three[2]:
                    stop_reason = "no_progress"
                    break

            # Prepare next iteration.
            prior_code = implementation.code
            prior_error = test_run.stderr

            if verbose and n < max_attempts:
                print(f"  -> revising with error feedback...")

        total_latency_ms = int((time.perf_counter() - start) * 1000)
        total_in = sum(a.usage.input_tokens for a in attempts)
        total_out = sum(a.usage.output_tokens for a in attempts)
        total_cost = round(sum(a.usage.cost for a in attempts), 6)

        return GenerationResult(
            spec_name=spec.name,
            attempts=attempts,
            final_code=final_code,
            success=success,
            stop_reason=stop_reason,
            total_input_tokens=total_in,
            total_output_tokens=total_out,
            total_cost=total_cost,
            total_latency_ms=total_latency_ms,
        )
    finally:
        if not keep_temp:
            shutil.rmtree(work_dir, ignore_errors=True)
        elif verbose:
            print(f"\n(--keep-temp: temp files preserved at {work_dir})")


# ---------- CLI / printing ----------


def _print_header(spec: CodeSpec, model: str, max_attempts: int) -> None:
    test_count = spec.test_source.count("def test_")
    print(f"=== Function Forge ===")
    print(f"Spec:           {spec.source_path}")
    print(f"Function:       {spec.name}")
    print(f"Tests:          {test_count} tests")
    print(f"Model:          {model}")
    print(f"Max attempts:   {max_attempts}")


def _print_result(result: GenerationResult) -> None:
    print("\n=== Result ===")
    status = "passed" if result.success else "FAILED"
    print(f"Status:         {status} (stop_reason={result.stop_reason})")
    print(f"Attempts used:  {len(result.attempts)}")
    print(f"Total tokens:   in={result.total_input_tokens} out={result.total_output_tokens}")
    print(f"Total cost:     ${result.total_cost:.6f}")
    print(f"Total time:     {result.total_latency_ms/1000:.1f}s")
    if result.final_code:
        print(f"\n=== Final code ===")
        print(result.final_code)


def main() -> None:
    parser = argparse.ArgumentParser(description="Function Forge — test-driven code generator (Module 14)")
    parser.add_argument("spec", help="Path to a Python spec file (signature + docstring + def test_* functions)")
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=DEFAULT_MAX_ATTEMPTS,
        help=f"Maximum generate+test rounds (default {DEFAULT_MAX_ATTEMPTS})",
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Preserve the temp directory so you can inspect generated files",
    )
    parser.add_argument("--model", default=MODEL, help=f"Model override (default: {MODEL})")
    args = parser.parse_args()

    try:
        spec = parse_spec(args.spec)
    except (FileNotFoundError, ValueError) as e:
        parser.error(f"could not parse spec: {e}")

    _print_header(spec, args.model, args.max_attempts)

    result = generate_with_retries(
        spec=spec,
        max_attempts=args.max_attempts,
        model=args.model,
        keep_temp=args.keep_temp,
    )

    _print_result(result)

    sys.exit(0 if result.success else 1)


if __name__ == "__main__":
    main()
