# Project: Test-Driven Code Generator ("Function Forge")

A CLI that takes a Python stub file (function signature + docstring + `def test_*` functions), generates the implementation via LLM, runs the tests in a subprocess, and iterates on failure up to N attempts.

## What you'll build

- A `parse_spec(path)` function that extracts the signature, docstring, imports, and test source from a `.py` stub
- An `extract_code(raw)` function that strips markdown fences and AST-validates the result
- A `generate_implementation(...)` step that prompts the model and returns code + usage
- A `run_tests(...)` step that writes a temp file and executes it via `subprocess.run` with a 10s timeout
- A `generate_with_retries(spec, max_attempts)` orchestrator that drives the iterate-on-failure loop
- A CLI with `spec`, `--max-attempts`, `--keep-temp`, and `--model` flags

The project demonstrates:

- **Workflow with revision sub-loop:** outer pipeline is deterministic (Module 13); retry loop is bounded (Module 12 critique pattern)
- **Code-output prompting:** format constraints, role framing, examples
- **Sandboxed execution:** subprocess + timeout + temp file (the threat-model discussion is in the module README)
- **Per-attempt observability:** `StepUsage` per attempt, totals at the end

## Prerequisites

- [Module 13 (Workflows & Chains)](../../13-workflows-chains/) and [Module 12 (Multi-Agent Systems)](../../12-multi-agent-systems/) — the deterministic outer pipeline and the bounded critique loop both come from there.
- Completed reading the [Module 14 README](../README.md) so the threat model of running model-generated code is fresh.
- Python 3.11+ with the project venv already installed from the repo root. No new dependencies beyond what Module 13 required.

## Setup

`.env` at the repo root supplies your API key. `LLM_MODEL` defaults to `anthropic/claude-sonnet-4-20250514` if unset; pass `--model` to override at runtime without touching `.env`. The script resolves `.env` relative to the source file, so you can run it from any cwd.

### Project layout

```text
project/
├── README.md          this file
├── solution.py        the full pipeline (~500 lines)
└── specs/
    ├── fizzbuzz.py
    ├── fibonacci.py
    └── parse_log_line.py
```

Read `solution.py` end-to-end before you run it. Every step is independently callable from a REPL.

## How it works

```text
spec.py → parse → generate → write+execute → pass? ──yes──→ done
                                  │            │
                                  │            no
                                  │            ↓
                                  └─── revise(error) ←──┐
                                       ↑                │
                                       └─ until max attempts ─┘
```

- **Parse** uses `ast.parse` on the stub file to pull out the target function's signature, docstring, and any top-level imports, plus the source of every `def test_*` function. No LLM call.
- **Generate** sends the spec to the model with the `GENERATOR_PROMPT` on the first attempt, or the `REVISER_PROMPT` plus the previous failure on later attempts. Returns the raw response and a `StepUsage` record.
- **Extract** strips markdown code fences (`python`, `py`, `json` tags all handled) from the response, then `ast.parse`s the result to confirm it is syntactically valid Python before it ever touches disk.
- **Execute** writes the implementation + the original tests + an inline test runner into a temp file, runs `python <tempfile>` via `subprocess.run` with a 10s timeout, and captures exit code, stdout, stderr, and wall time into a `TestRun`.
- **Iterate** loops generate → execute → check-pass, feeding the previous failure into the next prompt, until tests pass or `max_attempts` is reached. The loop has three stop conditions: tests passed, attempts exhausted, or extraction failed twice in a row (the model is producing un-parseable output and another attempt is unlikely to fix it).
- **Report** prints the final status, attempts used, totalled tokens and cost, total wall time, and the final code (or the last attempt's code on failure). The full `GenerationResult` is also returned so a caller can persist every attempt for later analysis.

The shape is workflow-first with a bounded sub-loop. The outer steps (parse, extract, execute, report) are deterministic. Only the generate step is a model call, and the revise loop around it is hard-capped by `max_attempts` — there is no chance of an unbounded retry, no nested decision the model gets to make about whether to continue. That bounded-ness is the whole reason this pattern is preferable to "wrap an agent around `python -c`" for this problem class.

## Build it step by step

1. **Define the Pydantic models** (`CodeSpec`, `Implementation`, `TestRun`, `StepUsage`, `Attempt`, `GenerationResult`). `CodeSpec` carries the parsed signature, docstring, imports list, and test source. `Attempt` bundles one (`Implementation`, `TestRun`, `StepUsage`) triple so the orchestrator can return an ordered list of attempts in `GenerationResult`.
2. **Write the two system prompts** (`GENERATOR_PROMPT`, `REVISER_PROMPT`). The generator gets a clean spec; the reviser gets the spec plus the prior implementation plus the captured stderr. Both prompts pin output format to a single fenced code block — no prose, no multi-block answers.
3. **Implement `_strip_code_fence` and `_usage_from_response` helpers.** Generalize fence-stripping to handle `python`, `py`, and `json` language tags (and no-tag fences) so the same helper works in Module 14 and the JSON-mode helpers from earlier modules. `_usage_from_response` builds a `StepUsage` from a litellm response.
4. **Implement `parse_spec(path)`.** Use `ast.parse` to walk the module, find the first non-`test_*` `FunctionDef`, and `ast.unparse` its signature and docstring. Collect every `Import` and `ImportFrom` node at module level. Collect every `def test_*` function's full source. Return a `CodeSpec`.
5. **Implement `extract_code(raw_response)`.** Strip the fence, then `ast.parse` the result. If parsing raises `SyntaxError`, raise a domain-specific `ExtractionError` with the offending source so the orchestrator can surface it as a failed attempt rather than crashing.
6. **Implement `generate_implementation(...)`.** On the first attempt, format the `GENERATOR_PROMPT` with the spec. On revise attempts, format the `REVISER_PROMPT` with the spec plus the prior `Implementation.code` plus the prior `TestRun.stderr`. Return `(Implementation, StepUsage)`.
7. **Implement `run_tests(implementation_code, spec, work_dir, timeout=10)`.** Concatenate the imports, the implementation, the test functions, and the inline test runner into one source string. Write to a uniquely-named temp file under `work_dir`. Call `subprocess.run([sys.executable, tempfile], capture_output=True, timeout=timeout)` and wrap the result in a `TestRun`. Catch `subprocess.TimeoutExpired` and surface it as `TestRun(passed=False, exit_code=-1, stderr="TIMEOUT")`.
8. **Implement the inline test runner.** This is the `if __name__ == "__main__":` block that gets concatenated into every temp file. It iterates module globals, picks every callable starting with `test_`, calls each in a try/except, prints `PASS:` or `FAIL: <name>: <exception>` per test, and `sys.exit(1)` if any test failed. Keep it pure stdlib — the temp file should not import anything beyond what the spec imports.
9. **Implement `generate_with_retries(spec, max_attempts, model, keep_temp)`.** The orchestrator. Loop from 1 to `max_attempts`. Each iteration: call `generate_implementation`, call `extract_code` (on `ExtractionError`, record the attempt and continue), call `run_tests`, append an `Attempt`. Stop on first pass. After the loop, return a `GenerationResult` with `stop_reason` set to `passed`, `max_attempts_exhausted`, or `extraction_failed`. Cleanup temp files unless `keep_temp` is true.
10. **Implement the print helpers** (`_print_header`, `_print_result`). The header prints the spec path, function name, test count, model, and max-attempts before the run starts. The result prints status, attempts used, total tokens, total cost, total wall time, and the final code block. Keep them pure-print — no logic, just formatting `GenerationResult`.
11. **Wire up the CLI with `argparse`.** Positional `spec` (required path). `--max-attempts INT` (default 4). `--keep-temp` (flag, default false). `--model NAME` (default `os.environ.get("LLM_MODEL", "anthropic/claude-sonnet-4-20250514")`). Parse args, call `parse_spec`, call `_print_header`, call `generate_with_retries`, call `_print_result`. Exit nonzero if the final result did not pass.

Each step is small and independently testable. Steps 4, 5, and 7 in particular should pass on their own before you wire up the orchestrator — `parse_spec` against a spec file, `extract_code` against a hand-written response string, `run_tests` against a known-good implementation. If those three are solid, the orchestrator is just plumbing.

## Run it

```bash
python solution.py specs/fizzbuzz.py
python solution.py specs/fibonacci.py --max-attempts 6
python solution.py specs/parse_log_line.py --keep-temp
```

Expected console output (exact values vary):

```text
=== Function Forge ===
Spec:           specs/fibonacci.py
Function:       fib
Tests:          3 tests
Model:          anthropic/claude-sonnet-4-20250514
Max attempts:   4

[Attempt 1] generating...
  -> extracted 14 lines
  -> running tests (timeout 10s)...
  -> FAIL (exit 1, 0.4s)
     FAIL: test_fib_zero: AssertionError:
  -> revising with error feedback...

[Attempt 2] generating...
  -> extracted 16 lines
  -> running tests (timeout 10s)...
  -> PASS (exit 0, 0.3s)

=== Result ===
Status:         passed (stop_reason=passed)
Attempts used:  2/4
Total tokens:   in=412 out=287
Total cost:     $0.003200
Total time:     5.1s

=== Final code ===
def fib(n: int) -> int:
    if n < 2:
        return n
    a, b = 0, 1
    for _ in range(n - 1):
        a, b = b, a + b
    return b
```

Use `--keep-temp` when an attempt fails in an unexpected way — the temp file is exactly what the subprocess ran, so you can `python /tmp/forge_xxx.py` yourself and inspect what blew up.

### Common pitfalls

- **Forgetting to AST-validate before executing.** `extract_code` runs `ast.parse` for a reason. If you skip it, the subprocess will crash on `SyntaxError` and you waste a generation attempt on a failure mode that is detectable in-process for zero cost.
- **Forwarding the full traceback into the reviser.** Stderr can be thousands of bytes when a test loops or a recursion explodes. Trim to the last ~2KB before feeding it back — the model only needs the assertion line and surrounding context, not the entire stack.
- **Letting the timeout be the only sandbox.** A 10-second timeout stops infinite loops; it does nothing about `os.system("rm -rf /")`. Treat the subprocess as a convenience for the happy path, not as a security boundary. The Docker extension is the real story.

## Extensions

Once the base pipeline works, these are the natural next experiments:

1. **Add a `--judge` flag** that runs a second LLM as a code-quality critic on the final implementation (Module 12 critique pattern). The judge takes the spec plus the final code and returns a structured rubric score. Useful for catching the "passes tests but is awful code" case.
2. **Replace plain `assert` with `pytest`.** What changes in the inline test runner? In the parser? The parser stays the same (you're still pulling `def test_*` source); the runner changes from a hand-rolled loop to `pytest.main([__file__])`. The temp file now needs `pytest` available — a constraint worth noticing.
3. **Add Docker-based execution behind a `--sandbox=docker` flag.** Same `run_tests` interface; different backend. Mount the temp file read-only, drop capabilities, set a memory limit, no network. This is the threat-model story from the module README made concrete.
4. **Add a `--retrieve` flag** that pulls 3 related functions from a local code corpus and includes them as context (Module 07 pattern). Helps on specs where the function should follow conventions from existing code.
5. **Stream the first attempt's generation** so the learner sees the model think (Module 05 pattern). Swap `litellm.completion` for `litellm.completion(stream=True)` on attempt 1 only; the revise attempts stay buffered because their output goes straight into the next prompt.

## Reference

Cross-links for context:

- [Module 14 README](../README.md) — code-generation prompting, sandboxed execution threat model, when to use this pattern vs. agentic code-gen.
- [Module 13 (Workflows & Chains)](../../13-workflows-chains/) — the deterministic outer pipeline shape and `StepUsage` accounting carry over directly.
- [Module 12 (Multi-Agent Systems)](../../12-multi-agent-systems/) — the critique-loop pattern is the model for the revise sub-loop, and `_strip_code_fence` is the same helper.

**Next:** Module 15 lifts this single-function loop into a multi-file code-generation agent — same iterate-on-feedback shape, but now the unit of work is a project directory rather than one function. The per-attempt `Attempt` records you produced here become the raw material for that module's planning step.
