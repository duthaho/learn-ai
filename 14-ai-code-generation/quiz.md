# Module 14 Quiz: AI Code Generation

Self-assessment questions for Module 14. Test your understanding before revealing each answer.

---

### Q1: What makes AI code generation different from AI text generation?

<details>
<summary>Answer</summary>

Text generation succeeds if it reads well; code generation succeeds if it runs and passes tests. That executable ground truth changes everything: it unlocks iterate-on-failure loops, makes evaluation mechanical, and narrows the output format. Where prose generation needs human judgement to grade, code generation gets a binary pass/fail from the runtime.

</details>

---

### Q2: Sketch the canonical code-generation workflow.

<details>
<summary>Answer</summary>

Parse the spec, generate the code, execute it against tests, and if it fails, feed the error back to the model and retry — bounded by a max-attempts limit. The outer pipeline is a deterministic workflow (Module 13); the inner retry loop is a bounded revision loop (Module 12's critique pattern, but the critic is `pytest`).

</details>

---

### Q3: Why won't `response_format={"type": "json_object"}` help with code generation?

<details>
<summary>Answer</summary>

Forcing JSON output gives you `{"code": "..."}` — the value is still a string of code you have to extract before you can run it. You pay for the wrapper without gaining any structure that helps you. For code, plain text plus a fence-parser is simpler and produces the same shape of post-processing work.

</details>

---

### Q4: Name two failure modes for extracting code from a raw LLM response.

<details>
<summary>Answer</summary>

1. The model wraps the code in a markdown fence (` ```python ` or plain ` ``` `), so a naive `json.loads` or `eval` will fail.
2. The model returns multiple code blocks (e.g., the function plus a usage example), and picking the wrong one gives broken or non-callable code.

AST validation before execution catches most extraction bugs early.

</details>

---

### Q5: Why is `subprocess` preferred over in-process `exec()` for running generated code? Even with `subprocess`, what's the threat model?

<details>
<summary>Answer</summary>

`exec()` runs in your process — an infinite loop or `sys.exit` from the generated code kills your script, not just the generated code. `subprocess` runs in a child process you can kill via `timeout`, and exceptions stay over there. The threat model: `subprocess` is safe for trusted personal use (you wrote the spec) but is NOT a hardened sandbox. For untrusted input, use Docker with `--network=none`, Pyodide, or a hosted sandbox like e2b.dev.

</details>

---

### Q6: What feedback should you send back to the LLM after a failed test run — the full traceback, or just the failing assertion?

<details>
<summary>Answer</summary>

Send the full stderr, but cap it (about 2000 chars) to keep context budget under control. The prior implementation should also go back in the conversation so the model knows what it tried. Sending only the assertion line is too thin — the traceback often contains the line number and stack context that lets the model localize the bug.

</details>

---

### Q7: Name three things test-pass rate does not tell you about generated code quality.

<details>
<summary>Answer</summary>

1. **Readability** — variable names, function decomposition.
2. **Complexity** — cyclomatic complexity, length, structure.
3. **Side effects** — writes to disk, debug prints, global mutations.

Tests can pass on a hard-coded solution that overfits to the test cases, or on a 50-line monstrosity that solves the problem the wrong way. Test-pass is necessary but not sufficient.

</details>

---

### Q8: How does the single-function workflow scale up to a Claude Code / Cursor architecture?

<details>
<summary>Answer</summary>

The pipeline grows tools and context. RAG over the codebase becomes a workflow step (Module 07). Edit-by-diff replaces full-file output. File-system operations (read, write, grep) become tool calls (Module 06). The whole thing gets wrapped in an agent loop (Module 11) so the model can decide which file to look at next. The shape stays the same — workflow with a revision sub-loop — but the surface area expands.

</details>
