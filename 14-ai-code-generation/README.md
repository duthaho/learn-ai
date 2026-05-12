# Module 14: AI Code Generation

**What you'll learn:**
- Why generating code is different from generating prose — and how executability changes everything
- The canonical code-generation workflow: parse spec → generate → execute → feed errors back → iterate
- Code-output prompting: format constraints, why `response_format=json_object` doesn't help, role framing
- Extracting code from raw LLM responses (markdown fences, multi-block responses, AST sanity checks)
- Executing generated code safely with `subprocess` + timeout + temp file
- The iterate-on-failure loop and how it converges (or doesn't)
- Evaluating generated code beyond test-pass-rate
- How single-function generation scales up to Claude Code / Cursor / Copilot

| Detail        | Value                                                                          |
|---------------|--------------------------------------------------------------------------------|
| Level         | Intermediate–Advanced                                                          |
| Time          | ~3.5 hours                                                                     |
| Prerequisites | Module 08 (Structured Output), Module 12 (Multi-Agent Systems — critique loop), Module 13 (Workflows & Chains) |

---

## Table of Contents

1. [Why AI Code Generation Is Different](#1-why-ai-code-generation-is-different)
2. [The Code Generation Workflow](#2-the-code-generation-workflow)
3. [Prompting for Code Output](#3-prompting-for-code-output)
4. [Extracting Code from Model Output](#4-extracting-code-from-model-output)
5. [Executing Generated Code Safely](#5-executing-generated-code-safely)
6. [The Iterate-on-Failure Loop](#6-the-iterate-on-failure-loop)
7. [Evaluating Generated Code](#7-evaluating-generated-code)
8. [AI Code Generation in the Stack & Real-World Tools](#8-ai-code-generation-in-the-stack--real-world-tools)

---

## 1. Why AI Code Generation Is Different

Every LLM-powered feature you have built in this curriculum so far has had the same final-step shape: the model produces text, and a human (or a downstream system) decides whether that text is good. A summary is good if it reads accurately. A classification is good if the label matches what a labeler would have chosen. A drafted reply is good if a human reviewer would have sent something similar. The quality bar is human judgment, applied after the fact.

Code generation breaks that pattern. When the model's output is code, "good" is no longer a matter of taste. The code either runs or it doesn't. The tests either pass or they don't. The function either returns the right value for the input or it returns the wrong one. The bar shifts from "reads well to a human" to "passes a deterministic check that a computer can run in milliseconds." That shift sounds small. It is not. It is the single most important fact about this module.

### Executability as ground truth

The phrase to internalise is *ground truth*. Most LLM outputs have no ground truth: there is no objectively correct summary of an article, no single right way to phrase an apology, no canonical translation between two languages. Two equally good summaries can disagree on what to include. Two reviewers can rank them in different orders. The quality signal is fuzzy, slow, and human-mediated.

Code has ground truth in a stronger sense than almost any other artifact an LLM produces. Given a function `def fib(n: int) -> int:` and the test `assert fib(10) == 55`, there is exactly one correct answer about whether a generated implementation passes: run it, see if the assertion fires. The check takes a few hundred milliseconds. It does not depend on a reviewer's mood, a market's preferences, or a culture's sense of style. It depends only on whether `1 == 1` evaluates to true.

This is the property that unlocks everything else in the module. When you can mechanically verify whether an output is correct, you can build a loop that *uses that signal*. Generate. Check. If wrong, tell the model what was wrong. Generate again. Repeat. This is the iterate-on-failure loop, and it is the killer feature of code generation over prose generation. You cannot do this with a summary, because there is no `pytest` for summaries. You can do it with code because the runtime is the critic.

### The output format is narrower

The second shift is in the shape of the output itself. When you ask a model to write a summary, the space of acceptable outputs is enormous: any natural-language paragraph that conveys the right ideas in roughly the right order is fine. Length can vary, phrasing can vary, structure can vary, and the result is still a good summary. The acceptable set is a vast, fuzzy region of possible strings.

When you ask a model to write a function, the acceptable set shrinks dramatically. The output must be syntactically valid Python. It must match a specific signature. It must produce a specific typed result for specific inputs. Most strings the model could produce — even most *Python-looking* strings — are not acceptable. The acceptable set is a narrow target inside an enormous space of plausible-looking failures.

This narrowness is both a constraint and a gift. The constraint: the model has to land precisely, not just approximately. Off-by-one errors, missing type coercions, slightly-wrong recursion bases — any of these turn a "nearly correct" output into a failure. The gift: because the target is so narrow, the prompt can be tight. You can tell the model exactly what shape its output must take ("return only the function body, no explanation, no markdown fences, matching this signature"), and the model can comply. Compare this to prompting for prose, where any specification of the output's shape leaves a hundred ways to satisfy it.

### Why this module sits after Module 13

This module is placed deliberately after [Module 13 (Workflows & Chains)](../13-workflows-chains/) because *code generation is a workflow*. Look at the pipeline: parse the spec, generate the code, execute the code, decide whether it passed, and if it didn't, feed the error back and try again. That is a sequential chain — exactly the shape Module 13 covers — with a bounded revision sub-loop bolted onto the execute step. The outer pipeline is deterministic; the inner revision loop is bounded; both shapes are familiar.

Code generation is not its own paradigm. It is workflows applied to a particular kind of output, where the output happens to be checkable mechanically. The novelty is the checker — `pytest`, `subprocess`, `ast.parse` — not the orchestration. Once you internalise that, the architecture in `solution.py` will look like a Module 13 workflow with an extra retry knob, which is exactly what it is.

The module also leans on the critique loop from [Module 12 (Multi-Agent Systems)](../12-multi-agent-systems/). In Module 12, the loop was *generator agent → critic agent → revise → re-critique*, where the critic was an LLM evaluating the generator's output against quality criteria. In this module, the same shape appears, but the critic is no longer an LLM. The critic is the Python runtime. The revision feedback is the traceback. Everything else — the iteration, the stop conditions, the cost-per-attempt bookkeeping — is the same.

### Vibes-based versus binary

The clearest way to feel the difference is to contrast the two evaluation regimes side by side.

In prose generation, evaluation is *vibes-based*. You read the output. You form a holistic judgment. You decide whether the result is publishable, sendable, or shippable based on intuitions about voice, accuracy, and tone that you cannot easily articulate to a computer. Quality is graded on a smooth continuum from "terrible" to "excellent," with most outputs falling somewhere in the middle, and the threshold for acceptance is itself a judgment call. Two graders working the same output can disagree on whether it crosses the bar, and neither is wrong.

In code generation, evaluation is *binary on the first pass*. The tests pass or they don't. If they pass, the code is at least correct for the test cases provided. If they don't, the code is broken in a specific, locatable way: a particular assertion fired with a particular value. There is no "kind of works" middle ground at this stage. The output is either green or red.

That binary signal is what you trade for in this module. You spend prompt-engineering effort to convince the model to produce a narrow, executable artifact. In exchange, you get evaluation that is fast, cheap, and unambiguous — and the iterate-on-failure loop, which only exists because the evaluation is mechanical. The vibes-based stuff comes back in Section 7, when we ask whether code that *passes the tests* is also *good code* (readable, secure, maintainable). But the foundation is the binary check, and the rest of the module is built on it.

---

## 2. The Code Generation Workflow

The pipeline shape for code generation is the same every time. There are five distinct stages, in a fixed order, with one of them (execute) carrying an attached revision sub-loop that may re-enter the generation step. Once you see the shape, you see it in every code-generation product on the market, from one-function demos to Claude Code itself. The number of stages does not change. The breadth of what each stage handles does.

### The canonical pipeline

```text
┌─────────────────────────────────────────────────────────────────┐
│                CODE GENERATION WORKFLOW                          │
│                                                                 │
│   spec                                                          │
│     │                                                           │
│     ▼                                                           │
│  ┌──────────┐                                                   │
│  │  Parse   │  signature + docstring + tests + imports         │
│  └──────────┘                                                   │
│       │                                                         │
│       ▼                                                         │
│  ┌──────────┐  ◄────────── prior code + error feedback ───┐     │
│  │ Generate │                                              │     │
│  └──────────┘                                              │     │
│       │                                                    │     │
│       ▼ raw LLM response                                   │     │
│  ┌──────────┐                                              │     │
│  │ Extract  │  strip fences, AST-validate                  │     │
│  └──────────┘                                              │     │
│       │                                                    │     │
│       ▼ valid Python source                                │     │
│  ┌──────────┐                                              │     │
│  │ Execute  │  subprocess + temp file + timeout            │     │
│  └──────────┘                                              │     │
│       │                                                    │     │
│       ▼ exit code, stdout, stderr                          │     │
│       │                                                    │     │
│   pass? ──── no ──────────► revise(stderr) ───────────────┘     │
│       │                     ▲                                    │
│       │ yes                 │ (until max_attempts)               │
│       ▼                                                          │
│    final code                                                   │
└─────────────────────────────────────────────────────────────────┘
```

The five boxes are the workflow. The dashed line from "no" back to "Generate" is the revision sub-loop. Read the diagram as a Module 13 sequential chain (parse → generate → extract → execute) with one branch on the test outcome: success exits the loop, failure routes back to a "revise" call that feeds the error and the prior implementation into the generator, and the loop runs again until the tests pass or the attempt budget runs out.

### Each stage in one sentence

**Parse.** Read the spec file and pull out the function-under-test's signature and docstring, the source of each `def test_*` function, and any top-level imports — everything the generator and the executor will need. Parsing is pure Python (no LLM call), uses the `ast` module, and produces a typed `CodeSpec` object that the rest of the pipeline reads.

**Generate.** Make an LLM call. On the first attempt, the prompt contains the signature, docstring, and visible test source. On subsequent attempts, the prompt additionally contains the prior code and the trimmed stderr from the failed run. The output is a raw string of model response — possibly wrapped in markdown fences, possibly with surrounding commentary.

**Extract.** Strip any markdown fences around the code (Section 4 covers this in detail) and run `ast.parse()` on what's left to confirm it is syntactically valid Python. An AST-parse failure short-circuits the rest of the pipeline: it counts as a failed attempt with a synthetic error message ("not valid Python"), and the workflow loops back to the generator.

**Execute.** Write the extracted code, plus the spec's imports, plus the test functions, plus a small inline runner, to a temporary file. Invoke `python <tempfile>` via `subprocess.run` with a timeout. Capture stdout and stderr. Read the exit code: 0 means all tests passed, nonzero means at least one test failed (or the process timed out).

**Iterate.** If the exit code is zero, the loop ends with `success=True`. If not, take the captured stderr, trim it to a context-safe size, and feed it back to the generator along with the prior code and a "revise" instruction. Repeat until success, until `max_attempts` is reached, or until the generator returns the same code multiple times in a row (a sign that it is stuck).

### Deterministic outer, bounded inner

Notice the workflow's split personality. The outer pipeline — parse, generate, extract, execute — is deterministic in Module 13's sense: the order is fixed, each step has a typed input and output, the orchestrator wires them together in code that you wrote before any input arrived. There is nothing the LLM "decides" about the pipeline's structure. It is a chain.

The inner revision loop is the only place where dynamic behavior shows up, and even there the dynamism is bounded by design: a small integer (`max_attempts`, default 4) plus a "stuck detector" that fires when the model produces identical code three times in a row. The loop cannot run forever, cannot grow new steps it didn't have at design time, and cannot exit through any path the orchestrator didn't anticipate. This is exactly the shape Module 12's critique loop took, where a critic and a generator alternated until the critic was satisfied or the iteration budget was exhausted. The same control-flow guarantees apply here: a bounded loop is operationally well-behaved in a way an unbounded agent loop is not.

The mental model worth carrying forward: this module is Module 13's workflow shape with Module 12's revision shape nested inside one of its steps. The "novelty" — the part that wasn't covered by either prior module — is the *checker*. Where Module 12's critic was an LLM with a rubric, this module's critic is `python <tempfile>`. Where Module 13's steps each made one LLM call and returned a typed result, the execute step here makes *no* LLM call — it runs a subprocess and reports what happened. The new building block is mechanical evaluation. Everything else is composition of patterns you already know.

### Cost: each retry is a fresh generation

The revision sub-loop is the most expensive part of the pipeline. Each attempt is a full LLM call, with the prompt growing as failed attempts accumulate (the prior code goes into the messages, the prior error goes into the messages). On the third attempt, the prompt is roughly 2–3x the size of the first. On the fifth attempt, it can be 4–5x. Token cost grows roughly linearly with attempt count, and latency grows by the same factor.

That growth is why the loop must be bounded. An infinite revision loop — one that keeps retrying until the tests pass, no matter how many attempts — has unbounded cost in the worst case. A spec that the model genuinely cannot solve (because it is underspecified, contradictory, or beyond the model's capability) will burn tokens forever. Setting `max_attempts` to a small number — 4 is the project's default — guarantees the loop terminates in a known budget. The trade-off is that some problems that would have solved with five attempts now fail with four. The right `max_attempts` value depends on your cost tolerance and your typical convergence behavior; Section 6 covers how to think about it.

Cross-link to [Module 13's parallel-vs-cost discussion](../13-workflows-chains/#worked-latency-vs-cost-example) is worth making here. In that module, the lesson was *parallel saves latency, not tokens*. In this module, the analogous lesson is *retries save correctness, not tokens*. Every retry costs you a full generation. Budget accordingly.

---

## 3. Prompting for Code Output

Prompting for code has its own genre conventions, distinct from prompting for prose. The conventions are not arbitrary — they exist because the output is checkable and narrow, which changes which prompt elements actually move the quality needle. This section is a tour of the patterns that work, the patterns that look like they should work but don't, and the reasoning behind each.

### Explicit format hints

The single most useful sentence in a code-generation prompt is one that tells the model exactly what to produce. "Return only the function body, no explanation, no markdown fences" is the canonical form. Each piece does work:

- **"Return only the function body"** tells the model not to repeat the signature, not to include a `def` line above the body, not to add an `if __name__ == "__main__":` block at the bottom. The model knows the signature already (you sent it). The orchestrator does not need the signature back; it has its own copy. Asking for "only the body" is asking the model to fill in the blank, not redraw the whole function.
- **"No explanation"** is the request that does the most work in practice. By default, instruction-tuned models *want* to explain their work. They want to write a paragraph before the code summarising what they're about to do, then a paragraph after the code explaining how it works. That commentary makes the response harder to extract and adds tokens to every retry. Explicitly suppressing it usually works on the first try.
- **"No markdown fences"** asks the model to skip the ` ```python` wrappers. This is less reliable than the first two — many models have been trained so heavily on fenced code in their assistant responses that they will include fences anyway, especially on multi-line outputs. You should suppress them in the prompt and still write an extractor that handles them, because the model will produce them sometimes regardless. Section 4 covers the extraction.

The combined effect of these hints is a response that is *almost* always just the code you wanted, with occasional regressions to fenced or explained output. The extractor cleans up the occasional regressions; the prompt does most of the work.

### Why `response_format={"type": "json_object"}` doesn't help

A natural-feeling instinct, especially after Module 08, is to reach for structured output. "Force the model to return a JSON object with a `code` field" sounds like it should solve the format problem cleanly: the model returns `{"code": "def fib(n): ..."}`, you grab the `code` field, you skip the fence-stripping entirely. No ambiguity, no extraction logic, just a clean string.

It doesn't work in practice. There are two reasons.

The first reason is that the value inside the JSON object is still a string of code. It needs to be parsed for syntactic validity, written to a file, and executed — exactly the same downstream pipeline as if it had come back as raw text. You haven't avoided extraction; you've added an extra layer of unwrapping. The structured-output benefit you got in Module 08 was that complex *shaped* data (objects with multiple fields, nested types, enumerated values) became typed Python objects without manual parsing. Code generation has no equivalent benefit — the "shape" is just "a string of Python source" — so the structured wrapper buys you nothing.

The second reason is that JSON encoding mangles code. Multi-line strings inside JSON require either explicit `\n` escapes or a parser that handles unescaped newlines, and neither path is comfortable. The model has to emit `\n` in the right places, which it sometimes does wrong (especially around `\\` for backslash escapes inside the code, which now have to be double-escaped through the JSON layer). What you save in extraction logic you pay back in escape-handling bugs, which are more annoying because they appear non-deterministically and only on inputs that contain certain characters.

You are also paying for the round-trip: the model now has to spend tokens producing the JSON wrapper (the keys, the quotes, the escapes), tokens you'd otherwise have spent on actual code. That cost is small per call, but it is pure overhead, and it adds up across thousands of generations.

The right tool for the job is a plain text response with a small extractor on the receiving end. The extractor (Section 4) handles the regular variations — fences, language tags, occasional commentary — and is about 20 lines of code. You write it once and reuse it for every code-generation surface in your system. The structured-output route is the kind of "obvious" optimisation that looks cleaner on the architecture diagram but costs you more in practice.

### Few-shot examples in the prompt

Few-shot prompting — showing the model a couple of input-output pairs before the real task — works exceptionally well for code generation, often better than for prose. The reason is again the narrowness of the output format: when the model can see exactly the shape of response you want for a similar problem, it produces a response in that shape for the real problem. A single full example often beats two paragraphs of prose instruction about the desired format.

A useful structure for a few-shot code prompt:

```text
[System] You are a Python code generator. Return only the function body...

[User] Here's an example.
       Signature: def square(n: int) -> int:
       Docstring: Return n squared.
       Tests:
         def test_square_basic():
             assert square(3) == 9
       Output:
       return n * n

[User] Now the real task.
       Signature: def cube(n: int) -> int:
       Docstring: Return n cubed.
       Tests:
         def test_cube_basic():
             assert cube(2) == 8
```

The example shows the model what counts as a correct response: a function body, no surrounding text, matching the signature, satisfying the test. The model then mirrors that shape for the real task. The cost is the example's tokens on every call (which is why you want to keep it short), but the quality lift on the first attempt is large enough that the bookkeeping pays for itself in retries-avoided.

This is a stronger version of the structured-output discipline from [Module 08](../08-structured-output/): instead of a JSON schema *telling* the model what shape to produce, a worked example *shows* it. Shown examples generalise better than abstract specifications when the output's shape is hard to describe in words but easy to recognise.

### Role framing

A small but real lever is the system message's role description. The instruction "you are writing a function that will be run as-is" anchors the model differently from "help the user with their Python question." The first framing positions the model as a producer of executable artifacts; the second positions it as an explainer of code. Models pick up the difference and shift their output style accordingly.

The framing matters most at the boundary cases. When the task is ambiguous — should the function handle this edge case, should it raise on bad input, should it use a stricter type — the framing influences which way the model leans. A model framed as "code that will be run as-is" tends to produce defensive, type-correct code with the assumption that something else will catch malformed inputs upstream. A model framed as "answer the user's question" tends to produce explanatory code with print statements, asserts, and helpful comments, which is exactly what you don't want when the output goes through a subprocess runner.

The right framing is not a magic incantation, but it is worth being deliberate about. Module 12 spent a section on role framing for multi-agent systems precisely because framing changes specialist behavior; the same effect operates here. Treat the system message as setting the model's *job*, not just its *expertise*, and the outputs become more consistent with what the rest of the pipeline expects.

### Give the model the test source

The single highest-leverage piece of context you can put in a code-generation prompt is the test source itself. Not a description of what the tests do, not a summary of expected behavior — the actual `def test_*` function bodies, verbatim.

Why this works so well: when the model can see exactly how its code will be exercised, it produces code that fits the exercise. If the test does `assert fib(0) == 0`, the model knows the base case is required. If the test does `assert parse_log_line("malformed") is None`, the model knows the function must return `None` on bad input rather than raise an exception. The test is the most precise possible specification of the function's behavior — more precise than any docstring, because the docstring is the contract in English and the test is the contract in Python.

This is a special case of a more general principle: *show the model the evaluator*. In domains where the evaluator can be exposed, exposing it dramatically improves first-attempt accuracy. In domains where the evaluator is humans (most prose tasks), there is no equivalent — you cannot show the model "the human reading this." But code generation has the evaluator right there in the spec, and you should always feed it to the model.

There is one mild risk: if the model overfits to the visible tests by hard-coding their inputs, the function passes the visible suite but would fail on hidden tests. Section 7 discusses this failure mode at length ("overfit to tests"). For the basic module project, you only have visible tests, so the risk is theoretical; in production code-generation systems with held-out test suites, it is a real evaluation problem to manage.

### A worked example: generalist vs code-focused

A side-by-side prompt comparison makes the difference concrete. Same task — implement `fizzbuzz` — same model, two prompts.

**Generalist prompt:**

```text
You are a helpful Python expert. Please help me write a function called fizzbuzz
that returns a list of strings for the numbers 1 through n, following the classic
rules: "fizz" for multiples of 3, "buzz" for multiples of 5, "fizzbuzz" for
multiples of both, and the number as a string otherwise.
```

**Code-focused prompt:**

```text
You are a Python code generator. You are given a function signature, docstring,
and test cases. Return ONLY the function implementation as Python code. Do not
include explanations, comments, or markdown fences. Match the signature exactly.

Signature: def fizzbuzz(n: int) -> list[str]:
Docstring: Return fizzbuzz output for 1..n inclusive.

Tests:
def test_fizzbuzz_basic():
    assert fizzbuzz(5) == ["1", "2", "fizz", "4", "buzz"]

def test_fizzbuzz_fifteen():
    assert fizzbuzz(15)[-1] == "fizzbuzz"
```

The generalist prompt typically produces something like this (the response shown verbatim, including the fences the model added):

```text
Here's a classic implementation of fizzbuzz:

    def fizzbuzz(n):
        result = []
        for i in range(1, n + 1):
            if i % 15 == 0:
                result.append("FizzBuzz")
            elif i % 3 == 0:
                result.append("Fizz")
            elif i % 5 == 0:
                result.append("Buzz")
            else:
                result.append(i)
        return result

This loops through the numbers from 1 to n and checks each one...
```

Note the problems: capitalised "FizzBuzz" instead of lowercase "fizzbuzz", integers instead of strings in the `else` branch, surrounding prose and a fence. Both tests fail. The extractor has to strip the fence and the prose. The model has produced a *recognisable* fizzbuzz but not the *correct* one for the given tests.

The code-focused prompt typically produces:

```python
return ["fizzbuzz" if i % 15 == 0 else "fizz" if i % 3 == 0 else "buzz" if i % 5 == 0 else str(i) for i in range(1, n + 1)]
```

No prose, no fence, correct casing, strings in all positions, signature matched. Both tests pass on the first attempt. Stylistically it's a one-liner that you might or might not love — Section 7 returns to whether passing tests is sufficient — but functionally it is correct, which is the bar this pipeline cares about.

The difference is not subtle. It is the kind of difference that turns "the first attempt usually passes" into "the first attempt rarely passes," and every retry costs another generation. Investing in the prompt is the cheapest possible quality improvement.

---

## 4. Extracting Code from Model Output

The model's response is a string. Sometimes that string is just code. Sometimes it's code wrapped in markdown fences. Sometimes it's code wrapped in two fences with prose between them. Sometimes it's a paragraph of "here's how I'd approach it" followed by code followed by another paragraph of "and here's how to use it." Sometimes the model refuses to produce code at all and apologises instead.

Your extractor has to handle all of these. It is the smallest piece of code in this module that has the most variation in its inputs, and getting it right matters because every failure here either feeds garbage into the subprocess (which then fails confusingly) or wastes an entire generation by treating a syntactically valid response as an extraction failure.

### How models wrap code, in practice

After running a few thousand code-generation requests, the empirical taxonomy of model output looks like this:

**No fence.** The model returns just the code, no wrapper. This is the cleanest case and the one a well-crafted prompt should produce most often. Extraction is the identity function.

**Single fence with language tag.** The classic ` ```python ` opening, the code, then a ` ``` ` closing. This is what every chat interface in the world has trained the model to produce by default; suppressing it is hard. The language tag is usually `python` but can be `py`, `python3`, or occasionally `Python` or even (frustratingly) `json` when the model has gotten confused about the response shape.

**Single fence without language tag.** Just ` ``` ` on both ends, no language indicated. Slightly less common but still routine.

**Both fences on one line.** A small but real failure mode where the model emits ` ```python def foo(): return 1 ``` ` all on one line. Some extractors break on this case because they assume the opening fence is followed by a newline.

**Multiple fences.** The model explains its approach, then provides the code in a fenced block, then says "and here's an example of how to use it" with another fenced block showing test invocations. Now you have two fenced blocks and need to pick the right one. Sometimes the model gives you three.

**Refusal or commentary.** The model declines to write the code ("I cannot do that") or returns a question ("Could you clarify what you mean by...?"). There is no code to extract. Treating this as an extraction failure is correct: there is no Python here, and feeding the refusal back to the model as the "prior code" makes the model try again.

### A fence-parsing strategy

The extractor needs to handle the above without becoming a maintenance burden. The approach that has held up well across the curriculum is layered: strip what looks like a fence, fall back gracefully if it doesn't look like one, validate the result with `ast.parse()` as a final sanity check.

A sketch of the strategy:

1. Strip leading and trailing whitespace from the raw response.
2. If the response starts with ` ``` `, strip the opening fence: the three backticks plus an optional language tag (`python`, `py`, `python3`, `json` — be permissive) plus an optional newline.
3. If the response (now stripped of an opening fence) ends with ` ``` `, strip the closing fence.
4. Strip whitespace again.
5. Run `ast.parse()` on the result. If it parses, return it. If it doesn't, raise.

This is the `_strip_code_fence` helper that has appeared in Modules 12 and 13 in lighter form (for JSON outputs) and is generalised in this module's project to handle the broader range of language tags. The extraction logic is short — perhaps 15–25 lines — and the test surface is the variety of fence patterns the model produces.

A more robust version, when you anticipate multiple fenced blocks, scans the response for *all* fenced blocks and applies a selection rule (covered next) rather than just stripping the outermost fence pair.

### Choosing the right block

When the response contains multiple fenced blocks, you have to pick one. There are a few rules of thumb, in order of how reliably they work:

**Pick the first block that AST-parses cleanly.** The first block is usually the one the model intended as the answer; subsequent blocks tend to be usage examples or alternative approaches. If the first block is valid Python, take it. This rule works for perhaps 80% of multi-block cases.

**Pick the longest block.** If the first block is short and looks like a sketch ("here's the rough idea") and a later block is longer and looks like the real implementation, the longer block is usually the answer. This rule kicks in when the first-block rule misses, especially when models hedge with a "minimal sketch" followed by a "full version."

**Combine: pick the longest block that AST-parses.** A reasonable default that handles both cases. Iterate all fenced blocks, keep only those that `ast.parse()` accepts, pick the longest.

None of these rules are perfect, and a model determined to be unhelpful (writing two equally valid implementations and asking you to pick) will defeat any rule. In practice, with a tight prompt that says "return only the function body, no explanation," multi-block responses are rare enough that the "first that parses" rule covers nearly everything that arises. The Module 14 project uses a simpler strategy: assume one block, strip it, validate it, and treat anything else as an extraction failure. That is enough for the project's spec files; production code-generation surfaces typically need the multi-block selector.

### Handling commentary around the block

A common pattern: "Here's the code:" before the fence, then a paragraph after the closing fence explaining how to use it. The cleanest way to handle this is to not handle it specially — strip the fences and what's between them is the code, ignoring everything before and after. The fence-pair extraction is the whole answer.

The case to watch for is when the *opening* fence is missing but a closing fence is present, or vice versa. If the model says "Here's the code: def foo(): return 1" with no fence at all, your extractor's "starts with ` ``` `" check fails, and the result of stripping no fence is the entire response including "Here's the code:". `ast.parse()` then chokes on the leading prose. This case is rare with a tight prompt but worth handling: if `ast.parse()` fails on the whole response, you can fall back to scanning for the first line that looks like Python (`def `, `class `, `import `, `from `) and parsing from there. The Module 14 project doesn't bother with this fallback — it lets AST failures count as extraction failures and feeds them back to the model — and that turns out to be enough.

### `ast.parse()` as the syntactic gate

Whatever extraction strategy you use, the final sanity check before handing the code to the executor should be `ast.parse()`. The parse takes microseconds, requires no subprocess, and catches every kind of "this is not actually Python" failure: refusals, prose responses, half-finished code where the model ran out of tokens, code with mismatched parentheses or unclosed strings.

The benefit is twofold. First, it short-circuits the rest of the pipeline: there is no point in writing the response to a temp file and running it if you already know it isn't valid Python. Second, it produces a meaningful error message ("invalid syntax at line 7") that you can feed back to the generator as the revision prompt. A short-circuit at AST-parse time is functionally equivalent to a fast-failing test, except it skips the subprocess round-trip.

There's a subtler property: `ast.parse()` accepts code that doesn't actually do anything (an empty function body, a function that just `pass`es, code that uses undefined names) as long as it's syntactically valid. AST validation does not prove the code works — only that it parses. The actual correctness check is the test run. AST parsing is the *cheap* gate; subprocess execution is the *expensive* gate. Running both gives you a layered defense against malformed outputs.

### What to do when extraction fails

When extraction fails — the AST parser raises, or no fenced block parses cleanly, or the model returned an apology with no code at all — treat it as a failed attempt. Construct a synthetic error message ("Extraction failed: response was not valid Python. First 200 chars: ...") and feed it into the revision sub-loop as if it were a test failure. The model gets the message, sees that its previous response was unparseable, and tries again.

This is the canonical pattern for connecting a non-test failure into a test-shaped loop: wrap the failure in the same shape the loop expects, so the loop's logic stays uniform. The orchestrator does not need a special branch for "extraction failed" vs "tests failed" — both are "we don't have a valid implementation yet, here's the error, please revise." Section 6 covers the iterate-on-failure loop in detail; the relevant point here is that *anything* that prevents you from getting a valid, executable implementation should plug into the same loop as a regular failure.

The Module 14 project encodes this by giving extraction failures their own exit-code sentinel (`-2`) in the `TestRun` model: it's not a real subprocess exit code, but it's distinguishable in the per-attempt logs, and it tells the orchestrator "this attempt failed in extraction, not in testing." Tests-vs-extraction is observable; from the loop's point of view, both count.

---

## 5. Executing Generated Code Safely

The executor is the part of the pipeline that *runs the code the model just wrote*. It is the single most operationally consequential step in the workflow, because it is the only step that touches the outside world. The generator is just an HTTP call to an LLM provider; the extractor is a few string operations and an AST parse; the parser reads a file you trusted enough to feed to your own script. The executor *runs untrusted Python in your process's vicinity*. How you do it matters.

### Why NOT `exec()` / `eval()` in-process

The seductive idea — just `exec()` the generated code in your process and inspect what happens — is wrong in several stacked ways. Each of them is independently disqualifying.

**Memory access.** Code run via `exec()` has full access to your process's memory: the variables in scope, the modules you've imported, the global state of your orchestrator. The generated code can read your API keys (loaded from `.env` into `os.environ`), iterate over `sys.modules`, walk the stack, anything. Even for trusted personal use, this means a confused model that produces wrong-but-curious code (`print(os.environ)`) leaks more than you wanted into the logs. For untrusted input — code generated from a user's prompt — this is a remote code execution vulnerability with no isolation between the user's request and your secrets.

**Infinite loops in your process.** If the generated code contains `while True: pass`, `exec()` hangs forever and you have no way to stop it from outside the call. You can't set a timeout on `exec()`. You can't interrupt it. The orchestrator is now stuck inside the user's code with no recovery path. The only way out is killing the whole Python process, which loses any in-progress state in the orchestrator. With `subprocess`, by contrast, you set a `timeout=` parameter and the subprocess gets killed when it expires, while your orchestrator keeps running.

**Exception leakage.** Exceptions raised inside `exec()` propagate up into your orchestrator's exception handlers. If the generated code raises a `KeyError`, your handler that was watching for `KeyError` from your own logic catches the generated code's `KeyError` instead, and now your error-handling logic runs on a payload it didn't expect. With a subprocess, the generated code's exceptions become stderr text and a nonzero exit code — well-shaped data that you handle with a single uniform check.

**Module pollution.** `import` statements inside `exec()`d code modify `sys.modules` permanently. The next time your orchestrator's code does `import foo`, it gets the version `foo` left around — possibly a monkey-patched one, possibly a stale one. This is the source of some of the worst debugging stories in Python history (it's why Jupyter notebooks have an "import autoreload" extension), and you do not want to invite it into a production pipeline. Subprocess `import`s don't touch your parent process.

These four failure modes compound. In total they mean `exec()` is not appropriate for this workflow regardless of how trusted you believe the model's output is. The shape of the failure is too easy to trigger and too costly to recover from. Use a subprocess.

### `subprocess.run` with `sys.executable`

The subprocess approach is short, clean, and gets you all the isolation you need. The canonical shape:

```python
import subprocess
import sys
from pathlib import Path

def run_code(code: str, work_dir: Path, timeout: int = 10) -> tuple[int, str, str]:
    file_path = work_dir / "attempt.py"
    file_path.write_text(code)
    try:
        result = subprocess.run(
            [sys.executable, str(file_path)],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired as e:
        return -1, e.stdout or "", e.stderr or "TIMEOUT after {}s".format(timeout)
```

A few small things deserve emphasis. `sys.executable` is the path to the Python interpreter that is currently running your orchestrator. Using it (rather than hardcoding `"python"`) means the subprocess runs in the same virtual environment as your orchestrator, so any packages the generated code imports are the ones you have installed. This is what you want: the test code and the orchestrator code share a dependency set. If the spec file imports `requests`, the subprocess will see `requests` because your venv has it.

`capture_output=True` makes `subprocess.run` collect stdout and stderr into the returned object rather than streaming them to your terminal. `text=True` decodes them as strings (using your locale's encoding by default) so you don't have to deal with bytes. The `timeout` parameter is the safety belt: if the subprocess runs for longer than `timeout` seconds, Python raises `TimeoutExpired` and (importantly) kills the subprocess before raising. You don't have to do the killing yourself.

The pattern from the project's `solution.py` follows this shape, with the addition that the temp file path goes through `tempfile.mkdtemp()` (so each run gets its own directory that can be cleaned up at exit) and the failed-test detection is exit-code-plus-stderr-parse rather than just exit-code.

### What stdout and stderr each tell you

Capturing both separately is worth doing because they carry different signals.

**stdout** is what the generated code chose to print. If the model wrote a function that includes a debug `print("got here")`, that line ends up in stdout. If the model included demo code at the bottom of the file (`print(fib(10))`), the result is in stdout. In a well-behaved generation, stdout is empty: the test runner exits with status 0 (or 1 on failure) and prints nothing. A non-empty stdout is a soft signal that the model produced side-effecting code, which Section 7 flags as a quality concern.

**stderr** is where errors go: Python tracebacks, the test runner's `FAIL:` lines (in this module's project), assertion messages, exception text. It is the primary signal for the iterate-on-failure loop. A failed run with no stderr is almost always a TimeoutExpired (the process was killed before it could write); otherwise, stderr contains the information the model needs in order to revise.

Reading them separately means you can route the signals differently. The orchestrator typically passes only the *stderr* into the revision prompt — it is the actionable error message — and leaves stdout out, since it is mostly noise from the model's accidental prints. Some pipelines log both for debugging but only show stderr to the model. Module 14's project follows that convention.

There is a degenerate case worth naming: a test that itself prints diagnostic output to stdout before asserting. The diagnostic shows up in stdout, the assertion failure shows up in stderr. Both are useful for the developer staring at the run, but only the stderr is useful for the model's next attempt. The split capture preserves the option to use them differently.

### The threat model

Now the uncomfortable part. The subprocess approach gives you process isolation: a crash, a hang, an infinite loop, or a thrown exception in the generated code stays in the subprocess and does not propagate into your orchestrator. That is real, and it is enough for the threat model this module assumes.

The threat model is *trusted personal use*. You wrote the spec yourself, or someone you trust wrote it. The model is being asked to fill in a function that you defined. The "untrusted input" is the model's *output*, but not adversarially crafted — it's just whatever the model happened to produce when asked to solve a problem you stated.

Under that threat model, the subprocess approach is appropriate. The kinds of failure it protects against — infinite loops, syntax errors, runtime exceptions, naive `exec()`-style cross-contamination — are exactly the failures a confused model produces. The subprocess timeout catches the loops. The exit code catches the failures. The captured stderr tells you why. Done.

**The subprocess approach is NOT a hardened sandbox.** Be explicit about this. The generated code has the same privileges as your orchestrator. It can:

- Read environment variables, including any secrets your orchestrator's `.env` loaded.
- Read and write any files your user account can read and write.
- Open network connections to any address your network allows.
- Eat your CPU up to the timeout limit (10 seconds per run by default).
- Spawn its own subprocesses, which inherit those same privileges.

If the spec is malicious — written by an attacker, or pasted from somewhere you don't trust — none of those capabilities are restricted by anything in this module's pipeline. A spec that says "write a function that reads `.env` and prints it" will succeed at that, and the test runner will dutifully report that the function did what was asked.

The right reaction to that list is *not* to bolt extra checks onto the subprocess approach. It is to recognise that subprocess isolation solves a different problem (operational safety from a confused model) than sandboxing (security from a malicious input), and that if you have a malicious-input threat model, you need different tools.

### Harder threat models, and the tools for them

For situations where the spec or the model output cannot be trusted, the toolbox steps up:

**Docker with `--network=none` and read-only mounts.** Run the subprocess inside a Docker container with networking disabled and the filesystem mounted read-only (except for a small writable volume for the temp file). The generated code now lives inside the container's OS and cannot see your host's filesystem or talk to the internet. The overhead is real — starting a container takes a second or two — but for batch use the cost amortizes well. Several production code-execution services (including some hosted offerings) use this pattern under the hood.

**Pyodide / WebAssembly.** Run the code in a Python interpreter compiled to WebAssembly, executing inside a browser or a WASM runtime. Pyodide runs in the browser's sandbox by default, with no filesystem access except what you grant it explicitly and no network access except what `XMLHttpRequest` is allowed. This is the right tool when the code execution needs to happen client-side — for example, an in-browser playground for AI-generated code where you cannot ship a containerised backend.

**Hosted sandboxes like e2b.dev or codeinterpreter-style APIs.** Several services offer a "run this code in our sandboxed VM and return the result" API. The provider handles the isolation; you pay them for the compute. This is the right tool when you have a malicious-input threat model and don't want to operate the sandbox yourself. Pricing is per-second of execution; latency is higher than local subprocess.

**Restricted Python (`RestrictedPython` library, AST-rewrite sandboxes).** Older, less battle-tested approaches that rewrite the AST to remove access to dangerous builtins. They tend to have a long tail of escape routes — Python's introspection makes truly restricting it from within Python very hard — and are not recommended for new code in 2026.

None of these are needed for this module's project. The project's spec files are written by you, the model's output is bounded by the function signature you provided, and the worst-case failure (an infinite loop or a stack overflow) is contained by the subprocess timeout. If you later build a system that takes specs from untrusted users, revisit this section and pick the right hardening tool from the list.

### Temp files are debugging aids

A nice side effect of the temp-file pattern is that the generated code is *inspectable*. After a failed run, the file is still on disk (in the temp directory) and you can open it in your editor, see exactly what the model produced, manually run the tests, add print statements, and diagnose what went wrong. The model's output is no longer ephemeral — it is artifact that lives in a known location.

The Module 14 project exposes this via a `--keep-temp` CLI flag. By default, the temp directory is cleaned up when the orchestrator exits (via `shutil.rmtree`). With `--keep-temp`, the directory is preserved and its path is printed at the end of the run, so you can `cd` into it and look at `attempt_1.py`, `attempt_2.py`, and so on. Each attempt produces its own file, so you can also see how the model's output *changed* between attempts — was it a small tweak, or did the model rewrite from scratch?

This is the kind of small operational convenience that pays for itself the first time you have to debug a confusing run. Without `--keep-temp`, you'd be left squinting at the final stderr and trying to reconstruct what the model wrote; with it, you have the actual file the subprocess ran. The pattern generalises: any time your pipeline writes a file as part of processing, keeping it (under a flag, or for failed runs only) makes debugging meaningfully easier.

---

## 6. The Iterate-on-Failure Loop

When the test run fails, the loop turns over. The orchestrator takes the stderr, packages it into a revision prompt, and asks the model to try again. This sub-loop is the part of the workflow that gives code generation its characteristic shape — the part that makes it feel like a *conversation with the runtime* rather than a one-shot generation. Done well, the loop converges quickly on most reasonable specs. Done poorly, it spirals into an expensive round-trip with no convergence.

### What feedback to send back

The first design question is *what information goes into the revision prompt*. The candidates are:

- **The full traceback / stderr.** Everything the subprocess wrote, verbatim.
- **A trimmed version of the traceback.** The same content, capped at a maximum size (say, the first 2000 characters).
- **Just the failing assertion line.** Hand-extracted, no context.
- **A structured summary** ("test_fib_zero failed: expected 0, got 1") parsed out of the stderr.

The right default is **the full stderr, trimmed at ~2000 characters**. Here's why each of the alternatives is worse.

Sending only the failing assertion line loses critical context. The model needs to see which test failed, what the assertion was, and what value the function returned. Just "AssertionError" tells it the function was wrong but not *how*. The model will then guess at what to fix, often regenerating the same broken implementation with a small tweak.

Sending a parsed structured summary sounds elegant but is brittle. Test runners emit traceback shapes that vary across Python versions, library versions, and stack depths. A regex that extracts "the failing line and the values" works until it doesn't, and when it doesn't, you've lost the information you needed and added a layer of parsing logic to maintain. The model handles raw stderr just fine — it has seen plenty of tracebacks in its training data — so the parsing overhead buys you nothing.

Sending the full untrimmed stderr is fine for most runs but pathological in the worst case. A test that has a deep recursion or a long stack trace can produce kilobytes of stderr. A few of those in a row blow your context budget, slow the next generation, and may push you over the model's context window. The cap at ~2000 characters is a soft guardrail: keep enough that the model can see the failure, drop the deep stack frames if there are too many.

The trimming should keep the *most useful* portion. The end of stderr typically contains the actual failure message, while the middle contains the call stack. Trimming from the middle (keep the first N lines and the last M lines, drop the middle) preserves both the test name and the assertion details. A simpler strategy — keep the last 2000 characters — also works because Python's traceback puts the exception at the end. Pick whichever fits your aesthetic; both are better than no trimming or no cap.

### Include the prior implementation

The second piece of context that matters is the *prior implementation*. The model needs to know what it tried last time. Without it, the model has no anchor for the revision — it might rewrite from scratch, possibly making the same mistake, possibly introducing a new one. With it, the model is doing a diff: "here's what you wrote, here's what failed, fix it."

The structure of the revision prompt's user message:

```text
Your previous implementation did not pass the tests.

Previous implementation:
def fib(n: int) -> int:
    if n == 0:
        return 1
    ...

Test runner stderr:
FAIL: test_fib_zero: AssertionError: assert fib(0) == 0

Please revise the implementation to fix the failures.
```

The "previous implementation" block primes the model to think in terms of revision rather than regeneration. The "test runner stderr" block tells it what specifically broke. Together they form the minimal context the model needs to produce a useful next attempt.

You might wonder whether to also include the original spec (signature, docstring, test source) on every retry, or only on the first attempt. The answer: include it every time. The cost is a few hundred extra tokens per retry, the benefit is that the model has the full context of what it's solving even if its earlier reasoning had drifted. Models can mis-remember the spec between turns when the conversation gets long; re-grounding them on every retry prevents drift.

### Stop conditions

A revision loop needs explicit stop conditions, or it will run forever and burn tokens until your wallet gives out. There are three useful ones:

**Success — the tests pass.** The happy path. `exit_code == 0` and no failures in stderr. The orchestrator records the result and exits the loop with `stop_reason="passed"`.

**`max_attempts` reached.** The pessimistic path. After N tries, give up. The default in this module's project is `max_attempts=4`, which is enough for most easy and medium problems and short enough that a hard problem doesn't burn a budget. The orchestrator records the result with `stop_reason="max_attempts"`, returns the (still-failing) implementations as part of the result so the user can inspect them, and exits the CLI with a nonzero status.

**No-progress detection — the model returns identical code.** A subtle but important stop condition. If the model produces the same code on attempts 2 and 3 (and the same error on both), there is no reason to expect attempt 4 to differ. The model is stuck. Continuing would burn another generation for no benefit. The detector compares the extracted code from the most recent N attempts (typically 2 or 3) and stops early if they match. The orchestrator records `stop_reason="no_progress"` and exits.

The no-progress check is the most operationally important of the three. `max_attempts` is a hard ceiling, but it lets you burn the full ceiling on a hopeless case. No-progress detection catches the hopeless case earlier, often saving 1–2 generations. Implementation is short: keep the last few implementations in a list, normalise whitespace, compare strings. A more sophisticated version normalises the code via `ast.dump()` so that semantically-identical implementations with whitespace differences are caught; the project uses the simpler string-compare and lives with the occasional false negative.

### Convergence behavior

Empirically — across the spec files this module ships and across roughly the kinds of problems learners try — the convergence pattern is:

**Easy problems** (fizzbuzz, factorial, palindrome check, anything you'd ask in a phone screen) almost always pass on attempt 1. The model has seen these patterns a thousand times in training; it produces correct code on the first try. If the first attempt fails, it usually fails because of an output-format issue (wrong return type, wrong list shape) that one retry fixes. Easy problems converge in 1 attempt, sometimes 2.

**Medium problems** (recursive functions with multiple base cases, regex parsing, recursive directory walks, things where the docstring leaves a few choices to the model) sometimes pass on attempt 1 but more often need a retry. The first attempt is usually close but has a subtle bug — the wrong base case, an off-by-one, a missing edge case in the test. The error message from the failed run usually contains enough information for the next attempt to fix it. Medium problems converge in 1–3 attempts.

**Hard problems** (problems where the spec is genuinely ambiguous, or where the right implementation requires non-obvious algorithms, or where the test set is unusually large) may converge or may not. When they converge, it's usually in 3–5 attempts. When they don't converge in 5, they usually don't converge in 10 either — the model is stuck on the wrong approach and just keeps revising the wrong approach.

A useful operational rule: **if convergence doesn't happen in 5 attempts, the spec is the problem, not the attempt count.** Increasing `max_attempts` to 10 doesn't unlock convergence on hard cases; it just spends more money to find out the spec is bad. The right move is to look at the failing attempts, find what the model keeps getting wrong, and either tighten the spec (add a test that pins down the ambiguity) or break the spec into smaller pieces (split a function that does two things into two functions).

### Same shape as Module 12's critique loop

Take a step back and the loop's shape is exactly the writer/critic loop from [Module 12 (Multi-Agent Systems)](../12-multi-agent-systems/). A producer generates an artifact. A critic evaluates it. If the critic approves, the loop ends. If not, the critic's feedback is fed back to the producer for revision, and the loop runs again. Both loops have bounded iteration. Both have stop conditions for "the producer isn't making progress." Both expose per-iteration cost.

The difference is in the critic. In Module 12, the critic was an LLM with a rubric ("here are the criteria, does the artifact meet them?"). In this module, the critic is the Python runtime ("here's the code, did it run and pass?"). The LLM-critic was useful in Module 12 because the artifact was prose and there was no mechanical check available; the runtime-critic is useful here because the artifact is code and mechanical checks are exactly what the runtime does well.

The two patterns are not in competition — they are the same pattern with different evaluators. Code generation systems sometimes use both: a runtime critic for "do the tests pass" and an LLM critic for "is the code readable and well-structured." The combined loop runs the runtime check first (because it's cheap and binary), then the LLM check on the surviving implementations. Section 7 returns to this composition.

### Cost: budget aggressively

A reminder that closes the section. Each retry is a full generation. With prompt growth across attempts (prior code, prior error, full spec re-grounded each time), retries can cost more than the first generation. A four-attempt run with a medium-length prompt can use 5–8x the tokens of a one-attempt run that succeeded on the first try.

This is the reason to invest heavily in Section 3's prompting techniques: every first-attempt-pass is a retry avoided. The expected cost of generation is approximately `(probability of first-attempt pass) * 1 + (probability of needing retries) * E[retries]`, where E[retries] is the expected attempt count for a problem that doesn't pass first. Improving the first-attempt pass rate even modestly — from 60% to 80% — collapses the right-hand term and produces an outsized cost reduction.

In numbers: at 60% first-attempt-pass and an average of 2 extra attempts on failures, expected attempts is `0.6 * 1 + 0.4 * 3 = 1.8`. At 80% first-attempt-pass with the same retry behavior, expected attempts is `0.8 * 1 + 0.2 * 3 = 1.4`. A 22% reduction in expected attempts, achieved entirely by prompt improvements that pay for themselves on the first run. This is the operational case for treating prompt engineering as cost engineering.

---

## 7. Evaluating Generated Code

Test-pass-rate is the binary check at the heart of this module's pipeline. Once it passes, the iterate-on-failure loop is satisfied and the code is, by the workflow's definition, "done." But test-pass-rate is necessary, not sufficient. A function can pass every test and still be wrong, unreadable, slow, dangerous, or rude to your future self. This section is the catalogue of what *else* you should check, and the tools that check it.

### Why test-pass is not enough

Imagine the model is asked to implement `def fizzbuzz(n: int) -> list[str]:` and is given two tests:

```python
def test_fizzbuzz_basic():
    assert fizzbuzz(5) == ["1", "2", "fizz", "4", "buzz"]

def test_fizzbuzz_fifteen():
    assert fizzbuzz(15)[-1] == "fizzbuzz"
```

A perfectly correct implementation passes both. So does this:

```python
def fizzbuzz(n: int) -> list[str]:
    if n == 5:
        return ["1", "2", "fizz", "4", "buzz"]
    if n == 15:
        return ["1", "2", "fizz", "4", "buzz", "fizz", "7", "8", "fizz", "buzz", "11", "fizz", "13", "14", "fizzbuzz"]
    return []
```

This is the **overfit-to-tests failure**. The function passes every visible test by hard-coding the expected outputs, and would fail catastrophically on any input not in the test suite. It is technically correct on the test set, completely wrong on the underlying task, and unless you look at the code you'd never know.

This failure mode is more common than learners expect, especially when the model is under pressure to pass tests it has been shown verbatim. The fix is partly *evaluation-side* (hold out a set of tests the model never sees, so it cannot overfit; cover more of the input space) and partly *prompt-side* (don't show the model the exact test cases; describe the behavior in the docstring and let the visible tests be only a subset). Neither fix is perfect on its own. The deeper lesson is that *the tests are not the contract — they are a sample of the contract*, and a code generator that treats the sample as the full contract has missed the point.

### Readability

Code that passes tests can still be unreadable. The model may produce a single-line list comprehension where a four-line loop would have been clearer. It may use one-letter variable names because the docstring didn't specify naming conventions. It may inline a constant rather than extracting it to a named value. None of these issues affect correctness; all of them affect the experience of any human who later has to read, debug, or extend the code.

Things to check for:

- **Variable names.** Are they descriptive? `result`, `n`, `total` are usually fine. `r`, `x`, `tmp` are usually not — they require the reader to remember context.
- **Function decomposition.** If the function is doing two distinct things (parse a log line, then validate it), are they split into helpers, or crammed into one function?
- **Comments.** Are there comments where the code is doing something non-obvious? Are there *no* comments where the code is opaque? Both extremes (over-commented obvious code, under-commented complex code) are smells.
- **Style consistency.** Does the code match the surrounding codebase's idioms? If your codebase uses `snake_case` and the model wrote `camelCase`, that's a real problem when the code lands in a PR.

Most of these are subjective enough that a human reviewer needs to be in the loop, but they are also signal that an LLM-based reviewer can pick up on with reasonable accuracy. Section 7's closing discussion covers LLM-as-judge for this purpose.

### Complexity

A function can be both correct and *too complex*. Common forms:

- **Cyclomatic complexity** — the number of independent paths through the code. A function with one if-else has cyclomatic complexity 2. A function with five nested if-elses has cyclomatic complexity 16. Above ~10, the function is hard to test exhaustively (you need 10+ test cases to cover every path) and hard to reason about.
- **Line count.** A correct 50-line solution to a 5-line problem is a signal that the model is either being defensive (handling edge cases that don't exist) or doing something inefficiently (writing manual loops where a built-in would suffice). Look at the line count relative to a hand-written version.
- **Function length.** Functions longer than ~30 lines are hard to keep in your head. The right move is usually to split them, but the model has no incentive to do so unless the prompt asks.

Tools that measure complexity automatically — `radon`, `flake8-complexity`, `mccabe` — can be added to the pipeline as a *secondary* check after tests pass. The Module 14 project does not include them, but a production code-generation surface would.

### Side effects

The generated code should produce the right output for its inputs and *nothing else*. Common side-effect failures:

- **Writes to disk.** The model spuriously opens a file, writes a log line, then continues. The function works (the disk write doesn't affect the return value), but your filesystem now has stray files.
- **Print statements.** The model leaves debug `print()` calls in the final code. The function works, but it pollutes stdout in production.
- **Global mutation.** The model assigns to a global (`results.append(x)` where `results` is a module-level list) instead of returning the value. Now the function's behavior depends on call order, which is a recipe for non-determinism.
- **Sleep or wait calls.** The model adds `time.sleep(1)` because the docstring mentioned "wait," even though no waiting is required. The function works, but each call takes a second longer than it needs to.

Some of these are caught by stdout capture in the executor (Section 5) — a non-empty stdout for a function that should return silently is a soft warning. Others (disk writes, global mutation) are not visible unless you actively look for them. A static linter or a careful code reviewer catches them; the model's runtime check does not.

### Security

This category is small but important. Generated code can introduce vulnerabilities:

- **`eval()` of user input.** The model writes `result = eval(user_string)` to parse "any kind of input." This is a remote code execution vulnerability if `user_string` comes from outside.
- **Shell command construction without escaping.** `subprocess.run(f"ls {user_path}", shell=True)` is a shell injection if `user_path` is `; rm -rf /`. The model has no way of knowing whether the caller has sanitised the input.
- **SQL string interpolation.** `cursor.execute(f"SELECT * FROM users WHERE id = {user_id}")` is a SQL injection if `user_id` is `1 OR 1=1`. The model is doing what the prompt seemed to ask, but the result is unsafe.
- **Hardcoded secrets.** The model copies an example API key from its training data into the generated code. Now your repo has a credential to rotate.

These are the kind of issue that a security review catches and a test suite usually does not. For high-stakes generation, a static-analysis pass (Bandit, Semgrep) on the generated code as a second-stage gate is appropriate. For this module's project, the spec is small enough and personal enough that explicit review is sufficient.

### LLM-as-judge for code review

For the qualitative axes — readability, style, naming, decomposition — running a *second* LLM as a code reviewer is a productive pattern. The model has read millions of code reviews in its training data. Asking "is this code readable, well-decomposed, and idiomatic?" produces useful structured feedback.

This is the same critic pattern from [Module 12 (Multi-Agent Systems)](../12-multi-agent-systems/), now applied after the runtime check has already verified correctness. The composition looks like this:

```text
generate → execute → tests pass → llm_review → ship
                          │             │
                          no            issues
                          ↓             ↓
                      revise        revise (style)
```

The first critic (runtime) checks correctness. The second critic (LLM) checks style. Each runs in its own pass; each has its own stop conditions. The LLM-as-judge call is cheaper than another generation because the reviewer doesn't have to *produce* the code, only *evaluate* it — the output is a short structured report (a few rubric scores and a few sentences of suggested fixes), not another implementation.

Module 14's project does not include the LLM-as-judge pass — it stays focused on the runtime check — but adding it is a small extension exercise. The mental model: runtime is the binary critic, LLM is the qualitative critic, both critics feed the same revision generator. Cross-link to [Module 12's critic-loop discussion](../12-multi-agent-systems/) for the details of how to structure the critic's prompt and how to use its rubric.

### What humans still need to verify

Some things only a human can sign off on. The list is short but real:

- **License-correct dependencies.** Did the model `import` a package whose license is incompatible with your project? GPL code cannot be linked into a permissive-licensed project. The model has no way to know your project's licensing constraints.
- **Performance under realistic inputs.** The model's tests run on small inputs. Does the function work on the inputs production will actually see? A function that's O(n^2) in the input size is fine for n=100 and dies for n=100,000.
- **Alignment with codebase conventions.** Does the function fit the style, the error-handling conventions, the logging conventions of the rest of your codebase? The model only sees the spec; it doesn't see the surrounding codebase.
- **Behavioral specification gaps.** Does the function do what was *meant* by the spec, even if the tests don't pin it down? The spec is always an incomplete description; humans fill the gaps with judgment that the model cannot replicate.

These are not failures of the model — they are failures of the spec to encode everything a human would have wanted. The right operational answer is to keep humans in the loop on anything that matters, and to use code generation as an accelerator rather than a replacement for review. Even Claude Code, which is the most ambitious code generation product on the market, ships with a "every diff goes through a human" assumption baked into its UX.

### Forward pointer: Module 15

Module 15 (Evaluation & Testing) builds directly on this section. The qualitative axes here — overfit detection, complexity scoring, readability evaluation, LLM-as-judge methodologies — become first-class topics in Module 15, with frameworks for running them at scale, dashboards for tracking them over time, and patterns for combining mechanical and LLM-based evaluation into a coherent quality signal. If Section 7 left you wanting more rigor, Module 15 is the answer.

---

## 8. AI Code Generation in the Stack & Real-World Tools

The pipeline you've built in this module — single function, single spec, single retry loop — is the seed of every AI-pair-programming product on the market. Real tools scale up in three dimensions: the *unit of generation* grows from a function to a file to a repo; the *context* grows from a spec to a codebase plus a project history; and the *execution* grows from a subprocess to a full IDE-integrated tool surface. This section is a tour of how those expansions look in practice and where they intersect with the rest of this curriculum.

### A sidebar on real tools

The four production tools that define the current landscape, each with a different bet on which expansion matters most:

**Claude Code** — the agent-loop reference architecture. Claude Code wraps the pipeline you built in this module inside an open-ended ReAct loop (cross-link [Module 11](../11-building-ai-agents/)) with filesystem and execution tools. The agent reads files, edits them, runs commands, observes results, and continues. The generation step is one among many — alongside file reads, file writes, grep, and bash invocations. The loop ends when the agent decides the task is done. Where Module 14's project is "generate one function until tests pass," Claude Code is "do whatever it takes to ship the feature." The architectural family is the same; the scope is dramatically larger.

**Cursor** — the IDE-integrated chat plus tab-completion bet. Cursor lives in the editor. The user types; Cursor suggests completions inline. The user opens a chat panel; Cursor reads the current file, uses repo-scoped vector search (cross-link [Module 07 (RAG)](../07-rag/)) to pull in relevant context from elsewhere in the codebase, and answers questions or proposes edits. The bet: most code generation in practice is *editing existing code in context*, not generating a function from scratch. The IDE integration gives the model continuous access to "what file is open, what was just changed, what does the user seem to be doing."

**GitHub Copilot** — the autocomplete-first incumbent. Copilot's primary surface is inline completion: you type the first half of a line, Copilot suggests the second half. The chat surface came later. The bet: most code generation in practice is *finishing the line you started*. Copilot succeeds when its suggestions are short, contextually correct, and accepted with a tab key — meaning the friction is near-zero and the model is rewarded for being a reliable partner on the keystroke rather than a verbose oracle.

**Aider** — the CLI git-aware refactorer. Aider lives in the terminal, reads your git status, and proposes edits as diffs. You apply or reject each diff. The bet: most code generation in practice is *small, targeted edits to existing files*, and the right interface is the same one developers already use for code review — diffs against HEAD. Aider's loop is "describe the change you want, review the diff, accept or reject, repeat." It is the closest to a "code review with an LLM" workflow.

Each of these tools makes different architectural choices, and each has found a productive niche. Claude Code aims at "agent that can ship a feature end-to-end." Cursor aims at "AI co-editor inside the IDE." Copilot aims at "AI partner at the keystroke level." Aider aims at "AI participant in the git-diff review loop." None of them is the future of code generation; all of them are co-evolving as the underlying model capabilities grow.

### What changes at repo scale

The biggest gap between Module 14's project and these tools is the unit of work. The project generates one function from one spec. Production tools work over a repository — many files, many directories, many interacting modules. Several things change at that scale, and most of them correspond to other modules in this curriculum:

**RAG over the codebase becomes a workflow step.** A function that needs to call other functions in the codebase needs to *know* about those functions. A naive approach — paste the whole codebase into the prompt — works for tiny projects and fails for anything real. The real approach is exactly what [Module 07](../07-rag/) covered: embed the codebase, search for relevant chunks by similarity to the current task, and inject them as context. Cursor's "repo-aware" feature is RAG over a vector index of the user's repo. Claude Code does similar retrieval via grep-style tools (rather than embeddings), trading recall for precision. Either way, retrieval becomes a step inside the generation workflow.

**Edit-by-diff replaces full-file output.** Generating a whole file every time the user wants to change a function is wasteful in tokens, slow to display, and easy to get wrong (the model might rewrite a function that wasn't supposed to change). Production tools shift to generating *diffs* — patches against the current file — which are smaller to produce, easier to review, and trivial to apply with standard tools (`patch`, `git apply`). Aider's whole interface is diff-first. Cursor's "edit" command shows a diff before applying. The pattern is universal in the production tools.

**Multi-file consistency becomes a constraint.** Changing one function often requires changing its callers. Adding a new field to a Pydantic model requires updating every place that reads the model. The model has to either edit multiple files in one turn (Claude Code's pattern) or repeatedly re-engage the generation loop on each affected file (which is slower but lets the user review each step). Either way, the workflow is no longer single-function; it is a small graph of dependent edits.

**File-system tools become tool calls.** The "read file," "write file," "search files," "run tests," "run command" operations that this module's project does directly in Python become *tools* the model can call inside an agent loop (cross-link [Module 06 (Tool Use & Function Calling)](../06-tool-use-function-calling/)). The agent reads the file it needs to edit, writes a new version, runs the tests, observes the output, and decides what to do next. The shape is unchanged from Module 11's agent loop — the *tools* are filesystem-and-exec operations rather than search-and-summarise operations.

**The agent loop wraps everything.** At the outermost level, the production tools are agents. The user states an intent in plain English ("add a `parse_json` function with error handling"); the agent decides which files to read, what edits to make, when to run tests, and when to stop. The generation step is one inside the agent's loop. The pattern is exactly [Module 11](../11-building-ai-agents/) — ReAct over filesystem tools instead of web-search tools. The module project is a small workflow; Claude Code is the same shape with the workflow itself wrapped by an agent that decides when to invoke it.

### Relationship to other modules

The module cross-references for code generation point in every direction:

**[Module 06 (Tool Use & Function Calling)](../06-tool-use-function-calling/)** — file-system and execution tools (read, write, grep, run) are exactly the tools an agent would call. The same tool-calling pattern, but the tool implementations talk to disk and shell instead of web APIs.

**[Module 07 (RAG)](../07-rag/)** — codebase RAG is the way an AI tool brings relevant context to a generation task. Embed the repo, search by similarity to the task, inject the matched chunks. The RAG pipeline you built in Module 07 is the same shape, the corpus is just code instead of documents.

**[Module 11 (Building AI Agents)](../11-building-ai-agents/)** — the agent loop is the outermost shell of every production code-generation tool. The agent decides what to do next based on previous tool results, until a stop condition is met.

**[Module 12 (Multi-Agent Systems)](../12-multi-agent-systems/)** — the writer/critic loop is exactly this module's iterate-on-failure loop, with the critic role taken by `pytest`. Some production tools layer a second LLM critic on top for code review.

**[Module 13 (Workflows & Chains)](../13-workflows-chains/)** — the workflow shape (parse → generate → extract → execute) is the inner spine of code generation. The agent loop calls workflow-shaped tools; the workflow handles the predictable parts; the agent handles the open-ended parts.

This module's project shows the pattern at the smallest possible scale — one function, one spec, one subprocess — because the smallest scale is the cleanest place to see the pattern. Scaling up multiplies the surface (more files, more context, more tools) but does not change the underlying shape.

### Forward pointer: Module 15 and Phase 4

**Module 15 (Evaluation & Testing)** builds directly on Section 7 of this module. The qualitative axes that Section 7 introduces — overfit-to-tests detection, complexity scoring, security review, LLM-as-judge methodologies — become Module 15's central topics. Module 15 also covers held-out test sets, evaluation harnesses, regression detection across model upgrades, and dashboards for tracking generation quality over time. If this module's Section 7 left you wanting a more systematic answer, Module 15 is it.

**Phase 4** of the curriculum then covers the operational themes that production code-generation systems require: caching (cache the model's output across identical specs to save retries; cache the embedding index across runs to amortise the indexing cost), observability (per-attempt telemetry like this module's `StepUsage`, but at production volume with dashboards and alerts), and deployment (running these systems behind APIs, on schedules, or as background workers, rather than as CLI scripts). The instrumentation patterns from this module — typed step records, per-run total costs, attempt-by-attempt tracking — become the foundation for the Phase 4 topics.

The arc from Module 13 through Module 15 is roughly: *learn the workflow shape (Module 13), apply it to a checkable artifact (Module 14), measure the artifact's quality systematically (Module 15)*. Phase 4 then takes that whole stack and teaches how to operate it. Code generation is the bridge between "workflows of LLM calls" and "production AI systems," because it's the first workflow in this curriculum whose output can be evaluated mechanically — which is exactly what production systems need from their evaluation tooling.

### Module cross-reference map

| This module's component | Prior module it builds on |
|---|---|
| Pydantic types as step communication contracts (`CodeSpec`, `Implementation`, `TestRun`, `Attempt`, `GenerationResult`) | [Module 08](../08-structured-output/) — structured output and schema validation |
| The parse/generate/extract/execute chain | [Module 13](../13-workflows-chains/) — sequential workflows and typed step boundaries |
| The iterate-on-failure revision sub-loop | [Module 12](../12-multi-agent-systems/) — writer/critic loop with bounded iteration |
| Per-step `StepUsage` records and per-run cost report | [Module 13](../13-workflows-chains/) — observability through fixed-shape logs |
| `_strip_code_fence` extractor (generalised for `python`/`py`/`json` tags) | [Module 12](../12-multi-agent-systems/) / [Module 13](../13-workflows-chains/) — fence-strip helper for structured LLM outputs |
| Subprocess + `sys.executable` + timeout pattern | New in this module — mechanical execution as the critic |
| `ast.parse()` as a syntactic gate before execution | New in this module — cheap pre-check that short-circuits invalid responses |
| File-system and execution tools as workflow steps (forward pointer to production tools) | [Module 06](../06-tool-use-function-calling/) — tool calling pattern, now applied to disk and shell |
| RAG over codebase as the context-injection step in production tools | [Module 07](../07-rag/) — embedding, retrieval, and context injection |
| The agent shell that production tools wrap around the workflow | [Module 11](../11-building-ai-agents/) — ReAct loop, stop conditions, and observability |

The unifying theme, as in Module 13: the techniques from prior modules are the *building blocks*, and this module composes them into a workflow whose distinctive feature — mechanical evaluation of the output — unlocks a revision loop that earlier workflows could not have. Internalising this composition (workflows on the outside, agents around them in production, RAG and tool-use as the steps inside) is the design principle that holds Phase 3 together. Module 15 and Phase 4 take that toolbox and teach how to evaluate and operate it.
