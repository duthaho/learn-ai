# Module 15: Evaluation & Testing

**What you'll learn:**
- Why evaluating LLM systems is harder than evaluating traditional software
- The eval loop: dataset + system-under-test + evaluators → scorecard
- Where eval datasets come from (golden curated, synthetic, production traffic) and their tradeoffs
- Mechanical evaluators — exact match, normalized match, schema validation, regex, set/list compare
- LLM-as-judge — rubric scoring, paired comparison, known biases, and self-consistency tactics
- Combining mechanical and LLM evaluators into a single scorecard
- Regression detection: comparing runs across prompt or model changes
- Where eval fits in the AI stack — Promptfoo, Phoenix, Langfuse, Ragas, DeepEval

| Detail        | Value                                                                                                  |
|---------------|--------------------------------------------------------------------------------------------------------|
| Level         | Intermediate–Advanced                                                                                  |
| Time          | ~3.5 hours                                                                                             |
| Prerequisites | Module 02 (Prompt Engineering), Module 08 (Structured Output), Module 13 (Workflows & Chains), Module 14 (AI Code Generation) |

---

## Table of Contents

1. [Why Evaluating LLM Systems Is Hard](#1-why-evaluating-llm-systems-is-hard)
2. [The Eval Loop](#2-the-eval-loop)
3. [Eval Datasets: Where They Come From](#3-eval-datasets-where-they-come-from)
4. [Mechanical Evaluators](#4-mechanical-evaluators)
5. [LLM-as-Judge](#5-llm-as-judge)
6. [Combining Mechanical + LLM Evaluators](#6-combining-mechanical--llm-evaluators)
7. [Regression Detection and the Iteration Loop](#7-regression-detection-and-the-iteration-loop)
8. [Eval in the AI Stack](#8-eval-in-the-ai-stack)

---

## 1. Why Evaluating LLM Systems Is Hard

Every module before this one has produced *some* kind of LLM-powered output: a summary, a classification, a tool-using agent, a multi-step workflow, a generated function. In every case, the question that should have been nagging you — and that this module finally takes seriously — is *how do you know it's good?* For the code-generation module that closed Phase 3, there was a clean answer: run the tests, see if they pass. The runtime was the oracle, the assertion was the contract, the bar was binary. That was the easy case. It was the easy case *because* the output was code, and code happens to be the one artifact an LLM produces that has an unambiguous mechanical evaluator built into the universe.

For everything else, the question is harder. A summary either reads well or it doesn't, and "reads well" is not a function you can call. A reply either lands with the recipient or it doesn't, and "lands" is not a value that comes back from `assert`. A classification either matches the label or it doesn't — but the label itself was assigned by a human who might have disagreed with another human on the same input. When you move from code generation back to the rest of the LLM design space, the evaluation problem stops being "did the tests pass" and starts being "what is this system supposed to do, in numbers, on a distribution of realistic inputs, and how can I tell when I've broken it?"

That problem is the subject of this module. It is the meta-module that closes Phase 3 because every other module in the curriculum has been quietly waving its hands at evaluation. We talked about prompts in Module 02 without saying how to compare two prompts rigorously. We talked about RAG in Module 07 without saying how to measure whether retrieval was actually helping. We talked about agents in Module 11 and 12 without saying how to score a multi-turn trace. The implied subtext was always: *you'll know it when you see it.* This module replaces that with: *you'll know it when you measure it, on a dataset you trust, with evaluators you understand, summarised in a scorecard you can compare across runs.*

### Non-determinism is the floor, not the ceiling

Start with the most fundamental difference. A deterministic function, given the same input, produces the same output every time. `fib(10)` returns `55` on every call, on every machine, in every Python version. An LLM, given the same input, produces *different* outputs across runs — and sometimes substantially different ones. Run the same prompt at the same temperature ten times and you will see ten distinct outputs, most of them similar in spirit, some of them notably different in detail. Drop the temperature to zero and you reduce the variance but you do not eliminate it: providers have their own internal nondeterminism (parallel decode, batching, hardware), and "deterministic" in LLM-API parlance means "less variable," not "the same bytes every time."

This matters for evaluation because a single sample is not the system. Asking "did the model get this right?" is the wrong unit of analysis. The right unit is "what's the distribution of outcomes the model produces on this input, and where does the distribution sit relative to the bar I care about?" A deterministic test runs once and answers yes or no. An LLM eval runs many times — across many inputs, sometimes across many trials per input — and produces a *rate*: 87% pass, mean score 8.2 out of 10, 3 of 20 rows failed. The bar is no longer a boolean; it is a threshold on a distribution.

### No ground truth for prose

The second wall is the absence of an oracle for most LLM outputs. "Is `fib(10) == 55` correct?" has a single right answer that a computer can check in microseconds. "Is this summary of the article good?" has no single right answer, no machine-checkable definition, and no two human reviewers will fully agree on it. Two well-written summaries can disagree on what to include, what to emphasise, and what to leave out — and both can be objectively *fine*. The grader's rubric is fuzzy; the reviewer's mood matters; the cultural context shapes what counts as "good."

That fuzziness is not a defect of evaluation; it is a property of the artifact. There is no canonical English summary of any non-trivial article. There is no canonical translation between two languages. There is no canonical reply to a customer complaint. Evaluation for these artifacts has to live with the absence of a hard oracle and substitute *proxies* — rubrics, paired comparisons, similarity to a reference, agreement with a panel — none of which give you the clean pass/fail signal of `assert`. Section 5 covers the modern proxy for prose: LLM-as-judge. The whole reason the section exists is that the runtime is no longer the critic.

### Prompts are brittle in ways that defy intuition

A change you would call cosmetic — "let's add the word *carefully* to the system prompt" — can shift behavior measurably. Reordering two bullets in the instructions can change which kind of output the model favors. Adding a single example to the few-shot block can reduce the variance on edge cases and have no effect on the easy ones. Removing one trailing newline can, occasionally and inexplicably, make a JSON-mode model produce malformed JSON.

The brittleness is not random — it is a consequence of how the model interprets context — but it is dense, hard to predict, and impossible to reason about from the prompt text alone. The only way to know whether a prompt change is an improvement is to run it against a dataset and compare. *"It feels better"* is not a method. *"My three favorite test cases passed"* is not a method either, because those three cases are not the system. The method is to fix the dataset, fix the evaluators, run before and after, and read the numbers.

### Drift between model versions

Providers update their models. Sometimes the update is announced with a new model ID (`gpt-4` → `gpt-4-turbo`), sometimes it is a silent revision under the same ID, sometimes it is a routing change in their inference fleet that nobody documents. Whatever the mechanism, the model you are calling today is not guaranteed to be the same model you were calling six months ago, even if your code says `model="anthropic/claude-sonnet-4-20250514"`.

The implication is that a system you shipped six months ago, against a model that behaved a certain way then, may behave differently today. The scorecard from launch and the scorecard from this morning would tell different stories. Without a dataset, evaluators, and scorecards committed to your repo, *you cannot detect this drift*. The system seems to work because the things you happen to try still work; the long tail of inputs you never personally re-tested may have quietly degraded. Section 7 returns to this with a scheduled-re-run pattern.

### The vibes-iteration trap

This is the failure mode worth naming explicitly, because it is the one most learners will recognise. You change the prompt. You try three examples. The three examples feel better — the tone is more confident, the formatting is cleaner, the answer addresses the question more directly. You ship the change. A week later, you notice the system is doing strange things on inputs you didn't try. You go back and find that, on a representative dataset, the change actually *regressed* on rows you cared about. The three examples felt better because you cherry-picked the ones you cared about most; the aggregate told a different story.

Module 14 introduced a milder version of this trap (the visible-tests-only pitfall in code generation). This module is how you escape the trap in general. The escape route is not "try harder to find the right three examples." The escape route is to maintain a real dataset, fix evaluators that measure properties you care about, and run them before every prompt change. The numbers will sometimes disagree with your gut. The numbers are right and your gut is wrong, because your gut sampled three inputs and the numbers sampled the distribution.

### Deterministic code: run once, get a bit. LLM systems: run on a distribution, get a distribution

The contrast is worth stating cleanly. In deterministic software, you write a test, you run the test, you read the bit it returns. Pass or fail. The bit is what testing produces, and the testing toolchain (pytest, JUnit, Mocha) is shaped around producing that bit cheaply, in CI, on every commit.

In an LLM system, you write an eval dataset — many rows, each with an input and (sometimes) an expected output. You run the system on every row. You run a list of evaluators on every result. You aggregate. What you get back is not a bit; it is a *scorecard*: per-evaluator pass rates, per-row outcomes, total cost, total latency, a `run_id` you can compare against the last run. The toolchain (Promptfoo, Phoenix, the harness in this module's project) is shaped around producing that scorecard cheaply, comparing it across runs, and surfacing what changed.

The mental shift is from *test, get a bit* to *eval, get a distribution.* Once you internalize that shift, the rest of the module is mechanical. Datasets are how you bound the distribution. Evaluators are how you summarise per-row outcomes. Scorecards are how you make distributions comparable. The discipline is everything traditional software engineering already does — committed test files, CI runs, regression detection — but applied to a regime where the unit of work is a distribution rather than a single assertion.

### Why this discipline didn't exist five years ago

It's worth pausing on the historical accident that put us here. Five years ago, evaluating an LLM was an academic activity. The community had benchmarks — MMLU, HumanEval, GSM8K — but those were benchmarks for *model capability*, not for *production system quality*. The benchmarks lived in research papers, the systems lived in production, and the gap between them was so wide that most production systems didn't have systematic evaluation at all. People shipped on vibes because the tools to ship on data didn't exist.

What changed is the cost structure of LLM inference and the maturity of the prompt-engineering practice. Cheaper inference made it economical to run thousands of evaluation calls per prompt change. Better libraries (LiteLLM, the LangChain eval modules, Promptfoo, this module's harness pattern) made the engineering trivial. And practitioners learned — sometimes painfully — that shipping LLM changes without measurement is shipping bugs you cannot see. The discipline you're learning now is two or three years old in most teams. Many production systems still don't have it. The teams that build it earliest end up with a substantial operational advantage: they catch regressions before users do, they iterate on prompts with confidence, and their deployments stop being magic incantations and start being engineering changes with predictable outcomes.

This module is the bridge between "I know how to use LLMs" and "I know how to operate LLM systems." Phase 4 is what comes after the bridge — production observability, online evaluation, A/B testing — but none of it works without the offline eval discipline first. Treat this module as the foundational practice that the rest of your LLM engineering career sits on top of.

---

## 2. The Eval Loop

The eval loop is the single most important diagram in this module. Once you see its shape, every eval framework you encounter — Promptfoo, Phoenix, Ragas, DeepEval, the harness in this module's project — will look like a variation on the same theme. Three ingredients go in (dataset, system-under-test, evaluators); one artifact comes out (scorecard); the loop closes when you change the system, re-run, and compare.

### The pipeline shape

```text
┌─────────────────────────────────────────────────────────────────────┐
│                          EVAL LOOP                                   │
│                                                                     │
│   ┌────────────────┐                                                │
│   │  Eval dataset  │   N rows of (input, expected, metadata)        │
│   └────────────────┘                                                │
│           │                                                         │
│           ▼                                                         │
│   ┌──────────────────────────┐                                      │
│   │  Run SUT on each row     │   parallel fan-out (ThreadPool)      │
│   │  (system-under-test)     │   one LLM call per row               │
│   └──────────────────────────┘                                      │
│           │                                                         │
│           ▼ N (row, actual) pairs                                   │
│   ┌──────────────────────────┐                                      │
│   │  Per-row evaluators      │   sequential per row                 │
│   │  exact_match | schema |  │   each evaluator returns             │
│   │  llm_judge   | ...       │   EvalResult(passed, score, reason)  │
│   └──────────────────────────┘                                      │
│           │                                                         │
│           ▼                                                         │
│   ┌──────────────────────────┐                                      │
│   │  Aggregate               │   per-evaluator pass rate            │
│   │                          │   overall pass rate (AND across      │
│   │                          │   evaluators per row)                │
│   └──────────────────────────┘                                      │
│           │                                                         │
│           ▼                                                         │
│   ┌──────────────────────────┐                                      │
│   │  Scorecard               │   console + JSON file                │
│   │  run_id, timestamp,      │   diffable across runs               │
│   │  fingerprints, totals    │                                      │
│   └──────────────────────────┘                                      │
│                                                                     │
│   Change prompt/model ──► re-run ──► new scorecard ──► diff         │
└─────────────────────────────────────────────────────────────────────┘
```

Read the diagram top-to-bottom. The dataset is fixed input — a JSONL file in the project, but it could be a database table, a remote API, or a stream sampled from production. The system-under-test (SUT) is the thing you're evaluating: a callable that takes a row's input and returns an output. The evaluators are a list of objects with a uniform `.evaluate(input, expected, actual)` interface; each one looks at the row's output and decides whether it passes its own criterion. The scorecard is the aggregate — per-evaluator pass rates, per-row outcomes, total cost, total latency — written to disk as JSON and printed to the console.

### Each piece in one sentence

**Dataset.** A list of rows. Each row has an `input` (what the SUT receives), usually an `expected` (the labeled ground truth — what good output looks like), and `metadata` (anything else: row id, difficulty tag, source). The dataset is *fixed*: same rows, same expected labels, across all runs of a given eval. If you change the dataset, you are running a different eval, not the same eval on a different system.

**System-under-test.** A callable. Given an `input`, it returns an `actual` — the output you want to evaluate. In this module's project, the SUT is a small sentiment classifier (LiteLLM call → JSON sentiment label). In a real eval, the SUT could be any LLM-powered function: a summariser, a tool-using agent, a multi-step chain, a RAG pipeline. The eval harness doesn't care; it just calls the SUT and captures what comes back.

**Evaluators.** A list of objects with a `.evaluate(input, expected, actual) -> EvalResult` method. Each one looks at one property of the output. `ExactMatchEvaluator` checks that `actual["sentiment"]` equals `expected["sentiment"]`. `SchemaEvaluator` checks that `actual` validates against a Pydantic model. `LLMJudgeEvaluator` asks a stronger LLM to rate the output against a rubric. Each evaluator returns an `EvalResult` with a `passed` flag, a normalized `score`, and a human-readable `reason`.

**Aggregate.** A small set of functions that turn N per-row results into a summary. Per-evaluator: pass count, total, pass rate, mean score. Per-row: "did all evaluators pass for this row?" Per-run: total cost (SUT cost + evaluator cost), total wall-clock latency, overall pass rate.

**Scorecard.** The aggregate plus per-row detail plus metadata (run id, timestamp, dataset path, SUT identifier, model, concurrency). Written to disk as a Pydantic JSON dump so you can diff one run against another later. Printed to the console as a human-readable table.

### The eval loop IS the development loop

Once you have a dataset and a set of evaluators committed to your repo, the eval loop becomes the development loop for prompt-based systems. You change a prompt. You run `python solution.py --dataset datasets/sentiment.jsonl`. You read the scorecard. Did the pass rate go up? Did any rows flip from pass to fail? Did the cost go up enough to matter? You decide whether to keep the change.

This is exactly the role `pytest` plays for traditional code. Without `pytest`, you change the code, you run a few examples manually, you tell yourself it looks fine, you ship it. The bugs you didn't try come back to bite you a week later. With `pytest`, you change the code, you run the test suite, you read the report, you decide based on what the report says. The discipline is the same: codify what you care about, run the codification automatically, trust the report over your gut.

The analogy holds tightly enough that "the eval loop is `pytest` for LLM systems" is a useful one-liner. The differences are real but secondary: an eval is slower than a unit test (because each row makes an LLM call), more expensive (because LLM calls cost money), and noisier (because LLM calls return distributions, not bits). The shape — codified criteria, automated execution, report-based decisions — is identical.

### The eval pipeline is a workflow

Look back at the diagram. Dataset → fan-out to SUT calls → fan-in to per-row evaluator sequences → fan-in to aggregate → scorecard. That is exactly the workflow shape from [Module 13 (Workflows & Chains)](../13-workflows-chains/): a sequential chain of steps, with a parallel fan-out inside one of the steps, with each step's output feeding the next step's input via typed Pydantic models. The harness in this module's project is a workflow with three steps (SUT runs, evaluator runs, aggregation) and a `ThreadPoolExecutor` parallelizing the first step.

The workflow framing matters because it means the eval pipeline inherits all of Module 13's discipline. Each step has typed inputs and outputs. Each step is independently testable. The orchestrator wires them together in code you wrote, not in code the LLM decided to invoke. The pipeline cannot grow new steps at runtime; cannot exit through a path you didn't anticipate; cannot hide cost or latency from the per-step accounting. *The eval pipeline is a workflow, full stop.*

### Cross-link: the iterate-on-failure loop is the same shape

The other piece worth pinning down is the relationship to [Module 14 (AI Code Generation)](../14-ai-code-generation/). Module 14's iterate-on-failure loop is: generate code, run tests, on fail feed the error back, generate again. That is an eval loop with one row, one evaluator (the test runner), and a re-generation step bolted on. Module 14 was a degenerate case where the "dataset" was a single spec, the "evaluator" was binary (tests pass / tests fail), and the response to a fail was to call the SUT again with revision feedback. This module generalises that shape: many rows, many evaluators, none of them necessarily binary, with the response to fails being a *scorecard* you read rather than an automatic regeneration.

Code generation got to have a runtime critic because code happens to be executable. The rest of the LLM design space rarely has that luxury. This module is what evaluation looks like when the critic is not the runtime — when the critic is a list of evaluators you write, mechanical where you can, LLM-as-judge where you must.

---

## 3. Eval Datasets: Where They Come From

The dataset is the foundation. Everything downstream — the evaluators, the scorecard, the cross-run comparisons — depends on having a dataset that actually represents the inputs your system will face. A perfect harness running on a bad dataset produces confident numbers that mean nothing. Three sources dominate in practice, each with distinct tradeoffs, and most real production systems mix all three.

### Golden curated

A *golden* dataset is one a human assembled by hand: each input was picked deliberately, each expected output was labeled deliberately, and the curator could explain why every row is in there. The classical version is a spreadsheet a domain expert sat with for an afternoon and filled in. Golden rows are slow to produce — a careful labeler might do 50 rows in an hour for an easy task, 10 rows in an hour for a hard one — but the *signal* per row is as high as it gets. Each row was chosen because it tested something the curator cared about; each label was chosen because the curator was willing to defend it.

Golden datasets are where you encode the cases that *matter most*. The edge case that broke production last quarter. The prompt that the legal team flagged. The category of input where the model has historically been unreliable. A golden set of 50–200 rows is often enough to give you confidence on the cases you can name, and it doubles as documentation of what you've decided "good behavior" looks like on those cases. The cost is the human time; the value is that the bar is set by a human who knows the domain.

The risk with golden datasets is *blindness to what you didn't think to include*. If your curator never thought about non-English inputs, the golden set has no non-English rows, and a model that quietly broke non-English handling will pass the golden eval with flying colors. Golden coverage is bounded by the curator's imagination, which is bounded by their experience. The mitigation is to combine golden with the other two sources below.

### Synthetic

A *synthetic* dataset is one an LLM generated for you, usually from a small seed of curated examples. You write a meta-prompt that says "here are five rows of valid eval data; generate fifty more in the same format, with this distribution of difficulty and this distribution of input style," and a strong model produces the rows. You spot-check a sample, you reject the obvious failures, and you keep what's left. Total time: an hour to write the meta-prompt, plus however long the LLM takes to generate.

Synthetic datasets win on speed and breadth. You can generate a thousand rows in the time it takes a human curator to produce ten, and you can cover input styles a human curator wouldn't have thought to write. They are particularly good for *stress-testing coverage*: vary the input length, the input language, the input formality, the input domain, and see whether the SUT holds up across the variation. They are also good for *cold-starting* a new eval — you don't have production traffic yet, you don't have time to curate by hand, but you want something to evaluate against now.

The cost is signal quality. The synthetic generator's biases leak into the dataset. Whatever the generator considers a "typical" input will be over-represented. Whatever the generator finds hard to imagine will be missing. If the generator and the SUT come from the same model family, you risk evaluating the SUT against rows it secretly knows how to handle (because both models share the same training-data quirks). Use a different model for synthesis than for SUT where possible. Treat synthetic rows as *plausible coverage* rather than *real coverage*, and never let synthetic rows be the entire dataset.

### Production traffic

A *production-traffic* dataset is sampled from real usage. Real users asked your system real questions; you logged the inputs (and ideally the outputs and their downstream signals: thumbs-up, did-they-retry, did-they-edit-the-output); you sample a subset, label it, and add it to your eval set. This is the most *representative* dataset of the three — it is literally the distribution your system faces — and it is the only one that reflects real-world input drift over time.

Production traffic is the eval source that mature systems lean on most. It catches the user behaviors no curator would have predicted ("oh — people are pasting in stack traces and asking for explanations"), the long-tail input shapes ("there are forty-two different ways our users format dates"), and the silent shifts in distribution ("the median input length doubled after we changed the onboarding"). For a system that's been live for a while, production traffic is the most honest signal about what the system actually has to do.

The cost is labeling. Production traffic comes in unlabeled — you have the input, but you don't have an `expected` field. You have to *label* the rows before they're usable as eval data, which means a human (or a strong LLM, or both) decides what the right output should have been. Labeling tooling matters a lot here: a queue of unlabeled rows, a labeler UI, a way to track inter-annotator agreement, a way to flag rows where labelers disagreed. Production-traffic datasets also require a *sampling strategy*: pure random sampling tends to over-represent the easy cases, so stratified sampling (over difficulty buckets) or adversarial sampling (over rows the system seemed to struggle with) is often better.

### Labeling, agreement, and borderline cases

Labels are not free even when they look free. Two labelers labeling the same row will disagree on a substantial fraction of any non-trivial task. *Inter-annotator agreement* — the rate at which two labelers agree on the same row — is the headline number for label quality, and on subjective tasks (tone, helpfulness, "is this summary good?") it often sits in the 70–85% range even with detailed rubrics. That ceiling is the upper bound on what any evaluator, mechanical or LLM-based, can hope to achieve: you cannot evaluate above the agreement of your labelers.

Borderline rows — the ones two labelers split on — deserve special treatment. A useful pattern is to maintain a `disagreement_set` separate from the main eval: rows where labelers consistently disagreed go into the disagreement set rather than being forced into a pass/fail label. The main eval is graded normally; the disagreement set is reported separately as "rows where the human bar is itself uncertain." A model that handles disagreement-set rows well is doing something interesting; a model that fails them might just be picking the other labeler's side.

Label *noise* — labelers being wrong, not just disagreeing — sets a different ceiling. If 5% of your labels are flat-out wrong (mistype, copy-paste mistake, momentary confusion), then 5% of "failures" in any eval are noise rather than signal. You will not improve below the noise floor by improving the SUT; you can only improve below it by improving the labels.

### Dataset size: 20 / 100 / 1000

A useful set of thresholds:

- **20 rows** — a smoke test. Enough to catch catastrophic failures (the system is broken; the prompt doesn't work). Not enough for a stable rate; the per-evaluator pass rates will swing by 5–10 percentage points run-to-run from sampling noise alone. This module's bundled dataset is 20 rows — small enough to read end-to-end, big enough to make the harness do real work, small enough to run on every prompt change without thinking about cost.
- **100+ rows** — a meaningful scorecard. Pass-rate swings drop to 2–5 percentage points; you can plausibly tell whether a prompt change moved the number or was within noise. This is the typical size for a small product's main eval set, and it usually represents 1–3 days of curator effort or a few hours of synthetic generation plus spot-checking.
- **1000+ rows** — a stable signal under noise. Pass-rate swings drop below 1 percentage point; you can reliably detect small regressions and small improvements. Cost and latency both grow proportionally, so 1000-row evals usually run on a schedule rather than on every prompt change. For high-stakes systems (anything user-facing, anything regulated, anything that handles money), the 1000-row tier is the right target eventually.

You don't have to start at the 1000-row tier. Start at 20 to build the habit; scale to 100 when the eval is part of your workflow; scale to 1000 when the cost of a regression demands it. *Some* eval is overwhelmingly better than none. Module 15's project is the 20-row tier; you scale up by adding rows, not by changing the harness.

### Composition: golden + synthetic + production, mixed

The right answer for a mature system is not "pick one source"; it is "mix all three, deliberately." A practical composition that works for production systems:

- **30–40% golden curated rows** at the core. These are the cases you care about most. They are the contract — the bar you commit to. They never change unless you deliberately decide to change them.
- **30–40% synthetic rows** for coverage. These extend the golden set to input shapes the curator did not think of. They are refreshed periodically (every quarter, or whenever the underlying system shifts) because synthetic biases are a moving target.
- **20–40% production-sampled rows** for representativeness. These are the long tail — the inputs real users send that no curator would have written. They are the freshest layer; they get updated every cycle.

The exact proportions depend on the system. RAG over a fixed corpus leans heavier on golden (because you know the corpus); a chat product with open-ended user inputs leans heavier on production-sampled (because you cannot anticipate the distribution). The principle is composition: each source covers what the others miss, and the eval set is *all three* rather than any one of them.

Maintaining the composition is itself a workflow. Every quarter, review the production-sampled subset and rotate in fresh rows; every six months, audit the golden subset for relevance (cases that no longer matter get retired); every year, regenerate the synthetic subset against the latest model so its biases reflect current reality. The eval set is not a static artifact; it is a living one that grows and shifts alongside the system it evaluates.

---

## 4. Mechanical Evaluators

A *mechanical* evaluator is one that decides pass/fail without calling an LLM. It is a Python function that reads `expected` and `actual`, compares them with deterministic logic, and returns a result. Mechanical evaluators are the cheapest, fastest, and most reliable kind, and they should always be your first line of evaluation — for the simple reason that they cost microseconds and zero dollars, while LLM-as-judge costs seconds and real money per call.

### Five flavors

**Exact match.** `actual[field] == expected[field]`. The simplest evaluator possible. Pass if the two values are identical, fail otherwise. Use for fields with a small enumerated set of valid values (a category label, a yes/no flag, a fixed string). The project's `ExactMatchEvaluator` is this evaluator, generalised to compare any named field of `expected` against the same field of `actual`.

**Normalized exact match.** `actual[field].lower().strip() == expected[field].lower().strip()`. Same as exact match, but the values are normalized first — lowercased, whitespace stripped, sometimes punctuation removed. Use for free-text fields where you care about content but not casing or surrounding whitespace. In practice, normalized exact match is what people *mean* when they say "exact match" — the literal byte-equal check fails too often on trivial differences.

**Schema validation.** Run the actual output against a Pydantic model (or any schema validator). Pass if it validates, fail if it doesn't. The failure reason is the validation error. This is the same pattern from [Module 08 (Structured Output)](../08-structured-output/), now reframed as an evaluator: "did the model produce output that matches the contract?" Schema validation is a free, deterministic check on the *shape* of the output, independent of whether the contents are correct. It catches the failure mode where the model returns prose instead of JSON, returns JSON with the wrong keys, or returns a value out of the allowed range.

**Regex match.** `re.match(pattern, actual[field])`. Pass if the pattern matches. Use for outputs that should follow a known format — phone numbers, dates, identifiers, URLs. A regex is a more permissive check than exact match (any number of strings can match the same pattern) but a stricter check than schema validation (the schema accepts any string; the regex accepts only strings of a particular shape).

**Set / list compare (order-insensitive).** `set(actual[field]) == set(expected[field])`. Pass if the two collections contain the same elements, regardless of order. Use for fields where the model is asked to produce a collection (a list of tags, a set of categories, a list of extracted entities) and order is incidental. The naive `actual[field] == expected[field]` would fail for the same-contents-different-order case, which is usually not what you want.

### A short example

The implementation is so small it's worth showing inline. The project's `ExactMatchEvaluator` is roughly this:

```python
class ExactMatchEvaluator:
    name = "exact_match"

    def __init__(self, field: str):
        self.field = field

    def evaluate(self, input_value, expected, actual) -> EvalResult:
        a = str(actual.get(self.field, "")).strip().lower()
        e = str(expected.get(self.field, "")).strip().lower()
        passed = a == e
        return EvalResult(
            evaluator_name=self.name,
            passed=passed,
            score=1.0 if passed else 0.0,
            reason=f"actual={a!r} expected={e!r}",
            latency_ms=0,
            cost=0.0,
        )
```

That is the entire evaluator. Ten lines, one comparison, a normalized-exact-match check on a single named field. It runs in microseconds, costs nothing, and produces a structured `EvalResult` the aggregator can summarise. A schema evaluator, a regex evaluator, and a set-compare evaluator each look about the same — a class with a `name`, a constructor that holds the configuration, and an `evaluate` method that returns an `EvalResult`. Adding a fifth evaluator is a copy-paste of the shape with a new comparison inside.

### What mechanical evaluators catch

The catch list is short and reliable:

- The model returned the wrong category when the label was binary or small-enumerated.
- The model returned malformed JSON when the schema required a specific shape.
- The model returned a string in the wrong format (a date that isn't a date, a phone number with letters).
- The model returned a collection with the right elements but the wrong order (or the wrong elements regardless of order).
- The model returned a value outside the allowed range (`confidence: 1.7` when the schema said `0.0 <= confidence <= 1.0`).

These are the failures that should never reach a human reviewer or an LLM judge. They are mechanical, repetitive, and obvious. Catching them with a mechanical evaluator is the eval-loop equivalent of a static type check: cheap, automated, and the right place to surface the failure.

### What they miss

The miss list is the more interesting one. Mechanical evaluators only check the properties you wrote down. They miss:

- **Technically correct but useless outputs.** The model returns the right category for the wrong reason. The label is "positive" because the review contains the word "great" and the model latched onto that word, ignoring the surrounding sarcasm. The label matches expected; mechanical evaluator passes; you'd never know without inspecting reasoning.
- **Plausible-looking failures.** The model returned text that *looks* like valid output but encodes the wrong meaning. The summary is grammatical and on-topic but contradicts the source document. The translation is fluent but wrong. Mechanical exact-match doesn't apply (these are open-ended outputs); schema validation passes (the shape is fine); the failure is invisible to mechanical evaluators.
- **Style and tone violations.** The model wrote the email in a confrontational tone instead of a diplomatic one. The summary used profanity inappropriate for the audience. The reply was correct but rude. Style is subjective; no mechanical evaluator can check it.
- **Subtle reasoning errors.** The model arrived at the right answer through wrong reasoning. The math checks out by accident. The classification matches because of a confound. Mechanical evaluators see only the final answer; they cannot see the reasoning path.

The pattern is: mechanical evaluators catch the *easy* failures, miss the *hard* failures, and that is exactly the right division of labor. The hard failures go to the LLM-as-judge in Section 5. The easy failures stay here, where they are cheap.

### Normalization is where most of the value lives

A practical note that bears repeating: most "exact match" failures in production are case or whitespace mismatches, not real content disagreements. The model returned `Positive` instead of `positive`, or ` positive ` with leading whitespace, or `Positive.` with a trailing period. These are not failures of the model; they are failures of the evaluator to normalize.

Always normalize. Lowercase. Strip whitespace. Strip trailing punctuation if the schema allows it. Pre-process both `expected` and `actual` through the same normalizer before comparing. The fix is one line of code and it removes ~half of the false positives in a typical mechanical-evaluator setup. Most "the eval is too strict" complaints from learners boil down to "you forgot to normalize."

### Mechanical evaluators as gatekeepers

A final framing. Think of mechanical evaluators as the *gatekeepers* of the eval pipeline. They run first, they run cheaply, and their job is to weed out the obvious failures so the expensive LLM-judge doesn't have to look at them. A row that fails the schema check is already a failure; there is no point spending $0.01 on an LLM judge to confirm it. A row that fails exact-match on the label is already a failure; the judge would just be reading the same wrong label.

Section 6 returns to this composition (mechanical first, LLM-judge on what survives) as the canonical eval recipe. The split-of-labor only works if the mechanical evaluators are doing their job — catching cheap failures, leaving expensive judgments to the expensive critic. That is what this section's evaluators are designed for.

---

## 5. LLM-as-Judge

When the property you want to measure has no mechanical evaluator — when "good" is subjective, when the output is open-ended prose, when the failure modes are "sensible-looking but wrong" — the option that opens up is *LLM-as-judge*. You write a rubric. You ask a model (usually a different and stronger one than the SUT) to grade the output against the rubric. You read the score. The judge plays the role that a human reviewer would have played, at a fraction of the cost and a fraction of the latency, with all the caveats that come with delegating judgment to a model.

LLM-as-judge is the dominant pattern for evaluating open-ended LLM outputs in 2026. It is also the pattern most people get wrong — by picking the wrong model, by ignoring the biases, by trusting a single judge call when they should be voting across multiple. This section is a tour of the three forms, the biases, the mitigations, and when the cost is worth paying.

### Three forms

**Paired comparison.** Show the judge two outputs (A and B) for the same input, and ask which is better. The judge returns A, B, or tie. Use when you have a baseline output to compare against — for example, "is the new prompt better than the old one on this row?" Paired comparison is the most direct form for measuring change: it does not require an absolute quality scale, only a relative one.

**Rubric scoring.** Show the judge one output and ask it to rate the output against a rubric. The rubric is a short prose document: "Rate this summary on a 0–10 scale. 10 means: complete, accurate, well-organized, no hallucinations. 5 means: mostly accurate but missing important detail or with minor errors. 0 means: factually wrong or off-topic." The judge returns a number and a brief reason. Use when you want an absolute quality score that's comparable across runs, prompts, or systems.

**Single-answer grading.** Show the judge one output and ask a yes/no question about it. "Does this output meet criterion X?" "Is this reply respectful?" "Did the answer hallucinate any facts not in the source?" Use when the property is binary and the judge's main job is to spot the failure mode rather than place the output on a continuum. Single-answer grading is also the cheapest of the three because the judge's output is one token (yes / no) plus a short reason.

The project's `LLMJudgeEvaluator` uses rubric scoring on a 0–10 scale with a configurable pass threshold (default 7). Pass means `score >= threshold`; the raw score and the judge's reason are stored in the `EvalResult` for inspection. Switching to paired comparison or single-answer grading is a prompt change, not an architecture change — the harness shape stays identical.

### Known biases

LLM judges are not impartial. They have predictable, well-documented biases that affect their scores. The three you should know:

**Position bias.** In paired comparison, judges prefer the *first* option they see more often than they should. The effect is small per call but real in aggregate — across many comparisons, the bias toward "A" (or whichever option was presented first) can shift the apparent winner. Studies have measured this at anywhere from 3% to 15% bias toward position 1, depending on the model and the task. *Mitigation:* swap positions and average. Run each comparison twice, once with output P in position A and output Q in position B, once with the positions reversed. Average the scores. The position bias washes out.

**Length bias.** Judges rate longer outputs higher than shorter ones, even when length is not part of the rubric. This is the LLM equivalent of the human bias toward longer essays in standardized tests: there's an implicit assumption that "more text = more thought = more value." The bias can be substantial — a verbose-but-padded answer can score above a concise-but-correct one. *Mitigation:* either include length explicitly in the rubric ("a good answer is concise; deduct points for unnecessary padding") or normalize for length post-hoc by stratifying scores by output length and looking at the trend. Sometimes the simplest fix is to ask the judge to provide a reason and look for length-flavored language in the reasons; if "the answer is thorough" is the most common reason for high scores, length bias is dominating.

**Self-preference bias.** A judge from the same model family as the SUT prefers outputs that *look like* outputs the judge would have written. A Claude-judged Claude output gets higher scores than the same output would get from a GPT judge; the reverse is also true. The bias is real and varies with how similar the two models are. *Mitigation:* judge with a *different* model than the SUT, ideally a stronger one. If your SUT is Claude Sonnet, judge with Claude Opus, or with a strong GPT variant. The cost is one extra model in your API setup; the benefit is a meaningful reduction in self-flattery.

There are other biases — verbosity bias overlaps length bias, formatting bias (judges prefer outputs in the format the judge expects), and ordering effects in rubrics where the order you list criteria affects which criteria the judge weights highest. They are less central than the three above but worth knowing. The general principle: assume the judge has biases, design your judge prompts and evaluation protocols to neutralize the known ones, and treat the judge's output as a noisy signal rather than ground truth.

### Self-consistency tactics

A single judge call on a single output is a noisy measurement. The same judge, called again with the same input, will often produce a slightly different score. Across a small dataset, that per-call noise is non-trivial. The mitigations:

**Vote across multiple judges.** Call the judge three or five times for each output, with temperature > 0 (so the calls differ), and take the median or majority vote. This is the LLM equivalent of an ensemble in classical ML — averaging across independent samples reduces variance. The cost scales linearly with the vote count, so use it for high-stakes evals and skip it for cheap ones.

**Swap positions in paired comparison.** As covered above. The cheapest variance reduction available; should be standard for any paired-comparison eval.

**Anchor with rubric examples.** Show the judge worked examples of what a 10/10 looks like, what a 5/10 looks like, what a 0/10 looks like. The examples calibrate the judge's scale. Without anchors, two judges will use different scales — one runs hot ("most answers are 8/10"), one runs cold ("most answers are 5/10"), and you cannot compare across them. With anchors, the scales align.

**Use a stronger judge than the SUT.** Already covered under self-preference bias, worth repeating. The judge has to be at least as capable as the SUT, ideally more so, or the judge will systematically misjudge cases the SUT got right by being smart in ways the judge can't follow.

**Calibrate against a small golden subset.** Hand-label 10–20 rows with absolute scores you trust. Run the judge on those rows. Look at the disagreement. If the judge is systematically too harsh or too lenient, adjust the threshold; if the judge is wildly inconsistent, the rubric isn't tight enough.

### When LLM-judge is worth the cost

LLM-judge is *expensive* relative to mechanical evaluators. A judge call takes 1–5 seconds and costs $0.001–$0.01 per row depending on the model and prompt length. On a 1000-row eval, that is $1–$10 per run and several minutes of wall-clock latency. On every prompt change, against a dataset you re-run frequently, the cost adds up quickly.

The right answer is to use LLM-judge *only when mechanical evaluators cannot do the job*. The properties where it is worth the cost:

- **Clarity.** Is the output easy to read? Mechanical check: none.
- **Helpfulness.** Did the output actually answer the user's question? Mechanical check: not really, unless the question is multiple-choice.
- **Tone.** Is the output diplomatic / friendly / professional / matching brand voice? Mechanical check: none.
- **Factual coherence on novel claims.** Does the output's factual claims hang together logically, given the source material? Mechanical check: none, unless you have a separate fact-checking model.
- **Reasoning quality.** Did the output arrive at the right answer through correct reasoning? Mechanical check: none, since reasoning is open-ended prose.

When the property *can* be expressed mechanically — schema validation, regex, exact match — use the mechanical evaluator. When it cannot, LLM-judge is the right tool, and the cost is the price of getting a signal you couldn't have gotten any other way.

### A short example

The project's `LLMJudgeEvaluator.evaluate` is roughly this shape:

```python
class LLMJudgeEvaluator:
    name = "llm_judge"

    def __init__(self, rubric: str, model: str, pass_threshold: int = 7):
        self.rubric = rubric
        self.model = model
        self.pass_threshold = pass_threshold

    def evaluate(self, input_value, expected, actual) -> EvalResult:
        prompt = (
            f"Rubric:\n{self.rubric}\n\n"
            f"Input: {input_value}\nExpected: {expected}\nActual: {actual}\n\n"
            "Return JSON: {\"score\": 0-10, \"reason\": \"...\"}."
        )
        response = completion(model=self.model, messages=[{"role": "user", "content": prompt}])
        raw = response.choices[0].message.content
        parsed = json.loads(_strip_code_fence(raw))
        score = int(parsed["score"])
        passed = score >= self.pass_threshold
        return EvalResult(
            evaluator_name=self.name,
            passed=passed,
            score=score / 10.0,
            reason=f"judge={score}/10 — {parsed['reason']}",
            latency_ms=...,
            cost=...,
        )
```

Fifteen lines. One LLM call. JSON parsing with the same fence-stripping helper from Module 14. A pass threshold that turns a 0–10 raw score into a boolean pass/fail. The `EvalResult` it returns has the same shape as every other evaluator's result, so the aggregator does not need to know whether the evaluator was mechanical or LLM-based — it just reads `passed` and `score`.

That uniform interface is the key. Every evaluator is just a `.evaluate(input, expected, actual) -> EvalResult` method. Mechanical or LLM-judge, the harness treats them identically. Adding a fourth evaluator means writing one class with that method. The cost difference between evaluators is hidden inside their `cost` field; the orchestrator just runs them in order and aggregates.

### Chain-of-thought in the rubric

A small detail that meaningfully improves judge quality: ask the judge to *reason before scoring*. The prompt becomes "First, write 2–3 sentences analysing the output against the rubric. Then return the JSON with the score and a one-line reason." The judge's reasoning happens in the visible response, and the score it produces is conditioned on that reasoning rather than coming from nowhere.

This is the same chain-of-thought pattern from [Module 02 (Prompt Engineering)](../02-prompt-engineering/), applied to the judge. The empirical effect is consistent: chain-of-thought judges produce more calibrated scores, are less susceptible to length bias (because their reasoning has to engage with the content rather than the form), and produce reasons that are more useful for debugging when the judge disagrees with the human label. The cost is a few hundred more output tokens per call, which is real but small compared to the calibration improvement.

The project's harness does not enforce chain-of-thought (the rubric is configurable), but the bundled rubric template uses it: the judge produces its analysis first, then the score, in the same JSON response. Inspect the `reason` field of any judge result and you'll see the reasoning that drove the score. The reason field is what makes a judge debuggable; a score with no reason is a number with no story behind it, and that story is exactly what you need when the judge disagrees with what you expected.

### Cost calibration: when the judge gets expensive

A worked example. On a 100-row dataset with one judge evaluator at $0.005 per call, the per-run cost of the judge is $0.50. Run-all (judge on every row) gives you the full signal. Filter-then-judge (mechanical first, judge only on rows that survived) might cut that to $0.40 if 20% of rows fail mechanical evaluators. The savings are real but modest at this scale.

Now scale up. On a 10,000-row dataset, the run-all cost is $50. The filter-then-judge cost is $40 if the mechanical filter rate is 20%, or $25 if it's 50%. The savings are substantial. On a 100,000-row dataset (less common, but real for production eval pipelines that sample from millions of daily calls), the run-all cost is $500 and the savings can be $200–$400.

The right design choice depends on which scale you're at. At the project's 20-row scale, run-all is the obvious default and the cost difference is rounding-error. At the 10,000-row scale, the filter-then-judge optimization is worth the engineering. At the 100,000-row scale, you also start thinking about *sampling* — instead of running the judge on every survivor, run it on a 10% sample of survivors and extrapolate. The harness shape stays the same; the policy on which evaluators run on which rows is what varies.

---

## 6. Combining Mechanical + LLM Evaluators

A real eval pipeline rarely uses just one evaluator. Mechanical evaluators catch the cheap failures; LLM-judge catches the subjective ones; combined, they form a single scorecard that grades each row across multiple properties. This section is about how to combine them — the canonical recipe, the aggregation choices, and the worked example that makes the combination concrete.

### The canonical recipe

Run cheap evaluators first. Run expensive evaluators on what survives, or on every row if you can afford it.

The reason is straightforward: mechanical evaluators take microseconds and cost nothing; LLM-judge takes seconds and costs real money per call. On a 1000-row dataset with a 10% obvious-failure rate, running mechanical-first weeds out 100 rows before the judge runs, saving 100 LLM calls — perhaps $1.00 in API costs and several minutes of wall-clock latency. On larger datasets the savings compound. The order matters.

The two patterns in common use:

**Filter-then-judge.** Mechanical evaluators run first. Rows that pass them all proceed to the LLM-judge. Rows that fail any mechanical evaluator are recorded as failures (with the specific evaluator that failed) and the judge does not run on them. This is the maximum-cost-savings pattern. It is also the pattern that loses signal: you don't get a judge score on the failing rows, so you can't tell whether they were "obviously wrong" or "wrong in a way the judge would have explained interestingly."

**Run-all.** Every evaluator runs on every row, regardless of which other evaluators passed or failed. This is the pattern the project's harness uses. The cost is higher (the judge sees the failing rows too) but the signal is richer: you can see, for every row, which evaluators passed and which failed and why. Over time, the per-evaluator pass rates tell you which evaluators are doing the work — is the schema check catching anything anymore, or has the model been reliably returning valid JSON for the last six months?

For small datasets (the project's 20 rows), run-all is the right default. Total cost is bounded by the dataset size and dominated by the SUT calls, not the evaluators. For larger datasets where every cent matters, filter-then-judge is the cost-conscious choice. The harness can be configured either way by changing the per-row evaluation logic; the project ships with run-all because the learning value is in seeing every evaluator's verdict on every row.

### Per-row pass/fail vs aggregate scoring

Two distinct ways to summarise a row's results:

**Per-row pass/fail (AND across evaluators).** A row passes if *all* of its evaluators passed. Any single failure means the row is a failure. This is the strict interpretation: "did the system get this row right, completely?" The overall pass rate is the fraction of rows that pass every evaluator. Use this when you want a clear binary signal for each row, and when the evaluators are all measuring required properties (the output must validate AND must match the label AND must be coherent).

**Aggregate scoring (weighted sum).** Each evaluator contributes a weighted score to a row's overall score. The row's score is a weighted average; the row "passes" if the weighted score is above a threshold. Use this when the evaluators measure properties of different importance, and when partial credit makes sense (schema validation is required, but a slightly bad reasoning score shouldn't fail the row outright).

The project's harness uses per-row pass/fail for the headline `overall_pass_rate` (all evaluators must pass for the row to count as a pass) AND prints per-evaluator aggregates separately, so you can see both views at once. The scorecard makes the distinction visible: the headline number ("85% overall pass rate") is the strict AND; the per-evaluator table ("schema 100%, exact_match 85%, llm_judge 95%") is the loose decomposition.

Mixing both views is the most informative output. A strict overall pass rate gives you a single number to track across runs; a per-evaluator breakdown shows you *which* evaluators are responsible for the gaps. The two views together let you answer "did I regress?" and "where did I regress?" in the same scorecard.

### The eval pipeline is a workflow

Picking up the thread from Section 2, the combination of mechanical and LLM evaluators makes the workflow shape concrete. The pipeline is:

```text
dataset
  │
  ├──► fan-out to SUT calls (parallel via ThreadPoolExecutor)
  │     │
  │     ▼ N (row, actual) pairs
  │
  └──► for each row: run evaluators in sequence
        │
        ├──► mechanical evaluators (microseconds each)
        ├──► LLM-judge evaluators (seconds each)
        │
        ▼ N row-outcomes (each with list of EvalResults)
  │
  └──► aggregate per-evaluator + overall pass rate
        │
        ▼
       scorecard
```

This is a [Module 13](../13-workflows-chains/) workflow with one parallel fan-out (the SUT calls) and one sequential inner step (the per-row evaluator runs). Each row is independent — the workflow has no cross-row state — and the per-row evaluators are independent of each other within a row. You could parallelize the evaluators too if you wanted; the project's harness keeps them sequential for clarity, accepting that the LLM-judge call is the wall-clock bottleneck per row.

The workflow framing inherits all of Module 13's discipline. Each step has a typed input and a typed output. Each step's cost and latency are captured. The orchestrator wires them in code; no LLM "decides" what comes next. The pipeline cannot grow new steps at runtime; cannot exit through unexpected paths; cannot hide failures from the per-step accounting. *The eval pipeline is a workflow, end-to-end.*

### LLM-judge IS the critic agent pattern

Cross-link to [Module 12 (Multi-Agent Systems)](../12-multi-agent-systems/). The critic agent in Module 12 was an LLM that read another agent's output and produced a structured critique against a rubric. The LLM-judge in this module is the same pattern, with two differences: there is no generator-critic loop (the judge produces a final verdict, not feedback for revision), and the judge runs across a whole dataset rather than on a single artifact.

The pattern reuse is deliberate. Once you understand the writer/critic loop from Module 12, the LLM-judge is recognisable instantly — same prompt shape, same model selection considerations (judge stronger than writer), same self-consistency tactics (vote across calls, swap positions, anchor with examples). The eval harness is *the critic pattern at dataset scale*, with the loop removed and the dataset added.

That reuse means the engineering investment in Module 12 pays off here. Your existing critic prompts can become eval-judge prompts with minor edits. The harness in this module's project would work if you swapped in a Module 12 critic class — the `.evaluate(input, expected, actual) -> EvalResult` interface accepts any callable that produces a structured verdict. The shape generalises: critics and judges are the same thing wearing different hats.

### Worked example: a 20-row sentiment task

Make it concrete. The project's bundled SUT is a sentiment classifier: given a movie review, return `{"sentiment": "positive" | "negative" | "neutral", "confidence": float}`. The dataset is 20 movie reviews with labeled sentiments. Three evaluators run on every row:

- `ExactMatchEvaluator(field="sentiment")` — does the predicted sentiment match the label?
- `SchemaEvaluator(schema=SentimentLabel)` — does the output validate against the Pydantic schema?
- `LLMJudgeEvaluator(rubric=...)` — given the review, the label, and the predicted output, does the prediction reflect well-grounded sentiment analysis?

The scorecard might look like this:

```text
Aggregates:
  exact_match    17/20 pass (85.0%)
  schema         20/20 pass (100.0%)
  llm_judge      19/20 pass (95.0%, mean 8.4/10)

Overall pass rate (all-must-pass): 85.0% (17/20)
```

Now decompose the failures. Three rows failed `exact_match`:

- Row r03: a mixed review where the label was "positive" but the model predicted "neutral." The reviewer's tone was hedged ("it was fine, I guess"), and the model picked the safer middle option. `exact_match` fails, `schema` passes, `llm_judge` gives a 6/10 — it agrees that the output is reasonable but not quite right. A *correct* failure caught by both mechanical and LLM evaluators.
- Row r12: a similar mixed review, model again picked "neutral" where the label was "negative." Same pattern. `exact_match` fails, `schema` passes, `llm_judge` gives a 7/10 — borderline pass on the judge's scale, since the model's read of the review is defensible even if the label disagrees. The label might be the one to revisit, not the model.
- Row r15: a sarcastic review where the label was "negative" and the model predicted "positive." Surface-level positive words ("amazing," "incredible") with sarcastic framing. `exact_match` fails, `schema` passes, `llm_judge` fails (3/10 — "model missed the sarcasm; the review is clearly negative"). The judge catches what the schema cannot.

This is the value of running all three evaluators on every row. Exact-match alone would say "3 failures." Schema alone would say "0 failures." LLM-judge alone would say "1 failure." Together they say: "3 failures total, 2 of which are mixed-review judgment calls, 1 of which is a sarcasm-handling gap that's worth fixing." The decomposed view is what drives action — the mixed-review failures might be label noise; the sarcasm failure is a real product issue.

This is also where Section 5's "correct label but bad reasoning" promise becomes visible. The judge would have caught a sarcastic-but-positive-label case that the exact-match accidentally passed — the eval would have shown `exact_match` passing on a row where `llm_judge` failed, flagging it for review. Running both evaluators on every row is what surfaces that pattern.

### Aggregation in practice: the scorecard's pivot tables

A subtle benefit of capturing per-row, per-evaluator results is that the scorecard becomes a pivot table you can slice by any axis. The project's `Scorecard` Pydantic model holds the raw data; aggregation is a thin layer on top. Once you have the raw data, you can ask questions the original aggregation didn't anticipate:

- **By difficulty.** If the dataset's metadata tags each row as `easy` / `hard`, you can compute per-difficulty pass rates. A system that scores 95% on easy rows and 60% on hard rows tells a different story than one that scores 80% on both.
- **By evaluator combination.** Which rows failed only `exact_match`? Which failed only `llm_judge`? Which failed both? The intersection of failures is where the most actionable insights live — rows that failed every evaluator are unambiguous failures; rows that failed only one evaluator are interesting because they reveal what each evaluator can and cannot see.
- **By cost band.** Sort rows by SUT cost (longer inputs cost more). Is the model getting worse on long inputs? A pattern of low-cost-passing / high-cost-failing rows points at a context-length issue or a token-budget issue, not a quality issue with the prompt.
- **By time.** If you run the eval daily, you can plot the per-evaluator pass rate over time. The graph tells you about silent drift, slow regressions, and changes that helped on one axis at the cost of another.

The project's harness produces a single console scorecard plus a JSON file; the JSON file is the input to whatever pivot you want to do. A 50-line Python script that reads the JSON and produces a difficulty-stratified table is a natural next exercise; a notebook that plots pass-rate time series across many scorecards is a natural one after that. The scorecard is the primitive; the analyses are downstream.

---

## 7. Regression Detection and the Iteration Loop

A scorecard from a single run is useful. A scorecard you can compare against last week's scorecard is *transformative*. Once you have two runs in the same format, you can diff them, see which rows flipped, see which evaluators moved, and answer the question that motivates most prompt-engineering work: *did this change help?* This section is about turning single-run evals into a regression-detection workflow.

### Versioning every run

The first move is to commit to versioning. Every run gets:

- A `run_id` — a unique identifier, typically a timestamp-based string like `scorecard_20260413_142103` so it sorts chronologically.
- A `timestamp` — ISO 8601 string of when the run started.
- A `dataset_path` — the dataset file used, so you know which inputs were tested.
- A `sut` field — the module:callable spec of the system-under-test, so you know what was evaluated.
- A `model` field — the underlying LLM model, in case it changed between runs.
- A `concurrency` field — the parallelism setting, for reproducibility of wall-clock numbers.

In the project, all of these are fields on the `Scorecard` Pydantic model. When the harness writes `results-{run_id}.json` at the end of a run, every one of these fields is in the file. You can read the file weeks later and know exactly what was tested, against which inputs, with which model.

For more rigorous setups, add *fingerprints*: a hash of the prompt text, a hash of the dataset contents, a hash of the SUT source code. Fingerprints let you detect "is this run actually comparable to that run?" — if the prompt fingerprint changed, the runs are not comparable on prompt quality; if the dataset fingerprint changed, the runs are not comparable at all. The project's harness does not fingerprint by default (the spec called for a minimal scorecard), but adding it is a small extension and a worthwhile one for any system that's been iterated on for a while.

### Comparing scorecards

With two runs in hand, the comparison is mechanical. The dimensions worth looking at:

**Per-evaluator pass rates.** Did `exact_match` go up or down? Did `llm_judge` move? A change in one evaluator's pass rate while others held steady tells you what kind of change the prompt edit produced. `exact_match` up + `llm_judge` flat probably means the model is getting more labels right but the underlying reasoning quality is unchanged. `exact_match` flat + `llm_judge` up means the model is producing similar labels but with better justification.

**Overall pass rate.** The headline number. Up is good, down is bad. A drop of more than the dataset's noise floor (3–5 percentage points on a 20-row set, 1–2 percentage points on a 100-row set) is worth investigating.

**Flipped rows.** The single most actionable comparison. Walk through the rows in both runs and identify rows that passed in run A and failed in run B, or vice versa. Flipped-to-fail rows are *regressions* — your change broke something specific. Flipped-to-pass rows are *fixes* — your change addressed something specific. Both lists are short (typically a handful out of dozens of rows), and both are read row-by-row.

**Total cost and latency.** The change cost you something. Did it slow down the system? Did it cost more in tokens (a longer prompt, more retries)? A pass-rate improvement that costs 3x in tokens is a real tradeoff to evaluate.

The project's JSON files are designed to be diffable. Two scorecard JSON files of the same dataset have the same shape, the same row IDs, and the same evaluator names, so a `jq` script or a small Python comparison program can produce a diff that highlights all four dimensions above. Comparison tooling is a natural extension of the project — the harness does not ship with a `compare` subcommand, but writing one is roughly 50 lines of Python (load A, load B, walk rows, emit a flipped-rows table). It is the obvious second project for a learner who wants to push the eval discipline further.

### The "did this prompt change help?" workflow

The canonical workflow for a single prompt iteration:

1. **Run baseline.** `python solution.py --dataset datasets/sentiment.jsonl --save results-baseline.json`. Read the scorecard. Note the overall pass rate, the per-evaluator breakdown, the failing rows.
2. **Make the change.** Edit the prompt (or the model, or the temperature, or whatever variable you're testing).
3. **Run candidate.** `python solution.py --dataset datasets/sentiment.jsonl --save results-candidate.json`.
4. **Compare.** Diff `results-baseline.json` against `results-candidate.json`. Look at per-evaluator pass rates, overall pass rate, flipped rows.
5. **Decide.** Keep the change if it improved the numbers without introducing regressions. Revert if it regressed. Investigate further if the result is mixed (some evaluators up, some down).

This is the same shape as the test-driven-development loop for traditional code, with `pytest` replaced by `python solution.py` and the per-row diff replaced by per-row scorecard comparison. The discipline of "make a change, run the suite, look at the report" is identical; only the suite and the report are different.

### How often to re-run

The cadence has three layers:

**On every prompt change.** This is the always-on layer. Every time you edit the system prompt, the few-shot examples, the rubric, the chain-of-thought instructions — any input to the LLM — re-run the eval before you decide whether to keep the change. Without this layer, prompt iteration is vibes-based.

**On every model change.** When you bump the model version, swap providers, or change temperature, re-run the eval. The model is part of the system; changing it changes the system; running the eval verifies the change is an improvement. This is also when fingerprint changes — the model fingerprint shifts, and any cross-run comparison should treat the fingerprint-change as a known confounder.

**On a schedule.** Run the eval against the latest model versions on a cron — daily, weekly, or whatever cadence matches your deployment frequency. The reason is silent drift (Section 1): the provider may update the model under you, and a scheduled re-run catches the resulting score shifts before they hit production. A weekly scheduled run that emails the team when the pass rate moves by more than 5 percentage points is a real-world early warning system.

The three layers serve different purposes. The on-change runs catch your own breakage. The scheduled runs catch the provider's breakage. Together they keep the system's measured quality grounded against time-varying threats.

### What to do when a regression lands

When a re-run shows a regression, the first move is *not* to ship anyway and hope nobody notices. The first move is to investigate. The flipped-to-fail rows are the smoking gun: they are the specific rows that got worse, and reading them in detail will usually reveal the root cause.

Three responses, in order of severity:

**Revert.** If the regression is meaningful (more flipped-to-fail than flipped-to-pass; overall pass rate dropped), revert the change. The eval found the bug your gut would have missed. Trust the eval; revert; pick a different change to try.

**Dig into the flipped rows.** Sometimes the regression is real but the flipped rows reveal a coverage gap, not a quality drop. A row that flipped because it was an edge case the model now mis-handles is a real regression. A row that flipped because the *expected* label was wrong (label noise) is not a regression in the model — it is a regression in your dataset's quality. Inspect each flipped row; categorize it; act accordingly.

**Add flipped rows to the golden dataset.** If the regression revealed a real coverage gap — a class of input the model used to handle by accident and now mis-handles — adding those rows (with carefully-labeled expected outputs) to your golden dataset is the right move. The next time you iterate on the prompt, the eval will know to watch for these rows. The dataset gets stronger over time; future regressions are caught earlier.

The fourth option — *increase max attempts, lower the bar, accept the regression because the change was important for other reasons* — is sometimes the right call, but it should be a deliberate decision based on the numbers in front of you, not an unconscious slide. The whole point of the eval discipline is to make the tradeoffs visible.

### Comparison tooling is a natural extension

The project ships single-run capability: produce one scorecard, write one JSON file. The comparison logic is left as an extension because the JSON files are already in a format designed to be compared. A small comparison program — `compare.py results-A.json results-B.json` — reads both files, walks rows by ID, emits a diff. Implementation is straightforward enough that a learner can write it in an afternoon, and the result is immediately useful for the workflow above.

The exercise is a good test of whether the eval discipline has clicked. If you can write the comparison program from scratch, you understand the scorecard's shape; if you can read its output and explain which rows regressed and why, you've internalized the regression-detection loop. The project is the seed; the comparison program is the next petal.

### Statistical significance: when "the number moved" is real

A practical complication for small datasets. On 20 rows, the difference between 17/20 passing (85%) and 18/20 passing (90%) is a single row. Run the same eval twice with no changes and you might see 85% one run and 90% the next, just from LLM non-determinism. The number moved, but it didn't move because of anything you did.

The right framing is *signal vs noise*. The noise floor on a binary-pass eval scales roughly as `sqrt(p(1-p)/n)` — for 20 rows at 80% pass rate, that's roughly 9 percentage points of one-standard-deviation noise. Changes smaller than that floor are not meaningful; changes larger than 2x the floor are. On a 100-row eval, the noise floor drops to about 4 percentage points; on 1000 rows, to about 1.3 percentage points. The thresholds in Section 3's "20 / 100 / 1000" guidance are roughly chosen so that the noise floor is small enough relative to the changes you typically care about.

For small datasets, two practical workarounds: re-run the eval multiple times (say, 3–5 runs) and average the pass rates to estimate the true mean, or focus on *flipped rows* (which are individually attributable to specific changes) rather than aggregate pass rates (which mix sampling noise with real signal). The project's harness supports the first approach trivially — run it 3 times, average the per-evaluator pass rates — and the second approach via the per-row outcomes in the JSON.

This is the same statistical discipline that A/B testing applies to product changes (Phase 4 returns to it). For now, the takeaway is: on a 20-row dataset, treat changes of less than ~10 percentage points as noise; on a 100-row dataset, less than ~5 points; on a 1000-row dataset, less than ~2 points. Above those thresholds, the signal is real; below them, you need more data or more runs before deciding the change helped.

---

## 8. Eval in the AI Stack

The harness you'll build in this module's project is small, in-process, and single-team. It is the minimal viable eval setup. Real production systems use bigger tools — frameworks built for declarative eval configuration, observability platforms with eval baked in, RAG-specific scorers — and this section is a tour of the landscape. The goal is not to teach any one tool deeply (that's their docs' job) but to make sure you know what's out there, when to reach for it, and when to skip it.

### A sidebar on real-world eval tools

The frameworks worth knowing in 2026, each with a different bet on what eval should look like:

**Promptfoo.** Declarative YAML eval configs, a web UI for browsing results, CLI for running. You write a `promptfooconfig.yaml` that lists prompts, providers, and test cases; you run `promptfoo eval`; you get a side-by-side comparison view of how different prompts/providers performed on the same dataset. The bet: prompt iteration is a configuration problem, and the right UI is a spreadsheet of configurations × results. Good for iterating on prompts when you have a clear test set and want to compare multiple variants at once. Less good when the eval logic itself is complex (custom evaluators in Python work but feel bolted-on).

**Phoenix / Arize.** Production tracing plus offline eval, with dashboards over time. Phoenix is the open-source observability layer for LLM apps (traces, spans, eval results); Arize is the hosted commercial version. The bet: eval is one piece of the broader observability picture, alongside per-call tracing, latency monitoring, and user-feedback collection. Good for production systems where you want eval results to live alongside real-time telemetry — you can see a regression in your scorecard *and* the production traces that explain it. Less good when you just want a one-shot eval on a dataset.

**Langfuse.** Open-source LLM observability with built-in eval support, similar scope to Phoenix but with a different UX. Langfuse is hosted-or-self-hosted, with strong SDK support across multiple languages. The bet: observability and eval are the same product, and the right abstraction is a single SDK that captures both. Good for teams that want a single platform for production telemetry and offline eval. Less good when you want eval to live separately from production observability.

**Ragas.** RAG-specific metrics — faithfulness (does the answer cite the retrieved context?), answer relevancy (does the answer address the question?), context precision (was the retrieved context useful?), context recall (did the retrieval find what it needed?). The bet: RAG has eval needs distinct from general LLM output, and packaged metrics for those needs are worth a dedicated library. Good for [Module 07 (RAG)](../07-rag/) systems where you want standard scores out of the box. Less good for non-RAG systems where the metrics don't apply.

**DeepEval.** Pytest-style assertions for LLM outputs. You write `assert_close(output, expected, metric="answer_relevancy")` inside `def test_*` functions; you run `pytest`; the eval runs as part of your test suite. The bet: eval should integrate with the testing tools developers already use, and the right interface is `pytest` syntax. Good for teams that want eval in CI alongside their existing tests. Less good when the eval workflow is more about ad-hoc comparison than about pass/fail on every commit.

Each of these tools makes different architectural choices, and each has found a productive niche. Promptfoo aims at prompt iterators. Phoenix/Arize and Langfuse aim at production observability. Ragas aims at RAG-specific scoring. DeepEval aims at the pytest-integration crowd. None of them is the only right answer; all of them coexist because LLM eval is a wide-enough problem to support multiple shapes.

### When to reach for one vs skip

The decision matrix is roughly:

**Skip frameworks and use a custom in-process harness (like this module's project) when:** the eval is single-team, the dataset is small, the evaluators are project-specific, the workflow is "prompt iteration on a developer's laptop," and there is no production observability requirement. The custom harness is the lowest-friction option — no config files, no service to run, no SDK to learn. Module 15's project covers this case fully.

**Reach for Promptfoo when:** you are iterating on prompts at scale, want a UI for comparing variants, and your eval logic is mostly declarative (compare two outputs, score against a rubric). Promptfoo's YAML config is the sweet spot for prompt-comparison workflows.

**Reach for Phoenix or Langfuse when:** you have a production system in the loop, want eval and tracing in the same place, and care about dashboards over time. The investment pays off when the same eval signal also tells you about live-traffic problems.

**Reach for Ragas when:** your system is RAG and you want standard metrics without writing them yourself. Faithfulness and context precision are non-trivial to implement; using a library that has them is a real time-saver.

**Reach for DeepEval when:** your team treats eval as continuous testing and you want pytest assertions to run in CI. The integration with existing test workflows is the main draw.

The "skip everything and roll your own" option is also a real option, especially for learning and for systems with unusual requirements. The harness in this module's project is roughly 500 lines of Python; the conceptual surface is what this README covers. If your needs grow beyond the project's harness, you can always graduate to a framework. The framework will not surprise you, because you'll already understand what it's doing under the hood.

### Relationship to other modules

Eval intersects with most of Phase 3 directly:

**[Module 02 (Prompt Engineering)](../02-prompt-engineering/).** The eval loop is the prompt-engineering iteration loop made rigorous. Where Module 02 asked "how do you write a good prompt?" Module 15 answers "how do you tell whether one prompt is better than another?" Without eval, prompt engineering is vibes-based; with eval, it is data-driven. The two modules are the same workflow seen from opposite ends.

**[Module 08 (Structured Output)](../08-structured-output/).** Schema validation as a free evaluator. Module 08 taught you how to constrain LLM outputs to a Pydantic schema; this module reframes schema validation as a mechanical evaluator that catches "did the model produce the right shape?" failures for the cost of a function call. The schemas you wrote for Module 08 are the schemas you reuse here as evaluators.

**[Module 12 (Multi-Agent Systems)](../12-multi-agent-systems/).** LLM-as-judge IS the critic agent pattern. The critic-loop architecture from Module 12 (writer produces, critic evaluates, optionally revise) is the same pattern as the LLM-judge evaluator (judge evaluates, no revise). One is a loop, the other is a one-shot, but the critic role is the same. Engineering investment in critic prompts pays off as eval-judge prompts here.

**[Module 13 (Workflows & Chains)](../13-workflows-chains/).** The eval pipeline is a workflow. Dataset → fan-out to SUT calls → fan-in → per-row evaluators → fan-in → aggregate is exactly Module 13's workflow shape, with a parallel fan-out inside one of the steps. The orchestrator in this module's project is a Module 13 workflow with three steps. The discipline (typed steps, fixed shape, per-step cost accounting) is identical.

**[Module 14 (AI Code Generation)](../14-ai-code-generation/).** The iterate-on-failure loop in Module 14 is the eval loop with one row, one evaluator (the test runner), and an automatic regeneration step. Module 14 got to have a runtime critic because code is executable; this module generalises the pattern to systems without a runtime critic, where evaluators are mechanical or LLM-based and the response to failure is a scorecard rather than automatic regeneration.

### Forward pointer: Phase 4

This module is the meta-module for Phase 3 — the part that says "you've built a lot of LLM systems; here's how to evaluate them." Phase 4 takes the discipline established here and applies it to production. The themes that show up:

**Production observability (Module 18).** Eval offline (this module) becomes eval online: traces from real user calls are sampled, evaluated against the same evaluators you developed offline, and the resulting scores feed dashboards. The eval harness shape stays; the dataset becomes a live sample from production rather than a fixed JSONL file.

**Online evaluation against live traffic.** Sometimes the eval can run *against the user's real call* in near-real-time — for example, a schema validator that fires on every response, or an LLM-judge that runs on a sampled 1% of calls. Online eval extends the offline patterns from this module to live data, with all the cost and latency tradeoffs that implies.

**A/B testing harnesses.** When you want to compare two prompt versions in production, you split traffic, run both versions on real users, and compare downstream signals (eval scores, user feedback, conversion). The eval discipline from this module is the foundation — without offline evaluators you trust, you have no scoring to apply to the A/B comparison.

**Monitoring for distribution shift.** Production inputs drift over time; users do new things; the system that scored well on yesterday's distribution may score worse on today's. Phase 4 covers shift detection and the operational responses to it (re-train, re-prompt, expand the eval set with rows from the new distribution).

The Phase 4 modules build on the eval discipline established here. If you've internalized "dataset + evaluators → scorecard, comparable across runs," you have the foundation for the production patterns. If you haven't, the Phase 4 patterns will be hard to operate because you won't have a way to tell whether they're working.

### Module cross-reference map

| This module's component | Prior module it builds on |
|---|---|
| Pydantic types as scorecard contracts (`EvalRow`, `EvalResult`, `RowOutcome`, `EvaluatorAggregate`, `Scorecard`) | [Module 08](../08-structured-output/) — structured output and schema validation |
| The dataset → SUT → evaluators → scorecard pipeline | [Module 13](../13-workflows-chains/) — sequential workflows with parallel fan-out and typed step boundaries |
| `ThreadPoolExecutor` to fan out SUT calls across rows | [Module 13](../13-workflows-chains/) — parallel-vs-sequential workflow patterns |
| `SchemaEvaluator` checking outputs against Pydantic models | [Module 08](../08-structured-output/) — schema validation, now reframed as a mechanical evaluator |
| `LLMJudgeEvaluator` calling a stronger model with a rubric prompt | [Module 12](../12-multi-agent-systems/) — critic agent pattern at dataset scale |
| `_strip_code_fence` helper for parsing the judge's JSON output | [Module 14](../14-ai-code-generation/) — same helper, reused for judge response parsing |
| Per-evaluator + overall aggregates printed as a console scorecard | [Module 13](../13-workflows-chains/) — observability through fixed-shape per-step logs |
| `results-{run_id}.json` written for every run, diffable across runs | New in this module — versioned scorecards as the regression-detection primitive |
| Mechanical-evaluator-first, then LLM-judge composition | [Module 12](../12-multi-agent-systems/) + [Module 13](../13-workflows-chains/) — cheap critic first, expensive critic on what survives, as a workflow |
| Forward pointer to online eval and production observability | [Phase 4 (Module 18)](../) — eval discipline applied to live traffic and trace sampling |

The unifying theme, as in Module 14: the techniques from prior modules are the *building blocks*, and this module composes them into a discipline whose distinctive feature — comparable scorecards across runs — unlocks regression detection that earlier modules could not have done. Internalising this composition (workflows on the outside, evaluators as the units, scorecards as the artifact, datasets as the input) is the design principle that holds Phase 3 together. Phase 4 takes that toolbox into production.

### Phase 3 in retrospect

Phase 3 has been about *composition*. Module 11 introduced the agent loop — one LLM driving its own behavior through tool calls. Module 12 introduced multi-agent patterns — multiple LLMs cooperating through structured roles. Module 13 introduced workflows — predetermined orchestration of LLM calls into typed pipelines. Module 14 introduced code generation — workflows whose output is mechanically checkable. Module 15 introduces evaluation — the discipline that makes any of the above measurable, comparable, and improvable.

Each module on its own is a building block. Composed, they form the production-system toolkit: an agent for open-ended decisions, multi-agent patterns for cooperation, workflows for the predictable parts, code generation for the executable parts, and eval throughout to keep the system honest. The composition is what production systems actually look like — not pure agent loops, not pure workflows, but a mix where each pattern is applied to the part of the system it fits best.

Phase 4 will take this toolkit into production. Observability, caching, deployment, A/B testing, online evaluation — all of it builds on the evaluation discipline established here. If you've finished Phase 3 with a working harness, a 20-row dataset, and a habit of running the eval before shipping prompt changes, you have the foundation. The Phase 4 modules layer the operational practices on top. The eval discipline does not change; it scales out, grows online, and becomes the heartbeat of a system that's running 24/7 for users you'll never meet.
