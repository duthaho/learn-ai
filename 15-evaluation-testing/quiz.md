# Module 15 Quiz: Evaluation & Testing

Self-assessment questions for Module 15. Test your understanding before revealing each answer.

---

### Q1: Why is evaluation harder for LLM systems than for traditional software?

<details>
<summary>Answer</summary>

LLM systems are non-deterministic — the same prompt yields different outputs across runs. There is no oracle for most prose outputs (you can't write `assert summary == expected`). Prompts are brittle, so small wording changes shift behavior. And silent model updates from the provider can move behavior between yesterday and today. Traditional software gets graded by `pytest` with binary pass/fail; LLM systems need to be graded across a distribution of inputs with a mix of mechanical and judgment-based evaluators.

</details>

---

### Q2: Name three sources of an eval dataset.

<details>
<summary>Answer</summary>

1. **Golden curated** — hand-picked inputs with hand-labeled outputs. Slow to build, but highest signal.
2. **Synthetic** — an LLM generates inputs and labels from seed examples. Fast and broad, but lower signal per row.
3. **Production traffic** — sampled from real usage. Most representative, but requires labeling tooling and a sampling strategy.

Most production systems use all three.

</details>

---

### Q3: Sketch the eval loop in one sentence.

<details>
<summary>Answer</summary>

For each row in an eval dataset, run the system-under-test, run a list of evaluators on the output, and aggregate the per-evaluator results into a scorecard you can compare across runs.

</details>

---

### Q4: When should you use exact-match vs LLM-as-judge?

<details>
<summary>Answer</summary>

Use mechanical evaluators (exact-match, schema validation, regex) for any property you can express in code — they are fast, free, and reliable. Use LLM-as-judge for properties that are genuinely subjective (clarity, tone, helpfulness, factual coherence on novel claims) where a mechanical check would be wrong or impossible. In practice you combine both: mechanical first to catch obvious failures, then LLM-judge on what survives.

</details>

---

### Q5: What are two known biases of LLM-as-judge evaluators?

<details>
<summary>Answer</summary>

1. **Position bias** — in paired comparisons, judges prefer the first option more often than they should. Mitigation: swap positions and average.
2. **Length bias** — judges rate longer outputs higher even when they shouldn't. Mitigation: include length as a feature in the rubric or normalize for it.

A third common one is **self-preference bias**: a model judge prefers outputs that look like its own.

</details>

---

### Q6: Why does the canonical recipe run mechanical evaluators before LLM-as-judge?

<details>
<summary>Answer</summary>

Mechanical evaluators are microseconds-fast and free. LLM-judge takes seconds and costs real money per row. Running mechanical first lets you filter out the obvious failures (wrong category, malformed JSON) for the cost of a function call, and reserve the expensive judge for rows where the question is actually nuanced. On a 1000-row dataset this ordering can turn a $5 eval run into a $0.50 eval run with no signal loss.

</details>

---

### Q7: What does a scorecard let you do that printing pass/fail doesn't?

<details>
<summary>Answer</summary>

A scorecard makes the run comparable across versions. It records the run_id, timestamp, prompt fingerprint, model, per-row outcomes, per-evaluator aggregates, total cost, and total latency. After a prompt or model change, you can diff scorecard A against scorecard B and see exactly which rows flipped and which evaluators moved. Printed pass/fail evaporates the moment the terminal scrolls.

</details>

---

### Q8: Name three eval frameworks and what each specializes in.

<details>
<summary>Answer</summary>

1. **Promptfoo** — declarative YAML eval configs with a web UI; good for prompt-iteration workflows.
2. **Phoenix / Arize** — production tracing plus offline eval, with dashboards over time.
3. **Ragas** — RAG-specific metrics like faithfulness, answer relevancy, and context precision.

Other valid answers: **Langfuse** for open-source LLM observability, **DeepEval** for pytest-style assertions on LLM outputs.

</details>
