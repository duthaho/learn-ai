# Module 13 Quiz: Workflows & Chains

Self-assessment questions for Module 13. Test your understanding before revealing each answer.

---

### Q1: What distinguishes a workflow from an agent?

<details>
<summary>Answer</summary>

Workflows have a fixed, design-time control flow — every step and the order between them is decided when the code is written. Agents decide what to do at runtime by choosing a tool to call. Workflows are predictable, testable, and cheap; agents are flexible but less predictable. Choose workflow when the steps are known; choose agent when they're not.

</details>

---

### Q2: Sketch the three workflow patterns in one sentence each.

<details>
<summary>Answer</summary>

- **Sequential chain:** step N+1 takes step N's output as input.
- **Branching/router:** an upstream classify step picks which downstream step runs.
- **Parallel fan-out + fan-in:** independent steps run concurrently and their outputs are joined.

</details>

---

### Q3: Why pass Pydantic models between workflow steps instead of raw strings?

<details>
<summary>Answer</summary>

Typed contracts make handoffs explicit. The next step's signature documents exactly what it expects, validation catches malformed handoffs at the boundary, and IDE/type-checker support makes the pipeline easier to refactor.

</details>

---

### Q4: When is a router/branching step preferable to letting an agent decide what to do?

<details>
<summary>Answer</summary>

When the set of downstream actions is small and well-defined. A classify-then-route workflow runs in one LLM call plus a Python branch; an agent reasoning over the same choice takes multiple steps and is non-deterministic. Use the router when the branching is bounded and the cost-per-choice matters.

</details>

---

### Q5: Why use `ThreadPoolExecutor` for parallel LLM calls in Python? Name one situation where it would NOT give a speedup.

<details>
<summary>Answer</summary>

LLM calls are I/O-bound — the thread is waiting on HTTP. Python releases the GIL during I/O, so multiple threads can wait in parallel. It does NOT speed things up when the provider rate-limits you (the requests just queue server-side), when only one step actually exists, or when one step depends on another's output.

</details>

---

### Q6: List three reasons workflows are easier to test than agents.

<details>
<summary>Answer</summary>

1. Each step is a pure function with a typed input and output — same input gives same output.
2. Per-step token/cost/latency tracking is built in; agent scratchpads have to be parsed.
3. Failure is isolated: when step 3 fails, you still have step 1 and step 2 outputs.

</details>

---

### Q7: Give one task that should be a workflow and one that should be an agent — and explain why.

<details>
<summary>Answer</summary>

- **Workflow:** Triage incoming support tickets — the steps (classify → route → extract+draft → assemble) are known and the order is fixed.
- **Agent:** Open-ended research where the next search query depends on what was just found — the model needs to decide at runtime what to do next.

</details>

---

### Q8: What does Module 13 buy you over Module 12, and what does it give up?

<details>
<summary>Answer</summary>

- **Buys:** Predictability, lower cost (fewer tokens per run because there is no orchestrator deliberation), per-step testability, easier observability.
- **Gives up:** Runtime adaptability — if the input does not fit the predefined steps, a workflow cannot pivot. Use workflows when the task is known; use multi-agent when it is not.

</details>
