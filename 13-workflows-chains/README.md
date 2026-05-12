# Module 13: Workflows & Chains

**What you'll learn:**
- The three workflow patterns: sequential chain, branching/router, parallel fan-out + fan-in
- When deterministic workflows beat dynamic agents
- Passing structured state between workflow steps with Pydantic
- Running independent LLM calls in parallel with `concurrent.futures.ThreadPoolExecutor`
- Per-step token, cost, and latency tracking
- Choosing between workflow, agent, and multi-agent for a given task
- Where workflows fit with tools (Module 06), RAG (Module 07), agents (Module 11), and multi-agent systems (Module 12)

| Detail        | Value                                                                 |
|---------------|-----------------------------------------------------------------------|
| Level         | Intermediate–Advanced                                                 |
| Time          | ~3 hours                                                              |
| Prerequisites | Module 08 (Structured Output), Module 11 (Building AI Agents), Module 12 (Multi-Agent Systems) |

---

## Table of Contents

1. [From Agents to Workflows](#1-from-agents-to-workflows)
2. [The Three Patterns](#2-the-three-patterns)
3. [Chains: Linear Sequential Steps](#3-chains-linear-sequential-steps)
4. [Branching: Routing on Classification](#4-branching-routing-on-classification)
5. [Parallel Steps: Fan-Out and Fan-In](#5-parallel-steps-fan-out-and-fan-in)
6. [Workflow vs Agent: Choosing the Right Tool](#6-workflow-vs-agent-choosing-the-right-tool)
7. [Observability & Testing](#7-observability--testing)
8. [Workflows in the AI Stack](#8-workflows-in-the-ai-stack)

---

## 1. From Agents to Workflows

[Module 11](../11-building-ai-agents/) built an agent: an LLM loop that decides at runtime which tool to call next, evaluates the result, and keeps looping until it reaches a final answer. [Module 12](../12-multi-agent-systems/) extended that idea to multiple agents, where an orchestrator (which is itself usually an LLM or a small Python coordinator) decides which specialist to invoke and in what order. In both cases the control flow — the question of "what runs next?" — is decided dynamically, at the moment the system is running, by an LLM looking at intermediate results.

A workflow is the same shape of system with one decision flipped: the control flow is decided *at design time*, in code, by the developer. Workflows are the deterministic sibling of agents. Where an agent looks at a partial result and chooses the next move, a workflow follows a path you laid out before the system was ever given an input. The steps are known, the order is fixed, and the only variation between runs comes from the LLM's text outputs inside each step — not from the structure of the pipeline itself.

This is not a small distinction. The same task can often be solved either way: as an agent that figures out what to do, or as a workflow that does the steps in a known order. When you choose one over the other, you are making a deliberate trade between flexibility and reliability. Agents handle the unknown well. Workflows handle the known *better*.

### When you don't need an agent

Most LLM-powered features in production turn out to be workflows in disguise. The team building them often starts by reaching for an agent — because agents feel general and capable — and then slowly realises that the agent is being constrained back into a fixed sequence anyway. After a few rounds of "make sure the agent always classifies before responding" and "add a system rule that the agent must call the extractor before the drafter," the system has become a workflow that happens to be implemented with an agent loop. The dynamic control flow was never used.

The signs that you do not need an agent are usually visible up front:

**The steps are known.** You can sit down and list them. Classify the ticket. Pull out the order ID. Write a response. There is no point in the pipeline where the LLM legitimately needs to decide between two different sequences of action — the sequence is just *those three steps, in that order*.

**The order is fixed.** You will never extract the order ID *before* you know whether the ticket is a refund. You will never write the response *before* you know what the ticket is about. The dependency graph between steps is a straight line (or a small tree), not a runtime decision.

**Success doesn't depend on adaptive behavior.** A good result for this task doesn't require the system to discover an unusual path through the problem. The path is normal. The variation is in the *content* of each step's output, not in *which steps run*.

When these three conditions hold, the agent's defining capability — runtime control flow — is dead weight. You pay for the agent's loop and reasoning tokens without using them. A workflow does the same job faster, cheaper, and more predictably.

A useful diagnostic: write down, in plain English, what your system should do for a representative input. If the sentences come out in a fixed order — first this, then this, then this — and the order does not change when you imagine other representative inputs, you have a workflow on your hands. If different inputs would produce sentences in a different order, or would introduce sentences that don't appear at all on other inputs, you have something that genuinely needs runtime control flow.

The diagnostic catches most of the false starts. A team building "an agent that handles refund requests" will almost always end up describing the same three-step sentence — classify the request, pull out the order details, write a response — for every example they think of. The agent framing was decorative; the underlying task is a workflow. Naming this honestly at the start of design saves weeks of accidentally re-implementing a workflow under an agent's coat.

### The four wins from being deterministic

Determinism in control flow buys you four concrete things, and each of them compounds as the system grows from a prototype into something other people depend on.

**Predictability.** Given an input that classifies as `refund`, you know exactly which step functions will run, in what order, and with what inputs. You don't have to read an agent trace to discover what happened. You can describe the path in a sentence and the description is correct on every run.

**Testability.** Each step is a pure function with a typed input and a typed output. You can test the extractor on its own, with a hand-written ticket, and assert against its output without standing up the whole pipeline. You can swap a fake LLM into one step and feed real inputs to the others. You can write a test that says "given this ticket, the classifier returns `category=refund`" and have it mean something stable across runs. Agent traces, by contrast, branch differently across runs even when the inputs are identical, which makes unit-level testing genuinely hard.

**Lower cost.** A workflow that calls one LLM per step pays for exactly N LLM calls — where N is the number of steps in the chosen path. An agent pays for those same N calls *plus* the reasoning tokens it spent deciding what to do at each step. For tasks where the decision is obvious in advance, those reasoning tokens are pure waste. At scale — thousands of tickets a day — the difference between "one classify + one extract + one draft" and "an agent that reasons about which of those three to call next" is real money.

**Easier observability.** Because the structure is fixed, your logs have a known shape. Every successful run produces the same set of records, in the same order, with the same fields. You can build a dashboard against those records without writing parsing logic that handles "the agent decided to do something different this time." Latency budgets are knowable up front: the workflow takes at most `sum(step_latencies)` (or less if steps run in parallel), and you can alert when that envelope is breached.

These four wins compound. Predictability makes failures locatable; locatable failures make tests writable; writable tests make regressions catchable; catchable regressions make the system maintainable across model upgrades. The opposite is also true: a system without these wins is operationally fragile, even when its individual LLM calls are well-tuned. Most of what makes production LLM systems painful to run is the absence of these four properties, not the LLM calls themselves.

### The one cost of being deterministic

There is exactly one cost, and it is the price of admission: a workflow has no runtime flexibility. If the input arrives in a shape the workflow's steps don't fit, the workflow can't pivot. An agent would notice "this isn't actually a refund ticket, even though it mentions an order — it's a question about returns policy" and route accordingly. A workflow, with its classifier wired to three categories, will stuff the input into whichever category looks closest and proceed. The result will be coherent but slightly wrong.

This is real, but it is bounded. Most of the time the inputs *do* fit the categories. When they don't, you can build a fallback (we'll come back to this in section 4) that catches the off-distribution cases and either degrades gracefully or escalates to a human. What you cannot do is have a workflow that suddenly grows new steps it didn't have at design time. If that's the requirement, you need an agent or a multi-agent system.

The rule of thumb: pick the simplest pattern that handles your common case well, and design the fallback for the rare case that doesn't fit. Most production LLM features land on "workflow with a fallback," not "agent for everything."

Put another way: the choice between agent and workflow is not the choice between "smart" and "dumb." It is the choice between paying for flexibility you'll use and paying for flexibility you won't. Workflows are not less capable than agents — they are *more focused*, and that focus is what gives you the operational properties of the next sections.

### When a workflow is the right tool

Four concrete examples where workflows are clearly the right pattern, drawn from real product surfaces:

**Form-to-record extraction.** You receive a free-text submission — an insurance claim, a job application, a contact-form message — and need to turn it into a structured database row. The steps are fixed: extract the named entities, validate them against a schema, write the row. There is no decision to make at runtime about *which* steps run. A workflow with `extract → validate → persist` does this perfectly and is easy to test.

**Document summarization.** Take a long document, chunk it if necessary, summarize each chunk, then summarize the summaries. Map-reduce in shape, but laid out at design time. There is no point at which the LLM needs to decide *whether* to chunk or *whether* to summarize the summaries — both decisions are encoded in the pipeline. The variation is in the content of the summaries themselves.

**Classify-then-route triage.** This module's project. An incoming ticket, email, or message gets a category, and based on that category, a specific handler runs. The router is a tiny upstream classifier; the downstream handlers are specialised on their categories. Nothing about the routing decision benefits from a long agent reasoning trace — the classifier knows the category in one call.

**Scheduled batch transformation.** A nightly job that walks a queue of records and applies a fixed transformation: classify each, extract entities, write a report. There is no human in the loop, no ambiguity about what the system should do, and a hard latency budget. Workflows are tailor-made for this — they run identically every time, they're trivial to retry on per-record failure, and they emit logs whose shape you control.

Each of these examples shares the same underlying structure: the steps are nameable, the order is fixed, and the variation between runs is in the content of each step's output rather than in the structure of the pipeline.

If your problem looks like one of these, you almost certainly want a workflow. If it looks like "open-ended research, where I don't know in advance what tools will be needed," you want an agent (Module 11) or a multi-agent system (Module 12). Section 6 of this module makes the choice explicit.

A subtler observation: many of these workflow shapes were *implemented* as agents in early LLM products because the framing of "an AI that handles your tickets" sounds more impressive than "a pipeline that handles your tickets." That framing is starting to fade as teams discover that the workflow versions are cheaper to run, easier to monitor, and more pleasant for the users who depend on consistent behavior. The marketing copy says "agent"; the production architecture is increasingly a workflow with one or two agentic steps.

---

## 2. The Three Patterns

Workflows come in three structural shapes. Almost every workflow you will write in practice is one of these three, or a small composition of them. Knowing each shape — its diagram, its strengths, and where it breaks down — is the bulk of the design work. Once you have picked the right shape, writing the code is mechanical.

### Sequential chain

The simplest pattern, and the one most people mean when they say "chain" without further qualification. The output of step N is the input of step N+1. There is no branching, no looping, no parallelism. Each step takes the previous step's output as a typed object, does its work, and hands its own typed output to the next step.

```text
┌─────────────────────────────────────────────────────────┐
│                   SEQUENTIAL CHAIN                       │
│                                                         │
│   input                                                 │
│     │                                                   │
│     ▼                                                   │
│  ┌─────────┐                                            │
│  │ Step A  │                                            │
│  └─────────┘                                            │
│       │                                                 │
│       ▼ (A's output)                                    │
│  ┌─────────┐                                            │
│  │ Step B  │                                            │
│  └─────────┘                                            │
│       │                                                 │
│       ▼ (B's output)                                    │
│  ┌─────────┐                                            │
│  │ Step C  │                                            │
│  └─────────┘                                            │
│       │                                                 │
│       ▼                                                 │
│    output                                               │
└─────────────────────────────────────────────────────────┘
```

Sequential chains are the natural pattern for tasks that have a clear "pipeline" feel: transform the data, then transform it again, then format the result. They are easy to read, easy to test (each step is a function), and easy to extend (add another step at the end).

### Branching / router

Branching introduces a single decision point. A small upstream step — usually a classifier — looks at the input and produces a label. The label selects which downstream step runs. The downstream steps are siblings: only one of them runs per input.

```text
┌─────────────────────────────────────────────────────────┐
│                  BRANCHING / ROUTER                     │
│                                                         │
│   input                                                 │
│     │                                                   │
│     ▼                                                   │
│  ┌──────────┐                                           │
│  │ Classify │                                           │
│  └──────────┘                                           │
│       │                                                 │
│       │  category in {A, B, C}                          │
│       │                                                 │
│   ┌───┴───┬───────────┬───────────┐                     │
│   ▼       ▼           ▼           ▼                     │
│ ┌───┐   ┌───┐       ┌───┐       ┌───┐                   │
│ │ A │   │ B │       │ C │       │def│  (fallback)       │
│ └───┘   └───┘       └───┘       └───┘                   │
│   │       │           │           │                     │
│   └───────┴───────────┴───────────┘                     │
│                       │                                 │
│                       ▼                                 │
│                    output                               │
└─────────────────────────────────────────────────────────┘
```

The decision point is the whole reason the pattern exists. If the right downstream step depends on a property of the input that you cannot infer cheaply with code, a classifier is the natural way to make the decision. The downstream steps are specialised — each one can use a tighter system prompt and a smaller Pydantic schema than a generalist would need — and the router itself is small, fast, and cheap.

### Parallel fan-out + fan-in

When two or more steps don't depend on each other, there is no reason to run them in series. Fan-out launches them concurrently. Fan-in collects their results once they're all done. The wall-clock cost of the parallel section is the maximum step latency, not the sum.

```text
┌─────────────────────────────────────────────────────────┐
│                PARALLEL FAN-OUT + FAN-IN                │
│                                                         │
│   input                                                 │
│     │                                                   │
│     ▼                                                   │
│  ┌──────┐                                               │
│  │ Fork │                                               │
│  └──────┘                                               │
│     │                                                   │
│   ┌─┴─────────┬─────────┐                               │
│   ▼           ▼         ▼                               │
│ ┌─────┐   ┌─────┐    ┌─────┐                            │
│ │  A  │   │  B  │    │  C  │  (independent; concurrent) │
│ └─────┘   └─────┘    └─────┘                            │
│   │           │         │                               │
│   └───┬───────┴─────────┘                               │
│       ▼                                                 │
│   ┌──────┐                                              │
│   │ Join │                                              │
│   └──────┘                                              │
│       │                                                 │
│       ▼                                                 │
│    output                                               │
└─────────────────────────────────────────────────────────┘
```

The pattern only makes sense when the parallel steps genuinely do not depend on each other. If step B needs A's output, you can't fan them out — you'd just be paying a thread-pool cost for serial work. Section 5 covers the cases where fan-out doesn't help.

### Pattern comparison

| Pattern | When to use | Control flow | Strengths | Weaknesses |
|---|---|---|---|---|
| Sequential chain | Each step depends on the previous step's output; the pipeline is a straight line | Linear, deterministic | Simplest to write and reason about; trivial to test step-by-step; clear data flow | No conditional logic; no parallelism; total latency is the sum of step latencies |
| Branching / router | The right downstream step depends on a property of the input that needs a classifier to identify | Linear up to the router, then one of N siblings | Specialised downstream prompts; lower cost on average (only one branch runs); easy to extend by adding a branch | Router is a single point of failure; off-category inputs fall through if there's no fallback; testing requires covering every branch |
| Parallel fan-out + fan-in | Two or more steps are independent of each other and can be run concurrently | Diverge at the fork, converge at the join | Wall-clock latency drops to `max(step)` instead of `sum(steps)`; cost is unchanged | Requires real I/O concurrency (threads or async); rate limits can serialize what should be parallel; introduces a join point that must wait for the slowest step |

### Real workflows mix all three

These three patterns are rarely encountered alone. A real workflow — and the project at the heart of this module is a good example — typically uses all three together. This module's Support Ticket Triage Pipeline starts with a sequential chain (`ticket → classify → handle → assemble`), branches inside the chain via a router (refund / technical / general / fallback), and *inside each branch* fans out two LLM calls (entity extraction and response drafting) in parallel before joining their results. Five steps, three patterns, one workflow.

This composition is the norm, not the exception. Once you internalize the three shapes, designing a workflow becomes the exercise of identifying which sub-task uses which shape and stitching them together with typed function boundaries. That stitching is what the next three sections cover, one pattern at a time.

A useful mental drill before reading on: take any LLM feature you've used recently — a code-completion suggestion, a meeting summarizer, a smart-reply email tool, a content moderation filter — and try to sketch its workflow on a napkin. Where is the chain? Is there a router? Are any of its steps independent enough to run in parallel? Most polished LLM features turn out to have a clean answer in one of the three shapes, even if their marketing presents them as a single magic "AI" feature. Workflows are the shape under the surface of the AI features you already use.

---

## 3. Chains: Linear Sequential Steps

The sequential chain is the foundational workflow pattern, and most of the discipline of building good workflows comes from getting chains right. The shape is simple: each step is a function, the output of step N is the input of step N+1, and the orchestrator is the small piece of code that wires the functions together in the right order.

What makes chains worth treating carefully — despite their apparent simplicity — is that the boundaries between steps are where most production bugs live. Get those boundaries wrong, and your pipeline becomes a chain of string-parsing hacks where each step has to guess at the format the previous step actually produced. Get them right, and each step is a self-contained function you can debug, test, and replace without touching the rest of the system.

### Pydantic models make handoffs safe

The single most useful technique for building reliable chains is to declare a Pydantic model for the data that flows between each pair of steps. The model is the contract: the producing step *must* return an instance of that model, and the consuming step receives one as input. Anything outside the model's schema either doesn't exist or fails validation up front.

This is a direct extension of the techniques from [Module 08 (Structured Output)](../08-structured-output/). The same `response_format` parameter and Pydantic model approach you used to get structured output from a single LLM call doubles, in a chain, as the data contract between successive steps. The model that the writer step *returns* is the same model the next step *receives*. There is no parsing layer in between, no string format to negotiate, no field name to remember.

The three concrete benefits, which all show up in the same way they showed up for multi-agent systems in [Module 12](../12-multi-agent-systems/):

- **Parseability.** The model either parses or it doesn't. Malformed LLM output fails validation immediately with a clear error, not silently with a wrong-shape dictionary that breaks the next step in a confusing way.
- **Versioning.** Changing the model — adding or removing a field — is a typed change. Every step that uses that model shows an error in your editor until you update it. There is no place where a stale field name lives in a string template and breaks only at runtime.
- **Debugging.** When a chain misbehaves, you can print the intermediate Pydantic object at any boundary and see exactly what the previous step produced. You don't have to reconstruct it from logs.

The Pydantic models for this module's project are defined at the top of `solution.py`, before any of the step functions, so the data contract between steps is the first thing a reader sees. Section 7 of [Module 12](../12-multi-agent-systems/) follows the same convention — typed boundaries make pipelines auditable.

### A two-step chain in code

Here is a minimal sequential chain: a `summarize` step followed by an `extract_keywords` step. Each step is a function that takes a typed input, makes one LLM call with structured output, and returns a typed result. The orchestrator is two lines.

```python
from litellm import completion
from pydantic import BaseModel

class Summary(BaseModel):
    summary: str

class Keywords(BaseModel):
    keywords: list[str]

def summarize(text: str) -> Summary:
    resp = completion(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": f"Summarize in two sentences:\n\n{text}"}],
        response_format=Summary,
    )
    return Summary.model_validate_json(resp.choices[0].message.content)

def extract_keywords(s: Summary) -> Keywords:
    resp = completion(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": f"Extract 3-5 keywords:\n\n{s.summary}"}],
        response_format=Keywords,
    )
    return Keywords.model_validate_json(resp.choices[0].message.content)

# Orchestrator: two-step chain.
result = extract_keywords(summarize(long_article))
```

Notice what is *not* in this code: no string parsing, no regex, no "did the LLM put the keywords on a single line or split them across lines?" question. The `summarize` step returns a `Summary`. The `extract_keywords` step takes a `Summary` and returns a `Keywords`. The orchestrator just composes the functions. The whole pipeline fits in twenty lines and is easy to test step by step — feed any string to `summarize`, assert on the returned `Summary`; feed any `Summary` to `extract_keywords`, assert on the returned `Keywords`.

This is the shape every step in a workflow should have. One input, one output, one responsibility, typed boundaries on both sides. The project in this module follows the same convention with a small extension: each step also returns a `StepUsage` record alongside its result, so the orchestrator can build a per-run cost and latency report. We'll come back to that instrumentation in section 7.

### What counts as a step

A "step" in a workflow is a small idea. It is *any unit with one input, one output, and one responsibility*. The unit does not have to be an LLM call, although it usually is. A step can be:

- **An LLM call** — the most common case. The unit's job is to take some structured input, ask the model to do one thing, and return structured output. The bulk of this module's steps are LLM calls.
- **A tool call.** A function that does no LLM work at all but produces output the next step needs. A database lookup, an HTTP fetch, a hash computation, a date parse — any of these can be a step. Tools used inside workflows are the same tools you saw in [Module 06 (Tool Use & Function Calling)](../06-tool-use-function-calling/), just wired in directly rather than being called by an agent that decides when to use them.
- **A pure Python function.** A transformation that happens entirely in code. A normalization step that lowercases and strips whitespace, a sorting step that orders a list, a math step that computes a derived field. These are usually the cheapest steps in the pipeline and the most reliable.
- **A RAG retrieval.** Embedding a query and fetching the top-k relevant chunks from a vector index. The output is a list of chunks, the input is a query — both can be Pydantic types, and the step slots into a chain like any other. The RAG pipelines you built in [Module 07](../07-rag/) become workflow steps without modification.
- **A smaller workflow.** A step can itself be an orchestrator over a smaller chain. This is how complex workflows are composed: rather than having one orchestrator with twenty steps, you have a top-level orchestrator that calls three sub-workflows of five steps each. Same control flow, easier to read.

The single test for whether something belongs as a step on its own: can you describe its job in one sentence, and can you give it a typed input and a typed output? If yes, it's a step. If the description needs "and then it also..." in the middle, it's two steps wearing the same name.

This unit discipline — single responsibility, typed boundaries — is the same discipline that made specialist agents in [Module 12](../12-multi-agent-systems/) reliable. The mental model is identical; the difference is that in a workflow, *you* decide the order at design time, rather than asking an orchestrator agent to decide it at runtime.

### Where step boundaries fail

Chains break most often at the boundary between two steps that have an *implicit* contract. The producing step returns a Pydantic model with five fields; the consuming step uses four of them and quietly ignores the fifth. Six months later someone adds a sixth field; six months after that someone renames the fifth, the consuming step keeps ignoring whatever's in that slot, and a feature that was supposed to use the new field never actually does. The pipeline is "working" in the sense that nothing throws an error, but it's silently producing slightly-wrong results.

The cure is to make the contract explicit. Document which fields each step *reads* from its input, not just what the input type is. In code, that's usually a brief docstring on the step function — three lines, no more — that names the fields the function actually uses. When the producing step's schema changes, the consuming step's docstring is the place reviewers look to decide whether the change is a breaking one.

A weaker cure, but still useful, is to keep the Pydantic models as small as possible. A model with five fields, all of which are used by every downstream consumer, is harder to misuse than a model with twenty fields where each consumer uses three. Resist the temptation to put "everything you might want to know about this step's output" into one fat model. The right size is *the smallest model that captures the contract* — and you can always add a second model for richer telemetry that lives in a different code path.

---

## 4. Branching: Routing on Classification

A pure sequential chain assumes every input takes the same path. That assumption breaks the moment you have inputs of meaningfully different types. A refund ticket and a technical issue ticket should not be processed by the same extractor — refunds care about order IDs and amounts, technical issues care about products and error messages. Forcing both through one generic extractor produces mediocre output on both.

The branching pattern handles this by inserting a small upstream step — a classifier — that decides which downstream step runs. The classifier is a router. Its only job is to look at the input and assign a label. The label selects the next step.

### Classify-then-route

The structure is always the same. One LLM call (or sometimes a heuristic) up front produces a label. A piece of routing code maps the label to a handler. The handler runs and produces the output. The router itself does not produce the final result — it produces only the decision.

```text
ticket → [classify] → category ∈ {refund, technical, general}
                          │
                          ▼
                      [router]
                          │
                ┌─────────┼─────────┐
                ▼         ▼         ▼
        [refund handler] [tech handler] [general handler]
```

The classifier is upstream and small. Each handler is specialised. The handlers do not know about each other — each one is the only handler for its category. The router itself does not need to understand what any handler does; it just dispatches.

### Two forms of router

There are two practical ways to implement the router, and both are correct in different situations.

**Code-as-router.** The classifier returns a label as part of its structured output, and a piece of Python code — typically a `match` statement or a small `if/elif` ladder — picks the handler. The LLM's job ends with producing the label. The dispatch is in code.

This is preferred whenever the set of possible labels is bounded and known in advance. The dispatch is transparent (you can read it), free (no extra LLM call), and trivial to test (give it a label, assert which handler was selected). The router code is the *contract* between the classifier's output and the rest of the system — when you add a new category, the place to update is obvious.

**LLM-as-router.** The classifier's output *is* the dispatch decision in a richer sense: the model is asked not just "what category is this?" but "what should we do with this?" The model might return a tool name, a function description, or a free-form action. The orchestrator then calls whatever the model picked.

This is the right pattern when the set of downstream handlers is large, dynamic, or hard to enumerate in advance — and when you trust the LLM to pick correctly. It is essentially what an agent's tool-use loop does inside a single iteration. The cost is that you've added a degree of runtime flexibility, which is exactly the thing workflows are designed to avoid; for that reason, code-as-router is usually the better fit for a workflow setting, and LLM-as-router is the better fit for agentic settings.

This module's project uses code-as-router. The classifier returns one of three category strings; a `match` statement picks the handler:

```python
def route(category: TicketCategory) -> tuple[Callable, Callable]:
    """Return (extract_fn, draft_fn) for the given category."""
    match category:
        case "refund":    return extract_refund, draft_refund
        case "technical": return extract_technical, draft_technical
        case "general":   return extract_general, draft_general
        case _:           return extract_general, draft_general   # fallback
```

The whole router is ten lines. Its behavior is exhaustive over the categories the classifier is supposed to return, and the `case _` is a fallback for the rare case where the classifier returns something unexpected. We come back to that fallback below.

### Why the router prompt should be small and cheap

The classifier runs on *every* input. If a workflow processes a million tickets a month, the classifier is called a million times. That makes it the single highest-leverage prompt in the whole pipeline from a cost-and-latency perspective.

Two design rules follow from that:

**Use the cheapest model that's reliable.** A `gpt-4o-mini`-class model classifies most short text into 3–10 categories with near-perfect accuracy. There is no reason to use a frontier model for this — the marginal accuracy gain is small, the marginal cost is large, and the classifier is the highest-volume step in the system. The handlers, which run only one out of N times per ticket, can use a stronger model if quality requires it. Save the expensive tokens for the steps where they matter.

**Keep the prompt tight.** The classifier's system prompt should be one short paragraph: name the categories, define each in one sentence, say what to do when none match. No examples unless examples genuinely move accuracy. Every token in the system prompt costs you on every input. A 200-token system prompt over a million inputs is 200 million tokens of overhead before you've even sent a ticket.

Phase 4 of this curriculum covers cost optimization in more detail — caching, prompt compression, batched inference — but the routing step is the easiest place to feel the effect: a 5x cheaper model on the highest-volume step in your pipeline is a 5x cost reduction on the entire workflow's classification budget.

### Failure modes: when the classifier returns something the router doesn't know about

The router has one structural failure mode: the classifier produces a label that no `case` arm matches. This happens for a few reasons:

- **Off-distribution input.** The user's message doesn't really fit any of the categories. The classifier picks the closest one but it's a bad fit, or the classifier (despite the prompt) returns a free-text answer like `"unknown"` or `"other"` that you didn't anticipate.
- **Prompt drift.** Someone updates the classifier's system prompt to add nuance and accidentally introduces a new label without adding a corresponding handler. The classifier dutifully returns the new label; the router doesn't know about it.
- **Model upgrade behavior change.** A new model version interprets the prompt differently and returns labels in a slightly different format (`"Refund"` vs `"refund"`, or `"refund_request"` vs `"refund"`). Pydantic with a `Literal` type catches most of this at validation time, but you still need to decide what to do.

The recommended pattern is a *default* or *general* fallback handler. The `case _` arm of the `match` statement routes the unknown label to a safe handler — typically the most generic one, which is designed not to make category-specific assumptions. In this module's project, the fallback is the general handler: the `general` handler extracts only a topic and a key question, and drafts a polite, non-committal response. If the classifier produces something weird, the user still gets a sensible reply; nothing in the pipeline blows up.

A weaker option is to raise an error on unknown labels and let it bubble up to the batch loop, which logs the failed ticket and continues with the rest. This is appropriate when the workflow has a human-review queue for failures. It is the wrong choice when you have no human in the loop — silently dropping tickets is worse than producing a generic response.

The general principle: *every router should have an exhaustive set of arms plus a default*. If you write a router without a default, you are betting that the classifier will never surprise you. In production, the classifier will eventually surprise you.

### Confidence as a routing signal

A more advanced version of the same pattern adds a *confidence* signal to the classifier's output. Instead of returning just a category, the classifier returns `(category, confidence)` where confidence is a number between 0 and 1 (or a label like `low`/`medium`/`high`). The router can then use the confidence to decide whether to trust the category at all.

A typical policy: if confidence is `high`, route normally; if confidence is `medium`, route to the chosen handler but also log the ticket for human review; if confidence is `low`, send the ticket to the fallback handler regardless of category. This gives you a smooth degradation from "fully automated" to "human-in-the-loop" to "safe default" as the input gets weirder, rather than the binary "either the classifier nails it or the fallback catches it."

The confidence value itself is unreliable when produced by an LLM directly (models are notoriously poorly calibrated about their own certainty). A better approach when stakes are high is to run the classifier twice on the same input — possibly with different system prompts — and compare. If the two runs agree, confidence is high; if they disagree, confidence is low. This doubles the classifier cost but only on inputs where the answer is genuinely ambiguous, and the cost is still small because the classifier is the cheapest step in the pipeline.

This module's project does not implement confidence-based routing — it sticks with the simpler "category plus fallback" pattern — but the project is small enough that you can add it as an exercise once you have the basic version running.

---

## 5. Parallel Steps: Fan-Out and Fan-In

The sequential and branching patterns assume each step waits for the previous one. That's the right model when each step needs the previous step's output. It is the *wrong* model when two steps need only the original input — or some shared upstream output — and don't need each other's results at all.

In the support-ticket pipeline, once the classifier has labeled a ticket as `refund`, the next two things that need to happen are: extract the refund-specific entities (order ID, amount, reason) and draft the customer-facing response. Both of those steps take the *same input* (the original ticket text). Neither depends on the other's output. Forcing them to run in series wastes wall-clock time for no reason — by the time the extractor finishes, the drafter could already have produced most of its reply.

### When independence enables parallelism

The mathematical observation is simple. If two independent steps take `t1` and `t2` seconds when run in series, the total wall-clock latency is `t1 + t2`. If they can be run in parallel, the wall-clock latency drops to roughly `max(t1, t2)`. The total work (CPU and tokens) is unchanged — you do the same number of LLM calls, send the same prompts, generate the same outputs — but the wall-clock time can drop substantially.

This matters most when the individual step latencies are comparable. If one step takes 2 seconds and another takes 0.1 seconds, parallelism saves you 0.1 seconds at best, which is barely worth the structural complexity. If two steps both take 1.5–2.5 seconds — which is typical for LLM calls — parallel execution can roughly halve user-perceived latency.

### Why `ThreadPoolExecutor` works for LLM calls

Python is famously single-threaded in the sense that the Global Interpreter Lock (GIL) prevents two threads from running pure-Python code simultaneously. People who know that fact often conclude that threads are useless in Python. For *CPU-bound* work, they are right: a pool of threads doing pure-Python arithmetic does not run faster than one thread doing the same arithmetic.

LLM calls are not CPU-bound. They are *I/O-bound*. Almost all the time spent in a `litellm.completion()` call is spent waiting on an HTTP request to OpenAI or Anthropic. The Python code involved is tiny: serialize a request, send it, sleep on the socket, receive the response, deserialize it. The bulk of the elapsed wall-clock time is the socket sleep.

The GIL releases during blocking I/O. While one thread is sleeping on the OpenAI response, another thread is free to acquire the GIL and run. With `concurrent.futures.ThreadPoolExecutor`, you can issue N HTTP requests on N threads, and they all sleep on their respective sockets at the same time. The wall-clock latency of the whole batch is the latency of the slowest request, not the sum of all latencies.

This is real parallelism for I/O-bound work. It does not require async/await, an event loop, or any change to the synchronous shape of `litellm.completion()`. The threads do the I/O concurrently while the Python interpreter happily juggles them, and the result is a clean speedup.

### A two-call fan-out in code

The pattern in code is short. Submit each independent step to the pool, collect the futures, then call `.result()` to gather the outputs.

```python
from concurrent.futures import ThreadPoolExecutor

def run_handler(text: str, extract_fn, draft_fn):
    with ThreadPoolExecutor(max_workers=2) as pool:
        ex_future = pool.submit(extract_fn, text)
        dr_future = pool.submit(draft_fn, text)
        entities, ex_usage = ex_future.result()
        response, dr_usage = dr_future.result()
    return entities, response, [ex_usage, dr_usage]
```

`pool.submit()` returns a `Future` immediately — it does *not* block. Both LLM calls start at almost the same instant, on two different threads. The first `.result()` call blocks the orchestrator until the extractor finishes; by the time it returns, the drafter is either already done or close to it. The total wall-clock latency is approximately `max(extract_latency, draft_latency)`, not the sum.

The `with` block guarantees that the pool is shut down cleanly when the handler returns — no zombie threads, no resource leak.

### When parallel does NOT help

Parallelism looks like a free lunch, but it has several failure modes worth naming explicitly.

**Rate-limited providers.** If your LLM provider has a tokens-per-minute or requests-per-minute limit, running calls in parallel does not increase your throughput — it just clusters your requests in time, which can push you over the rate limit faster. Either you'll hit 429 errors that retry on a backoff, or your provider's queueing will serialize your calls behind the scenes. The wall-clock speedup vanishes.

**Single-token-budget situations.** Some providers price differently for parallel vs serial usage. More commonly, if your *organisational* budget is the constraint — a single API key with a low daily limit — parallel calls let you burn through that budget faster, which is rarely what you wanted.

**Steps that actually depend on each other.** If step B uses step A's output, you cannot run them in parallel no matter how much you'd like to. Make sure the steps you're parallelising are truly independent. A surprising number of "parallel" pipelines that look correct on paper turn out to have a subtle dependency — for example, step B uses a field from step A's output that you forgot to thread through the inputs.

**Very short steps.** If both steps take 50ms, you've saved 50ms by parallelising them, at the cost of a thread pool. Not worth the complexity. Parallelism is for steps where the latency is measured in seconds.

### Worked latency-vs-cost example

Consider the support ticket pipeline's handler stage for a refund ticket. The two steps inside the handler are `extract_refund` and `draft_refund`, each making one LLM call. Suppose, for a representative ticket, the latencies are:

- `extract_refund`: 1.4 seconds
- `draft_refund`: 2.1 seconds

**Serial.** Run `extract_refund` first, then `draft_refund`. Wall-clock latency = `1.4s + 2.1s = 3.5s`. Total tokens used: `(extract_tokens + draft_tokens)`. Total cost: 100% of the baseline (we use this run as the baseline for the ratio).

**Parallel.** Submit both to a `ThreadPoolExecutor`. Wall-clock latency = `max(1.4s, 2.1s) = 2.1s`. Total tokens used: `(extract_tokens + draft_tokens)` — *unchanged*. Total cost: 100% of the baseline — *unchanged*.

The parallel version is `(3.5 - 2.1) / 3.5 ≈ 40%` faster in wall-clock time, at exactly the same token cost. The drafter and the extractor send the same prompts and receive the same outputs in both cases — there is no token saving from parallelism. What you save is *time*.

That is the rule to remember: **parallel saves latency, not tokens.** If your bottleneck is cost (you're trying to fit inside a budget), parallelism doesn't help. If your bottleneck is latency (the user is waiting for a response), parallelism is a clean, cheap win.

There is a small operational subtlety worth naming. The `total_latency_ms` your workflow reports is the *wall-clock* time of the whole run, which is what the user feels. The sum of per-step latencies is the total *work* done, which is what your provider invoices against (in spirit — they actually invoice on tokens, not seconds, but the time spent on the provider side correlates with both). When you parallelize, those two numbers diverge: the total report shows roughly `max(step)` but the sum of step latencies is still `sum(step)`. Both numbers are useful. Wall-clock latency tells you about user experience; sum of step latencies tells you about how busy your workers are. Reports that show both side-by-side make it obvious whether parallelism is actually being achieved or whether the steps have silently re-serialized due to a rate limit.

In this module's project, the parallel handler reduces the per-ticket wall-clock latency by roughly the gap between the slower step and the sum of both — typically 1–1.5 seconds per ticket. Across a million tickets a month, that adds up to a measurable improvement in throughput on the same fleet of machines, which becomes a real infrastructure saving even though no individual ticket costs less to process.

### When to reach for asyncio instead

`ThreadPoolExecutor` is the right tool here because (a) the number of concurrent LLM calls inside a single workflow is small — typically 2–4 — and (b) `litellm.completion()` is a synchronous function. Threads slot in without changing the shape of the surrounding code. The mental model stays simple.

If you are running *hundreds* of concurrent requests inside one process — for example, a batch endpoint that fans out a single user query across hundreds of documents — threads start to feel heavy. Each thread has stack overhead, and at the high end, context switching costs become visible. `asyncio` is the natural fit for that scale: lighter coroutines, an event loop instead of OS threads, and `litellm`'s async equivalent (`acompletion`) plugs into it naturally.

For everything below that scale — which covers the overwhelming majority of LLM workflows — threads are sufficient and the simpler choice. This module sticks with threads.

### What about more than two parallel steps?

The two-step fan-out shown above (extract and draft running in parallel) is the simplest version of the pattern. Nothing about `ThreadPoolExecutor` is restricted to two workers — you can submit any number of independent tasks to a pool sized for the workload and join all their results. A workflow that needs to fan out across four independent analyses, or score a single document against six rubric items in parallel, uses exactly the same shape: submit N futures, call `.result()` N times.

Two caveats become important once N grows beyond a handful. First, the pool's `max_workers` setting is real — if you submit N tasks to a pool of size M < N, the extra tasks queue behind the first M, and the wall-clock latency approaches `(N/M) * average_step_latency` rather than `max(step)`. Size the pool to the number of concurrent calls you actually want, not the default. Second, your provider's rate limit becomes the binding constraint earlier than you'd expect when fan-out widens — five parallel requests per ticket times a hundred concurrent tickets is five hundred concurrent requests, which most providers will throttle.

The right rule for sizing parallelism: parallelize as wide as the work genuinely allows, but no wider than the provider will support. A two-step or three-step fan-out is usually well within any provider's headroom; a fifty-step fan-out almost always isn't.

---

## 6. Workflow vs Agent: Choosing the Right Tool

The first six modules of Phase 3 (modules 10 through 13) cover four different ways to wire up an LLM-powered system: a single LLM call with structured output, a single agent that loops with tools, a multi-agent system with specialist agents and an orchestrator, and now a workflow with deterministic control flow. By the end of this module, you have all four in your toolbox. Picking the right one for a given task is the central design skill.

There is no single right answer in the abstract — each pattern is the best fit for some tasks and the wrong fit for others. The choice is driven by the task's structure, the importance of predictability vs flexibility, and your tolerance for cost and latency.

### Comparison across dimensions

| Dimension | Workflow (this module) | Agent (Module 11) | Multi-agent (Module 12) |
|---|---|---|---|
| Control flow | Static — decided at design time | Dynamic — decided at runtime by the LLM loop | Dynamic, coordinated — orchestrator (or peer agents) decide which specialist runs |
| Predictability | High — same input class produces the same path | Low — the agent may take different paths through the same input on different runs | Medium — the orchestrator decides, so paths are more constrained than a free agent but still vary |
| Cost per run | Lowest — only the necessary LLM calls run | Variable — depends on how many tool iterations the agent takes | Highest — multiple agents, each with their own context, plus orchestrator overhead |
| Testability | Easy — each step is a pure function with typed I/O | Hard — the loop's path depends on model nondeterminism; tests are brittle | Hard — same as agents, plus the orchestrator adds another layer to trace |
| Failure isolation | Per step — if step 3 fails, you still have steps 1–2's outputs | Global — a mid-loop failure typically aborts the whole run | Per agent — a specialist failure is contained, but the orchestrator must handle it |
| Adaptive behavior | None — the structure is fixed | Yes — the agent can choose tools and stop conditions at runtime | Yes — the orchestrator (and individual specialists) can adapt |

The pattern of trade-offs is regular. Workflows are easier to operate (predictable, testable, cheap, isolated failures) but cannot adapt. Agents are flexible but harder to operate. Multi-agent sits between the two — more structured than a free agent, less rigid than a workflow.

### A decision tree (in prose)

When you have a new LLM-powered task and need to pick a pattern, walk through these questions in order. Each "yes" picks a pattern; each "no" moves on.

**Is the order of steps known up front?** That is, can you sit down and write the list of steps the system will execute, in order, before you've seen any input? If yes, you want a workflow. The deterministic control flow buys you predictability, testability, lower cost, and easier observability without giving up anything you actually need. Stop here.

If no — the right next step really does depend on what previous steps discovered, in a way you can't enumerate ahead of time — move on.

**Do agents collaborate, or is one agent sufficient?** That is, does the task have separable sub-tasks that each warrant their own specialist (researcher, writer, critic), or is it one fuzzy task that one agent should handle end-to-end with a tool loop? If many specialists are clearly warranted, you want a multi-agent system (Module 12). If one agent is enough — open-ended research, code generation, a single complex task that needs flexible tool use — you want a single agent (Module 11).

That's the whole tree. Three questions, four leaf nodes (workflow, single agent, multi-agent, and "fall back to a single LLM call" — the implicit default when none of the above is needed). The answers usually become obvious within a minute of thinking about the task.

The most common mistake is skipping the first question. Teams often jump to "we'll build an agent" because agents feel more capable, then spend weeks adding rules to constrain the agent's behavior — rules that, taken together, describe a workflow. The agent is doing the same fixed work each time, just paying for the loop overhead on every run. Asking "is the order known up front?" honestly, at the start, prevents this.

### Nesting: workflows and agents compose

The four patterns are not mutually exclusive within a single system. They nest cleanly:

**Workflows can call agents as one of their steps.** A workflow step might be "do open-ended research on the topic," which is genuinely the kind of task an agent handles well. The step function wraps a small ReAct agent (as built in Module 11), returns its final structured output, and the workflow continues. The orchestrator does not need to know that the step internally ran a tool loop; it just gets back a typed result.

This is a common production pattern. The workflow gives you the predictable spine — same steps every time, in the same order — and the agent inside the research step handles the part of the task that genuinely benefits from runtime tool selection. You get most of the benefits of workflows (predictable structure, easy testing of non-agent steps) while still being able to do open-ended work where it is required.

**Agents can call workflows as tools.** Conversely, a single agent might have access to a tool called `triage_ticket(text: str) -> TriagedTicket`. From the agent's perspective, it's a tool. Internally, that tool runs a full workflow — classify, route, fan-out, fan-in. The agent doesn't know or care; it just gets back a structured result and decides what to do next based on it.

Neither composition is "better." They solve different problems. The first wraps unpredictable sub-tasks inside a predictable shell. The second exposes a predictable sub-task to an agent that needs it as one of many tools. In a large system you will often see both: workflows that call agent-shaped steps for the parts that are genuinely open-ended, and agents that call workflow-shaped tools for the parts that have known structure.

The deeper point: workflow vs agent is not a religious choice about which paradigm to commit to. It is a per-sub-task decision. Some sub-tasks have known structure — workflow them. Some have unknown structure — agent them. The right system uses both where each fits.

---

## 7. Observability & Testing

A workflow is easier to observe and test than an agent for the same fundamental reason it is cheaper to run: its structure is known in advance. There is no "what did the agent decide to do this time?" question — the structure of every successful run is the same. That regularity is the foundation everything else in this section builds on.

### Each step is a pure function

In a well-constructed workflow, each step is a pure function from input to output. Same input → same output, modulo the LLM's nondeterminism inside the step itself. The function does not consult global state, it does not hold hidden context from prior runs, and it does not depend on the order in which the orchestrator happens to call it (the orchestrator's call order is fixed at design time).

This is a stronger claim than it sounds. In an agent, the "step" is really an iteration of a loop that depends on the loop's prior history — the scratchpad, the previous tool results, the running summary. Two runs of an agent on the same input can take different paths because the model's stochasticity in step 2 changes what step 3 even *is*. There is no clean unit to unit-test.

In a workflow, by contrast, step 3 is the same function-call regardless of what step 2 produced. The *content* of step 3's input changes from run to run (because the model's output in step 2 changes), but the *identity* of step 3 — its name, its system prompt, its expected input type, its expected output type — does not. That stability is what lets you treat each step as an independent unit and write tests against it.

In a workflow, every step is a function with a typed signature. To test it, you call it with a representative input and assert against its output. You can mock the LLM call (or run it for real, depending on the test), feed in deterministic data, and check the shape and content of the result. The test runs in isolation; it doesn't require setting up the rest of the pipeline.

Per-step testing is *tractable* in a workflow in a way it genuinely is not in an agent. You can write a unit test for each step function. You can write integration tests that compose a small subset of the steps (classify-only, classify-then-extract). You can write end-to-end tests that exercise the full pipeline on canonical inputs. All three levels are normal things to write. Agent traces, by contrast, are usually tested only at the end-to-end level — with snapshot tests that are brittle to model updates — because the intermediate steps are not stable units.

### Per-step token, cost, and latency tracking

A second consequence of having known step boundaries is that you can attribute cost and latency to each step. This module's project puts a `StepUsage` record next to every step's output:

```python
class StepUsage(BaseModel):
    step: str            # "classify" | "extract" | "draft"
    input_tokens: int
    output_tokens: int
    cost: float
    latency_ms: int
```

Every step function returns a tuple of `(result, StepUsage)`. The orchestrator collects the `StepUsage` records into a list and produces a per-run report at the end:

```text
=== Step Usage ===
step       in     out    cost      latency
classify     85    62  $0.001200    1234ms
extract      90    48  $0.000900    1421ms
draft        92   140  $0.002100    2102ms
TOTAL                  $0.004200    3336ms  (parallel: extract+draft overlap)
```

Three things become visible from this report that are obscured in agent traces:

**Which step is the cost driver.** In the example above, the drafter is the most expensive step. If costs are growing across a quarter of usage, the report tells you immediately whether the drafter or the classifier or the extractor is responsible, without you having to instrument anything additional.

**Which step is the latency bottleneck.** The drafter is also the slowest step. If you want to cut wall-clock latency further, the drafter is where to put the effort — maybe a smaller model, maybe a shorter system prompt. The classifier, despite being upstream of everything, is not the bottleneck; speeding it up by 50% saves you only 600ms.

**Whether parallel actually helped.** The `TOTAL` line shows wall-clock time, which is less than the sum of the per-step latencies when steps ran in parallel. If the total is `extract_latency + draft_latency` (sum), your fan-out has degraded into serial. If the total is closer to `max(extract_latency, draft_latency)`, parallelism is working.

In a multi-agent system you can get something similar by instrumenting each specialist call, as covered in [Module 12 (section 6)](../12-multi-agent-systems/#6-coordination-and-control). The difference is that in a workflow the *set* of step records is fixed and predictable, so dashboards can be built against a stable schema. In an agent run, the number of steps and their nature varies per run, so dashboards must aggregate over wildly different traces.

### Failure isolation: one step doesn't poison the rest

When step 3 fails — the LLM returns malformed JSON, the model API times out, the input violates a schema — the workflow has a well-defined state at the moment of failure: steps 1 and 2 produced valid outputs, step 3 raised an exception, steps 4–N never ran. The orchestrator can decide what to do based on what it has.

In single-ticket mode, the right behavior is usually to surface the error to the caller with the raw text and the step at which it failed, so a developer can investigate. The earlier step outputs are still available for debugging — you can see exactly what step 3 was given as input and reason about why it failed.

In batch mode, failure isolation matters far more. The orchestrator processes each ticket independently. If ticket 17 of 100 fails at the extractor step, that does not affect tickets 1–16 (already done) or tickets 18–100 (still to do). The orchestrator catches the per-ticket exception, logs it, appends an error record to the results, and moves on. The other 99 tickets succeed normally.

```python
def triage_batch(tickets: list[str]) -> list[TriagedTicket | dict]:
    results = []
    for text in tickets:
        try:
            results.append(triage(text))
        except Exception as e:
            results.append({"error": str(e), "input_text": text})
    return results
```

Compare this to a long-running agent that fails halfway through. The agent has accumulated a scratchpad of partial state, may have side effects in flight (a half-written file, a half-issued API call), and the failure typically aborts the whole run. Resuming requires either replaying the scratchpad or starting from scratch. Failure isolation in a workflow is *structural*; in an agent it requires deliberate engineering (checkpointing, idempotent tools, recovery logic), which is real work that most agent implementations skip.

### Compared to agent traces

The contrast with agent observability sharpens the point. An agent trace is an unbounded scratchpad whose shape depends on what the model chose to do. You log every iteration: the thought, the tool call, the tool result, the new state. After ten iterations, you have ten log entries. After fifteen, fifteen. The *order* of the entries was decided by the model. The *number* of entries varies per run. The *content* of entry N depends on entries 1 through N-1.

Two observations follow:

**Workflow logs have a known shape; agent logs do not.** A successful workflow run produces a fixed set of step records. You can build a dashboard that groups by `step` and aggregates `latency_ms` across thousands of runs. Agent runs have variable shape; building the same dashboard requires deciding how to bucket steps that don't have stable names.

**Workflows are auditable; agents need replay tooling.** If you want to know "why did this ticket get routed to the refund handler?", the workflow's log answers it directly: the classifier produced `category=refund`, the router dispatched. If you want to know "why did the agent decide to call `web_search` instead of `fetch_url`?", you need to read the model's thoughts, infer its reasoning, and possibly replay the run with logging at a finer grain. Both questions are answerable, but the first is a SQL query and the second is an investigation.

**Workflow regressions are localisable; agent regressions are not.** When a workflow starts producing worse output after a model upgrade, the per-step records tell you which step is responsible — the extractor's outputs got vaguer, or the drafter's responses got longer, or the classifier started splitting `general` more aggressively. You can pin the model on just the regressing step while leaving the others alone. In an agent, the same regression manifests as "the agent's trace looks different now," which is much harder to act on because there's no clean unit to revert.

This is not a knock on agents — open-ended tasks genuinely need the flexibility agents provide, and the auditability cost is the price of that flexibility. The point is that when the task is workflow-shaped, taking the workflow path also gives you the operational benefit of cleaner observability for free. You're not paying extra for it.

### What you still need to test

Workflows are easier to test than agents, but they are not *self*-testing. There are specific things to verify that won't catch themselves:

**The classifier's routing accuracy.** Does the classifier produce the right category for representative inputs in each class? Build a small labeled set — a handful of refunds, technical issues, general inquiries, and intentionally ambiguous tickets — and run the classifier on them. The expected category should be the actual category at least 90% of the time on the easy cases and ideally close to 100% on the unambiguous ones. If accuracy drops on a model upgrade, you'll see it here before it shows up as a production incident.

**The router's coverage.** Every category the classifier can produce must map to a handler. The `case _` fallback catches anything the classifier produces that you didn't anticipate. A simple test: for each value of the `TicketCategory` Literal, call `route()` and assert that it returns a non-None pair of handlers. This guards against the prompt-drift failure mode where the classifier learns to produce a new label but the router was never updated.

**End-to-end latency, especially parallel sections.** Run a representative ticket through the full pipeline and inspect the `total_latency_ms` versus the sum of per-step latencies. If `total_latency_ms ≈ sum(per_step_latency)`, your parallel fan-out is silently serial — usually because of a thread pool misconfiguration, a rate limit, or a step that accidentally has a sequential dependency. Catch this in a test rather than in production.

**Schema stability of step outputs.** Every step returns a typed Pydantic object. Snapshot the schema of each output type and watch for drift. If a developer accidentally renames a field on `TicketClassification`, every downstream step that uses that field breaks — but the breakage shows up at validation time, immediately, with a clear error message. This is the same property that made structured output worth the discipline back in [Module 08](../08-structured-output/).

**Fallback-handler behavior.** The `case _` arm of the router is the rarely-exercised path that catches off-distribution inputs. Because it rarely runs, it rarely gets tested in production by accident — which means you have to test it on purpose. Feed the workflow a few inputs that are deliberately weird (a ticket that's just emoji, a message in another language, a request that fits no category) and verify the fallback handler produces a sensible response rather than blowing up. The point of the fallback is that the system degrades gracefully on bad inputs; that promise only holds if the fallback path is exercised in tests.

Together these tests give a workflow a level of operational confidence that an equivalent agent rarely achieves. They are not free — you have to write them — but they are *possible to write* in a way that the equivalent agent tests usually are not.

---

## 8. Workflows in the AI Stack

This module's project is a single-process Python script with no external dependencies beyond `litellm` and `pydantic`. That is on purpose. The point of this module is to teach the structural patterns of workflows without anchoring them to any particular framework. Once you've built one workflow by hand, you understand what the frameworks are doing and you can choose one (or not) deliberately.

### Real-world workflow tools

Several mature frameworks exist to build production workflows on top of the same structural ideas this module covers. Each one trades a different bundle of features for the small lift of adopting a new abstraction:

**LangChain LCEL** (LangChain Expression Language) gives you a small set of composable operators — `RunnableSequence`, `RunnableParallel`, `RunnableBranch` — that map directly to the three patterns in this module. The big win is built-in streaming, batching, and tracing through LangSmith. You give up some Python clarity (the operators are more declarative than functional) in exchange for less boilerplate around the cross-cutting concerns.

**LangGraph** is a step up from LCEL when your workflow has loops, conditional re-entry, or genuinely graph-shaped control flow that isn't a clean chain. You define nodes and edges; the graph engine handles state passing, checkpointing, and resumption. It is the right framework when a workflow needs to be durable across process restarts or when human-in-the-loop steps need to pause and resume the pipeline.

**Haystack** is the mature option for RAG-flavored workflows specifically. It comes with pipeline primitives for retrieval, reranking, and generation, plus an opinionated set of components for common tasks. If your workflow is "retrieve, augment, generate, evaluate" with light variations on each step, Haystack lets you assemble it from components rather than writing each step from scratch.

**Prefect / Temporal** are general workflow orchestrators, not LLM-specific. They give you durable execution (a workflow can survive process crashes), retries with backoff, scheduled runs, distributed task execution, and a UI for inspecting workflow history. You reach for them when your "workflow" is really a piece of production infrastructure — running on a schedule, processing thousands of items, needing recoverability — and the LLM calls are one part of a larger pipeline that includes database writes, queue management, and external API integrations.

These four are not the only options — there are dozens of smaller libraries and a few platforms (Inngest, Trigger.dev) aimed at the same space — but they cover the design space well. LCEL and LangGraph are LLM-native and lightweight; Haystack is RAG-flavored and component-driven; Prefect and Temporal are general-purpose and durable. Most production LLM systems end up using either no framework at all (small enough not to need one) or one of these four (large enough to want the operational features).

### When to reach for one — and when to skip

The decision is mostly about scale and operational requirements, not about LLM-specific concerns:

**Reach for a framework** when you have production workloads with non-trivial reliability requirements. If the workflow needs to retry on transient failures, survive process restarts, run on a schedule, expose a tracing UI to non-engineers, or distribute work across machines, you want the orchestrator features. Building these from scratch is a real project, and the frameworks have years of operational hardening you'd otherwise reinvent badly.

**Skip the framework** for a single-process script — like the project in this module. When the workflow runs in one process, processes one input at a time (or a small batch), and doesn't need durability across crashes, the structural patterns of this module are the entire job. A framework would add a dependency, a learning curve, and an abstraction layer for no benefit. Plain Python with `concurrent.futures` and Pydantic is the right tool.

The transition point is usually a function of three things: (1) is this code running in production with real users depending on it, (2) does it need to recover from failures automatically, and (3) does it process enough volume that operational issues happen regularly enough to need infrastructure rather than manual investigation. Two or three "yes" answers and you want a framework. Zero or one and you don't.

### Relationship to other modules

Workflows are the structural backbone that the prior modules' capabilities slot into. Each prior module describes a capability that becomes a step (or a piece of a step) inside a workflow:

**[Module 06](../06-tool-use-function-calling/) — tools as workflow steps.** Function-calling-style tools work as workflow steps with no modification. A workflow step that performs an HTTP fetch, a database lookup, or a shell command is just a tool wired into a chain at design time, rather than being made available to an agent that decides when to call it. Inside a workflow, the call is *unconditional* — the step always runs, the LLM is not the one deciding — which is exactly the point.

**[Module 07](../07-rag/) — RAG retrieval as a chain step.** A RAG pipeline (embed a query, search a vector index, retrieve top-k chunks) is itself a small workflow. As a step inside a larger workflow it becomes a function that takes a query and returns a list of chunks. This module's project doesn't use RAG, but adding a "retrieve relevant past tickets" step between `classify` and the handler would be a one-function change — the retrieval step slots in as cleanly as any other.

**[Module 11](../11-building-ai-agents/) — an agent inside a workflow step.** Section 6 covered nesting at the conceptual level: a workflow step can wrap a full ReAct agent. The step takes a typed input, the agent runs its loop, the step returns a typed output. The orchestrator doesn't know the step is internally agentic. This is how you keep the predictable spine of a workflow while allowing one sub-task to do open-ended tool use.

**[Module 12](../12-multi-agent-systems/) — a multi-agent system inside a workflow step.** The same nesting applies at the multi-agent level. A workflow step could wrap the orchestrator-plus-specialists pipeline you built in Module 12 — for example, "expand this user request into a researched blog post" might be one step in a larger content-publishing workflow. The workflow doesn't care that the step is internally a multi-agent collaboration; it just gets back a typed result.

The unifying theme: workflows are the *outer* shape, agents and multi-agent systems are the *inner* shape when a sub-task genuinely needs runtime control flow. Most production LLM systems are workflows-with-agentic-steps, not pure agents and not pure workflows. Knowing which shape goes outside and which goes inside is the design question this module's project is meant to make tangible.

A short worked example to ground the composition. A customer-support product might be a workflow with these top-level steps: receive ticket → triage (this module's project) → resolve. The triage step is itself a workflow (classify, route, extract, draft). The resolve step might be an agent — because actually resolving a ticket sometimes requires a database lookup, sometimes a refund API call, sometimes a follow-up question to the customer, and the right sequence genuinely depends on what the ticket asks. The outer shell is predictable (every ticket goes through these three steps), the middle is also predictable (every triaged ticket follows the same four-step workflow), and only the leaf step that genuinely needs runtime flexibility is agentic. The total system is observable and testable at every layer except inside the leaf agent, which is the smallest amount of agent-shaped work that gets the job done.

Compare that to the naive "make the whole thing an agent" architecture, which gives you one big opaque loop where the model decides at every iteration what to do next. The flexibility is there in principle; in practice it's spent re-deciding things the workflow already knew. The agent-as-everything design is rarely the right one once you've internalised that workflows can host agents as a step.

### Forward pointer: Module 14 and Phase 4

**Module 14 (AI Code Generation)** builds on the patterns you've just learned. AI code generation is, in practice, a workflow: parse the prompt, retrieve relevant context, generate the code, run tests, possibly iterate. Each step is bounded, typed, and observable in the way this module's project is. The agent loop that AI-pair-programming products feel like at the surface is, under the hood, a workflow with a tightly scoped revision sub-loop — much like the critique loop in [Module 12](../12-multi-agent-systems/) wrapped in a chain shape much like this module's.

Beyond Module 14, Phase 4 of the curriculum covers the operational themes that workflows make tractable: caching (cache hits across step boundaries, since each step's input and output is a typed object you can hash), observability (the per-step `StepUsage` records become the rows of a real metrics pipeline), and deployment (workflows are easier to ship to production than agents because their resource envelope is predictable). The instrumentation introduced in this module — `StepUsage`, per-run reports, parallel-vs-serial comparisons — is the foundation those Phase 4 topics build on.

The arc of the curriculum from this point on is: take the patterns you've learned to build LLM-powered systems (single calls, agents, multi-agent, workflows) and learn how to run them well. That means caching, monitoring, evaluating, and deploying. Workflows are the friendliest shape to operationalise, which is why this module sits between the *building* modules of Phase 3 and the *running* modules of Phase 4. Once you can build a workflow, you can run it — and most of what running it well looks like comes from the structural choices you make at design time, which this module has now put in your hands.

### Module cross-reference map

| This module's component | Prior module it builds on |
|---|---|
| Pydantic types as step communication contracts | [Module 08](../08-structured-output/) — structured output and schema validation |
| A tool call as a workflow step | [Module 06](../06-tool-use-function-calling/) — function calling and tool dispatch |
| A RAG retrieval as a workflow step | [Module 07](../07-rag/) — embedding, retrieval, and context injection |
| An agent wrapped in a single workflow step | [Module 11](../11-building-ai-agents/) — the ReAct loop, stop conditions, and observability |
| A multi-agent collaboration wrapped in a single workflow step | [Module 12](../12-multi-agent-systems/) — specialist agents and orchestration |

Each prior module's technique slots into a workflow without modification. The workflow is the spine; the techniques are the steps. Internalising this composition — workflows on the outside, agentic and tool-using techniques on the inside — is the design principle that ties Phase 3 of this curriculum together. Phase 4 will take that combined toolbox and teach the operational patterns that make it production-grade.

---
