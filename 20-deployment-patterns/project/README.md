# Project: Resilient LLM Wrapper

Build a `ResilientLLM` class that composes six policies (budget gate, fallback chain, circuit breaker, retry loop, timeout wrapper, alerting) around a `Provider` protocol, then prove each layer fires by running deterministic chaos scenarios against an in-process mock provider.

## What you'll build

- A `ResilientLLM` orchestrator that wires six policies into a single `call()` method.
- A `Provider` protocol with two implementations: `LiteLLMProvider` (real, used by `--ask`) and `MockProvider` (chaos-scripted, used by `--chaos`).
- Seven named chaos scenarios that demonstrate each policy: `happy`, `rate_limit_burst`, `primary_dead`, `circuit_trip`, `slow_burn`, `runaway`, `slow_primary`.
- A `MetricsRecorder` that aggregates per-call outcomes into a `RunSummary` and emits alerts to a pluggable `AlertSink`.
- A 4-mode CLI: `--ask`, `--chaos <name>`, `--list-scenarios`, `--report`.

## Prerequisites

- [Module 04 (AI API Layer)](../../04-ai-api-layer/) — the `litellm.completion()` surface and how cost capture works.
- [Module 15 (Evaluation & Testing)](../../15-evaluation-testing/) — the "scripted scenario + summary report" loop is the same shape as the eval harness from M15.
- [Module 17 (Caching & Cost Optimization)](../../17-caching-cost-optimization/) — the `try/except` cost-capture pattern around `litellm.completion_cost` is reused here.
- [Module 18 (Observability & Monitoring)](../../18-observability-monitoring/) — this module picks up the alerting thread M18 deferred. The `AlertSink` is the same shape as M18's `SpanSink`.

## Setup

`.env` at the repo root supplies your API key. The script resolves it three levels up (`parent.parent.parent / ".env"`).

Set `LLM_MODEL` in `.env` (or your shell environment) to pick the model for `--ask`. Default is `openai/gpt-4o-mini` if unset.

**No new dependencies.** Everything we use (`litellm`, `pydantic`, `python-dotenv`) is already in `requirements.txt` from earlier modules.

### Project layout

```text
project/
├── README.md                this file
├── solution.py              wrapper + policies + mock + CLI (~700 lines)
├── .gitignore               .resilient_state.json + alerts.jsonl
└── (created at runtime)
    ├── .resilient_state.json
    └── alerts.jsonl         only if --alert-file passed
```

Read `solution.py` top to bottom before running it. Each policy class is small (40–80 lines) and stands on its own. The `ResilientLLM.call()` method is the orchestrator and is the one place in the file where the policies meet.

## Walkthrough

Run these four steps in order. Each builds on the previous.

### Step A — List the scenarios

```
python solution.py --list-scenarios
```

Expected output (formatting matters; the table aligns on the scenario name):

```
Scenario               Description
--------------------------------------------------------------------------------
happy                  10 successful calls with low latency; baseline metrics
rate_limit_burst       primary rate-limits 2 times then recovers; retries fire with retry_after honored
primary_dead           primary returns auth_error forever; fallback chain engages every call
circuit_trip           primary returns 503 12 times in a row; circuit opens, fallback engages
slow_burn              primary alternates ok/503 for 20 calls; error-rate alert fires
runaway                both providers ok forever; CLI loops aggressively to trigger budget kill-switch
slow_primary           primary always takes 30s (well over the per-attempt timeout); fallback is fast
```

### Step B — Run the baseline scenario

```
python solution.py --chaos happy
```

Watch for: 10 calls, 10 ok, 0 retries, 0 fallback hops, 0 alerts. This is what a healthy system looks like. The `Saved to .resilient_state.json` line at the end means the summary persisted.

### Step C — Watch retries fire

```
python solution.py --chaos rate_limit_burst --fast
```

The `--fast` flag scales every latency and every retry_after by 100x, so the scenario runs in under 5 seconds instead of 30+ seconds. The report should show:
- Calls: 10 total, 10 ok, 0 failed
- Retries: greater than 0 (the primary rate-limited 2 times before recovering)
- Fallback hops: 0 (retries succeeded; no need to fall back)

The retries fired and worked. The fallback was never engaged. That's the right outcome.

### Step D — Watch the circuit open and fallback engage

```
python solution.py --chaos circuit_trip --fast
```

The report should show:
- Calls: 12 total, 12 ok, 0 failed
- Retries: greater than 0 (the primary retried before each fallback hop)
- Fallback hops: 12 (every call ended up at the fallback)
- Circuit trips: 1 (the primary's breaker tripped CLOSED to OPEN)
- Alerts: 2 fired (1 circuit_open, 1 latency_p95)
- Per-provider: `mock_primary` shows `circuit=OPEN`, `mock_fallback` shows `circuit=CLOSED`

You should see two `[ALERT]` lines printed to stderr before the summary. The first is the circuit opening; the second is the latency-p95 alert — the retry-plus-fallback path is slow enough under `--fast` to cross the threshold.

> **Why isn't the error-rate alert firing here?** `circuit_trip` exhausts retries on the primary and falls back to a healthy secondary. Every call ends up successful from the wrapper's point of view, so the wrapper-level error rate stays at 0%. The error-rate alert is wired to the wrapper outcome, not to per-provider failure rates — see `slow_burn` for a scenario where it does fire.

## Worked Exercise 1 — Tune the circuit breaker

Edit `solution.py`, find `DEFAULT_CIRCUIT_FAILURE_THRESHOLD = 5`, and change it to `2`.

Re-run:

```
python solution.py --chaos slow_burn --fast
```

Observe: the circuit trips earlier (after 2 consecutive failures instead of 5). Fewer wasted calls against the bad primary, but more flapping — the circuit will re-open and re-close as the alternating pattern hits it. Compare to the default threshold and reason about the tradeoff: low threshold reacts fast but is jumpy; high threshold is stable but slow. There is no "correct" value; it depends on how flaky the underlying provider is.

Restore the default before moving on.

## Worked Exercise 2 — Add a chaos scenario

Add a new entry to the `_SCENARIOS` dict in `solution.py`:

```python
"partial_outage": ChaosScenario(
    name="partial_outage",
    description="primary returns ok/503 alternating with bursts; fallback is healthy",
    primary_steps=[
        Step("ok", latency_ms=80),
        Step("server_error", latency_ms=80, status=503),
        Step("server_error", latency_ms=80, status=503),
        Step("ok", latency_ms=80),
        Step("ok", latency_ms=80),
        Step("server_error", latency_ms=80, status=503),
    ] * 4,
    fallback_steps=[Step("ok", latency_ms=80) for _ in range(24)],
    loop_n=24,
    seed=42,
),
```

Re-run:

```
python solution.py --list-scenarios
python solution.py --chaos partial_outage --fast
```

The `--list-scenarios` output should now include `partial_outage`. The `--chaos partial_outage` run should fire the `ErrorRateAlert` at least once because half of primary's responses fail.

## Live-test

Make sure your `.env` has a valid API key, then:

```
python solution.py --ask "what is exponential backoff in two sentences"
```

You should see the model's answer, followed by a `Summary` block showing 1 call, 1 ok, 0 retries, 0 fallback hops, a real `budget_spent_usd` value (typically $0.0001 to $0.001 for a small prompt), and `circuit=CLOSED` for the litellm provider.

## Where to go next

- **Replace `StderrAlertSink` with a webhook sink.** The `AlertSink` protocol is one method (`emit`); implementing a `SlackAlertSink` or `PagerDutyAlertSink` is ~20 lines using `requests`.
- **Migrate to `tenacity` for retries.** Compare the decorator-based API to the hand-rolled loop. Notice that `tenacity` handles all the cases this module's `RetryLoop` handles (and more), but the layering with the circuit breaker requires explicit composition.
- **Wire the alert sink into M18's trace recorder.** In production, the alert sink and the trace recorder both look at the same span stream — the alert sink filters, the trace recorder writes everything. Implementing this unification is the natural follow-on to both M18 and M20.
- **Use `litellm`'s built-in retries and fallbacks.** Pass `num_retries=3, fallbacks=[...]` to `completion()`. The library version handles many of the same cases. Read the LiteLLM docs to see what's overlap and what's missing.
