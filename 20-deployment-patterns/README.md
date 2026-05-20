# Module 20: Deployment Patterns

**What you'll learn:**
- Why production LLM applications need a dedicated resilience layer
- The six policies that compose into one wrapper, and why their order matters
- Retries with exponential backoff and jitter — what to retry and what never to retry
- The circuit breaker as Fowler describes it, applied per-provider for LLM apps
- Fallback chains, cheaper-model spare tires, and the silent-quality-drop trap
- The budget kill-switch as the runaway-loop guard
- Alerting on SLO breaches — the M18 thread, finally resolved
- The library landscape and when not to build this yourself

| Detail        | Value                                                                                                |
|---------------|------------------------------------------------------------------------------------------------------|
| Level         | Intermediate-Advanced                                                                                |
| Time          | ~3.5 hours                                                                                           |
| Prerequisites | Module 04 (AI API Layer), Module 15 (Evaluation & Testing), Module 17 (Caching & Cost Optimization), Module 18 (Observability & Monitoring) |

---

## Table of Contents

1. [Why Production LLM Apps Need a Resilience Layer](#1-why-production-llm-apps-need-a-resilience-layer)
2. [The Six Policies and Why This Order](#2-the-six-policies-and-why-this-order)
3. [Retries with Exponential Backoff + Jitter](#3-retries-with-exponential-backoff--jitter)
4. [The Circuit Breaker](#4-the-circuit-breaker)
5. [Fallback Chains: Cheaper Models as Spare Tires](#5-fallback-chains-cheaper-models-as-spare-tires)
6. [Budget Kill-Switch](#6-budget-kill-switch)
7. [Alerting on SLO Breaches](#7-alerting-on-slo-breaches)
8. [The Ecosystem & When Not to Build This Yourself](#8-the-ecosystem--when-not-to-build-this-yourself)

---

## 1. Why Production LLM Apps Need a Resilience Layer

Every module before this one has been able to assume that the model call, when made, completes. The prompt is rendered, the request is dispatched, the response comes back, and the application proceeds. The evals catch the cases where the response is wrong; the caching layer makes the response cheaper; the observability layer records what happened. None of those modules has had to grapple with what happens when the response does not come back at all, or when it comes back after thirty seconds, or when it comes back at the rate of a hundred 429s a second, or when it comes back ten thousand times in five minutes because a loop slipped its leash. Those failure modes are the subject of this module, and the layer that handles them is the last layer between a working prototype and a production system that can be on-call rotated.

### Provider outages are real, and more common than mature SaaS

The first thing to understand is that LLM providers fail more often than the SaaS infrastructure most engineers are used to depending on. Look at the public status pages over the last two years — `status.anthropic.com`, `status.openai.com`, the Google Vertex AI status dashboard, AWS Bedrock's regional health page — and the incident density is conspicuously higher than that of, say, Stripe or Twilio. The reasons are structural: the underlying model serving infrastructure is younger, the request shapes are heavier and harder to load-balance, the autoscaling decisions are made on token-throughput signals that lag behind request counts, and the rate-limit envelopes are tighter and recalibrated more often than the comparable knobs on mature payment infrastructure.

The observable availability of major LLM providers over the last two years has trended closer to three-and-a-half nines than to the four-or-five nines that Stripe or Twilio ship, judging by the cumulative downtime reported on their status pages. A system that depends on a single provider inherits that floor as its own ceiling. An application that has its own internal SLO of four nines — fifty-two minutes of downtime per year — cannot meet that SLO on top of a provider that itself only commits to three-and-a-half. The math forces the operator to either accept the provider's floor as the application's ceiling, or to build the redundancy that lets the application stay up when one provider is down. The fallback chain in Section 5 is the engineering answer to the second option; the rest of this module is what makes that fallback chain trustworthy.

The status pages are also operational tools, not just documentation. The on-call engineer reading the alert page in this module's Section 7 should know to check the provider's status dashboard before assuming the issue is in the application. Half of the incidents the alerting layer surfaces will be the provider's, not the application's, and the fastest path to resolution is recognising that early and switching to the fallback rather than debugging an application that is functioning correctly against a provider that is not.

### Soft failures: slow, partial, regressed

The hard failures — the call errored, the call timed out, the call returned a 500 — are the easy half of the problem. The soft failures are what wake people up. A prompt that genuinely takes thirty seconds because the model has been asked to do something hard, against a service whose p95 SLO is four seconds, is a soft failure: nothing errored, the call eventually returned, the response was fine — but the user already closed the tab and the dashboard already paged. The application has to decide whether thirty seconds is "still working" or "should have given up by now," and the answer depends on what the user was waiting for.

Partial streams are the second flavour. The streaming endpoint started, sent the first three hundred tokens, and then the connection dropped before the model finished. The application is now holding half an answer. Returning the partial to the user is misleading; discarding it is wasteful; retrying produces a different partial (because the model is non-deterministic) that the application cannot meaningfully merge with the first. The right answer is workload-specific — for chat, retry from scratch; for batch processing, keep the partial and mark it for review — but the wrapper has to surface the partial in a structured form so the application can decide.

Model regressions on the same version are the third, and the most insidious. The model identifier did not change. The model's weights, by the provider's account, did not change. But the inference stack underneath them did, and the output distribution shifted in a way the application's evals start noticing on the next nightly run. Tone got more formal. JSON-mode failures crept up. Refusal rates on a specific category of prompts climbed. None of these are visible to the resilience layer in real time — they look like normal responses — but the alerting layer's quality-regression signals (Module 15's eval-on-prod patterns) catch them and the operator's response is usually to pin the provider to a specific version or to switch the fallback chain's primary.

### Rate-limit thrash: the tier surprise

Rate-limit thrash is the third class of failure. LLM rate limits are denominated in tokens-per-minute as well as requests-per-minute, and the token denomination is the one that bites. A burst of traffic at three in the morning fans out across the application's worker pool, each worker hammers the provider, the per-organization token bucket drains in seconds, and every call after that returns 429 with a `retry-after` measured in tens of seconds. The application's retry layer, if it is naive, retries each 429 immediately, which triggers another 429, which triggers another retry — a thrash loop that does not produce a single successful call until the rate-limit window resets.

The classic incident: the on-call engineer learns at 3am that the "tier-1" plan they thought meant 30,000 requests per minute actually meant 30,000 *tokens* per minute, that their average prompt is 1,200 tokens, and that the application has been one bad cron job away from a sustained outage since launch. The provider's tier table reads in the documentation as a series of generous-sounding numbers, and the underlying denomination — tokens, not requests — is the asterisk that turns the generous number into a tight bound. The first time a real traffic burst hits the bound, the on-call learns the lesson.

The lesson is not that rate limits are surprising; the lesson is that the application needs a layer that absorbs them gracefully rather than letting them propagate to the user. Honoring `retry-after` (Section 3), throttling at the gateway when the token-rate gets close to the cap, and falling back to a different provider (Section 5) whose rate-limit bucket is independent of the primary's are all the right responses. The resilience layer's job is to keep the 429 noise from reaching the user; the user's experience should be a slightly-slower response, not a wall of errors.

### Cost runaway: the failure mode unique to LLMs

Cost runaway is the fourth class, and it is the one that has no analog in classical web infrastructure. Every other failure mode in this list — outages, slow responses, rate limits — also exists for Stripe, for Twilio, for Postgres. Cost runaway is unique to LLM systems because every call costs measurable cents and the call rate has no natural upper bound. A bug that produces a million Postgres queries makes the bill marginally higher; a bug that produces a million LLM calls makes the bill catastrophically higher.

The story that gets retold in every LLM engineering channel: a developer wires the assistant into a data pipeline, the pipeline hits an edge case that triggers a retry, the retry triggers another LLM call, the LLM call's response is parsed and triggers another stage of the pipeline, and ninety minutes later there is a $14,000 OpenAI bill that nobody noticed until the daily cost report ran at the end of the day. The pipeline had no infinite loop; it had a loop with an exit condition that happened never to fire on the data that arrived that night. The retry layer made it worse, not better — every retry was another paid call, and the retry loop's exponential backoff bought a few seconds of breathing room per failure but did not stop the loop from eventually firing.

Cost runaway is the failure mode that takes a one-week incident and turns it into a one-quarter incident, because the dollar number that comes out of the post-mortem is large enough to be a board-level conversation. The budget kill-switch in Section 6 exists to bound the worst case at a number the operator can afford to lose. Without it, the worst case is bounded only by the provider's willingness to keep accepting calls, which is to say not bounded at all on any timescale that matters.

### Why "just retry on error" makes most of these worse

The reason this module exists, and the reason the resilience layer is a layer rather than a handful of try/except blocks, is that "just retry on error" makes most of these failures worse rather than better. A retry against a provider that is genuinely down produces another error at the cost of latency and burned tokens. A naive retry against a 429 produces another 429 a millisecond later, plus the next call's 429, plus the next call's 429 — and a thousand simultaneously-retrying clients turn a one-second blip into a five-minute thundering herd. A retry around a buggy loop multiplies the cost without bounding it.

The right answer is a composition of several policies — retry, circuit-break, fallback, budget, timeout, alert — each of which addresses one failure mode and each of which assumes the others are doing their jobs. The retry layer handles transients only after the circuit has decided the provider is reachable. The circuit handles sustained provider failures only when the retry layer's per-call work is exhausted. The fallback handles per-provider outages only when the local policies have given up. The budget handles cost runaway before any of the other policies fire. The composition is the lesson; how the policies fit together is the load-bearing engineering content of this module.

---

## 2. The Six Policies and Why This Order

The wrapper this module ships composes six policies. The composition is not arbitrary — each policy assumes the policies inside it are running and runs only after the policies outside it have already had their say. The pipeline looks like this:

```
caller
  |
  v
1. Budget gate         (reject before spending)
2. Fallback chain      (iterate ordered providers)
     |
     v per provider
3. Circuit gate        (skip if open)
4. Retry loop          (exponential backoff + full jitter)
     |
     v per attempt
5. Timeout wrapper     (wall-clock cap per attempt)
6. Underlying call     (litellm OR mock provider)
```

Every request enters at the top, descends through the layers, and either returns a response or raises a terminal exception. The defense of the ordering is the defense of the architecture; once each pairwise choice is understood, the wrapper is implementation.

### Budget before fallback

The budget gate runs first because once the spending cap is hit, no provider in the fallback chain helps. Putting the budget check below the fallback chain — "try OpenAI, if it fails try Anthropic, then check the budget" — means the runaway loop gets to spend twice on every iteration before the kill-switch notices. Putting it at the very top means the runaway loop gets exactly one rejected call before every subsequent call is short-circuited at zero cost. The check is cheap, the cost of being wrong is enormous, and the right place is the outermost layer.

There is a secondary reason: the budget gate is the only policy whose decision does not depend on which provider is being called. Every other policy in the stack is per-provider — fallback iterates providers, circuit state is per-provider, retry policy is per-provider. The budget is the global control, and global controls belong outside per-target controls. Moving it inside the fallback would mean each provider has to know the budget state, which is bookkeeping the layer above already owns.

### Fallback wraps circuit and retry

The fallback chain wraps the per-provider policies because circuit state and retry counts are properties of individual providers, not of the operation as a whole. Each provider has its own breaker. Each provider has its own retry budget. When the fallback chain moves from OpenAI to Anthropic, it leaves OpenAI's circuit in whatever state OpenAI's circuit is in (often `OPEN` by that point, since the failure to call OpenAI is what triggered the fallback) and starts Anthropic with a fresh retry counter.

If the per-provider policies wrapped the fallback chain, every provider would share state, which means a string of OpenAI failures would cause Anthropic calls to be skipped — the opposite of what fallback is for. The breaker would open after five OpenAI failures, and the next call against Anthropic would see the open breaker and refuse to call Anthropic, even though Anthropic is fine. The whole point of having two providers in the chain is that their failure modes are independent; sharing the resilience state between them collapses that independence into a single point of failure.

### Circuit before retry

The circuit gate runs before the retry loop because retrying against an open circuit is pointless. If the breaker has determined that the provider is currently down, the retry loop's first attempt will be skipped by the circuit, the retry loop's second attempt will be skipped by the circuit, the retry loop's third attempt will be skipped by the circuit — and then the call will fail with `CircuitOpenError` after burning the retry budget on nothing. Checking the circuit first means the first attempt is the one that decides: if the breaker is open, the call hops to the next provider immediately rather than running through retries that will all fail the same way.

The reverse ordering — retry checks circuit on each attempt — also fails for a different reason, covered as the anti-example below. The general principle: a check whose answer cannot change within the duration of the inner loop belongs outside the inner loop. The circuit's open/closed state changes on timescales of seconds (the cooldown window); the retry loop runs on timescales of seconds too (the backoffs). The two are the same magnitude, and putting the circuit inside the retry means the inner loop spends most of its time looking at a state that has not changed since the loop started.

### Retry before timeout

The retry loop wraps the timeout because the two policies govern different time domains. The timeout governs *one attempt* — a single call to the underlying provider that must complete in, say, ten seconds. The retry loop governs *the operation* — the sequence of attempts that together implement the resilient call, with their own end-to-end budget.

If you put the timeout outside the retry loop, the wall-clock cap applies to the whole sequence: a single ten-second timeout has to contain three attempts and two backoffs, which means each attempt has roughly three seconds, which means the slow-but-eventually-successful response never arrives. The right model is the inverse: each attempt gets the full timeout, and the retry loop bounds how many attempts can happen before the operation gives up. A separate operation-level timeout (set by the caller's request-handling code, not by the resilience layer) is the right place to bound the total wall-clock time the user is willing to wait.

### The anti-example: circuit inside retry

The most common variant that looks correct and is not: putting the circuit check inside the retry loop, so that each retry consults the breaker. The logic seems appealing — "skip the retry if the breaker has opened during the loop" — and it is a category error. Here is what actually happens. The first attempt fails. The retry loop sleeps for the backoff. The second attempt fails. The retry loop sleeps for the backoff. The third attempt fails, and on the third failure the breaker trips and opens. The fourth attempt is skipped by the circuit. The retry loop reports failure after four attempts, three of which contributed to opening the breaker that skipped the fourth.

The intended benefit of the breaker — *skipping the call when the provider is known-bad* — never materializes, because the breaker only opens *during* the loop, not before. The whole point of the circuit is to fail fast on calls *after* the breaker is open, and the cross-call state has no chance to accumulate if the loop completes within one operation. The breaker is doing zero work; its existence is a comment in the code that does not affect the behaviour.

Moving the circuit check to *outside* the retry loop — the ordering in the diagram — makes the breaker do what it is supposed to do: the next user request, arriving a second after the first one's retries finished, sees the open breaker and hops to the fallback immediately. The first user pays the retry cost; every subsequent user pays no cost at all until the breaker half-opens. The breaker's state — accumulated across calls, persisted between them — is what gives the second user the cheap path. The retry-inside-circuit ordering wastes that state.

### The error taxonomy

The policies above only work if each kind of error is routed to the right policy. Routing happens through a small set of exception types that the wrapper raises (or that the underlying call's exceptions are mapped into). The taxonomy:

| Class | Retriable? | Hops fallback? | Trips circuit? | Notes |
|-------|-----------|----------------|----------------|-------|
| `RateLimitError(retry_after_s)` | yes | only after retries exhausted | yes | honors `retry_after_s` |
| `TimeoutError` | yes | only after retries exhausted | yes | per-attempt only |
| `TransientError` | yes | only after retries exhausted | yes | maps from 5xx |
| `AuthError` | no | yes | no | maps from 401/403 |
| `ModelNotFoundError` | no | yes | no | maps from "model not available" |
| `ContentFilterError` | no | yes | no | provider refused content |
| `BadRequestError` | no | **no** | no | caller bug — raises through fallback |
| `CircuitOpenError` | no | yes | n/a | breaker has already opened |
| `AllProvidersFailedError(causes)` | terminal | n/a | n/a | chain exhausted |
| `BudgetExceededError(spent, cap)` | terminal | n/a | n/a | budget gate raised |

### Reading the taxonomy

The most counterintuitive row is `BadRequestError`. A 400 from the provider means the caller sent a malformed request — wrong model id, malformed messages list, schema violation in a structured-output call. Hopping to the next provider with the same malformed request produces another 400. The bad request is not a provider problem; it is a caller bug, and the right response is to surface it loudly so the caller can fix it, not to mask it by trying every provider in the chain. The wrapper raises `BadRequestError` straight through the fallback layer, terminating the operation immediately. The cost is that one variety of recoverable error is not recoverable — but the benefit is that caller bugs are visible at the moment they happen rather than hidden inside an `AllProvidersFailedError` whose causes list happens to contain ten identical `BadRequestError` entries.

`AuthError` and `ModelNotFoundError` hop the fallback chain because they are provider-specific configuration mistakes, not request mistakes. The OpenAI key may have expired while the Anthropic key is fine; the model name `gpt-4o` is not valid on Anthropic but `claude-sonnet` is. These should hop, not retry — retrying with a bad key produces another auth error, and the next provider in the chain is the place where a working configuration is most likely to live.

`CircuitOpenError` is the wrapper's own signal that the breaker for this provider is open. It is not a real error from the provider; it is the wrapper saying "I would have called this provider, but I know from recent history that the call would fail." Hopping to the next provider is exactly the right response. The fallback chain treats `CircuitOpenError` identically to a real error — a failed call is a failed call from the chain's perspective — but the cost is different: a real call burns latency and a partial token charge, while a `CircuitOpenError` is raised in microseconds with no provider cost.

The two terminal errors — `AllProvidersFailedError` and `BudgetExceededError` — are the two ways the wrapper can give up. The first means the chain has run out of providers; the second means the operation has run out of budget. Both should be loud, both should be alertable, and both should carry enough context (the cause list, the spent amount, the cap) that the on-call engineer can decide what happened without having to dig into a separate log.

---

## 3. Retries with Exponential Backoff + Jitter

Retries exist because transient errors are real. A momentary network blip, a brief autoscaler hiccup, a single 503 from a provider rolling a deploy — all of these are events where the second attempt succeeds and no human needed to be involved. The right number of retries for these events is small (two or three) and the right behavior on success is to log the recovered call and proceed. The point of retries is to absorb the noise that exists in any distributed system, and on this dimension they earn their keep.

### What retries cost

The cost of retries is the part the implementation has to bound. Every retry adds latency to the operation — the wall-clock time the user waits is the sum of the attempt latencies plus the backoff sleeps. A three-attempt retry with one-second and two-second backoffs adds at least three seconds to the worst case, on top of whatever each attempt's own latency was. For interactive workloads where the user is waiting on the response, the cumulative cost is the user's patience.

Every retry doubles the cost on a partially-successful call: if attempt one streamed seven hundred tokens before erroring on the eight hundredth, those seven hundred tokens were billed and the retry has to pay for them again. The cost asymmetry between success and failure is what makes retries economically harmful on workloads where the failure rate is high — every percentage point of failure rate is a percentage point of doubled cost, and a 10% failure rate on a workload that pays a cent per call costs roughly 10% more on the bill than the same workload without retries.

Every retry amplifies the load on a provider that may already be struggling — a thousand clients all retrying in lockstep against a slow provider is how a soft degradation becomes a hard outage. The thundering-herd argument below makes this concrete, but the principle deserves to stand on its own: retries are not free, and the policy that decides how many, how often, and against what error is what separates a useful retry layer from a harmful one. A wrapper that retries everything three times with no backoff is worse than no wrapper at all, because it turns every transient blip into a 3x load amplification and every persistent failure into a 3x cost amplification.

### Backoff math: linear is wrong, exponential is right

Linear backoff — wait one second after the first failure, two after the second, three after the third — is gentle but wrong. The cumulative wait is small (1+2+3+4+5 = 15s over five attempts) and the provider gets hammered just as quickly on the fifth attempt as on the second. The curve is too flat; by the time the loop has decided to give up, the provider has already received nearly the same load it received in the first attempt.

Exponential backoff — wait `base * 2^attempt` — is the right curve. At attempt five with `base=1s`, linear waits five seconds and total elapsed wait is fifteen; exponential with `base=0.5s` waits eight seconds at attempt five and the next attempt would wait sixteen. The doubling is what makes the curve back off rather than push: by the fourth or fifth retry the wait has grown enough that the operation is effectively giving up gradually rather than hammering through to exhaustion. The provider gets longer between requests as the failure persists, which is the right shape — if the provider is genuinely down, you want to be one of the *less* aggressive clients on its way back up, not one of the most.

The base value matters too. A `base=0.5s` is the canonical default for LLM workloads because the typical transient failure (a single 503 from a provider deploying a new version) recovers within a second, and the half-second base lets the first retry land just after the recovery. A larger base (one or two seconds) is more conservative and right for workloads where the failures are slower to recover; a smaller base (a hundred milliseconds) is right for workloads where you want to retry aggressively against transient network blips that recover in milliseconds. The base is a configuration knob, not a constant; tune it on your own failure data.

### Jitter: why full jitter wins

Jitter is what separates exponential backoff from a thundering herd. Without jitter, every client that started its retry loop at roughly the same moment — say, all the clients that saw the same provider blip — wakes up at roughly the same moment for the second attempt. A thousand clients waking up simultaneously at t=1s and slamming the provider with a thousand requests recreates the outage the retry was supposed to absorb.

Marc Brooker's 2015 AWS Architecture Blog post, "Exponential Backoff and Jitter," works through three jitter strategies and concludes that *full jitter* — sleep a random duration in `[0, base * 2^attempt]` rather than the deterministic `base * 2^attempt` — beats both equal jitter (where half the sleep is deterministic and half is random) and decorrelated jitter (a different distribution that's better in some pathological cases) for the common case of thundering-herd suppression. The mechanism is intuitive: equal jitter has a deterministic floor, so the herd compresses but does not fully scatter; full jitter has no floor, so the herd spreads uniformly across the interval.

This module uses full jitter for the same reason every production retry library defaults to it. `tenacity`'s `wait_random_exponential`, the OpenAI SDK's internal retry, LiteLLM's `num_retries` path — all of them use full jitter under the hood. The choice is no longer controversial; the reason it shows up in every reference implementation is that the alternatives have known failure modes and full jitter does not.

### The thundering-herd worked example

The math is what makes jitter feel non-optional. Scenario: 1,000 clients, all hit the same provider at t=0, the provider returns 503 to all of them, the provider recovers at t=1s. With no jitter and a fixed 1-second backoff, all 1,000 clients wake at exactly t=1s, all 1,000 hit the provider in the same millisecond window, and the provider sees 1,000x its normal request rate at the moment it just started accepting traffic again. The provider buckles, returns 503 to most of them, and the cycle repeats. The recovery never sticks because the recovery is being immediately undone by the herd.

With full jitter at base=1s, the 1,000 clients spread their retries uniformly over the interval `[0s, 1s]`, so the provider sees roughly 1 client per millisecond — peak load drops by approximately 1,000x, and the provider has time to ramp its capacity back up between requests. To put numbers on the smoothed profile: in any 10ms window during the recovery period, the expected number of arriving retries is 10, not 1,000 — a load profile two orders of magnitude flatter than the no-jitter case. The peak instantaneous load (the largest gap-free burst in any short interval) follows a Poisson distribution with mean 10 over 10ms windows, putting the 99th-percentile peak at roughly 18–20 clients rather than the full 1,000-client wall. The arithmetic is what makes jitter mandatory; the absence of jitter is what makes naive retry loops dangerous. A retry layer without jitter is not a milder version of a retry layer with jitter; it is a thundering herd waiting to happen.

The same arithmetic scales to smaller numbers. Ten clients without jitter produce a 10x burst; with jitter, they spread to roughly 1x. A hundred clients without jitter produce a 100x burst; with jitter, roughly 1x. The benefit is greatest at the high end (where the burst would be most damaging) but exists at every scale. There is no scenario where the deterministic backoff is preferable to the jittered version.

### Honoring `retry_after`

The `retry_after` field is the one place where the provider knows better than the math does. When a provider returns 429 with `retry-after: 45`, it is telling the client that the rate-limit bucket will refill in forty-five seconds and that retrying earlier will produce another 429. Honoring `retry_after` is a contract with the provider that says: trust the signal it sends, do not second-guess it with the computed backoff.

The wrapper's `RateLimitError` carries a `retry_after_s` field; when present, it overrides whatever the exponential-jittered math would have produced. The log line distinguishes the two cases — `"honored retry_after=45s"` versus `"computed backoff=8.2s"` — because a debugger inspecting a slow operation needs to know whether the wait was provider-instructed or client-computed. The two failure modes have different fixes: provider-instructed waits suggest the application is over its rate-limit allocation and should consider a higher tier or fewer parallel workers; client-computed waits suggest the application is hitting transient errors that the retry is absorbing.

The header parsing has its own subtleties. Some providers return `retry-after` as a number of seconds; others return it as an HTTP-date timestamp. The wrapper normalises both to a float-seconds value before populating `retry_after_s`. A negative or unparsable value falls back to the computed backoff; the application should never see a negative sleep. These are small details, but they are the kind that turn a retry layer from "works most of the time" into "works on every provider the application has been pointed at."

### What not to retry

The taxonomy table covers this row by row, but the principle deserves one paragraph: retrying a 4xx generates more 4xxes. A 400 is a malformed request, and retrying it sends the same malformed request again. A 401 is a bad credential, and retrying it sends the same bad credential again. A 403 is forbidden, and retrying it is forbidden the same way. The model-not-found error and the content-filter rejection are the same pattern — the provider has told the client the request is unacceptable, and retrying is asking the provider to change its mind. The provider will not change its mind.

Retry only the errors where transience is plausible: 429 (rate limit recovers), 408 and 504 (timeout might succeed on retry), 5xx (server might recover), connection errors (network might unflake). Everything else is a caller bug or a configuration bug or a permanent provider state, and the right answer is to raise straight to the application layer where someone can fix it. The wrapper's exception taxonomy is the codification of this rule: `RateLimitError`, `TimeoutError`, and `TransientError` are marked retriable; everything else is not.

The cost of getting this wrong is measurable. A retry layer that treats 400s as transient produces three 400s for every one a caller would have seen, and the caller's debugging story now has three times the noise to sort through. A retry layer that treats content-filter rejections as transient produces three identical rejections, sometimes burning the provider's per-account budget on calls that were never going to succeed. The right policy is conservative on what counts as retriable; the wrong policy retries everything and hopes for the best.

### Retry budgets versus retry counts

The implementation has one more knob worth naming: whether the retry layer bounds attempts by count or by elapsed time. A count-based budget — at most three attempts — is the more common configuration and the one this module's wrapper uses. A time-based budget — at most ten seconds of total retry work — is the more nuanced configuration and is useful when the per-attempt latency varies widely.

The count-based budget is easier to reason about and easier to log, which is why it dominates production. The on-call engineer sees `attempts=3` in the span and knows immediately what happened; `elapsed_retry_ms=8240` requires arithmetic to map to "how many tries was that?" The count-based budget also composes cleanly with the timeout-per-attempt: three attempts of up-to-ten-seconds each gives an upper bound of thirty seconds plus backoffs, which the caller can plan around.

The time-based budget is more honest about the user's actual constraint — the user does not care whether the wrapper made one attempt or three, they care whether the response arrived within their patience window. Implementations that mix both — at most three attempts, but also at most ten seconds total — get the best of both shapes at the cost of more configuration. The wrapper this module ships uses count-only for simplicity; production deployments often add a time bound when the per-attempt latency is bimodal.

---

## 4. The Circuit Breaker

Martin Fowler's 2014 bliki entry, "CircuitBreaker" (`martinfowler.com/bliki/CircuitBreaker.html`), is the canonical reference for the pattern. The model is a three-state machine: `CLOSED` (the normal state, calls pass through to the protected operation), `OPEN` (the failed state, calls return immediately without attempting the operation), and `HALF_OPEN` (the recovery probe state, exactly one call is allowed through to test whether the protected operation has recovered). The transitions are deterministic: consecutive failures in `CLOSED` trip the breaker to `OPEN`; a cooldown timer in `OPEN` transitions to `HALF_OPEN`; success in `HALF_OPEN` returns to `CLOSED`; failure in `HALF_OPEN` returns to `OPEN`. The wrapper this module ships implements exactly that machine, with the failure threshold and the cooldown configurable per provider.

### The three states, in operational detail

In `CLOSED`, every call passes through to the underlying provider. The breaker counts consecutive failures: each failure increments a counter, each success resets it to zero. When the counter reaches the configured threshold — five, by default — the breaker transitions to `OPEN` and the failure timestamp is recorded as the moment the breaker opened. The transition is a one-way door from this side; nothing in `CLOSED` can move the breaker to `HALF_OPEN` directly.

In `OPEN`, every call is rejected immediately with `CircuitOpenError`. The provider is not called; no latency is incurred, no tokens are spent. The breaker stays in `OPEN` for the configured cooldown (default thirty seconds). When the cooldown elapses, the next call to be rejected instead transitions the breaker to `HALF_OPEN` and is allowed through as the probe. The transition is lazy — the breaker does not run a timer that fires automatically; it checks the elapsed time on each call and transitions when appropriate. The lazy design avoids needing a background thread and keeps the breaker's state changes synchronous with the call traffic that observes them.

In `HALF_OPEN`, exactly one call is allowed through to the provider. The result of that call determines the next state. If the call succeeds, the breaker transitions back to `CLOSED`, the failure counter is reset, and traffic flows normally. If the call fails, the breaker transitions back to `OPEN`, the failure timestamp is reset to the current time, and the cooldown begins again. While the probe is in flight, every other call sees the breaker as `HALF_OPEN` and is rejected with `CircuitOpenError` — only one client gets the probe; everyone else waits.

### Threshold tuning: too low, too high, and the architectural default

The threshold is the first knob that has to be tuned, and it is the knob where the trade-off is most visible. Set the threshold too low — say, two consecutive failures — and the breaker trips during normal jitter. Any provider has a baseline failure rate; two failures in a row over a thousand calls is a normal noise event, not a signal that the provider is down, and tripping on it produces a cascade of false fallbacks and unnecessary alerts.

Set the threshold too high — say, twenty-five consecutive failures — and the breaker is slow to react to a real outage. By the time the twenty-fifth call has failed, the application has spent twenty-five attempts' worth of latency and dollars to learn what a faster breaker would have learned in five. During a real outage, twenty-five calls might take a minute and cost a few dollars per worker; multiplied across the worker pool, the over-high threshold is the difference between "the breaker caught the outage immediately" and "the breaker caught the outage after a measurable financial loss."

The wrapper ships with a default threshold of 5 consecutive failures on the reasoning that real LLM-provider outages, when they happen, tend to be sustained rather than intermittent — once a provider's error rate climbs past 50%, it usually stays there for minutes to hours, not seconds. Five-in-a-row trips the breaker fast in that mode without false-tripping on the occasional one-off timeout. Operators should retune for their own provider mix and tolerance; this is a default, not a recommendation.

### The half-open probe: why exactly one

The half-open probe allows exactly one call. Two would be a design error, and the reasoning is the load-on-recovery argument. A provider that has just recovered from an outage is at its weakest moment — its caches are cold, its rate-limiter is at its conservative setting, its connection pools are draining the backlog from the outage. Two clients hitting it simultaneously doubles the load at that moment, which is precisely the moment when doubling the load is most likely to push it back into failure.

The single-probe rule says: one client volunteers to test the water; everyone else waits for that client to report back. If the probe succeeds, the breaker closes and traffic flows; if the probe fails, the breaker reopens and the cooldown timer restarts. The single-probe rule is also what makes the cooldown duration meaningful — a thirty-second cooldown with a single probe gives the provider thirty seconds of zero load before one client touches it again, which is the recovery window the provider needs. Without the single-probe rule, the cooldown would just delay the next thundering herd by thirty seconds instead of preventing it.

The implementation has to be careful to enforce the single-probe rule under concurrency. Two threads calling the wrapper simultaneously when the breaker is `HALF_OPEN` must not both be allowed through; one must win the probe and the other must see `CircuitOpenError`. The wrapper handles this with a small lock around the state-transition logic — the cost is a few microseconds per call, and the correctness benefit is that the single-probe rule actually holds under real traffic.

### Per-provider, not per-route

The breaker state lives on the provider object, not on a `(provider, endpoint)` pair. The reason is that LLM provider failures are almost always provider-wide: when Anthropic has an incident, every endpoint is affected — `/v1/messages`, `/v1/messages/count_tokens`, the streaming variants, the batch API. Routes do not fail independently the way they do in a service mesh, where each microservice has its own deployment health and the breaker is what isolates one bad service from the rest.

Service-mesh tools — Istio, Linkerd, Envoy — implement per-route breakers because they sit at a different layer; an LLM resilience wrapper sits at the application layer and the right granularity is the provider. The implementation is simpler too: one breaker per `ProviderClient`, persisted in memory for the life of the process, no per-route bookkeeping.

There is a refinement that some workloads want: per-model breakers within a provider, when the provider exposes multiple model tiers with independent availability. Anthropic's `claude-opus` and `claude-haiku` are usually correlated in their availability but not always; a deprecation event for Opus does not affect Haiku, and a per-model breaker would isolate the two. The wrapper this module ships uses provider-level breakers by default and leaves model-level breakers as an extension point — a `breaker_scope` parameter that defaults to `"provider"` and accepts `"model"` for workloads that need the finer granularity.

### The relationship to retries

The relationship to retries is the most useful framing once both layers exist. Retries are within-call resilience: a single logical operation gets multiple chances to succeed, with state — the retry counter, the backoff timer — that lives only for the duration of the call. The breaker is across-call resilience: state — the failure count, the open-timer — that persists across calls and lets one call inform the policy applied to the next.

They are complementary because they handle different failure modes. A burst of transient errors during a single operation is the retry's job; sustained errors across many operations is the breaker's job. A wrapper with retries but no breaker hammers a known-down provider on every new request; a wrapper with a breaker but no retries fails every operation that experiences a single transient blip. Both are needed, and the composition — retry inside breaker, breaker per-provider, providers iterated by fallback — is the architecture in Section 2.

The two layers also have different observability requirements. A retry attempt is a span attribute or a structured log field on the call's span (`attempt_number: 2`, `backoff_ms: 1234`). A circuit transition is a structured log event with no parent span, because the transition is a state change in the breaker, not a step in any one call. The trace recorder from Module 18 captures the per-call retries; a separate `circuit_state_change` log line captures the transitions. The on-call engineer reading the alert can scan the log for `OPEN` events to see which providers tripped during the incident, then drill into the trace for the specific calls that drove the failures.

### Cooldown tuning: matching the recovery time

The cooldown duration is the second knob and it interacts with the threshold in a way that is not obvious until both are in production. A short cooldown — five seconds — means the breaker probes recovery quickly; if the provider's outage was momentary, traffic returns fast. The cost is that if the outage lasts longer than the cooldown, the probe fails, the breaker reopens, and the cycle repeats every five seconds. The probe traffic itself contributes load to a recovering provider, and a too-short cooldown is the way to keep a struggling provider pinned in failure.

A long cooldown — five minutes — gives the provider real recovery time but delays the application's return to service. If the provider recovered after thirty seconds, the application waits four-and-a-half more minutes before it notices, during which every call hops to the fallback and the user experience is whatever the fallback offers. The thirty-second default is the empirical sweet spot for LLM workloads: long enough that a typical provider incident has time to recover, short enough that the application is not stuck in degraded mode for longer than necessary.

A team that has measured its own provider's incident-duration distribution can tune the cooldown to match. If most incidents last under a minute, a thirty-second cooldown is too short (probes too often); a sixty-second cooldown is right. If most incidents last several minutes, a thirty-second cooldown is far too short and a three-minute cooldown is closer. The measurement that drives the tuning is the duration of past incidents, available from the provider's status page or from the application's own historical logs.

### Failure-counter reset semantics

The failure counter has a non-obvious detail: when does it reset to zero? The simplest answer — reset on every success — is the right one for most workloads, and it is what the wrapper implements. Five failures in a row trips the breaker; one success in a row resets the counter; the next five failures (whether consecutive with the previous five or interleaved with successes) start counting from zero. The model is "consecutive failures from the most recent success."

The alternative — a rolling-window counter that tracks failures within the last N seconds — is more sophisticated and rarely worth the complexity. The rolling window catches the case where a provider has a high failure rate but never quite fails five-in-a-row (one failure, one success, one failure, one success, …), which the consecutive-counter misses. In practice, real provider failures do come in consecutive runs because the underlying cause (a deploy gone wrong, a regional outage, a rate-limit cliff) affects every call uniformly until it is resolved. The consecutive-counter catches the real cases; the rolling-window-counter catches the rare cases at the cost of more state to manage and tune.

---

## 5. Fallback Chains: Cheaper Models as Spare Tires

The fallback chain is what gives the application a path through the failure of any single provider. The chain is an ordered list of `ProviderClient` instances; the wrapper tries them in order; the first one that returns a response wins; if all of them fail the operation raises `AllProvidersFailedError` with the cause list attached. Two ordering choices dominate, and the right answer depends on the failure mode the chain is designed to absorb.

### Same model, different provider

When the secondary should be *the same model on a different provider*: this is the high-fidelity configuration. OpenAI's `gpt-4o` on Azure as a fallback for `gpt-4o` on the OpenAI API; Anthropic's `claude-sonnet` on AWS Bedrock as a fallback for the same model on the Anthropic API. The reasoning is provider-specific failure modes: when one region of one cloud has an incident, the same model running in a different region or on a different cloud is usually fine.

The quality is identical because the model is identical — the user gets the same response they would have gotten from the primary, just routed through a different path. This is the production-grade configuration for systems where quality cannot drop on fallback. A customer-support assistant that must give the same quality of answer regardless of which provider answered it should run the same model on two providers; a regulatory-compliance workflow that has been validated against a specific model version should not silently move to a different model version on fallback.

The operational cost of the same-model-different-provider configuration is that the application has to maintain two sets of provider credentials and two sets of cost-tracking entries. Per-call cost telemetry (Module 17) has to know which provider answered to attribute the cost correctly; the budget kill-switch has to aggregate spend across both providers; the alerting layer has to track error rates per provider so a degradation in the secondary does not get hidden by the primary's healthy numbers. None of these are obstacles, but they are the bookkeeping that comes with running a multi-provider configuration.

### Cheaper model, same provider

When the secondary should be *a cheaper model on the same provider*: this is the spare-tire configuration. The primary is `claude-sonnet`; the secondary is `claude-haiku`. The fallback fires when the failure is model-specific — the primary model was deprecated overnight, the fine-tune is unavailable, the model's rate limits are tighter than the cheaper tier's. The cost goes down on fallback (Haiku is meaningfully cheaper than Sonnet), and the quality goes down too, because Haiku is a smaller model.

The trade-off is acceptable when the alternative is total failure: a slightly-worse answer is better than no answer. It is unacceptable when the user cannot tell that they got the cheaper model and the worse answer is silently wrong about something the user trusted. The spare-tire configuration is right for casual chat, for non-critical summarisation, for workloads where the worst case of a wrong-on-the-margin answer is harmless. It is wrong for medical advice, for legal advice, for financial advice, for anywhere the user's decision-making depends on the answer being right.

The configuration choice — same-model versus cheaper-model fallback — is also not exclusive. A long chain can use both: `[claude-sonnet-anthropic, claude-sonnet-bedrock, claude-haiku-anthropic]` tries the primary, then the same model on a different provider, then the cheaper model on the original provider. The first hop maintains quality; the second hop also maintains quality; the third hop is the spare tire that absorbs the case where both providers' Sonnet tiers are down simultaneously. The cost grows with the chain length only when the earlier hops fail, which is exactly the right shape.

### The silent quality drop trap

This brings the **silent quality drop trap**, which is the failure mode the fallback chain creates and the one most operators do not anticipate when they ship it. The chain works in the sense that it returns a response; the response is plausible; the user reads it and acts on it; the application's error rate is zero because no exception was raised. But the response is from Haiku, not Sonnet, and Haiku is wrong about the specific fact the user needed. The user could not tell. The application could not tell either, because nothing in the response shape signals which model produced it.

The silent quality drop is the worst failure mode in a resilience system because it has no signal — the dashboards stay green, the alerts stay quiet, and the user trust quietly degrades. The wrapper that ships the fallback chain has, in the moment of shipping, also shipped a class of bugs that will not show up until the eval harness's quality metrics drift downward over a period of days or weeks. By the time the team notices, the cause — fallback firing more than expected — is one of many possible explanations, and the investigation has to start from scratch.

### The fix: surface the metadata

The fix is metadata, and it has to be metadata that survives the boundary between the wrapper and the application. Every response the wrapper returns should carry, alongside the model output, a small structured record: `provider_used` (the actual provider that answered), `model_used` (the actual model the provider routed to), `fallback_hops` (how many providers were tried before this one succeeded), `retry_attempts` (how many attempts within the successful provider).

The application can then surface this metadata to the user — UIs that render a small label, "answered by GPT-3.5 (fallback)," let users discount the answer appropriately; internal tools that log the fallback rate let operators see when the secondary is doing more work than expected and fix the primary. The metadata is the only thing standing between a working fallback chain and a fallback chain that silently degrades user-facing quality.

The wrapper's response object — the `ResilienceCallResult` in the project — wraps the provider's response and attaches this metadata as first-class fields. The application's code that consumes the response should access `result.text` for the model output and `result.provider_used` for the attribution, never bypassing the wrapper to get at the raw provider response. The discipline is the same as the cache wrapper in Module 17: the wrapper owns a typed return shape, and the application interacts with that shape, not with whatever the wrapper happens to have called internally.

The alerting layer in Section 7 should track the `fallback_hops` distribution as a primary signal. A healthy chain has the vast majority of calls completing at `hops=0` (the primary answered); a small fraction at `hops=1` (the secondary answered, primary failed); a rare event at `hops=2` or higher (multiple providers down). When the distribution shifts — when `hops=1` becomes ten percent of traffic instead of one percent — the alert fires and the operator investigates the primary's health.

### The bad-request exception

The bad-request exception is the one row in the table where the fallback chain does not hop. A `BadRequestError` from the primary means the application sent a malformed request — the messages list does not validate, the model id is wrong, the structured-output schema has a bug. Trying the next provider with the same malformed request produces the same error, with the additional confusion that the user-visible error mentions a provider the application never intended to call.

The wrapper raises `BadRequestError` straight through the chain so the application surfaces the real bug rather than masking it as "all providers failed." The principle is the one in Section 2: caller bugs are caller's problem; provider bugs are the chain's problem; the chain only handles the second class. A 400 that says "model 'gpt-4o' not found" when the application meant to call Anthropic's `claude-sonnet` is a configuration mistake in the application, and the right error message is "you asked for a model that doesn't exist on this provider," not "all of your providers are down."

The wrapper distinguishes `BadRequestError` from `ModelNotFoundError` because the latter is a per-provider configuration mismatch that the chain *can* resolve by hopping. `BadRequestError` is structural — the request shape is wrong on every provider; `ModelNotFoundError` is per-provider — the request shape is fine, but this provider doesn't have the model. The taxonomy table makes the distinction visible; the wrapper enforces it at the routing layer.

### Chain length and the tail-cost trade

The chain length is a configuration choice with a measurable cost. A two-provider chain (primary, secondary) is the minimum useful configuration and handles the common case of "one provider is down, the other is fine." A three-provider chain adds another fallback and handles the rare case of "two providers are down simultaneously." A five-provider chain is overkill for almost all workloads — the probability of five providers being down at once is so low that the engineering cost of maintaining the chain is not justified by the resilience benefit.

The tail cost is the latency users see when the early providers fail. A request that hops through three providers — each with its own retries, each with its own timeout — can take thirty seconds to ultimately succeed or fail, by which point the user has already given up. The wrapper has to bound the total operation time, not just the per-attempt time, and the bound is what makes long chains usable. The pattern is an operation-level timeout, set by the caller, that overrides the per-attempt timeouts in the chain — at twenty seconds, the operation gives up regardless of which provider is currently being tried.

The right chain length for most workloads is two. The right composition for those two is a same-model-different-provider pair, because that combination handles the dominant failure mode (provider-specific outages) without introducing the silent-quality-drop problem. A third provider is appropriate when the first two are correlated (both on AWS, both in the same region) and the third is in a different cloud or region. Beyond three, the marginal resilience benefit drops sharply and the marginal complexity cost rises.

---

## 6. Budget Kill-Switch

The budget kill-switch is the single most important LLM-specific control in this module. Every other policy — retries, circuit breakers, fallbacks, timeouts, alerting — has an exact analog in classical web infrastructure. The budget kill-switch does not, because classical web infrastructure does not have a failure mode where each call costs measurable cents and the call rate has no natural bound.

### Why this is the LLM-unique control

Postgres queries do not cost per-query; HTTP calls to internal services do not cost per-call; even Stripe API calls, which are billed, are billed at rates so low and against operations whose volume is so naturally throttled by the business logic above them that runaway loops never reach billing-significant volume. LLM calls have neither of these protections. The cost-per-call is high enough to matter individually, and the call rate is bounded only by the application's loop logic and the provider's rate limits.

The kill-switch is the policy that says: regardless of what the application is trying to do, regardless of what the provider is willing to serve, this operation will not exceed this dollar amount. The bound is what gives the operator the confidence to ship the system without a constant fear that the next bug will drain the bank account. Without the kill-switch, every loop in the codebase is a potential runaway; with the kill-switch, the worst case is bounded at a number the operator chose and the operator can afford.

The principle generalises: any system whose failure mode involves unbounded resource consumption needs a hard limit, not just a soft monitor. Cloud-budget alarms (which fire after the spend has happened) are not kill-switches; they are post-mortem notifications. The kill-switch is the policy that prevents the spend before it happens, by refusing to make the call that would push the total over the cap.

### Per-call cost telemetry is the prerequisite

The prerequisite is per-call cost telemetry, and this is the cross-link to [Module 17 (Caching & Cost Optimization)](../17-caching-cost-optimization/). Every call has to produce a `cost_usd` value the moment it returns, computed from the token counts in the provider's usage object multiplied by the per-token rate published in the model's metadata. The standard pattern is a `try` around `litellm.completion_cost(response)` with an `except` that falls back to a manual computation using known rates, so that a missing model in the LiteLLM cost table does not silently drop the cost telemetry.

Once every call produces a cost number, the budget gate is the simple part: a running sum of costs, a configured cap, and a comparison. The complexity is in the cost computation, not in the budget logic — getting the cost right per call, especially across providers with different billing models (per-token, per-character, per-completion), is what the cost-telemetry layer earns its keep on. The budget gate consumes the telemetry; it does not produce it.

The cost telemetry has to be persistent across the wrapper's lifetime, not just per-call. The running sum lives on the wrapper instance and accumulates across every call the wrapper makes. When the wrapper instance is recreated (process restart, new request handler), the running sum resets — which is correct for per-process or per-session budgets but wrong for global budgets that must survive restarts. Workloads with global budgets use an external store (Redis, a database, a metrics backend) to persist the running sum; the wrapper reads the current spend on initialisation and writes the update after each call. The trade-off is one extra network hop per call against the correctness of the cross-restart budget.

### Window versus lifetime caps

Window versus lifetime caps. The choice is operational, not technical. A *window cap* is a rolling N-second budget — at most $10 of spend in any 60-second window — which lets the system absorb a burst, recover, and then absorb another burst when the window has rolled past. A *lifetime cap* is an absolute number — at most $100 of spend over the life of this process or this user's session or this run — which gives an upper bound that does not depend on how the system is paced.

Window caps are right when the operation is long-running and the budget is meant as a rate limit. A customer-support assistant serving a steady stream of users has no natural "session" — the application runs continuously, and the meaningful budget is "no more than $X per hour, no matter what." The rolling window enforces the rate without forcing the application to track sessions.

Lifetime caps are right when the operation has a defined scope and the budget is the total resource available. An eval harness running a fixed set of test cases has a natural lifetime — the harness starts, runs the cases, ends — and the budget is "this run cannot cost more than $X total." Within the run, bursts are fine; across the run, the cap is absolute. A nightly data-processing job is the same shape: the run has an end, and the cap is the budget for that run.

The wrapper this module ships supports both, and the default is the lifetime cap because the lifetime cap is the conservative choice: an operation that exceeds its lifetime budget has done something the operator did not authorize, and stopping is safer than continuing. The window cap is the more permissive choice — it allows the operation to continue indefinitely as long as no single window exceeds the rate — and it requires the operator to have thought through what rate is acceptable.

### Pre-call check, not post-call

The check happens *before* the call, not after. Pre-call check: compute the projected cost of the call (estimated from prompt tokens at the input rate, with a small buffer for the response), compare to the remaining budget, raise `BudgetExceededError` if the projection would exceed the cap, otherwise proceed. Post-call check: make the call, observe the actual cost, add it to the running sum, and check the sum against the cap before allowing the *next* call.

The pre-call check is the right one for the runaway-loop case, because the runaway loop's danger is that it fires calls faster than the post-call check can react — a hundred calls per second of cost-feedback latency, even if each call is cheap, racks up real money before the kill-switch sees the first sum. Pre-call gating means the runaway loop is stopped at call one of the run that would have exceeded the budget, not at call one hundred.

The conservative projection — use input tokens as a lower bound for the cost, then add a buffer for the output — means the budget cap is honored within a small tolerance even when the actual cost is hard to predict. A call whose prompt is 1,000 tokens at $3 per million input tokens has a known input cost of $0.003; the output is unknown but bounded by `max_tokens` at the configured rate. The pre-call projection uses input cost plus the `max_tokens` worst-case output cost, and the actual cost is almost always lower. The trade-off is that the budget caps slightly under-utilise the configured cap (a $100 budget might allow $95 of actual spend before the projection refuses), which is the conservative direction.

### Graceful degradation versus hard stop

Graceful degradation versus hard stop. When the budget is exhausted, the wrapper has two reasonable responses: return a cached "we're at capacity, please try again later" response that the user sees as a brief, polite outage; or raise `BudgetExceededError` and let the application layer decide whether to degrade, cache, fail, or alert.

The wrapper this module ships raises the exception and lets the application choose, because the wrapper does not know enough about the application to make a good choice on its behalf — a customer-support assistant might want the polite-outage response, an internal data pipeline might want a hard exception that pages the on-call, an evaluation harness might want the operation to silently abort and the result row to be marked "budget exceeded." Each application has the right answer for its own context, and the wrapper's job is to surface the signal cleanly enough that the answer can be implemented in three lines of caller code.

The exception carries the context the caller needs: `BudgetExceededError(spent_usd=98.42, cap_usd=100.00, projected_call_usd=2.10)`. The caller can decide based on those numbers — a small projected cost might be worth re-trying after a brief wait (the rolling window will refresh), while a large projected cost is a sign that the operation is genuinely over its budget and should give up. The wrapper does not make this decision; the wrapper exposes the data.

### Reserving headroom for the alerting layer

A subtle interaction between Sections 6 and 7 deserves to be made explicit. The budget kill-switch fires when the spend crosses the cap; the budget-burn alert in Section 7 fires when the spend crosses some fraction of the cap (eighty percent by default). The gap between the two — the twenty percent headroom — is the operator's window to react before the operation is forcibly stopped. The headroom should be wide enough that the operator has time to either raise the cap, kill the runaway loop, or accept that the operation is going to be stopped.

A too-small headroom — say, ninety-five percent alert threshold against a hundred percent cap — gives the operator a few minutes at most. By the time the page is acknowledged and the dashboard is open, the kill-switch has already fired. A too-large headroom — say, fifty percent alert threshold — pages the operator far before there is anything actionable to do, training them to ignore the alert. Eighty percent is the empirical sweet spot for most workloads: a workload burning at a steady rate has a few tens of minutes between the alert and the cap, which is enough time to investigate without being so much that the alert feels premature.

---

## 7. Alerting on SLO Breaches

Module 18 shipped the trace recorder and explicitly deferred alerting. This is the section where the thread is picked back up. The model is the same as the rest of this module: a small number of well-defined policies, each handling one signal, each composing with the others without stepping on the others' toes.

### The three signals that matter

Three signals carry most of the weight. *Error rate* — the fraction of calls in a rolling window that ended in an exception. *Latency p95* — the 95th-percentile wall-clock duration of calls in a rolling window, where p95 is the right percentile for an alert because p50 misses tail problems and p99 is too noisy at low call volumes. *Budget burn* — the cumulative spend against the configured budget, with the alert firing when the burn crosses a threshold (typically 80% of the budget, leaving the operator time to react before the kill-switch fires).

The wrapper ships one rule per signal, with thresholds and windows configurable per deployment. Other signals exist and have their uses — cost-per-call (catches model-tier drift), fallback rate (catches silent quality drops from Section 5), circuit-trip count (catches provider degradation patterns) — but the three primary signals catch the majority of the incidents that matter, and an alerting layer that fires too often is one that gets muted.

The principle behind picking three rather than ten is that every alert rule has an ongoing maintenance cost. The rule has to be tuned to the deployment's normal behaviour; the threshold has to be reviewed when the workload changes; the alert's recipient list has to be kept current; the runbook for "what to do when this fires" has to be written and updated. Ten rules with the same level of care as three is hard to maintain; three rules with the same level of care as ten is easy. The right move is to start with three, ship them well-tuned and well-documented, and add a fourth only when there is concrete evidence that the three are missing an incident class.

### Cool-downs suppress storms

Cool-downs are what keep the alert layer useful during a sustained incident. An error-rate alert that fires once a second for ten minutes during an outage produces six hundred pages and zero new information after the first one. Each rule carries its own cool-down window: the error-rate rule fires once, then refuses to fire again for sixty seconds, then re-evaluates; the latency-p95 rule has the same shape with the same default; the budget-burn rule has no cool-down because the threshold crossing is a one-shot event — the budget is either at 80% or it is not, and a second crossing of the same threshold is impossible without a budget reset.

The cool-down is what separates an alert from a notification storm, and the right default for the rolling rules is *the time the operator needs to read the page and start investigating* — a minute is conservative; some teams configure five. Longer cool-downs are appropriate when the on-call is paged by a noisier channel (Slack, email) where the latency of acknowledgement is higher; shorter cool-downs are appropriate when the on-call is paged by PagerDuty or OpsGenie where the page is dispatched immediately and the operator is expected to acknowledge within seconds.

The cool-down also has to be re-armed correctly. After the cool-down expires, the rule re-evaluates the current state; if the breach is still in effect, the alert fires again. If the breach has cleared during the cool-down, the rule notes that the breach cleared and does not fire. The implementation has to be careful about the boundary case — a breach that clears and re-breaches within the cool-down should re-fire, because the second breach is a new incident, not a continuation of the first. The wrapper handles this by tracking the last-fired timestamp and the current-state-is-breached flag separately; the rule fires when the current state is breached *and* the last-fired timestamp is older than the cool-down.

### Pager versus dashboard

The pager-versus-dashboard split is the operational pattern this module assumes. Alerts wake humans; dashboards inform the humans once they are awake. The two surfaces have different consumers and should be designed to different requirements.

The pager surface — the `AlertSink` in this module's wrapper — is for the small number of events whose right response is "stop what you are doing and look at this." It is loud, infrequent, and structured to communicate one fact: which SLO is breached, by how much, since when. The pager alert is not the place for a detailed diagnostic; it is the trigger for the operator to open the dashboard and investigate.

The dashboard surface is for the operator who has been paged and now needs to investigate — and that surface is the [Module 18 (Observability & Monitoring)](../18-observability-monitoring/) trace recorder plus the `RunSummary` records this module's wrapper emits. The trace recorder gives the operator the per-call detail; the run summary gives the operator the operation-level aggregates; together they answer the questions the page raised. The dashboard is rich, the pager is sparse, and the two are designed not to overlap.

The discipline that holds the split together is that every pager alert carries a link to the corresponding dashboard view. The on-call operator reading "ERROR_RATE_HIGH on provider=anthropic since 03:14" should be one click away from "the trace view for all error spans on the anthropic provider over the last fifteen minutes," and that one click should land them on a view where the per-call detail is visible. The page is the trigger; the dashboard is the destination; the link is the path between them.

### Where this hooks into M18's trace recorder

The `AlertSink` is, architecturally, a trace consumer. It watches the same span stream the trace recorder writes; it filters that stream for breach conditions; it emits a structured alert event when the conditions are met. In a production system, both consumers would share infrastructure — one OpenTelemetry exporter feeding both a long-term trace store and a streaming alert engine; in this module they are separate components because M18 shipped the trace recorder first and explicitly deferred the alerting layer to this module.

The wrapper's `AlertSink` interface is deliberately small: a single `record(span)` method that the wrapper calls after every operation. The default implementation runs the three rules in-process and writes alerts to a structured log; production implementations replace the default with a sink that feeds PagerDuty, OpsGenie, or whatever notification system the team uses. The interface is what lets the project's test mode run alerts to stdout while the production deployment runs them to the on-call rotation, without any change to the wrapper's caller code.

That arrangement resolves the M18 handoff: the trace recorder is the substrate, the alerting layer is the policy that turns its output into pages, and pairing the policy with the resilience wrapper here makes the dependency explicit — alerts fire when the resilience policies are doing more work than usual, and the operator's response is to adjust the resilience configuration or escalate to the provider.

### Alert fatigue and the silent-success principle

The hardest part of running an alerting layer is keeping the alerts trustworthy. An alert that fires on a problem the operator does not need to act on — a transient blip the resilience layer already absorbed, a momentary latency spike from a heavy prompt, a budget warning ten minutes before the actual cap — teaches the operator to ignore the channel. Once the channel is ignored, the alert that does matter is missed too. The discipline that prevents this is conservative thresholds and the silent-success principle: most of the time, the resilience layer is doing its job and the operator should hear nothing.

The thresholds the wrapper ships with are deliberately conservative. The error-rate rule fires at five percent of calls erroring in the last five minutes, not at one. The latency-p95 rule fires at thirty seconds, not at ten. The budget-burn rule fires at eighty percent of cap, not at fifty. Each threshold is far enough from "normal" that crossing it is a real signal, not a noise event. A team that finds the alerts firing too often should tune the thresholds up, not down — the cost of a missed alert is far lower than the cost of a thousand spurious alerts that train the operator to mute the channel.

The silent-success principle also applies to the resilience layer's own behavior. A successful retry — the call failed once, retried, succeeded — is not an alert; it is a structured log event with `attempts=2` and the operator can query the log when they want to. A successful fallback hop — the primary failed, the secondary succeeded — is not an alert; it is a metadata field on the response and the alerting layer notices only when the hop rate becomes anomalous. The resilience layer's job is to absorb the noise; the alerting layer's job is to surface only the noise that requires human intervention.

---

## 8. The Ecosystem & When Not to Build This Yourself

The patterns in this module are well-known. None of them was invented here; this module's contribution is the composition, the order of operations, and the LLM-specific framing of the kill-switch. The pattern primitives — retries with jitter, circuit breakers, fallbacks — exist as battle-tested libraries in the Python ecosystem and the broader cloud-native ecosystem, and the right production answer is almost always to use those libraries rather than to hand-roll the wrapper from scratch.

### `tenacity` for retries

`tenacity` (`github.com/jd/tenacity`) is the Python standard for retry. It is decorator-based, supports exponential backoff with jitter out of the box, lets you specify which exception types to retry and which to raise through, and has good defaults. For any new code that needs a retry layer, `@retry(stop=stop_after_attempt(3), wait=wait_random_exponential(multiplier=1, max=10))` is the right starting point. The library is mature, the documentation is clear, and the implementation handles the edge cases (retry on certain exceptions only, callbacks on each attempt, integration with logging) that a hand-rolled version would have to reinvent.

The integration pattern is to wrap the underlying `completion()` call with the decorator, then let the rest of the wrapper compose around it. `tenacity` handles the within-call concerns — attempts, backoffs, jitter, exception filtering; the application owns the across-call concerns — circuit state, fallback iteration, budget tracking. The split is the same as the one this module's project implements, just with `tenacity` doing the inner work that the project's `RetryPolicy` does manually.

### `pybreaker` and `circuitbreaker` for breakers

`pybreaker` and `circuitbreaker` are the Python circuit-breaker libraries. `pybreaker` is the older, more configurable option with good support for state listeners and storage backends; `circuitbreaker` is a smaller, decorator-based library that is easier to drop into existing code. Either is appropriate; the choice depends on whether your application needs to share breaker state across processes (use `pybreaker` with a Redis backend) or whether per-process state is fine (use either; `circuitbreaker` is simpler).

`pybreaker` in particular has the listener pattern that makes it easy to integrate with the alerting layer in Section 7: a `state_change_listener` callback fires on every transition, and the callback can forward the state change to the trace recorder or to a metrics backend. The discipline of treating breaker state as observable is what lets the on-call engineer see "the Anthropic breaker has opened three times in the last hour" rather than just "the application is degraded."

### LiteLLM's built-ins

LiteLLM itself ships built-in support for the policies in this module. `completion(..., num_retries=3, fallbacks=["claude-haiku", "gpt-4o-mini"], timeout=10)` configures retries, the fallback chain, and the timeout in a single call. For applications that have already adopted LiteLLM as the provider abstraction (the recommendation from [Module 04 (AI API Layer)](../04-ai-api-layer/)), this is the path of least resistance — the retry, fallback, and timeout policies are configured in the same place the model is selected, and the library handles the bookkeeping.

Most applications do not need to wrap LiteLLM with their own resilience layer; the built-ins are sufficient for the common case. The wrapper this module ships exists for teaching purposes — to make the policies visible and inspectable — and for the workloads whose requirements exceed what LiteLLM's built-ins offer (custom budget logic, integration with an in-house alerting system, breaker state shared across processes). For everyone else, the LiteLLM built-ins are the right starting point.

The pattern that combines well: use LiteLLM's `num_retries` and `fallbacks` for the within-call and across-provider resilience, supplement with a small in-house budget gate that consults the cost telemetry from Module 17, and let LiteLLM's logging callbacks feed the trace recorder from Module 18. Three components, each from a different module, each owning a narrow responsibility — the same architecture this module's project implements, just split across libraries rather than concentrated in one wrapper.

### OpenAI SDK's `max_retries`

The OpenAI Python SDK has a `max_retries` parameter on the client. The retries are exponential and the implementation is solid, though the configuration is sparse compared to `tenacity` — there are fewer knobs, and the underlying policy is what OpenAI thinks is reasonable rather than what your application has measured. For straight OpenAI usage without LiteLLM, the built-in retries are usually enough, supplemented with a circuit breaker from `pybreaker` and a budget gate written in-house.

The Anthropic SDK has the same shape — a `max_retries` parameter on the client, sensible defaults, no exposed knobs for jitter or exception filtering. The pattern is consistent across providers: the SDKs ship a basic retry layer for the most common transient errors, and anything more sophisticated is the application's responsibility.

### Infrastructure-layer policies

At the infrastructure layer, AWS Resilience Hub and the Istio service mesh policies cover the same patterns at the network and orchestration layers. They are relevant when the LLM call is one hop in a larger service architecture — for example, a gateway service that fans out to multiple downstream LLM-powered services, where the resilience policies belong at the gateway rather than inside each service. The patterns are the same; the layer of abstraction is different.

For a single LLM-powered application, the in-process libraries above are simpler and have lower operational overhead than the mesh policies. For an application that is one service in a mesh, the mesh's policies should handle the network-layer resilience (retries on the HTTP call to the LLM provider, circuit-breaking at the egress proxy) and the application can focus on the LLM-specific concerns (the budget gate, the alerting on cost burn, the silent-quality-drop metadata). The split is by layer, not by feature: network-layer policies do network-layer work; application-layer policies do application-layer work.

### Use libraries in production, hand-roll for learning

Use a library in production. Hand-roll for learning. The code in this module is to teach the patterns — to make the six policies visible, the ordering defensible, the failure modes inspectable, and the trade-offs concrete in code that a reader can step through line by line. That is what a teaching project should do; it is not what a production codebase should ship.

In real code, reach for `tenacity` for retries, `pybreaker` for circuit breakers, LiteLLM's built-ins for fallbacks and timeouts, and a small in-house budget gate that consults the cost telemetry from [Module 17 (Caching & Cost Optimization)](../17-caching-cost-optimization/) and surfaces the configuration in the same place the rest of the application's operational settings live. The composition is the same; the implementation effort is an order of magnitude smaller; the production reliability is higher because the libraries have been hardened by years of use across thousands of deployments.

The teaching project's value is not that its implementation is better than the libraries — it is not — but that walking through the implementation forces the reader to understand the policies at a level that just configuring a library does not. After working through the project, the reader can read the `tenacity` source code and see exactly what `wait_random_exponential` is doing; can read `pybreaker`'s state machine and recognise the same three-state model from Section 4; can read LiteLLM's fallback logic and notice that it implements the same hopping rules from Section 5. The libraries are no longer black boxes; they are familiar implementations of familiar patterns.

### Pointer back to M18 for tracing

The other pointer back to earlier modules: every behavior this layer exhibits should be traceable in the [Module 18 (Observability & Monitoring)](../18-observability-monitoring/) trace recorder. Each retry attempt is a span with its own `attempt_number` attribute; each fallback hop is a span with its own `provider_used` attribute; each circuit transition is a structured log event; each budget gate decision is a structured log event with the `spent_usd` and `cap_usd` fields.

The resilience layer's job is to make the application keep running; the observability layer's job is to make the resilience layer's behavior inspectable; the alerting layer's job is to make the observability layer's signals actionable. The three layers compose into the production stack that the rest of the curriculum has been building toward. A team that has worked through Modules 17, 18, and 20 has the cost, the visibility, and the reliability — the three legs of a production LLM application, each owned by its own module, each integrated with the others through small, well-defined interfaces.

### What this module does not cover

A short note on the boundaries. This module covers the application-level resilience patterns — retries, breakers, fallbacks, budgets, alerting — and does not cover infrastructure-level concerns like load balancing, autoscaling, container orchestration, or multi-region deployment. Those are real production concerns and they interact with the patterns here (a multi-region deployment is the natural home for a multi-provider fallback chain, for example), but they belong to the platform engineering discipline rather than to the LLM-application discipline.

The patterns in this module are the ones that are specific to the LLM context — the budget kill-switch in particular has no analog in classical web infrastructure — and the patterns that apply uniformly to any remote dependency. A team running an LLM application on Kubernetes still needs the budget gate, the per-provider circuit, and the alerting on cost burn; the Kubernetes layer handles the orchestration, but the LLM-specific resilience belongs at the application layer where the wrapper sits.

The full set of production concerns for an LLM application is therefore broader than this module — it includes everything in this curriculum plus the platform engineering disciplines that any production system needs. What this module contributes is the layer that turns a sequence of LLM calls into a system that an operator can keep running through provider outages, rate-limit thrash, and the cost-runaway loops that no other module addresses.
