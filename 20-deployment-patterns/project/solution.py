"""
Module 20: Resilient LLM wrapper.

Composes six policies around an LLM call:
  1. Budget gate     (reject before spending)
  2. Fallback chain  (iterate ordered providers)
  3. Circuit gate    (skip if open)
  4. Retry loop      (exponential backoff + full jitter)
  5. Timeout wrapper (wall-clock cap per attempt)
  6. Underlying call (litellm OR mock)

Run `python solution.py --list-scenarios` to see the chaos scenarios.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Literal, Protocol

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict


# ============================================================================
# 1. Constants
# ============================================================================

DEFAULT_MODEL = "openai/gpt-4o-mini"
DEFAULT_BUDGET_USD = 1.00
DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_BASE_DELAY = 0.5
DEFAULT_MAX_DELAY = 8.0
DEFAULT_TIMEOUT_S = 10.0
DEFAULT_CIRCUIT_FAILURE_THRESHOLD = 5
DEFAULT_CIRCUIT_RECOVERY_S = 30.0
DEFAULT_COOL_DOWN_S = 60.0
DEFAULT_ALERT_WINDOW = 10
DEFAULT_ERROR_RATE_THRESHOLD = 0.20
DEFAULT_LATENCY_P95_MS = 5000.0
DEFAULT_BUDGET_BURN_PERCENT = 0.80

MOCK_COST_INPUT_USD = 0.001
MOCK_COST_OUTPUT_USD = 0.002
FAST_FACTOR = 0.01

STATE_FILE = Path(".resilient_state.json")

# .env at repo root: this file lives at <repo>/20-deployment-patterns/project/solution.py
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(REPO_ROOT / ".env")


# ============================================================================
# 2. Exception taxonomy
# ============================================================================

class ResilientLLMError(Exception):
    pass


class RateLimitError(ResilientLLMError):
    def __init__(self, message: str = "rate limited", retry_after_s: float | None = None):
        super().__init__(message)
        self.retry_after_s = retry_after_s


class TimeoutError(ResilientLLMError):
    pass


class TransientError(ResilientLLMError):
    pass


class AuthError(ResilientLLMError):
    pass


class ModelNotFoundError(ResilientLLMError):
    pass


class ContentFilterError(ResilientLLMError):
    pass


class BadRequestError(ResilientLLMError):
    pass


class CircuitOpenError(ResilientLLMError):
    pass


class AllProvidersFailedError(ResilientLLMError):
    def __init__(self, causes: dict[str, Exception]):
        msg = "all providers failed: " + ", ".join(
            f"{name}={type(e).__name__}({e})" for name, e in causes.items()
        )
        super().__init__(msg)
        self.causes = causes


class BudgetExceededError(ResilientLLMError):
    def __init__(self, spent_usd: float, cap_usd: float):
        super().__init__(f"budget exceeded: spent ${spent_usd:.4f} of ${cap_usd:.4f}")
        self.spent_usd = spent_usd
        self.cap_usd = cap_usd


RETRIABLE_ERRORS: tuple[type[Exception], ...] = (RateLimitError, TimeoutError, TransientError)
PROVIDER_PERMANENT_ERRORS: tuple[type[Exception], ...] = (AuthError, ModelNotFoundError, ContentFilterError)


# ============================================================================
# 3. Pydantic models
# ============================================================================

class ProviderResponse(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    text: str
    cost_usd: float
    model: str


class CallResult(BaseModel):
    text: str
    cost_usd: float
    latency_ms: float
    provider_used: str
    fallback_hops: int
    retries_used: int                        # retries within the serving provider only
    circuit_state_at_start: dict[str, str]   # provider_name -> "CLOSED"|"OPEN"|"HALF_OPEN"


class AlertRecord(BaseModel):
    kind: str
    message: str
    timestamp_s: float
    value: float | None = None
    threshold: float | None = None


class ProviderSummary(BaseModel):
    name: str
    calls: int
    ok: int
    failed: int
    circuit_state: str


class RunSummary(BaseModel):
    scenario: str | None
    calls_total: int
    calls_ok: int
    calls_failed: int
    retries_total: int
    fallback_hops_total: int
    circuit_trips_total: int
    budget_spent_usd: float
    budget_remaining_usd: float
    p50_latency_ms: float
    p95_latency_ms: float
    max_latency_ms: float
    alerts_fired: list[AlertRecord]
    per_provider: list[ProviderSummary]


# ============================================================================
# 4. Chaos scenario dataclasses
# ============================================================================

@dataclass
class Step:
    kind: Literal["ok", "rate_limit", "timeout", "server_error", "auth_error", "bad_request"]
    latency_ms: int = 50
    output: str = "mock response"
    retry_after_s: float | None = None
    status: int | None = None


@dataclass
class ChaosScenario:
    name: str
    description: str
    primary_steps: list[Step]
    fallback_steps: list[Step]
    seed: int = 0
    loop_n: int = 1  # how many times the CLI calls wrapper.call()


# ============================================================================
# 5. Provider protocol + LiteLLMProvider
# ============================================================================

class Provider(Protocol):
    name: str
    def complete(self, messages: list[dict], model: str, **opts: Any) -> ProviderResponse: ...


class LiteLLMProvider:
    def __init__(self, name: str = "litellm"):
        self.name = name

    def complete(self, messages: list[dict], model: str, **opts: Any) -> ProviderResponse:
        import litellm
        try:
            response = litellm.completion(model=model, messages=messages, **opts)
        except Exception as e:
            # translate to our exception taxonomy
            name = type(e).__name__.lower()
            msg = str(e)
            if "ratelimit" in name or "rate_limit" in name or "429" in msg:
                retry_after = getattr(e, "retry_after", None)
                raise RateLimitError(msg, retry_after_s=retry_after) from e
            if "auth" in name or "401" in msg or "403" in msg:
                raise AuthError(msg) from e
            if "badrequest" in name or "400" in msg:
                raise BadRequestError(msg) from e
            if "notfound" in name or "not_found" in name or "model" in msg.lower():
                # broad — better to err on the side of permanent than to retry-loop a typo
                raise ModelNotFoundError(msg) from e
            if "timeout" in name or "timed out" in msg.lower():
                raise TimeoutError(msg) from e
            if "content" in msg.lower() and "filter" in msg.lower():
                raise ContentFilterError(msg) from e
            raise TransientError(msg) from e

        text = response.choices[0].message.content or ""
        try:
            cost_usd = litellm.completion_cost(completion_response=response)
        except Exception:
            cost_usd = 0.0
        return ProviderResponse(text=text, cost_usd=cost_usd, model=model)


# ============================================================================
# 6. MockProvider (chaos replay)
# ============================================================================

class MockProvider:
    def __init__(self, name: str, steps: list[Step]):
        self.name = name
        self.steps = steps
        self._cursor = 0
        self._abort_event: threading.Event | None = None

    def set_abort_event(self, event: threading.Event) -> None:
        """Called by TimeoutWrapper before invoking complete(); allows cooperative abort."""
        self._abort_event = event

    def complete(self, messages: list[dict], model: str, **opts: Any) -> ProviderResponse:
        if self._cursor < len(self.steps):
            step = self.steps[self._cursor]
            self._cursor += 1
        else:
            step = self.steps[-1]  # repeat last step forever

        self._sleep_cooperatively(step.latency_ms / 1000.0)

        if step.kind == "ok":
            return ProviderResponse(
                text=step.output,
                cost_usd=MOCK_COST_INPUT_USD + MOCK_COST_OUTPUT_USD,
                model=model,
            )
        if step.kind == "rate_limit":
            raise RateLimitError("mock: rate limited", retry_after_s=step.retry_after_s)
        if step.kind == "timeout":
            # If the sleep took longer than the timeout, TimeoutWrapper already raised on the
            # main thread. If --fast scaled it down enough, the call "succeeds" with a placeholder.
            return ProviderResponse(text="mock late response", cost_usd=0.0, model=model)
        if step.kind == "server_error":
            raise TransientError(f"mock: HTTP {step.status or 503}")
        if step.kind == "auth_error":
            raise AuthError("mock: invalid api key")
        if step.kind == "bad_request":
            raise BadRequestError("mock: 400 bad request")
        raise RuntimeError(f"unknown step kind: {step.kind}")

    def _sleep_cooperatively(self, total_s: float) -> None:
        elapsed = 0.0
        chunk = 0.05
        while elapsed < total_s:
            if self._abort_event is not None and self._abort_event.is_set():
                return
            sleep_for = min(chunk, total_s - elapsed)
            time.sleep(sleep_for)
            elapsed += sleep_for


# ============================================================================
# 7. Named chaos scenarios
# ============================================================================

_SCENARIOS: dict[str, ChaosScenario] = {
    "happy": ChaosScenario(
        name="happy",
        description="10 successful calls with low latency; baseline metrics",
        primary_steps=[Step("ok", latency_ms=80) for _ in range(10)],
        fallback_steps=[Step("ok", latency_ms=80)],
        loop_n=10,
        seed=1,
    ),
    "rate_limit_burst": ChaosScenario(
        name="rate_limit_burst",
        description="primary rate-limits 2 times then recovers; retries fire with retry_after honored",
        primary_steps=[
            Step("rate_limit", latency_ms=30, retry_after_s=1.0),
            Step("rate_limit", latency_ms=30, retry_after_s=1.0),
            *[Step("ok", latency_ms=80) for _ in range(8)],
        ],
        fallback_steps=[Step("ok", latency_ms=80)],
        loop_n=10,
        seed=2,
    ),
    "primary_dead": ChaosScenario(
        name="primary_dead",
        description="primary returns auth_error forever; fallback chain engages every call",
        primary_steps=[Step("auth_error", latency_ms=30) for _ in range(20)],
        fallback_steps=[Step("ok", latency_ms=80) for _ in range(20)],
        loop_n=10,
        seed=3,
    ),
    "circuit_trip": ChaosScenario(
        name="circuit_trip",
        description="primary returns 503 12 times in a row; circuit opens, fallback engages",
        primary_steps=[Step("server_error", latency_ms=30, status=503) for _ in range(12)],
        fallback_steps=[Step("ok", latency_ms=80) for _ in range(12)],
        loop_n=12,
        seed=4,
    ),
    "slow_burn": ChaosScenario(
        name="slow_burn",
        description="both providers return 503 every call; wrapper fails every call; error-rate alert fires",
        primary_steps=[Step("server_error", latency_ms=80, status=503) for _ in range(20)],
        fallback_steps=[Step("server_error", latency_ms=80, status=503) for _ in range(20)],
        loop_n=20,
        seed=5,
    ),
    "runaway": ChaosScenario(
        name="runaway",
        description="both providers ok forever; CLI loops aggressively to trigger budget kill-switch",
        primary_steps=[Step("ok", latency_ms=20)],
        fallback_steps=[Step("ok", latency_ms=20)],
        loop_n=10_000,
        seed=6,
    ),
    "slow_primary": ChaosScenario(
        name="slow_primary",
        description="primary always takes 30s (well over the per-attempt timeout); fallback is fast",
        primary_steps=[Step("timeout", latency_ms=30_000) for _ in range(10)],
        fallback_steps=[Step("ok", latency_ms=80) for _ in range(10)],
        loop_n=10,
        seed=7,
    ),
}


# ============================================================================
# 8. BudgetTracker
# ============================================================================

class BudgetTracker:
    def __init__(self, max_cost_usd: float, window_seconds: int | None = None):
        self.max_cost_usd = max_cost_usd
        self.window_seconds = window_seconds
        self.spent_usd = 0.0
        self._events: deque[tuple[float, float]] = deque()  # (timestamp, cost)

    def check(self) -> None:
        self._trim_window()
        if self.spent_usd >= self.max_cost_usd:
            raise BudgetExceededError(self.spent_usd, self.max_cost_usd)

    def record(self, cost_usd: float) -> None:
        ts = time.time()
        self.spent_usd += cost_usd
        self._events.append((ts, cost_usd))
        self._trim_window()

    def _trim_window(self) -> None:
        if self.window_seconds is None:
            return
        cutoff = time.time() - self.window_seconds
        while self._events and self._events[0][0] < cutoff:
            _, cost = self._events.popleft()
            self.spent_usd -= cost
        if self.spent_usd < 0:
            self.spent_usd = 0.0

    @property
    def remaining_usd(self) -> float:
        return max(0.0, self.max_cost_usd - self.spent_usd)

    @property
    def percent_used(self) -> float:
        if self.max_cost_usd <= 0:
            return 0.0
        return self.spent_usd / self.max_cost_usd


# ============================================================================
# 9. CircuitBreaker (per provider)
# ============================================================================

class CircuitState(str, Enum):
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"


class CircuitBreaker:
    def __init__(self,
                 failure_threshold: int = DEFAULT_CIRCUIT_FAILURE_THRESHOLD,
                 recovery_seconds: float = DEFAULT_CIRCUIT_RECOVERY_S):
        self.failure_threshold = failure_threshold
        self.recovery_seconds = recovery_seconds
        self.state = CircuitState.CLOSED
        self.consecutive_failures = 0
        self.opened_at: float | None = None
        self.trip_count = 0  # number of CLOSED -> OPEN transitions over lifetime

    def allow(self) -> bool:
        if self.state == CircuitState.CLOSED:
            return True
        if self.state == CircuitState.OPEN:
            assert self.opened_at is not None
            if time.time() - self.opened_at >= self.recovery_seconds:
                self.state = CircuitState.HALF_OPEN
                return True
            return False
        # HALF_OPEN — a probe is outstanding; we don't admit a second one until it resolves
        return False

    def record_success(self) -> None:
        self.consecutive_failures = 0
        self.state = CircuitState.CLOSED
        self.opened_at = None

    def record_failure(self) -> None:
        self.consecutive_failures += 1
        if self.state == CircuitState.HALF_OPEN:
            self._trip()
            return
        if self.state == CircuitState.CLOSED and self.consecutive_failures >= self.failure_threshold:
            self._trip()

    def _trip(self) -> None:
        was_not_open = self.state != CircuitState.OPEN
        self.state = CircuitState.OPEN
        self.opened_at = time.time()
        if was_not_open:
            self.trip_count += 1


# ============================================================================
# 10. RetryLoop (per provider, per attempt)
# ============================================================================

class RetryLoop:
    def __init__(self,
                 max_attempts: int = DEFAULT_MAX_ATTEMPTS,
                 base_delay: float = DEFAULT_BASE_DELAY,
                 max_delay: float = DEFAULT_MAX_DELAY,
                 retriable: tuple[type[Exception], ...] = RETRIABLE_ERRORS,
                 rng: random.Random | None = None,
                 sleep_fn: Callable[[float], None] | None = None):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.retriable = retriable
        self.rng = rng or random.Random()
        self.sleep_fn = sleep_fn or time.sleep
        self.last_retries_used: int = 0

    def call(self, fn: Callable[[], ProviderResponse]) -> tuple[ProviderResponse, int]:
        """Returns (response, retries_used). Raises the last exception if all attempts fail."""
        retries = 0
        last_exc: Exception | None = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                self.last_retries_used = retries
                return fn(), retries
            except self.retriable as e:
                last_exc = e
                if attempt >= self.max_attempts:
                    break
                delay = self._compute_delay(attempt, e)
                self.sleep_fn(delay)
                retries += 1
            except Exception:
                raise
        assert last_exc is not None
        self.last_retries_used = retries
        raise last_exc

    def _compute_delay(self, attempt: int, exc: Exception) -> float:
        if isinstance(exc, RateLimitError) and exc.retry_after_s is not None:
            return exc.retry_after_s
        backoff = min(self.max_delay, self.base_delay * (2 ** (attempt - 1)))
        return backoff * self.rng.random()  # full jitter


# ============================================================================
# 11. TimeoutWrapper (per attempt)
# ============================================================================

class TimeoutWrapper:
    def __init__(self, per_attempt_seconds: float = DEFAULT_TIMEOUT_S):
        self.per_attempt_seconds = per_attempt_seconds

    def call(self, fn: Callable[[], ProviderResponse], provider: Provider) -> ProviderResponse:
        result: list[Any] = [None]
        exc: list[Any] = [None]
        done = threading.Event()
        abort = threading.Event()

        # cooperative abort signal: mocks listen, real litellm does not
        if hasattr(provider, "set_abort_event"):
            provider.set_abort_event(abort)

        def worker() -> None:
            try:
                result[0] = fn()
            except Exception as e:
                exc[0] = e
            finally:
                done.set()

        t = threading.Thread(target=worker, daemon=True)
        t.start()
        if not done.wait(self.per_attempt_seconds):
            abort.set()
            raise TimeoutError(f"call exceeded {self.per_attempt_seconds:.1f}s")
        if exc[0] is not None:
            raise exc[0]
        return result[0]


# ============================================================================
# 12. Alert sinks
# ============================================================================

class AlertSink(Protocol):
    def emit(self, alert: AlertRecord) -> None: ...


class StderrAlertSink:
    def emit(self, alert: AlertRecord) -> None:
        print(f"[ALERT] {alert.kind}: {alert.message}", file=sys.stderr, flush=True)


class FileAlertSink:
    def __init__(self, path: str):
        self.path = Path(path)

    def emit(self, alert: AlertRecord) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(alert.model_dump_json() + "\n")


class CompositeAlertSink:
    def __init__(self, sinks: list[AlertSink]):
        self.sinks = sinks

    def emit(self, alert: AlertRecord) -> None:
        for s in self.sinks:
            s.emit(alert)


# ============================================================================
# 13. Alert rules
# ============================================================================

def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    sorted_v = sorted(values)
    k = max(0, min(len(sorted_v) - 1, int(round(q * (len(sorted_v) - 1)))))
    return sorted_v[k]


class ErrorRateAlert:
    def __init__(self,
                 threshold: float = DEFAULT_ERROR_RATE_THRESHOLD,
                 window: int = DEFAULT_ALERT_WINDOW,
                 cool_down_seconds: float = DEFAULT_COOL_DOWN_S):
        self.threshold = threshold
        self.window = window
        self.cool_down_seconds = cool_down_seconds
        self.last_fired_at: float | None = None
        self.recent_outcomes: deque[bool] = deque(maxlen=window)

    def observe_call(self, ok: bool) -> AlertRecord | None:
        self.recent_outcomes.append(ok)
        if len(self.recent_outcomes) < self.window:
            return None
        failed = sum(1 for o in self.recent_outcomes if not o)
        rate = failed / len(self.recent_outcomes)
        if rate <= self.threshold:
            return None
        now = time.time()
        if self.last_fired_at is not None and now - self.last_fired_at < self.cool_down_seconds:
            return None
        self.last_fired_at = now
        return AlertRecord(
            kind="error_rate",
            message=f"ErrorRate exceeded threshold: {rate * 100:.1f}% over last {len(self.recent_outcomes)} calls (limit {self.threshold * 100:.1f}%)",
            timestamp_s=now,
            value=rate,
            threshold=self.threshold,
        )


class LatencyAlert:
    def __init__(self,
                 p95_ms: float = DEFAULT_LATENCY_P95_MS,
                 window: int = DEFAULT_ALERT_WINDOW,
                 cool_down_seconds: float = DEFAULT_COOL_DOWN_S):
        self.p95_ms = p95_ms
        self.window = window
        self.cool_down_seconds = cool_down_seconds
        self.last_fired_at: float | None = None
        self.recent_latencies: deque[float] = deque(maxlen=window)

    def observe_call(self, latency_ms: float) -> AlertRecord | None:
        if latency_ms <= 0:
            return None
        self.recent_latencies.append(latency_ms)
        if len(self.recent_latencies) < self.window:
            return None
        p95 = _percentile(list(self.recent_latencies), 0.95)
        if p95 <= self.p95_ms:
            return None
        now = time.time()
        if self.last_fired_at is not None and now - self.last_fired_at < self.cool_down_seconds:
            return None
        self.last_fired_at = now
        return AlertRecord(
            kind="latency_p95",
            message=f"LatencyP95 exceeded threshold: p95={p95:.0f}ms over last {len(self.recent_latencies)} calls (limit {self.p95_ms:.0f}ms)",
            timestamp_s=now,
            value=p95,
            threshold=self.p95_ms,
        )


class BudgetBurnAlert:
    def __init__(self, percent: float = DEFAULT_BUDGET_BURN_PERCENT):
        self.percent = percent
        self.has_fired = False

    def observe_budget(self, tracker: BudgetTracker) -> AlertRecord | None:
        if self.has_fired:
            return None
        if tracker.percent_used < self.percent:
            return None
        self.has_fired = True
        return AlertRecord(
            kind="budget_burn",
            message=f"BudgetBurn crossed {self.percent * 100:.0f}%: spent ${tracker.spent_usd:.4f} of ${tracker.max_cost_usd:.4f}",
            timestamp_s=time.time(),
            value=tracker.percent_used,
            threshold=self.percent,
        )


# ============================================================================
# 14. MetricsRecorder
# ============================================================================

class MetricsRecorder:
    def __init__(self,
                 budget: BudgetTracker,
                 alert_sink: AlertSink,
                 error_rule: ErrorRateAlert | None = None,
                 latency_rule: LatencyAlert | None = None,
                 budget_rule: BudgetBurnAlert | None = None):
        self.budget = budget
        self.alert_sink = alert_sink
        self.error_rule = error_rule or ErrorRateAlert()
        self.latency_rule = latency_rule or LatencyAlert()
        self.budget_rule = budget_rule or BudgetBurnAlert()

        self.calls_total = 0
        self.calls_ok = 0
        self.calls_failed = 0
        self.retries_total = 0
        self.fallback_hops_total = 0
        self.latencies: list[float] = []
        self.alerts_fired: list[AlertRecord] = []
        self.per_provider: dict[str, ProviderSummary] = {}

    def register_provider(self, name: str) -> None:
        if name not in self.per_provider:
            self.per_provider[name] = ProviderSummary(
                name=name, calls=0, ok=0, failed=0, circuit_state="CLOSED"
            )

    def add_failed_retries(self, n: int) -> None:
        self.retries_total += n

    def record_provider_attempt(self, provider_name: str, ok: bool, circuit_state: str) -> None:
        s = self.per_provider[provider_name]
        s.calls += 1
        if ok:
            s.ok += 1
        else:
            s.failed += 1
        s.circuit_state = circuit_state

    def record_call(self, result: CallResult | None) -> None:
        self.calls_total += 1
        if result is not None:
            self.calls_ok += 1
            self.retries_total += result.retries_used
            self.fallback_hops_total += result.fallback_hops
            self.latencies.append(result.latency_ms)
            self._fire_observations(ok=True, latency_ms=result.latency_ms)
        else:
            self.calls_failed += 1
            self._fire_observations(ok=False, latency_ms=0.0)

    def _fire_observations(self, ok: bool, latency_ms: float) -> None:
        candidates: list[AlertRecord | None] = [
            self.error_rule.observe_call(ok),
            self.latency_rule.observe_call(latency_ms),
            self.budget_rule.observe_budget(self.budget),
        ]
        for alert in candidates:
            if alert is not None:
                self.alerts_fired.append(alert)
                self.alert_sink.emit(alert)

    def emit_circuit_alert(self, provider_name: str, consecutive_failures: int) -> None:
        alert = AlertRecord(
            kind="circuit_open",
            message=f"CircuitBreaker {provider_name}: CLOSED -> OPEN ({consecutive_failures} consecutive failures)",
            timestamp_s=time.time(),
        )
        self.alerts_fired.append(alert)
        self.alert_sink.emit(alert)

    def build_summary(self,
                      scenario: str | None,
                      circuit_trips_total: int,
                      provider_final_states: dict[str, str]) -> RunSummary:
        for name, state in provider_final_states.items():
            if name in self.per_provider:
                self.per_provider[name].circuit_state = state
        return RunSummary(
            scenario=scenario,
            calls_total=self.calls_total,
            calls_ok=self.calls_ok,
            calls_failed=self.calls_failed,
            retries_total=self.retries_total,
            fallback_hops_total=self.fallback_hops_total,
            circuit_trips_total=circuit_trips_total,
            budget_spent_usd=self.budget.spent_usd,
            budget_remaining_usd=self.budget.remaining_usd,
            p50_latency_ms=_percentile(self.latencies, 0.5),
            p95_latency_ms=_percentile(self.latencies, 0.95),
            max_latency_ms=max(self.latencies) if self.latencies else 0.0,
            alerts_fired=list(self.alerts_fired),
            per_provider=list(self.per_provider.values()),
        )


# ============================================================================
# 15. ResilientLLM (orchestrator)
# ============================================================================

class ResilientLLM:
    def __init__(self,
                 providers: list[Provider],
                 budget: BudgetTracker,
                 metrics: MetricsRecorder,
                 retry_loop: RetryLoop,
                 timeout_wrapper: TimeoutWrapper,
                 breakers: dict[str, CircuitBreaker] | None = None):
        if not providers:
            raise ValueError("at least one provider required")
        self.providers = providers
        self.budget = budget
        self.metrics = metrics
        self.retry_loop = retry_loop
        self.timeout_wrapper = timeout_wrapper
        self.breakers = breakers or {}
        for p in providers:
            self.metrics.register_provider(p.name)
            if p.name not in self.breakers:
                self.breakers[p.name] = CircuitBreaker()

    def call(self, messages: list[dict], model: str, **opts: Any) -> CallResult:
        # 1. Budget gate (raises BudgetExceededError if over cap)
        self.budget.check()

        start = time.time()
        circuit_state_at_start = {name: br.state.value for name, br in self.breakers.items()}

        causes: dict[str, Exception] = {}

        # 2. Fallback chain
        for hop, provider in enumerate(self.providers):
            breaker = self.breakers[provider.name]
            # 3. Circuit gate
            if not breaker.allow():
                causes[provider.name] = CircuitOpenError(f"{provider.name} circuit is OPEN")
                self.metrics.record_provider_attempt(
                    provider.name, ok=False, circuit_state=breaker.state.value
                )
                continue

            # 4 + 5 + 6: retry loop wraps timeout wrapper wraps provider.complete
            try:
                response, retries_used = self.retry_loop.call(
                    lambda p=provider: self.timeout_wrapper.call(
                        lambda pp=p: pp.complete(messages, model, **opts),
                        p,
                    )
                )
            except BadRequestError:
                # caller bug — do not hop, do not trip circuit
                self.metrics.record_provider_attempt(
                    provider.name, ok=False, circuit_state=breaker.state.value
                )
                self.metrics.record_call(None)
                raise
            except RETRIABLE_ERRORS as e:
                # all retries exhausted; treat as provider-permanent
                was_closed = breaker.state == CircuitState.CLOSED
                breaker.record_failure()
                if was_closed and breaker.state == CircuitState.OPEN:
                    self.metrics.emit_circuit_alert(provider.name, breaker.consecutive_failures)
                self.metrics.add_failed_retries(self.retry_loop.last_retries_used)
                causes[provider.name] = e
                self.metrics.record_provider_attempt(
                    provider.name, ok=False, circuit_state=breaker.state.value
                )
                continue
            except PROVIDER_PERMANENT_ERRORS as e:
                # provider-permanent; hop, but do not trip circuit
                causes[provider.name] = e
                self.metrics.record_provider_attempt(
                    provider.name, ok=False, circuit_state=breaker.state.value
                )
                continue

            # success
            breaker.record_success()
            self.budget.record(response.cost_usd)
            latency_ms = (time.time() - start) * 1000.0
            result = CallResult(
                text=response.text,
                cost_usd=response.cost_usd,
                latency_ms=latency_ms,
                provider_used=provider.name,
                fallback_hops=hop,
                retries_used=retries_used,
                circuit_state_at_start=circuit_state_at_start,
            )
            self.metrics.record_provider_attempt(
                provider.name, ok=True, circuit_state=breaker.state.value
            )
            self.metrics.record_call(result)
            return result

        # all providers failed
        self.metrics.record_call(None)
        raise AllProvidersFailedError(causes)


# ============================================================================
# 16. CLI helpers
# ============================================================================

def _scale_scenario(scenario: ChaosScenario, factor: float) -> ChaosScenario:
    """Apply --fast scaling to all latencies and retry_after values."""
    def scale_step(s: Step) -> Step:
        return Step(
            kind=s.kind,
            latency_ms=max(1, int(s.latency_ms * factor)),
            output=s.output,
            retry_after_s=(s.retry_after_s * factor) if s.retry_after_s is not None else None,
            status=s.status,
        )
    return ChaosScenario(
        name=scenario.name,
        description=scenario.description,
        primary_steps=[scale_step(s) for s in scenario.primary_steps],
        fallback_steps=[scale_step(s) for s in scenario.fallback_steps],
        seed=scenario.seed,
        loop_n=scenario.loop_n,
    )


def _print_summary(summary: RunSummary) -> None:
    total_budget = summary.budget_spent_usd + summary.budget_remaining_usd
    percent = (100.0 * summary.budget_spent_usd / total_budget) if total_budget > 0 else 0.0
    lines = []
    lines.append("")
    lines.append("Summary")
    lines.append("-" * 41)
    lines.append(f"Calls:           {summary.calls_total} total | {summary.calls_ok} ok | {summary.calls_failed} failed")
    lines.append(f"Retries:         {summary.retries_total} total")
    lines.append(f"Fallback hops:   {summary.fallback_hops_total} total")
    lines.append(f"Circuit trips:   {summary.circuit_trips_total}")
    lines.append(f"Budget:          ${summary.budget_spent_usd:.4f} / ${total_budget:.4f} ({percent:.1f}% used)")
    lines.append(f"Latency:         p50={summary.p50_latency_ms:.0f}ms  p95={summary.p95_latency_ms:.0f}ms  max={summary.max_latency_ms:.0f}ms")

    kind_counts: dict[str, int] = {}
    for a in summary.alerts_fired:
        kind_counts[a.kind] = kind_counts.get(a.kind, 0) + 1
    if summary.alerts_fired:
        bits = ", ".join(f"{n} {k}" for k, n in kind_counts.items())
        lines.append(f"Alerts:          {len(summary.alerts_fired)} fired ({bits})")
    else:
        lines.append("Alerts:          0 fired")

    lines.append("")
    lines.append("Per-provider:")
    for ps in summary.per_provider:
        lines.append(f"  {ps.name:<18} {ps.calls} calls  {ps.ok} ok  {ps.failed} failed   circuit={ps.circuit_state}")
    lines.append("")
    print("\n".join(lines))


def _save_state(summary: RunSummary) -> None:
    STATE_FILE.write_text(summary.model_dump_json(indent=2))
    print(f"Saved to {STATE_FILE}", flush=True)


def _load_state() -> RunSummary | None:
    if not STATE_FILE.exists():
        return None
    raw = STATE_FILE.read_text()
    return RunSummary.model_validate_json(raw)


def _list_scenarios() -> None:
    print(f"{'Scenario':<22} Description")
    print("-" * 80)
    for name, sc in _SCENARIOS.items():
        print(f"{name:<22} {sc.description}")


# ============================================================================
# 17. main / argparse
# ============================================================================

def _run_chaos(name: str, fast: bool, budget_usd: float, max_attempts: int,
               timeout_s: float, alert_file: str | None, seed_override: int | None) -> None:
    if name not in _SCENARIOS:
        print(f"unknown scenario: {name}", file=sys.stderr)
        print(f"available: {', '.join(_SCENARIOS.keys())}", file=sys.stderr)
        sys.exit(2)
    scenario = _SCENARIOS[name]

    if fast:
        scenario = _scale_scenario(scenario, factor=FAST_FACTOR)
        latency_alert_p95 = DEFAULT_LATENCY_P95_MS * FAST_FACTOR
        timeout_s_effective = timeout_s * FAST_FACTOR
        # also scale the circuit recovery_seconds so HALF_OPEN can fire in --fast runs
        recovery_s_effective = DEFAULT_CIRCUIT_RECOVERY_S * FAST_FACTOR
    else:
        latency_alert_p95 = DEFAULT_LATENCY_P95_MS
        timeout_s_effective = timeout_s
        recovery_s_effective = DEFAULT_CIRCUIT_RECOVERY_S

    seed = seed_override if seed_override is not None else scenario.seed

    primary = MockProvider("mock_primary", scenario.primary_steps)
    fallback = MockProvider("mock_fallback", scenario.fallback_steps)

    budget = BudgetTracker(max_cost_usd=budget_usd)

    sinks: list[AlertSink] = [StderrAlertSink()]
    if alert_file:
        sinks.append(FileAlertSink(alert_file))
    alert_sink: AlertSink = sinks[0] if len(sinks) == 1 else CompositeAlertSink(sinks)

    metrics = MetricsRecorder(
        budget=budget,
        alert_sink=alert_sink,
        latency_rule=LatencyAlert(p95_ms=latency_alert_p95),
    )

    rng = random.Random(seed)
    retry_loop = RetryLoop(max_attempts=max_attempts, rng=rng)
    timeout_wrapper = TimeoutWrapper(per_attempt_seconds=timeout_s_effective)

    breakers = {
        "mock_primary": CircuitBreaker(recovery_seconds=recovery_s_effective),
        "mock_fallback": CircuitBreaker(recovery_seconds=recovery_s_effective),
    }

    wrapper = ResilientLLM(
        providers=[primary, fallback],
        budget=budget,
        metrics=metrics,
        retry_loop=retry_loop,
        timeout_wrapper=timeout_wrapper,
        breakers=breakers,
    )

    print(f"Scenario: {scenario.name}")
    print(f"Description: {scenario.description}")
    print(f"Providers: primary={primary.name}, fallback={fallback.name}")
    if fast:
        print(f"(--fast: latencies and timers scaled {int(1/FAST_FACTOR)}x; timeout={timeout_s_effective:.3f}s; p95_alert={latency_alert_p95:.0f}ms)")
    print(f"\nRunning up to {scenario.loop_n} calls...\n", flush=True)

    messages = [{"role": "user", "content": "what is exponential backoff?"}]
    stopped_early: str | None = None
    for i in range(scenario.loop_n):
        try:
            wrapper.call(messages, model="mock-model")
        except BudgetExceededError as e:
            stopped_early = f"budget exceeded after {i} calls: {e}"
            break
        except AllProvidersFailedError:
            # wrapper failure is expected in chaos scenarios; keep looping so
            # metrics (error_rate, etc.) accumulate observations across the run
            continue
    if stopped_early:
        print(f"\n{stopped_early}", flush=True)

    circuit_trips_total = sum(br.trip_count for br in breakers.values())
    summary = metrics.build_summary(
        scenario=scenario.name,
        circuit_trips_total=circuit_trips_total,
        provider_final_states={name: br.state.value for name, br in breakers.items()},
    )
    _print_summary(summary)
    _save_state(summary)


def _run_ask(prompt: str, budget_usd: float, max_attempts: int, timeout_s: float,
             alert_file: str | None) -> None:
    model = os.environ.get("LLM_MODEL")
    if not model:
        model = DEFAULT_MODEL
        print(f"using default model: {model}", file=sys.stderr)

    provider = LiteLLMProvider("litellm")
    budget = BudgetTracker(max_cost_usd=budget_usd)

    sinks: list[AlertSink] = [StderrAlertSink()]
    if alert_file:
        sinks.append(FileAlertSink(alert_file))
    alert_sink: AlertSink = sinks[0] if len(sinks) == 1 else CompositeAlertSink(sinks)

    metrics = MetricsRecorder(budget=budget, alert_sink=alert_sink)
    retry_loop = RetryLoop(max_attempts=max_attempts)
    timeout_wrapper = TimeoutWrapper(per_attempt_seconds=timeout_s)

    wrapper = ResilientLLM(
        providers=[provider],
        budget=budget,
        metrics=metrics,
        retry_loop=retry_loop,
        timeout_wrapper=timeout_wrapper,
    )

    messages = [{"role": "user", "content": prompt}]
    try:
        result = wrapper.call(messages, model=model)
        print(result.text)
        print("", flush=True)
    except (AllProvidersFailedError, BudgetExceededError) as e:
        print(f"call failed: {e}", file=sys.stderr)

    circuit_trips_total = sum(br.trip_count for br in wrapper.breakers.values())
    summary = metrics.build_summary(
        scenario=None,
        circuit_trips_total=circuit_trips_total,
        provider_final_states={name: br.state.value for name, br in wrapper.breakers.items()},
    )
    _print_summary(summary)
    _save_state(summary)


def main() -> None:
    parser = argparse.ArgumentParser(description="Module 20: resilient LLM wrapper.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--ask", metavar="PROMPT", help="real call against LiteLLMProvider")
    mode.add_argument("--chaos", metavar="SCENARIO", help="run a named chaos scenario against mock providers")
    mode.add_argument("--list-scenarios", action="store_true", help="print scenarios and exit")
    mode.add_argument("--report", action="store_true", help="re-print the last run from .resilient_state.json")

    parser.add_argument("--fast", action="store_true", help="scale chaos latencies and timers 100x")
    parser.add_argument("--budget-usd", type=float, default=DEFAULT_BUDGET_USD)
    parser.add_argument("--max-attempts", type=int, default=DEFAULT_MAX_ATTEMPTS)
    parser.add_argument("--timeout-s", type=float, default=DEFAULT_TIMEOUT_S)
    parser.add_argument("--alert-file", type=str, default=None, help="also append alerts to this JSONL file")
    parser.add_argument("--seed", type=int, default=None)

    args = parser.parse_args()

    if args.list_scenarios:
        _list_scenarios()
        return

    if args.report:
        summary = _load_state()
        if summary is None:
            print("no previous run found; try --chaos happy first")
            return
        _print_summary(summary)
        return

    if args.ask:
        _run_ask(args.ask, args.budget_usd, args.max_attempts, args.timeout_s, args.alert_file)
        return

    if args.chaos:
        _run_chaos(args.chaos, args.fast, args.budget_usd, args.max_attempts,
                   args.timeout_s, args.alert_file, args.seed)
        return


if __name__ == "__main__":
    main()
