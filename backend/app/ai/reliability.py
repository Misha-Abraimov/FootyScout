"""Central bounded reliability policy for external AI and web providers."""

from __future__ import annotations

import random
import time
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class ProviderReliabilityPolicy:
    max_retries: int = 1
    base_backoff_seconds: float = 0.25
    max_backoff_seconds: float = 2.0
    jitter_seconds: float = 0.1
    max_provider_calls: int = 4
    max_output_tokens: int = 4096
    estimated_cost_ceiling_usd: float | None = None

    def __post_init__(self) -> None:
        if not 0 <= self.max_retries <= 2:
            raise ValueError("max_retries must be between 0 and 2.")
        if self.base_backoff_seconds < 0 or self.max_backoff_seconds < 0:
            raise ValueError("Backoff values cannot be negative.")
        if self.max_provider_calls < 1:
            raise ValueError("max_provider_calls must be positive.")
        if self.max_output_tokens < 1:
            raise ValueError("max_output_tokens must be positive.")


@dataclass(frozen=True)
class RetryOutcome:
    attempts: int
    retry_count: int
    rate_limit_events: int


class RetryExhaustedError(RuntimeError):
    def __init__(self, error: Exception, outcome: RetryOutcome) -> None:
        super().__init__(str(error))
        self.error = error
        self.outcome = outcome


def call_with_retry[T](
    operation: Callable[[], T],
    *,
    policy: ProviderReliabilityPolicy,
    is_retryable: Callable[[Exception], bool],
    retry_after_seconds: Callable[[Exception], float | None] | None = None,
    sleeper: Callable[[float], None] = time.sleep,
    jitter: Callable[[float, float], float] = random.uniform,
) -> tuple[T, RetryOutcome]:
    """Execute one external operation with bounded transient-only retries."""
    retries = 0
    rate_limits = 0
    while True:
        try:
            return operation(), RetryOutcome(
                attempts=retries + 1,
                retry_count=retries,
                rate_limit_events=rate_limits,
            )
        except Exception as exc:
            if _http_status(exc) == 429:
                rate_limits += 1
            if not is_retryable(exc) or retries >= policy.max_retries:
                raise RetryExhaustedError(
                    exc,
                    RetryOutcome(
                        attempts=retries + 1,
                        retry_count=retries,
                        rate_limit_events=rate_limits,
                    ),
                ) from exc
            delay = (
                retry_after_seconds(exc)
                if retry_after_seconds is not None
                else None
            )
            if delay is None:
                delay = min(
                    policy.max_backoff_seconds,
                    policy.base_backoff_seconds * (2**retries),
                ) + jitter(0.0, policy.jitter_seconds)
            sleeper(max(0.0, min(delay, policy.max_backoff_seconds)))
            retries += 1


def is_transient_provider_error(exc: Exception) -> bool:
    """Classify only network, timeout, 429, and selected 5xx failures as transient."""
    status = _http_status(exc)
    if status == 429 or (status is not None and 500 <= status <= 599):
        return True
    name = type(exc).__name__.casefold()
    return any(marker in name for marker in ("timeout", "connection", "connecterror"))


def retry_after_from_headers(exc: Exception) -> float | None:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    value = headers.get("retry-after")
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _http_status(exc: Exception) -> int | None:
    status = getattr(exc, "status_code", None)
    response = getattr(exc, "response", None)
    if status is None and response is not None:
        status = getattr(response, "status_code", None)
    return status if isinstance(status, int) else None
