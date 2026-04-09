"""
Retry logic for LLM API calls using tenacity.

Provides a decorator that retries on transient errors with exponential backoff
and jitter, logging each retry attempt.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, TypeVar

from tenacity import (
    RetryCallState,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

# Import OpenAI exceptions, with fallback for environments without OpenAI
try:
    from openai import (
        APIConnectionError,
        APIStatusError,
        APITimeoutError,
        InternalServerError,
        RateLimitError,
    )
except Exception:  # pragma: no cover
    APIConnectionError = Exception
    APITimeoutError = Exception
    APIStatusError = Exception
    InternalServerError = Exception
    RateLimitError = Exception

RETRYABLE_STATUS_CODES = {408, 409, 429, 500, 502, 503, 504}

F = TypeVar("F", bound=Callable[..., Any])

__all__ = [
    "retry_llm_call",
    "is_retryable_exception",
    "extract_status_code",
]


def get_logger(name: str = "llm.retry") -> logging.Logger:
    """
    Return a configured logger for retry events.

    This logger prints to stdout with a timestamp and level.
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


def extract_status_code(exc: Exception) -> int | None:
    """
    Best‑effort extraction of an HTTP status code from an exception.

    Checks:
    - `exc.status_code`
    - `exc.response.status_code`
    - `exc.code` (for some API exceptions)
    """
    # Check for `status_code` attribute directly
    if hasattr(exc, "status_code"):
        try:
            return int(exc.status_code)
        except Exception:
            pass

    # Check for `response.status_code`
    response = getattr(exc, "response", None)
    if response is not None and hasattr(response, "status_code"):
        try:
            return int(response.status_code)
        except Exception:
            pass

    # Check for `code` attribute (used by some API clients)
    if hasattr(exc, "code"):
        try:
            return int(exc.code)
        except Exception:
            pass

    return None


def is_retryable_exception(exc: Exception) -> bool:
    """
    Return True if the exception is considered transient and should be retried.

    This includes:
    - OpenAI‑specific transient errors (RateLimitError, APITimeoutError, etc.)
    - HTTP status codes in RETRYABLE_STATUS_CODES (408, 429, 5xx, etc.)
    - Generic network errors (TimeoutError, ConnectionError)
    """
    # OpenAI‑specific exceptions
    if isinstance(
        exc, (RateLimitError, APITimeoutError, APIConnectionError, InternalServerError)
    ):
        return True

    # APIStatusError may contain a status code
    if isinstance(exc, APIStatusError):
        status_code = extract_status_code(exc)
        return status_code in RETRYABLE_STATUS_CODES

    # Generic network errors
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True

    # Any exception with a retryable status code
    status_code = extract_status_code(exc)
    if status_code in RETRYABLE_STATUS_CODES:
        return True

    return False


def log_before_sleep(retry_state: RetryCallState) -> None:
    """
    Log a retry attempt before sleeping.

    This function is called by tenacity before each sleep.
    """
    logger = get_logger()
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    attempt_number = retry_state.attempt_number

    if exc is None:
        logger.warning("Retry triggered without an exception object.")
        return

    status_code = extract_status_code(exc)
    if status_code is not None:
        logger.warning(
            "Transient LLM error on attempt %s. status_code=%s error=%s",
            attempt_number,
            status_code,
            repr(exc),
        )
    else:
        logger.warning(
            "Transient LLM error on attempt %s. error=%s",
            attempt_number,
            repr(exc),
        )


def retry_llm_call(max_attempts: int = 6) -> Callable[[F], F]:
    """
    Retry decorator for LLM API calls with exponential backoff and jitter.

    Args:
        max_attempts: Maximum number of attempts (including the first).

    Returns:
        A decorator that wraps the function with retry logic.

    Example:
        @retry_llm_call(max_attempts=3)
        def call_openai():
            ...
    """
    return retry(
        retry=retry_if_exception(is_retryable_exception),
        wait=wait_exponential_jitter(initial=1, max=30),
        stop=stop_after_attempt(max_attempts),
        reraise=True,
        before_sleep=log_before_sleep,
    )