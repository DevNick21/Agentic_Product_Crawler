"""Retry utilities with exponential backoff for API calls."""

import logging
import time
from typing import Callable, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar('T')


def retry_with_backoff(
    func: Callable[[], T],
    *,
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    max_delay: float = 60.0,
    retryable_exceptions: tuple = (TimeoutError, ConnectionError),
    logger_name: Optional[str] = None,
) -> T:
    """Retry a function with exponential backoff.

    Args:
        func: Function to retry (should take no arguments)
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds before first retry
        backoff_factor: Multiplicative factor for delay between retries
        max_delay: Maximum delay in seconds
        retryable_exceptions: Tuple of exceptions that trigger a retry
        logger_name: Optional logger name for logging

    Returns:
        Result of successful function call

    Raises:
        Last exception if all retries exhausted
    """
    log = logging.getLogger(logger_name) if logger_name else logger

    last_exception: Optional[Exception] = None
    delay = initial_delay

    for attempt in range(max_retries + 1):
        try:
            return func()
        except retryable_exceptions as e:
            last_exception = e

            if attempt >= max_retries:
                log.error(f"All {max_retries} retries exhausted: {e}")
                raise

            log.warning(
                f"Attempt {attempt + 1}/{max_retries + 1} failed: {e}. "
                f"Retrying in {delay:.1f}s..."
            )

            time.sleep(delay)
            delay = min(delay * backoff_factor, max_delay)
        except Exception as e:
            # Non-retryable exception - fail immediately
            log.error(f"Non-retryable error: {e}")
            raise

    # Should never reach here, but satisfy type checker
    if last_exception:
        raise last_exception
    raise RuntimeError("Unexpected: no result and no exception")


def is_retryable_api_error(error_msg: str) -> bool:
    """Check if an API error message indicates a retryable error.

    Args:
        error_msg: Error message string

    Returns:
        True if error is retryable (timeouts, rate limits, server errors)
    """
    if not error_msg:
        return False

    error_lower = error_msg.lower()

    # Retryable HTTP status codes and error types
    retryable_indicators = [
        "timeout",
        "timed out",
        "503",  # Service Unavailable
        "504",  # Gateway Timeout
        "429",  # Too Many Requests
        "502",  # Bad Gateway
        "500",  # Internal Server Error (sometimes transient)
        "overloaded",
        "rate limit",
        "temporarily unavailable",
        "connection reset",
        "connection refused",
        "network error",
    ]

    return any(indicator in error_lower for indicator in retryable_indicators)
