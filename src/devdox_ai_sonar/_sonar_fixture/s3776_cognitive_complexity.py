"""Fixture for python:S3776 (cognitive complexity).

One function deliberately exceeding the default cognitive complexity
threshold of 15. SonarCloud is expected to report exactly one
S3776 finding here, on `categorize_response`.

Routing: CognitiveComplexityHandler → REWRITTEN refactoring prompts.
"""

from typing import Any


def categorize_response(
    payload: Any,
    status_code: int,
    headers: dict,
    attempt: int = 0,
) -> str:
    if payload is None:
        if status_code >= 500:
            if attempt > 3:
                return "exhausted-empty-server-error"
            return "retry-empty-server-error"
        if status_code >= 400:
            return "empty-client-error"
        if status_code >= 200:
            return "empty-success"
        return "empty-unknown"
    if status_code >= 500:
        if attempt > 3:
            return "exhausted-server-error"
        if status_code == 503:
            if "Retry-After" in headers:
                return "retry-after-server-busy"
            return "server-busy"
        if status_code == 504:
            return "gateway-timeout"
        return "server-error"
    if status_code >= 400:
        if status_code == 401:
            if "WWW-Authenticate" in headers:
                return "auth-required"
            return "auth-missing"
        if status_code == 403:
            return "forbidden"
        if status_code == 404:
            return "not-found"
        if status_code == 429:
            if "Retry-After" in headers:
                return "rate-limited-retryable"
            return "rate-limited"
        return "client-error"
    if status_code >= 200:
        if status_code == 204:
            return "no-content"
        if status_code == 201:
            return "created"
        return "success"
    return "unknown"
