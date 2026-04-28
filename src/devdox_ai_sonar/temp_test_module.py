"""Temporary scaffold module — not part of the production code path.

This file exists solely as a target for verifying devdox_sonar's
python:S1192 fix path against real code (string-literal extraction
via fix_at_line). Delete after the verification round.
"""

import json
import logging
import time
import urllib.request
from typing import Any
OUTBOUND_REQUEST_FAILED = 'Outbound request failed'

OUTBOUND_REQUEST_STARTED = 'Outbound request started'

APPLICATION_JSON = 'application/json'



DEFAULT_TIMEOUT_S = 30
MAX_RETRIES = 3

logger = logging.getLogger(__name__)


def _json_headers(token: str) -> dict:
    return {
        "Content-Type": APPLICATION_JSON,
        "Authorization": f"Bearer {token}",
        "Accept": APPLICATION_JSON,
        "Cache-Control": "no-cache",
    }


def post_json(url: str, payload: Any, token: str) -> Any:
    req = urllib.request.Request(url, method="POST")
    req.add_header("Content-Type", APPLICATION_JSON)
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("Accept", APPLICATION_JSON)
    req.add_header("Cache-Control", "no-cache")
    req.add_header("X-Idempotency-Key", _idem(payload))
    body = json.dumps(payload).encode("utf-8")
    logger.debug(OUTBOUND_REQUEST_STARTED)
    try:
        return urllib.request.urlopen(req, data=body, timeout=DEFAULT_TIMEOUT_S)
    except Exception:
        logger.exception(OUTBOUND_REQUEST_FAILED)
        raise


def put_json(url: str, payload: Any, token: str) -> Any:
    req = urllib.request.Request(url, method="PUT")
    req.add_header("Content-Type", APPLICATION_JSON)
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("Accept", APPLICATION_JSON)
    req.add_header("Cache-Control", "no-cache")
    body = json.dumps(payload).encode("utf-8")
    logger.debug(OUTBOUND_REQUEST_STARTED)
    try:
        return urllib.request.urlopen(req, data=body, timeout=DEFAULT_TIMEOUT_S)
    except Exception:
        logger.exception(OUTBOUND_REQUEST_FAILED)
        raise


def patch_json(url: str, payload: Any, token: str) -> Any:
    req = urllib.request.Request(url, method="PATCH")
    req.add_header(
        "Content-Type",
        APPLICATION_JSON,
    )
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("Accept", APPLICATION_JSON)
    req.add_header("Cache-Control", "no-cache")
    body = json.dumps(payload).encode("utf-8")
    logger.debug(OUTBOUND_REQUEST_STARTED)
    try:
        return urllib.request.urlopen(req, data=body, timeout=DEFAULT_TIMEOUT_S)
    except Exception:
        logger.exception(OUTBOUND_REQUEST_FAILED)
        raise


def delete_json(url: str, token: str) -> Any:
    req = urllib.request.Request(url, method="DELETE")
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("Accept", APPLICATION_JSON)
    req.add_header("Cache-Control", "no-cache")
    logger.debug(OUTBOUND_REQUEST_STARTED)
    try:
        return urllib.request.urlopen(req, timeout=DEFAULT_TIMEOUT_S)
    except Exception:
        logger.exception(OUTBOUND_REQUEST_FAILED)
        raise


def post_with_retry(url: str, payload: Any, token: str) -> Any:
    last_exc: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return post_json(url, payload, token)
        except Exception as exc:
            last_exc = exc
            logger.warning("Retrying outbound request")
            time.sleep(0.5 * attempt)
    logger.exception(OUTBOUND_REQUEST_FAILED)
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("post_with_retry exhausted without exception")


_OPERATIONS = {
    "create": ("POST", APPLICATION_JSON),
    "update": ("PUT", APPLICATION_JSON),
    "modify": ("PATCH", APPLICATION_JSON),
    "remove": ("DELETE", APPLICATION_JSON),
}


def dispatch(op: str, url: str, payload: Any, token: str) -> Any:
    method, content_type = _OPERATIONS[op]
    req = urllib.request.Request(url, method=method)
    req.add_header("Content-Type", content_type)
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("Accept", APPLICATION_JSON)
    req.add_header("Cache-Control", "no-cache")
    logger.debug(OUTBOUND_REQUEST_STARTED)
    if payload is None:
        return urllib.request.urlopen(req, timeout=DEFAULT_TIMEOUT_S)
    body = json.dumps(payload).encode("utf-8")
    return urllib.request.urlopen(req, data=body, timeout=DEFAULT_TIMEOUT_S)


def _idem(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True).encode("utf-8")
    return raw.hex()


def parse_response(raw: bytes) -> Any:
    return json.loads(raw.decode("utf-8"))


def parse_error(raw: bytes) -> Any:
    try:
        return json.loads(raw.decode("utf-8"))
    except Exception:
        logger.exception(OUTBOUND_REQUEST_FAILED)
        return None


def stringify(payload: Any) -> bytes:
    return json.dumps(payload).encode("utf-8")


def categorize_response(
    payload: Any,
    status_code: int,
    headers: dict,
    attempt: int = 0,
) -> str:
    """Bucket an HTTP response into a coarse outcome label.

    Deliberately branchy — used as an S3776 fixture to exercise the
    cognitive-complexity refactor path of devdox_sonar.
    """
    if payload is None:
        if status_code == 204:
            if attempt > 0:
                return "empty_success_after_retry"
            return "empty_success"
        if status_code == 200:
            return "empty_ok"
        return "empty_failure"
    if status_code >= 500:
        if status_code == 503 and attempt < 3:
            return "retry_service_unavailable"
        if status_code == 504 and attempt < 2:
            return "retry_gateway_timeout"
        if status_code == 502 or status_code == 500:
            if attempt > 5:
                return "give_up_server_error"
            return "server_error"
        return "server_error"
    if status_code >= 400:
        if status_code == 401:
            if "X-Refresh-Token" in headers and attempt == 0:
                return "auth_refresh_needed"
            return "auth_failed"
        if status_code == 403:
            if headers.get("X-Reason") == "quota":
                return "forbidden_quota"
            return "forbidden"
        if status_code == 404:
            if "not_found" in headers.get("X-Hint", "") or attempt > 0:
                return "missing"
            return "client_error"
        if status_code == 409:
            if attempt > 0 and headers.get("X-Conflict-Retryable") == "true":
                return "conflict_retry"
            return "conflict"
        if status_code == 429:
            if attempt >= 5:
                return "rate_limit_exhausted"
            if "Retry-After" in headers:
                return "rate_limit_with_hint"
            return "rate_limit"
        return "client_error"
    if status_code >= 200:
        if status_code == 201 or status_code == 202:
            return "accepted"
        if status_code == 206:
            return "partial"
        return "ok"
    return "unknown"
