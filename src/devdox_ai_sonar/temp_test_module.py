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


DEFAULT_TIMEOUT_S = 30
MAX_RETRIES = 3

logger = logging.getLogger(__name__)


def _json_headers(token: str) -> dict:
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
        "Cache-Control": "no-cache",
    }


def post_json(url: str, payload: Any, token: str) -> Any:
    req = urllib.request.Request(url, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("Accept", "application/json")
    req.add_header("Cache-Control", "no-cache")
    req.add_header("X-Idempotency-Key", _idem(payload))
    body = json.dumps(payload).encode("utf-8")
    logger.debug("Outbound request started")
    try:
        return urllib.request.urlopen(req, data=body, timeout=DEFAULT_TIMEOUT_S)
    except Exception:
        logger.exception("Outbound request failed")
        raise


def put_json(url: str, payload: Any, token: str) -> Any:
    req = urllib.request.Request(url, method="PUT")
    req.add_header("Content-Type", "application/json")
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("Accept", "application/json")
    req.add_header("Cache-Control", "no-cache")
    body = json.dumps(payload).encode("utf-8")
    logger.debug("Outbound request started")
    try:
        return urllib.request.urlopen(req, data=body, timeout=DEFAULT_TIMEOUT_S)
    except Exception:
        logger.exception("Outbound request failed")
        raise


def patch_json(url: str, payload: Any, token: str) -> Any:
    req = urllib.request.Request(url, method="PATCH")
    req.add_header(
        "Content-Type",
        "application/json",
    )
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("Accept", "application/json")
    req.add_header("Cache-Control", "no-cache")
    body = json.dumps(payload).encode("utf-8")
    logger.debug("Outbound request started")
    try:
        return urllib.request.urlopen(req, data=body, timeout=DEFAULT_TIMEOUT_S)
    except Exception:
        logger.exception("Outbound request failed")
        raise


def delete_json(url: str, token: str) -> Any:
    req = urllib.request.Request(url, method="DELETE")
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("Accept", "application/json")
    req.add_header("Cache-Control", "no-cache")
    logger.debug("Outbound request started")
    try:
        return urllib.request.urlopen(req, timeout=DEFAULT_TIMEOUT_S)
    except Exception:
        logger.exception("Outbound request failed")
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
    logger.exception("Outbound request failed")
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("post_with_retry exhausted without exception")


_OPERATIONS = {
    "create": ("POST", "application/json"),
    "update": ("PUT", "application/json"),
    "modify": ("PATCH", "application/json"),
    "remove": ("DELETE", "application/json"),
}


def dispatch(op: str, url: str, payload: Any, token: str) -> Any:
    method, content_type = _OPERATIONS[op]
    req = urllib.request.Request(url, method=method)
    req.add_header("Content-Type", content_type)
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("Accept", "application/json")
    req.add_header("Cache-Control", "no-cache")
    logger.debug("Outbound request started")
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
        logger.exception("Outbound request failed")
        return None


def stringify(payload: Any) -> bytes:
    return json.dumps(payload).encode("utf-8")
