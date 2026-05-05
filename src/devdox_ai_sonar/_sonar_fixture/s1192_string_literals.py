"""Fixture for python:S1192 (string literals should not be duplicated).

Three string literals each repeated 3+ times across multiple
functions. SonarCloud is expected to report 3 S1192 findings here
(one per duplicated literal):
- 'application/json' (>= 6 occurrences)
- 'Outbound request started' (3 occurrences)
- 'Outbound request failed' (3 occurrences)

Routing: StringLiteralDuplicateHandler -> DEFAULT agent prompts.
"""

import json
import logging
import urllib.request
from typing import Any

logger = logging.getLogger(__name__)


def post_json(url: str, payload: Any, token: str) -> Any:
    req = urllib.request.Request(url, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Accept", "application/json")
    req.add_header("Authorization", f"Bearer {token}")
    body = json.dumps(payload).encode("utf-8")
    logger.debug("Outbound request started")
    try:
        return urllib.request.urlopen(req, data=body, timeout=30)
    except Exception:
        logger.exception("Outbound request failed")
        raise


def put_json(url: str, payload: Any, token: str) -> Any:
    req = urllib.request.Request(url, method="PUT")
    req.add_header("Content-Type", "application/json")
    req.add_header("Accept", "application/json")
    req.add_header("Authorization", f"Bearer {token}")
    body = json.dumps(payload).encode("utf-8")
    logger.debug("Outbound request started")
    try:
        return urllib.request.urlopen(req, data=body, timeout=30)
    except Exception:
        logger.exception("Outbound request failed")
        raise


def patch_json(url: str, payload: Any, token: str) -> Any:
    req = urllib.request.Request(url, method="PATCH")
    req.add_header("Content-Type", "application/json")
    req.add_header("Accept", "application/json")
    req.add_header("Authorization", f"Bearer {token}")
    body = json.dumps(payload).encode("utf-8")
    logger.debug("Outbound request started")
    try:
        return urllib.request.urlopen(req, data=body, timeout=30)
    except Exception:
        logger.exception("Outbound request failed")
        raise
