"""Fixture for multi-rule, single-file fix sequencing.

Combines four findings across different rules in one file:
- python:S3776 on `route_request` (cognitive complexity > 15)
- python:S1192 on the literal 'application/json' (>= 6 occurrences)
- python:S1481 on `unused_total` in `summarize`
- python:S125  on the commented block in `archive_record`

Tests whether fix_at_line preserves line offsets across multiple
fixes to the same file in one CLI run, AND whether the rewritten
S3776 prompt path coexists cleanly with the default path applied
to the other findings.
"""

import json
import urllib.request


def route_request(payload, status_code, attempt=0):
    if payload is None:
        if status_code >= 500:
            if attempt > 3:
                return "exhausted-empty"
            return "retry-empty"
        if status_code >= 400:
            return "empty-client"
        return "empty-other"
    if status_code >= 500:
        if attempt > 3:
            return "exhausted"
        if status_code == 503:
            return "busy"
        if status_code == 504:
            return "timeout"
        return "server-error"
    if status_code >= 400:
        if status_code == 401:
            return "auth"
        if status_code == 403:
            return "forbidden"
        if status_code == 404:
            return "not-found"
        return "client-error"
    return "ok"


def post_a(url, body, token):
    req = urllib.request.Request(url, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Accept", "application/json")
    req.add_header("Authorization", f"Bearer {token}")
    return urllib.request.urlopen(req, data=json.dumps(body).encode())


def post_b(url, body, token):
    req = urllib.request.Request(url, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Accept", "application/json")
    req.add_header("Authorization", f"Bearer {token}")
    return urllib.request.urlopen(req, data=json.dumps(body).encode())


def post_c(url, body, token):
    req = urllib.request.Request(url, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Accept", "application/json")
    req.add_header("Authorization", f"Bearer {token}")
    return urllib.request.urlopen(req, data=json.dumps(body).encode())


def summarize(values):
    unused_total = sum(values)
    return len(values)


def archive_record(record):
    # if record.archived:
    #     return None
    # record.mark_archived()
    return record
