#!/usr/bin/env python3
#
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#
"""Shared GitCode API client for batch scripts.

Extracted from fetch_issues.py and classify_issues.py to eliminate duplicated
HTTP / token / URL / time utilities. This module is imported by upper-layer
scripts via sys.path; it is NOT a standalone CLI tool.

Design notes:
- Token: passed as query param for GET (api_get), as form-data field for POST
  (api_post). This matches GitCode API conventions confirmed in gitcode-api.md.
- Retry: 3 attempts; 429 honors the server rate-limit window, while 5xx /
  ConnectionError / Timeout use linear backoff.
  Non-retryable 4xx raised immediately (GET) or returned (POST).
- Proactive limiting: callers may attach a cross-process rolling limiter. The
  default reserves one request every 60/45 seconds and persists only timestamps
  under a SHA-256 bucket derived from API host and token.
- Token safety: redact_token() / safe_error_text() strip access_token from
  URLs before logging. See token-config.md for token handling policy.

Usage (from an upper-layer script):
    import sys, os
    _HERE = os.path.dirname(os.path.realpath(__file__))
    _TOOLKIT = os.path.normpath(
        os.path.join(_HERE, '..', '..', 'gitcode-toolkit', 'scripts')
    )
    sys.path.insert(0, _TOOLKIT)
    from gitcode_client import (
        resolve_token, parse_repo_path, parse_issue_url, resolve_api_base,
        SharedRateLimiter, make_session, rate_limit_metrics,
        api_get, api_post, api_put, api_patch, redact_token, safe_error_text,
        parse_iso, DEFAULT_GITCODE_API_BASE, TZ_CHINA, STATE_MAP,
    )
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import logging
import os
import tempfile
import threading
import time
from datetime import datetime, timezone, timedelta
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import NamedTuple
from urllib.parse import urlparse

import requests

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
DEFAULT_GITCODE_API_BASE = "https://api.gitcode.com/api/v5"
TZ_CHINA = timezone(timedelta(hours=8))
STATE_MAP = {"opened": "open", "closed": "closed", "all": "all"}
DEFAULT_USER_AGENT = "cannbot/gitcode-client"
DEFAULT_RATE_LIMIT = 45
DEFAULT_RATE_WINDOW = 60.0
DEFAULT_RATE_BURST = 1
LOGGER = logging.getLogger(__name__)


class GitCodeClientError(ValueError):
    """Raised when local GitCode client configuration is invalid."""


class RateLimitPolicy(NamedTuple):
    """Stable rolling-window settings for one shared limiter."""

    limit: int = DEFAULT_RATE_LIMIT
    window: float = DEFAULT_RATE_WINDOW
    burst: int = DEFAULT_RATE_BURST


class RateLimitHooks(NamedTuple):
    """Injectable time hooks used by deterministic limiter tests."""

    clock: object = time.time
    sleeper: object = time.sleep


class SharedRateLimiter:
    """Cross-process rolling-window limiter keyed by API host and token digest."""

    _STATE_VERSION = 1

    def __init__(
        self,
        state_dir,
        *,
        policy=None,
        hooks=None,
    ):
        policy = policy or RateLimitPolicy()
        hooks = hooks or RateLimitHooks()
        if (
            int(policy.limit) <= 0
            or float(policy.window) <= 0
            or int(policy.burst) <= 0
        ):
            raise GitCodeClientError("Error: invalid GitCode rate-limit settings")
        self.state_dir = Path(state_dir)
        self.policy = RateLimitPolicy(
            int(policy.limit),
            float(policy.window),
            int(policy.burst),
        )
        self.hooks = hooks
        self._metrics_lock = threading.Lock()
        self._metrics = {
            "http_attempts": 0,
            "limiter_waits": 0,
            "limiter_wait_seconds": 0.0,
            "rate_limit_429s": 0,
        }

    @staticmethod
    def _bucket_id(url, token):
        host = urlparse(url).netloc.casefold()
        identity = f"{host}\0{token or 'anonymous'}".encode("utf-8")
        return hashlib.sha256(identity).hexdigest()

    @staticmethod
    def _write_state(state_path, payload):
        state_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=state_path.parent,
                prefix=f".{state_path.name}.",
                suffix=".tmp",
                delete=False,
            ) as file_obj:
                temp_path = Path(file_obj.name)
                json.dump(payload, file_obj, separators=(",", ":"))
                file_obj.flush()
                os.fsync(file_obj.fileno())
            os.replace(temp_path, state_path)
        finally:
            if temp_path and temp_path.exists():
                temp_path.unlink()

    def acquire(self, url, token):
        """Wait outside the file lock, then reserve one real HTTP attempt."""
        while True:
            def reserve(state, now):
                timestamps = state["timestamps"]
                waits = [state["cooldown_until"] - now]
                if timestamps and self.policy.burst == 1:
                    waits.append(
                        timestamps[-1]
                        + self.policy.window / self.policy.limit
                        - now
                    )
                if len(timestamps) >= self.policy.limit:
                    waits.append(timestamps[0] + self.policy.window - now)
                wait = max(0.0, *waits)
                if wait > 0:
                    return wait, False
                timestamps.append(now)
                state["cooldown_until"] = max(0.0, state["cooldown_until"])
                return 0.0, True

            wait = self._locked_state(url, token, reserve)
            if wait <= 0:
                with self._metrics_lock:
                    self._metrics["http_attempts"] += 1
                return
            with self._metrics_lock:
                self._metrics["limiter_waits"] += 1
                self._metrics["limiter_wait_seconds"] += wait
            self.hooks.sleeper(wait)

    def set_cooldown(self, url, token, delay):
        """Publish one 429 cooldown so other processes stop before retrying."""
        delay = max(0.0, float(delay))

        def update(state, now):
            state["cooldown_until"] = max(state["cooldown_until"], now + delay)
            return None, True

        self._locked_state(url, token, update)
        with self._metrics_lock:
            self._metrics["rate_limit_429s"] += 1

    def pause(self, delay):
        """Use the injected sleeper for non-rate-limit retry backoff."""
        self.hooks.sleeper(delay)

    def snapshot(self):
        """Return non-sensitive transport counters for run diagnostics."""
        with self._metrics_lock:
            result = dict(self._metrics)
        result["limiter_wait_seconds"] = round(result["limiter_wait_seconds"], 3)
        return result

    def _paths(self, url, token):
        bucket = self._bucket_id(url, token)
        return self.state_dir / f"{bucket}.json", self.state_dir / f"{bucket}.lock"

    def _load_state(self, state_path, now):
        try:
            payload = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            payload = {}
        if not isinstance(payload, dict) or payload.get("version") != self._STATE_VERSION:
            payload = {}
        timestamps = []
        for value in payload.get("timestamps") or []:
            try:
                parsed = float(value)
            except (TypeError, ValueError, OverflowError):
                continue
            if now - self.policy.window < parsed <= now + self.policy.window:
                timestamps.append(parsed)
        try:
            cooldown_until = float(payload.get("cooldown_until") or 0)
        except (TypeError, ValueError, OverflowError):
            cooldown_until = 0.0
        if cooldown_until > now + 3600:
            cooldown_until = 0.0
        return {
            "version": self._STATE_VERSION,
            "timestamps": sorted(timestamps),
            "cooldown_until": max(0.0, cooldown_until),
        }

    def _locked_state(self, url, token, update):
        state_path, lock_path = self._paths(url, token)
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a+", encoding="utf-8") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            now = float(self.hooks.clock())
            state = self._load_state(state_path, now)
            result, changed = update(state, now)
            if changed:
                self._write_state(state_path, state)
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        return result


# --------------------------------------------------------------------------- #
# Token resolution
# --------------------------------------------------------------------------- #
def resolve_token(cli_token=None):
    """Resolve GitCode API token. Priority: CLI arg > env var > error.

    Aligns with token-config.md (user message > env > initial preflight). The
    upper-layer skill asks from its initial credential preflight, or re-enters
    that preflight after a required endpoint returns 401/403. Downstream
    scripts fail fast instead of independently asking the user.
    """
    token = cli_token or os.environ.get("GITCODE_TOKEN")
    if not token:
        raise GitCodeClientError(
            "Error: API token not provided.\n"
            "Pass --token <token> or set the GITCODE_TOKEN environment variable."
        )
    return token


# --------------------------------------------------------------------------- #
# URL parsing
# --------------------------------------------------------------------------- #
def parse_repo_path(repo_url):
    """Extract (owner, repo) from a repository URL.

    Examples:
        https://gitcode.com/cann/ops-math        -> ('cann', 'ops-math')
        https://gitcode.com/group/sub/repo.git/  -> ('sub', 'repo')
    The last two non-empty path segments are used; trailing slashes and a
    trailing .git suffix are stripped.
    """
    parsed = urlparse(repo_url)
    parts = [p for p in parsed.path.split("/") if p]
    if len(parts) < 2:
        raise GitCodeClientError(f"Error: cannot parse owner/repo from URL: {repo_url}")
    repo = parts[-1]
    if repo.endswith(".git"):
        repo = repo[:-4]
    return parts[-2], repo


def parse_issue_url(issue_url):
    """Extract (owner, repo, number) from an issue URL.

    Examples:
        https://gitcode.com/cann/ops-math/issues/2170     -> ('cann', 'ops-math', '2170')
        https://gitcode.com/group/sub/repo/issues/2170/   -> ('sub', 'repo', '2170')
    Locates the 'issues' path segment; the two segments before it are
    owner/repo, the one after is the issue number.
    """
    parsed = urlparse(issue_url)
    parts = [p for p in parsed.path.split("/") if p]
    try:
        idx = parts.index("issues")
    except ValueError as exc:
        raise GitCodeClientError(
            f"Error: not an issue URL (no '/issues/<number>' segment): {issue_url}"
        ) from exc
    if idx < 2 or idx + 1 >= len(parts):
        raise GitCodeClientError(
            f"Error: cannot parse owner/repo/number from issue URL: {issue_url}"
        )
    owner = parts[idx - 2]
    repo = parts[idx - 1]
    if repo.endswith(".git"):
        repo = repo[:-4]
    return owner, repo, parts[idx + 1]


def resolve_api_base(cli_api_base=None, repo_url=None):
    """Resolve the API v5 base URL.

    Order: --api-base flag > GITCODE_API_BASE env > derived from repo URL.
    For gitcode.com / www.gitcode.com the API host is api.gitcode.com.
    For self-hosted, assume /api/v5 on the same host.
    """
    if cli_api_base:
        return cli_api_base.rstrip("/")
    env_base = os.environ.get("GITCODE_API_BASE")
    if env_base:
        return env_base.rstrip("/")
    if repo_url:
        parsed = urlparse(repo_url)
        netloc = parsed.netloc.lower()
        if netloc in ("gitcode.com", "www.gitcode.com"):
            return DEFAULT_GITCODE_API_BASE
        return f"{parsed.scheme}://{netloc}/api/v5"
    return DEFAULT_GITCODE_API_BASE


# --------------------------------------------------------------------------- #
# Session
# --------------------------------------------------------------------------- #
def make_session(
    user_agent=DEFAULT_USER_AGENT,
    *,
    rate_limiter=None,
    rate_limit_dir=None,
):
    """Create a requests.Session with headers and an optional shared limiter."""
    s = requests.Session()
    s.trust_env = True
    s.headers.update({"User-Agent": user_agent})
    if rate_limiter is None and rate_limit_dir:
        rate_limiter = SharedRateLimiter(rate_limit_dir)
    s.gitcode_rate_limiter = rate_limiter
    return s


def rate_limit_metrics(session):
    """Return limiter counters for a session without exposing bucket identity."""
    limiter = getattr(session, "gitcode_rate_limiter", None)
    if limiter is None:
        return {
            "http_attempts": 0,
            "limiter_waits": 0,
            "limiter_wait_seconds": 0.0,
            "rate_limit_429s": 0,
        }
    return limiter.snapshot()


def session_rate_limiter(session):
    """Return the limiter so related sessions can share in-process metrics."""
    return getattr(session, "gitcode_rate_limiter", None)


# --------------------------------------------------------------------------- #
# Token redaction (for safe logging)
# --------------------------------------------------------------------------- #
def redact_token(url):
    """Strip access_token=... from a URL for safe logging."""
    if not url:
        return url
    if "access_token=" in url:
        head, _, tail = url.partition("access_token=")
        token_val = tail.split("&", 1)[0]
        rest = tail[len(token_val):]
        url = f"{head}access_token=***{rest}"
    return url


def safe_error_text(resp):
    """Build an error string from a Response without leaking the token."""
    body_preview = str(getattr(resp, "text", "") or "")[:200]
    redacted = redact_token(resp.url or "")
    msg = f"HTTP {resp.status_code} for {resp.request.method} {redacted}"
    if body_preview:
        msg += f"\n  body: {body_preview}"
    return msg


# --------------------------------------------------------------------------- #
# HTTP GET (with retry)
# --------------------------------------------------------------------------- #
def _rate_limit_retry_delay(resp, fallback=61, max_wait=300):
    """Return a bounded wait using Retry-After or rate-limit reset headers."""
    headers = resp.headers or {}
    retry_after = headers.get("Retry-After")
    if retry_after:
        try:
            delay = float(retry_after)
        except (TypeError, ValueError):
            try:
                reset_at = parsedate_to_datetime(retry_after).timestamp()
                delay = reset_at - time.time()
            except (TypeError, ValueError, OverflowError):
                delay = None
        if delay is not None:
            return min(max(1, delay + 1), max_wait)

    reset_value = (
        headers.get("X-RateLimit-Reset")
        or headers.get("RateLimit-Reset")
        or headers.get("X-Rate-Limit-Reset")
    )
    if reset_value:
        try:
            reset_number = float(reset_value)
            if reset_number > 10_000_000_000:
                reset_number /= 1000
            delay = reset_number - time.time()
            if delay > 0:
                return min(max(1, delay + 1), max_wait)
        except (TypeError, ValueError, OverflowError):
            pass
    return min(max(1, fallback), max_wait)


def _retry_delay(exc, attempt):
    response = getattr(exc, "response", None)
    status = response.status_code if response is not None else 0
    if status == 429:
        delay = _rate_limit_retry_delay(response)
        LOGGER.warning(
            "GitCode API rate limit reached; retrying in %.0fs",
            delay,
        )
        return delay
    return 2 * (attempt + 1)


def _pause_before_retry(limiter, delay):
    if limiter is None:
        time.sleep(delay)
    else:
        limiter.pause(delay)


def _retry_transport_failure(attempt, limiter):
    if attempt >= 2:
        return False
    _pause_before_retry(limiter, 2 * (attempt + 1))
    return True


def _retry_http_failure(error, attempt, limiter, url, token):
    status = error.response.status_code
    if status == 429:
        delay = _retry_delay(error, attempt)
        if limiter is not None:
            limiter.set_cooldown(url, token, delay)
        if attempt >= 2:
            return False
        if limiter is None:
            time.sleep(delay)
        return True
    if status >= 500 and attempt < 2:
        _pause_before_retry(limiter, _retry_delay(error, attempt))
        return True
    return False


def _request_with_retry(
    request,
    *,
    return_client_errors=False,
    limiter=None,
    url="",
    token=None,
):
    """Execute one HTTP request with the shared bounded retry policy."""
    for attempt in range(3):
        if limiter is not None:
            limiter.acquire(url, token)
        try:
            response = request()
        except (requests.ConnectionError, requests.Timeout):
            if not _retry_transport_failure(attempt, limiter):
                raise
            continue

        status = response.status_code
        if status < 400:
            return response
        if return_client_errors and status != 429 and status < 500:
            return response

        error = requests.HTTPError(safe_error_text(response), response=response)
        if _retry_http_failure(error, attempt, limiter, url, token):
            continue
        raise error

    raise RuntimeError("unreachable retry state")


def api_get(session, url, token=None, *, params=None, timeout=30):
    """GET with optional token as query param + retry.

    - token=None or empty: no access_token param added (public API access).
    - 3 attempts. 429 honors Retry-After / rate-limit reset headers; 5xx uses
      linear backoff (2s, 4s).
    - Non-retryable 4xx raised immediately as HTTPError.
    - Returns parsed JSON (dict / list), or {"_status_code", "_text"} if
      response body is not JSON.
    """
    merged = dict(params or {})
    if token:
        merged.setdefault("access_token", token)
    effective_token = merged.get("access_token") or token
    limiter = getattr(session, "gitcode_rate_limiter", None)
    response = _request_with_retry(
        lambda: session.get(url, params=merged, timeout=timeout),
        limiter=limiter,
        url=url,
        token=effective_token,
    )
    try:
        return response.json()
    except ValueError:
        return {"_status_code": response.status_code, "_text": response.text}


# --------------------------------------------------------------------------- #
# HTTP POST (with retry)
# --------------------------------------------------------------------------- #
def api_post(session, url, token, *, data=None, timeout=30):
    """POST with token in form-data + retry.

    - 3 attempts. 429 honors server reset headers; 5xx uses linear backoff.
    - Non-retryable 4xx: returns the Response object (caller inspects body).
    - Success (2xx): returns the Response object.

    Caller checks ``resp.status_code < 400`` for success, or inspects
    ``resp.status_code`` / ``resp.text`` for error details.
    """
    merged = dict(data or {})
    merged.setdefault("access_token", token)
    effective_token = merged.get("access_token") or token
    limiter = getattr(session, "gitcode_rate_limiter", None)
    return _request_with_retry(
        lambda: session.post(url, data=merged, timeout=timeout),
        return_client_errors=True,
        limiter=limiter,
        url=url,
        token=effective_token,
    )


# --------------------------------------------------------------------------- #
# HTTP PUT (with retry)
# --------------------------------------------------------------------------- #
def api_put(session, url, token, *, json_data=None, timeout=30):
    """PUT JSON with the token in the query string and bounded retry.

    GitCode's custom Issue status-flow endpoint uses PUT with a JSON body.
    Retry and client-error semantics match :func:`api_patch`.
    """
    params = {"access_token": token}
    payload = dict(json_data or {})
    limiter = getattr(session, "gitcode_rate_limiter", None)
    return _request_with_retry(
        lambda: session.put(url, params=params, json=payload, timeout=timeout),
        return_client_errors=True,
        limiter=limiter,
        url=url,
        token=token,
    )


# --------------------------------------------------------------------------- #
# HTTP PATCH (with retry)
# --------------------------------------------------------------------------- #
def api_patch(session, url, token, *, json_data=None, timeout=30):
    """PATCH JSON with the token in the query string and bounded retry.

    GitCode's Issue update endpoint expects a JSON document. Retry semantics
    match :func:`api_post`: transient failures are retried, while callers can
    inspect non-retryable 4xx responses without exposing the token.
    """
    params = {"access_token": token}
    payload = dict(json_data or {})
    limiter = getattr(session, "gitcode_rate_limiter", None)
    return _request_with_retry(
        lambda: session.patch(url, params=params, json=payload, timeout=timeout),
        return_client_errors=True,
        limiter=limiter,
        url=url,
        token=token,
    )


# --------------------------------------------------------------------------- #
# Time utilities
# --------------------------------------------------------------------------- #
def parse_iso(ts):
    """Parse ISO 8601 timestamp string to datetime. Returns None on failure.

    Handles trailing 'Z' (UTC) by converting to +00:00 offset.
    """
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return None
