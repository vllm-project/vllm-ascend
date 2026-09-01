# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

TOOLKIT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = TOOLKIT_ROOT / "scripts" / "gitcode_client.py"
SPEC = importlib.util.spec_from_file_location("gitcode_client_under_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
CLIENT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CLIENT)


class FakeSession:
    def __init__(self, response) -> None:
        self.response = response
        self.calls = []

    def patch(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return self.response

    def put(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return self.response

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return self.response


class FakeGetResponse:
    def __init__(self, status_code, payload=None, headers=None) -> None:
        self.status_code = status_code
        self.payload = payload
        self.headers = headers or {}
        self.text = ""
        self.url = "https://api.example.test/issues"
        self.request = SimpleNamespace(method="GET")

    def json(self):
        return self.payload


class FakeGetSession:
    def __init__(self, responses) -> None:
        self.responses = list(responses)
        self.calls = 0

    def get(self, *_args, **_kwargs):
        response = self.responses[self.calls]
        self.calls += 1
        return response


class FakeClock:
    def __init__(self, now=1000.0) -> None:
        self.now = now
        self.sleeps = []

    def time(self):
        return self.now

    def sleep(self, delay):
        self.sleeps.append(delay)
        self.now += delay


def limited_session(tmp_path, clock, responses):
    limiter = CLIENT.SharedRateLimiter(
        tmp_path,
        hooks=CLIENT.RateLimitHooks(clock.time, clock.sleep),
    )
    session = CLIENT.make_session(rate_limiter=limiter)
    fake = FakeGetSession(responses)
    session.get = fake.get
    return limiter, session, fake


class TestRateLimit:
    @staticmethod
    def test_get_waits_for_server_reset_window_then_continues() -> None:
        session = FakeGetSession(
            [
                FakeGetResponse(
                    429,
                    headers={"X-RateLimit-Reset": "1010"},
                ),
                FakeGetResponse(200, payload=[{"number": 1}]),
            ]
        )

        with patch.object(CLIENT.time, "time", return_value=1000):
            with patch.object(CLIENT.time, "sleep") as sleep:
                result = CLIENT.api_get(
                    session,
                    "https://api.example.test/issues",
                    "token",
                )

        assert result == [{"number": 1}]
        assert session.calls == 2
        sleep.assert_called_once_with(11)

    @staticmethod
    def test_rolling_window_is_shared_across_instances(tmp_path) -> None:
        clock = FakeClock()
        first = CLIENT.SharedRateLimiter(
            tmp_path,
            policy=CLIENT.RateLimitPolicy(2, 10, 1),
            hooks=CLIENT.RateLimitHooks(clock.time, clock.sleep),
        )
        second = CLIENT.SharedRateLimiter(
            tmp_path,
            policy=CLIENT.RateLimitPolicy(2, 10, 1),
            hooks=CLIENT.RateLimitHooks(clock.time, clock.sleep),
        )

        first.acquire("https://api.example.test/issues", "secret-token")
        second.acquire("https://api.example.test/issues", "secret-token")
        first.acquire("https://api.example.test/issues", "secret-token")

        assert clock.sleeps == [5, 5]
        assert first.snapshot()["http_attempts"] == 2
        assert second.snapshot()["http_attempts"] == 1

    @staticmethod
    def test_429_publishes_cooldown_and_counts_retry_attempt(tmp_path) -> None:
        clock = FakeClock()
        responses = [
            FakeGetResponse(429, headers={"Retry-After": "7"}),
            FakeGetResponse(200, payload={"ok": True}),
        ]
        limiter, session, fake = limited_session(tmp_path, clock, responses)

        result = CLIENT.api_get(session, "https://api.example.test/issues", "token")

        assert result == {"ok": True}
        assert fake.calls == 2
        assert clock.sleeps == [8]
        assert limiter.snapshot() == {
            "http_attempts": 2,
            "limiter_waits": 1,
            "limiter_wait_seconds": 8.0,
            "rate_limit_429s": 1,
        }

    @staticmethod
    def test_5xx_retry_attempts_are_counted_without_real_sleep(tmp_path) -> None:
        clock = FakeClock()
        responses = [
            FakeGetResponse(500),
            FakeGetResponse(503),
            FakeGetResponse(200, payload={"ok": True}),
        ]
        limiter, session, fake = limited_session(tmp_path, clock, responses)

        result = CLIENT.api_get(session, "https://api.example.test/issues", "token")

        assert result == {"ok": True}
        assert fake.calls == 3
        assert clock.sleeps == [2, 4]
        assert limiter.snapshot()["http_attempts"] == 3

    @staticmethod
    def test_corrupt_state_recovers_without_persisting_token(tmp_path) -> None:
        clock = FakeClock()
        limiter = CLIENT.SharedRateLimiter(
            tmp_path,
            hooks=CLIENT.RateLimitHooks(clock.time, clock.sleep),
        )
        url = "https://api.example.test/issues"
        limiter.acquire(url, "do-not-store-this-token")
        state_path = next(tmp_path.glob("*.json"))
        state_path.write_text("not json", encoding="utf-8")

        limiter.acquire(url, "do-not-store-this-token")

        payload = json.loads(state_path.read_text(encoding="utf-8"))
        assert payload["version"] == 1
        for path in tmp_path.iterdir():
            assert "do-not-store-this-token" not in path.name
            assert "do-not-store-this-token" not in path.read_text(encoding="utf-8")

    @staticmethod
    def test_get_4xx_still_raises_and_retries_are_counted(tmp_path) -> None:
        clock = FakeClock()
        limiter, session, _ = limited_session(
            tmp_path,
            clock,
            [FakeGetResponse(403)],
        )

        try:
            CLIENT.api_get(session, "https://api.example.test/issues", "token")
        except CLIENT.requests.HTTPError:
            pass
        else:
            raise AssertionError("GET 403 must raise HTTPError")

        assert limiter.snapshot()["http_attempts"] == 1


class TestPatch:
    @staticmethod
    def test_patch_sends_token_as_query_and_state_as_json() -> None:
        session = FakeSession(SimpleNamespace(status_code=200))

        response = CLIENT.api_patch(
            session,
            "https://api.example.test/issues/42",
            "secret-token",
            json_data={"state": "closed"},
        )

        assert response.status_code == 200
        _, kwargs = session.calls[0]
        assert kwargs["params"] == {"access_token": "secret-token"}
        assert kwargs["json"] == {"state": "closed"}

    @staticmethod
    def test_non_retryable_client_error_is_returned_to_caller() -> None:
        session = FakeSession(SimpleNamespace(status_code=403))

        response = CLIENT.api_patch(
            session,
            "https://api.example.test/issues/42",
            "secret-token",
            json_data={"state": "closed"},
        )

        assert response.status_code == 403
        assert len(session.calls) == 1


class TestPut:
    @staticmethod
    def test_put_sends_token_as_query_and_status_as_json() -> None:
        session = FakeSession(SimpleNamespace(status_code=200))

        response = CLIENT.api_put(
            session,
            "https://web-api.example.test/status-flow/42",
            "secret-token",
            json_data={"status_before": "进行中", "status_current": "挂起"},
        )

        assert response.status_code == 200
        _, kwargs = session.calls[0]
        assert kwargs["params"] == {"access_token": "secret-token"}
        assert kwargs["json"] == {
            "status_before": "进行中",
            "status_current": "挂起",
        }

    @staticmethod
    def test_put_returns_non_retryable_client_error() -> None:
        session = FakeSession(SimpleNamespace(status_code=409))

        response = CLIENT.api_put(
            session,
            "https://web-api.example.test/status-flow/42",
            "secret-token",
            json_data={"status_current": "挂起"},
        )

        assert response.status_code == 409
        assert len(session.calls) == 1


class TestPost:
    @staticmethod
    def test_post_keeps_non_retryable_client_error_contract() -> None:
        session = FakeSession(SimpleNamespace(status_code=422))

        response = CLIENT.api_post(
            session,
            "https://api.example.test/issues/42/comments",
            "secret-token",
            data={"body": "text"},
        )

        assert response.status_code == 422
        assert session.calls[0][1]["data"] == {
            "body": "text",
            "access_token": "secret-token",
        }
