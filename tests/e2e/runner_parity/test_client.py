# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import io
import json

import pytest

from tests.e2e.runner_parity.client import stream_completion


def stream_bytes(events):
    return io.BytesIO(
        b"".join(b"data: " + json.dumps(event).encode() + b"\n\n" for event in events) + b"data: [DONE]\n"
    )


def test_special_tokens_have_latency_even_without_visible_text(monkeypatch):
    events = [
        {"choices": [{"text": "", "token_ids": [7]}]},
        {"choices": [{"text": "", "token_ids": [8]}]},
        {"choices": [], "usage": {"prompt_tokens": 1, "completion_tokens": 2}},
    ]
    ticks = iter((0.0, 0.1, 0.2, 0.3, 0.4))
    monkeypatch.setattr("tests.e2e.runner_parity.client.time.perf_counter", lambda: next(ticks))
    monkeypatch.setattr(
        "tests.e2e.runner_parity.client.urllib.request.urlopen", lambda *args, **kwargs: stream_bytes(events)
    )
    result = stream_completion("http://local", {"max_tokens": 2})
    assert result["generated_token_ids"] == [7, 8]
    assert result["ttft_s"] == pytest.approx(0.1)
    assert result["tpot_s"] == pytest.approx(0.1)
    assert result["first_visible_text_s"] is None


def test_incomplete_stream_is_saved_and_rejected(monkeypatch, tmp_path):
    events = [{"choices": [{"text": "", "token_ids": [7]}]}]
    monkeypatch.setattr(
        "tests.e2e.runner_parity.client.urllib.request.urlopen", lambda *args, **kwargs: stream_bytes(events)
    )
    with pytest.raises(AssertionError, match="stream/usage"):
        stream_completion("http://local", {"max_tokens": 2}, error_dir=tmp_path)
    files = list(tmp_path.glob("stream-error-*.json"))
    assert len(files) == 1
    saved = json.loads(files[0].read_text())
    assert saved["generated_token_ids"] == [7] and saved["events"][0]["data"] == events[0]
