"""System-level tests for load_balance_proxy_layerwise_server_example.

The layerwise proxy shares the non-streaming buffering logic with the main
proxy but is a separate script with its own startup wiring, so it gets its own
test module. The mock backend and the HTTP helpers are reused from
``test_load_balance_proxy_server``; this module only owns the layerwise
process and its ports.
"""

import contextlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest

from tests.ut.proxy.test_load_balance_proxy_server import (
    DONE,
    MODEL,
    _chat_req,
    _completion_body,
    _kill_port,
    _set_script,
    _wait_health,
    _worker_port,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
PROXY_SCRIPT = REPO_ROOT / "examples" / "disaggregated_prefill_v1" / "load_balance_proxy_layerwise_server_example.py"
MOCK_BACKEND_SCRIPT = Path(__file__).resolve().parent / "test_load_balance_proxy_server.py"

# Distinct base ports from the main proxy module so both modules can run in
# parallel xdist workers without colliding.
MOCK_PREFILL_PORT = _worker_port(19101)
MOCK_DECODE_PORT = _worker_port(19102)
PROXY_PORT = _worker_port(19181)


@pytest.fixture(scope="module")
def layerwise_proxy_server():
    """Start mock backends + the layerwise proxy, yield proxy URL, teardown."""
    for port in (MOCK_PREFILL_PORT, MOCK_DECODE_PORT, PROXY_PORT):
        _kill_port(port)
    time.sleep(1)

    procs = []
    try:
        for port in (MOCK_PREFILL_PORT, MOCK_DECODE_PORT):
            procs.append(
                subprocess.Popen(
                    [sys.executable, str(MOCK_BACKEND_SCRIPT), "--mock-backend", str(port)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                )
            )
        assert _wait_health(f"http://127.0.0.1:{MOCK_PREFILL_PORT}/health"), "mock prefill failed to start"
        assert _wait_health(f"http://127.0.0.1:{MOCK_DECODE_PORT}/health"), "mock decode failed to start"

        procs.append(
            subprocess.Popen(
                [
                    sys.executable,
                    str(PROXY_SCRIPT),
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(PROXY_PORT),
                    "--prefiller-hosts",
                    "127.0.0.1",
                    "--prefiller-ports",
                    str(MOCK_PREFILL_PORT),
                    "--decoder-hosts",
                    "127.0.0.1",
                    "--decoder-ports",
                    str(MOCK_DECODE_PORT),
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        )
        assert _wait_health(f"http://127.0.0.1:{PROXY_PORT}/healthcheck"), "layerwise proxy failed to start"

        yield f"http://127.0.0.1:{PROXY_PORT}"
    finally:
        for p in procs:
            with contextlib.suppress(Exception):
                os.killpg(os.getpgid(p.pid), signal.SIGTERM)
            with contextlib.suppress(Exception):
                p.kill()
        time.sleep(1)


@pytest.mark.parametrize(
    ("chunks", "expected_status"),
    [
        pytest.param([b'{"choices":'], 502, id="partial-json"),
        pytest.param([b""], 502, id="empty-body"),
        pytest.param([b" \n\t"], 502, id="whitespace-body"),
        pytest.param([DONE], 502, id="done-only"),
        pytest.param(
            [b"data: " + json.dumps(_completion_body("hi")).encode() + b"\n\n" + DONE],
            200,
            id="document-plus-done",
        ),
    ],
)
def test_nonstream_truncated_body_returns_error(
    layerwise_proxy_server: str,
    chunks: list[bytes],
    expected_status: int,
) -> None:
    """Non-streaming decode that ends without a document returns 502, not 200.

    Same matrix as the main proxy: a decoder that closes a 200 response without
    a parseable body (partial JSON, empty, whitespace-only, or a lone ``[DONE]``
    sentinel) must surface an upstream error, while a valid document followed by
    ``[DONE]`` still succeeds.
    """
    _set_script(
        MOCK_DECODE_PORT,
        [{"kind": "stream", "status": 200, "chunks": chunks}],
    )
    with httpx.Client(timeout=30) as c:
        r = c.post(
            f"{layerwise_proxy_server}/v1/chat/completions",
            json=_chat_req(stream=False),
            headers={"Content-Type": "application/json"},
        )
    assert r.status_code == expected_status
    body = r.json()
    if expected_status == 502:
        assert body["error"]["type"] == "upstream_error"
        assert body["error"]["code"] == "decode_backend_incomplete"
    else:
        assert body["choices"][0]["message"]["content"] == "hi"


def test_nonstream_recompute_in_unterminated_tail_retries(layerwise_proxy_server: str) -> None:
    """A recompute signal in the unterminated trailing frame still retries.

    A backend may close right after its last frame without sending the
    terminating blank line, so that frame is only decoded by ``flush()`` at EOF.
    When it carries ``stop_reason="recomputed"`` the layerwise proxy must retry
    instead of returning the partial answer.
    """
    tail = json.dumps(_completion_body("gen", stop_reason="recomputed")).encode("utf-8")
    _set_script(
        MOCK_DECODE_PORT,
        [
            {"kind": "stream", "status": 200, "chunks": [tail]},
            {"kind": "json", "status": 200, "body": _completion_body("ok")},
        ],
    )
    with httpx.Client(timeout=30) as c:
        r = c.post(
            f"{layerwise_proxy_server}/v1/chat/completions",
            json=_chat_req(stream=False),
            headers={"Content-Type": "application/json"},
        )
    assert r.status_code == 200
    body = r.json()
    # The retry ran and its content is appended to the aborted attempt.
    assert body["choices"][0]["message"]["content"] == "genok"


def test_nonstream_unterminated_usage_frame_merges(layerwise_proxy_server: str) -> None:
    """An unterminated usage frame at EOF is merged into the final document.

    A backend may emit a separate usage frame after the completion frame and
    close without the terminating blank line, so that frame is only decoded by
    ``flush()``. Its token counts must supersede the ones embedded in the
    completion frame; previously the flush branch was skipped once a document
    had been received and the stale counts (completion_tokens=1) were returned.
    """
    document = json.dumps(_completion_body("hi")).encode("utf-8") + b"\n\n"
    usage = json.dumps({"usage": {"prompt_tokens": 5, "completion_tokens": 9}}).encode("utf-8")
    _set_script(
        MOCK_DECODE_PORT,
        [{"kind": "stream", "status": 200, "chunks": [document, usage]}],
    )
    with httpx.Client(timeout=30) as c:
        r = c.post(
            f"{layerwise_proxy_server}/v1/chat/completions",
            json=_chat_req(stream=False),
            headers={"Content-Type": "application/json"},
        )
    assert r.status_code == 200
    body = r.json()
    assert body["choices"][0]["message"]["content"] == "hi"
    assert body["usage"]["completion_tokens"] == 9


def test_nonstream_unterminated_usage_after_metadata_frame(layerwise_proxy_server: str) -> None:
    """A choices-less metadata frame followed by an unterminated usage frame.

    The metadata frame populates ``final_document`` before EOF, so the flush
    merge branch takes the "merge into existing document" path. Previously
    ``trailing_usage`` was only assigned when the trailing frame carried
    choices, and the merge branch read it unconditionally — the resulting
    UnboundLocalError was swallowed and the client got an empty 200.
    """
    metadata = json.dumps({"id": "x", "model": MODEL, "created": 1}).encode("utf-8") + b"\n\n"
    document = json.dumps(_completion_body("hi")).encode("utf-8") + b"\n\n"
    usage = json.dumps({"usage": {"prompt_tokens": 5, "completion_tokens": 9}}).encode("utf-8")
    _set_script(
        MOCK_DECODE_PORT,
        [{"kind": "stream", "status": 200, "chunks": [metadata, document, usage]}],
    )
    with httpx.Client(timeout=30) as c:
        r = c.post(
            f"{layerwise_proxy_server}/v1/chat/completions",
            json=_chat_req(stream=False),
            headers={"Content-Type": "application/json"},
        )
    assert r.status_code == 200
    body = r.json()
    assert body["choices"][0]["message"]["content"] == "hi"
    assert body["usage"]["completion_tokens"] == 9
