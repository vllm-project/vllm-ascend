"""System-level tests for load_balance_proxy_server_example.

Tests the proxy's HTTP behavior (status codes + response bodies) using mock
backends and real subprocesses. Does NOT import or call internal proxy functions,
so it survives refactoring of the proxy internals.
"""

import base64
import contextlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import httpx
import pytest

# --------------------------------------------------------------------------- #
# Paths and constants
# --------------------------------------------------------------------------- #
REPO_ROOT = Path(__file__).resolve().parents[3]
PROXY_SCRIPT = REPO_ROOT / "examples" / "disaggregated_prefill_v1" / "load_balance_proxy_server_example.py"


def _worker_port(port: int) -> int:
    """Offset fixed ports per xdist worker so parallel workers never share ports.

    pytest-xdist sets PYTEST_XDIST_WORKER (e.g. "gw3") in worker processes;
    without it (serial runs) the base ports are used unchanged.
    """
    worker = os.environ.get("PYTEST_XDIST_WORKER", "")
    if worker.startswith("gw"):
        try:
            return port + int(worker[2:]) * 100
        except ValueError:
            pass
    return port


# The 100-port step exceeds the largest gap between any two of the fixed
# ports below, so no two logical ports of different workers can collide.
MOCK_PREFILL_PORT = _worker_port(19001)
MOCK_DECODE_PORT = _worker_port(19002)
PROXY_PORT = _worker_port(19080)
MODEL = "test-model"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _b64(b: bytes) -> dict:
    return {"__b64__": base64.b64encode(b).decode()}


def _encode_items(items: list[dict]) -> list[dict]:
    """Encode bytes fields for JSON transport."""
    out = []
    for it in items:
        it2 = dict(it)
        if isinstance(it2.get("body"), (bytes, bytearray)):
            it2["body"] = _b64(it2["body"])
        if "chunks" in it2:
            it2["chunks"] = [_b64(c) if isinstance(c, (bytes, bytearray)) else c for c in it2["chunks"]]
        out.append(it2)
    return out


def _sse(content: str = "hi", stop_reason: str | None = None) -> bytes:
    obj = {
        "id": "x",
        "choices": [
            {"index": 0, "delta": {"content": content}, "finish_reason": None, "stop_reason": stop_reason},
        ],
    }
    return b"data: " + json.dumps(obj).encode() + b"\n\n"


DONE = b"data: [DONE]\n\n"


def _completion_body(content: str = "hi", stop_reason: str | None = None) -> dict:
    """A realistic non-streaming completion JSON body (decode backend output)."""
    return {
        "id": "x",
        "choices": [
            {
                "index": 0,
                "message": {"content": content},
                "finish_reason": "stop",
                "stop_reason": stop_reason,
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
    }


def _set_script(port: int, items: list[dict]) -> None:
    httpx.post(
        f"http://127.0.0.1:{port}/__set_script__",
        json={"items": _encode_items(items)},
        timeout=5,
    )


def _wait_health(url: str, timeout: int = 30) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if httpx.get(url, timeout=2).status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(0.3)
    return False


def _kill_port(port: int) -> None:
    """Kill any process listening on *port* (uses netstat; lsof may not exist)."""
    try:
        out = subprocess.check_output(["netstat", "-tlnp"], stderr=subprocess.DEVNULL, text=True)
        for line in out.splitlines():
            if f":{port} " in line and "LISTEN" in line:
                pid = line.split()[-1].split("/")[0]
                with contextlib.suppress(Exception):
                    os.kill(int(pid), signal.SIGKILL)
    except Exception:
        pass


def _post(
    proxy_url: str,
    payload: dict,
    stream: bool = False,
    timeout: float = 30,
) -> tuple[int, bytes]:
    with httpx.Client(timeout=timeout) as c:
        if stream:
            with c.stream(
                "POST",
                f"{proxy_url}/v1/chat/completions",
                json=payload,
                headers={"Content-Type": "application/json"},
            ) as r:
                return r.status_code, b"".join(r.iter_bytes())
        r = c.post(
            f"{proxy_url}/v1/chat/completions",
            json=payload,
            headers={"Content-Type": "application/json"},
        )
        return r.status_code, r.content


def _chat_req(max_tokens: int = 20, stream: bool = False) -> dict:
    return {
        "model": MODEL,
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": stream,
    }


# --------------------------------------------------------------------------- #
# Mock backend ASGI app (self-contained, started as subprocess)
# --------------------------------------------------------------------------- #
def _create_mock_backend_app():
    """Create a FastAPI app that serves as a mock prefill/decode backend.

    Responses are controlled via a script queue set through POST /__set_script__.
    Each script item specifies kind (json/stream/abort), status, body, chunks,
    and optional mid-stream error.
    """
    import asyncio
    from dataclasses import dataclass, field

    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse, Response, StreamingResponse

    app = FastAPI()

    @dataclass
    class ScriptItem:
        kind: str = "json"
        status: int = 200
        body: Any = None
        chunks: list = field(default_factory=list)
        chunk_delay: float = 0.0
        mid_stream_error: str = ""
        error_after: int = 0

    class ScriptQueue:
        def __init__(self):
            self._items: list[ScriptItem] = []
            self._default = ScriptItem(kind="json", status=200, body={"kv_transfer_params": {"do_remote_decode": True}})

        def set(self, items: list):
            self._items = list(items)

        def pop(self) -> ScriptItem:
            if self._items:
                return self._items.pop(0)
            return self._default

    queue = ScriptQueue()

    def _dec_val(v):
        if isinstance(v, dict) and set(v.keys()) == {"__b64__"}:
            return base64.b64decode(v["__b64__"])
        if isinstance(v, list):
            return [_dec_val(x) for x in v]
        if isinstance(v, dict):
            return {k: _dec_val(x) for k, x in v.items()}
        return v

    @app.post("/__set_script__")
    async def set_script(req: Request):
        data = await req.json()
        items = [_dec_val(it) for it in data.get("items", [])]
        queue.set([ScriptItem(**it) if isinstance(it, dict) else it for it in items])
        return {"ok": True}

    async def _gen_stream(item: ScriptItem):
        emitted = 0
        for ck in item.chunks:
            if item.mid_stream_error and emitted >= item.error_after:
                raise ConnectionError(item.mid_stream_error)
            yield ck
            emitted += 1
            if item.chunk_delay:
                await asyncio.sleep(item.chunk_delay)
        if item.mid_stream_error and emitted >= item.error_after:
            raise ConnectionError(item.mid_stream_error)

    @app.post("/v1/chat/completions")
    @app.post("/v1/completions")
    async def handle(req: Request):
        item = queue.pop()
        if item.kind == "json":
            if isinstance(item.body, dict):
                return JSONResponse(status_code=item.status, content=item.body)
            return Response(content=item.body or b"", status_code=item.status)
        if item.kind == "stream":
            return StreamingResponse(_gen_stream(item), status_code=item.status, media_type="text/event-stream")
        if item.kind == "abort":
            return Response(content=b"", status_code=item.status)
        return JSONResponse(status_code=500, content={"error": "bad script"})

    @app.get("/health")
    async def health():
        return {"ok": True}

    @app.get("/v1/models")
    async def models():
        return {"data": [{"id": "test-model"}]}

    return app


if __name__ == "__main__" and "--mock-backend" in sys.argv:
    port = int(sys.argv[sys.argv.index("--mock-backend") + 1])
    import uvicorn

    uvicorn.run(_create_mock_backend_app(), host="127.0.0.1", port=port, log_level="warning")


# --------------------------------------------------------------------------- #
# Pytest fixture: start mock backends + proxy
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def proxy_server():
    """Start mock backends + proxy as subprocesses, yield proxy URL, teardown."""
    test_file = str(Path(__file__).resolve())
    procs = []

    # Kill any leftover processes on test ports
    for port in (MOCK_PREFILL_PORT, MOCK_DECODE_PORT, PROXY_PORT):
        _kill_port(port)
    time.sleep(1)

    try:
        # Start mock backends (each in its own process group for clean teardown)
        procs.append(
            subprocess.Popen(
                [sys.executable, test_file, "--mock-backend", str(MOCK_PREFILL_PORT)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        )
        procs.append(
            subprocess.Popen(
                [sys.executable, test_file, "--mock-backend", str(MOCK_DECODE_PORT)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        )
        assert _wait_health(f"http://127.0.0.1:{MOCK_PREFILL_PORT}/health"), "mock prefill failed to start"
        assert _wait_health(f"http://127.0.0.1:{MOCK_DECODE_PORT}/health"), "mock decode failed to start"

        # Start proxy
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
        assert _wait_health(f"http://127.0.0.1:{PROXY_PORT}/healthcheck"), "proxy failed to start"

        yield f"http://127.0.0.1:{PROXY_PORT}"
    finally:
        # Teardown — must run even when a startup assertion fails, otherwise
        # the subprocesses leak and cause port conflicts in later tests.
        for p in procs:
            with contextlib.suppress(Exception):
                os.killpg(os.getpgid(p.pid), signal.SIGTERM)
            with contextlib.suppress(Exception):
                p.kill()
        time.sleep(1)


# --------------------------------------------------------------------------- #
# Tests — HTTP behavior only (no internal function calls)
# --------------------------------------------------------------------------- #
def test_normal_nonstream(proxy_server):
    """Normal non-streaming request returns 200 + content."""
    _set_script(MOCK_DECODE_PORT, [{"kind": "stream", "status": 200, "chunks": [_sse("hello"), DONE]}])
    st, body = _post(proxy_server, _chat_req(stream=False))
    assert st == 200
    assert b"hello" in body


def test_normal_stream(proxy_server):
    """Normal streaming request returns 200 + SSE + [DONE]."""
    _set_script(MOCK_DECODE_PORT, [{"kind": "stream", "status": 200, "chunks": [_sse("hello"), DONE]}])
    st, body = _post(proxy_server, _chat_req(stream=True), stream=True)
    assert st == 200
    assert b"data: " in body
    assert b"[DONE]" in body


def test_overlimit_nonstream(proxy_server):
    """Decode 4xx (over-limit) returns real 400 + error body."""
    _set_script(
        MOCK_DECODE_PORT,
        [
            {
                "kind": "json",
                "status": 400,
                "body": {"error": {"message": "max_tokens too long", "type": "BadRequestError"}},
            },
        ],
    )
    st, body = _post(proxy_server, _chat_req(max_tokens=99999, stream=False))
    assert st == 400
    assert b"max_tokens" in body


def test_overlimit_stream(proxy_server):
    """Decode 4xx in streaming mode returns real 400 + error body."""
    _set_script(
        MOCK_DECODE_PORT,
        [
            {
                "kind": "json",
                "status": 400,
                "body": {"error": {"message": "max_tokens too long", "type": "BadRequestError"}},
            },
        ],
    )
    st, body = _post(proxy_server, _chat_req(max_tokens=99999, stream=True), stream=True)
    assert st == 400
    assert b"max_tokens" in body


def test_decode_5xx(proxy_server):
    """Decode 5xx (retry exhausted) returns real 500."""
    _set_script(MOCK_DECODE_PORT, [{"kind": "json", "status": 500, "body": {"e": 1}}] * 3)
    st, body = _post(proxy_server, _chat_req(stream=False))
    assert st == 500


def test_decode_connection_failure(proxy_server):
    """Decode connection failure returns 502."""
    dead_port = _worker_port(19081)
    _kill_port(dead_port)
    time.sleep(0.5)
    dead_proc = subprocess.Popen(
        [
            sys.executable,
            str(PROXY_SCRIPT),
            "--host",
            "127.0.0.1",
            "--port",
            str(dead_port),
            "--prefiller-hosts",
            "127.0.0.1",
            "--prefiller-ports",
            str(MOCK_PREFILL_PORT),
            "--decoder-hosts",
            "127.0.0.1",
            "--decoder-ports",
            str(_worker_port(19999)),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    try:
        # The health wait must also be inside `try` so a failed startup
        # assertion still reaches the cleanup below.
        assert _wait_health(f"http://127.0.0.1:{dead_port}/healthcheck", timeout=15)
        with httpx.Client(timeout=15) as c:
            r = c.post(
                f"http://127.0.0.1:{dead_port}/v1/chat/completions",
                json=_chat_req(stream=False),
                headers={"Content-Type": "application/json"},
            )
            assert r.status_code == 502
            assert b"decode_backend_unavailable" in r.content or b"proxy_error" in r.content
    finally:
        with contextlib.suppress(Exception):
            os.killpg(os.getpgid(dead_proc.pid), signal.SIGTERM)
        with contextlib.suppress(Exception):
            dead_proc.kill()
        _kill_port(dead_port)


def test_completions_prompt_endpoint(proxy_server):
    """/v1/completions (prompt) endpoint works."""
    _set_script(MOCK_DECODE_PORT, [{"kind": "stream", "status": 200, "chunks": [_sse("hello"), DONE]}])
    with httpx.Client(timeout=30) as c:
        r = c.post(
            f"{proxy_server}/v1/completions",
            json={"model": MODEL, "prompt": "hello", "max_tokens": 5, "stream": False},
            headers={"Content-Type": "application/json"},
        )
        assert r.status_code == 200


def test_v1_models(proxy_server):
    """/v1/models returns 200."""
    with httpx.Client(timeout=10) as c:
        r = c.get(f"{proxy_server}/v1/models")
        assert r.status_code == 200


def test_healthcheck(proxy_server):
    """/healthcheck returns 200."""
    with httpx.Client(timeout=10) as c:
        r = c.get(f"{proxy_server}/healthcheck")
        assert r.status_code == 200


def test_recompute(proxy_server):
    """Recompute (stop_reason=recomputed) retries and returns normal content."""
    _set_script(
        MOCK_DECODE_PORT,
        [
            {
                "kind": "stream",
                "status": 200,
                "chunks": [_sse("gen", stop_reason="recomputed")],
            },
            {"kind": "stream", "status": 200, "chunks": [_sse("done"), DONE]},
        ],
    )
    st, body = _post(proxy_server, _chat_req(stream=True), stream=True)
    assert st == 200
    assert b"done" in body
    assert b"recomputed" not in body


def test_mid_stream_error(proxy_server):
    """Mid-stream error: proxy does not crash, partial content forwarded."""
    _set_script(
        MOCK_DECODE_PORT,
        [
            {
                "kind": "stream",
                "status": 200,
                "chunks": [_sse("par")],
                "mid_stream_error": "aborted",
                "error_after": 1,
            },
        ],
    )
    st, body = _post(proxy_server, _chat_req(stream=True), stream=True, timeout=15)
    # Proxy should not crash; partial content may be forwarded
    assert st == 200 or b"par" in body


def test_concurrent_requests(proxy_server):
    """Multiple concurrent requests are handled without errors."""
    _set_script(
        MOCK_DECODE_PORT,
        [
            {"kind": "stream", "status": 200, "chunks": [_sse("r1"), DONE]},
            {"kind": "stream", "status": 200, "chunks": [_sse("r2"), DONE]},
            {"kind": "stream", "status": 200, "chunks": [_sse("r3"), DONE]},
        ],
    )
    import concurrent.futures

    def send():
        return _post(proxy_server, _chat_req(stream=False))

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
        results = list(ex.map(lambda _: send(), range(3)))
    for st, body in results:
        assert st == 200


def test_instances_add_remove(proxy_server):
    """Add and remove a decode instance via /instances/add and /instances/remove."""
    test_file = str(Path(__file__).resolve())
    temp_port = _worker_port(19003)
    _kill_port(temp_port)
    temp_proc = subprocess.Popen(
        [sys.executable, test_file, "--mock-backend", str(temp_port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    assert _wait_health(f"http://127.0.0.1:{temp_port}/health", timeout=15)
    try:
        # Add instance (goes into "waiting" state, activated by NodeListener)
        r = httpx.post(
            f"{proxy_server}/instances/add",
            json={"type": "decode", "instances": [f"127.0.0.1:{temp_port}"]},
            headers={"Content-Type": "application/json"},
            timeout=10,
        )
        assert r.status_code == 200
        assert f"{temp_port}".encode() in r.content
        # Remove instance (isolates then removes)
        r = httpx.post(
            f"{proxy_server}/instances/remove",
            json={"type": "decode", "instances": f"127.0.0.1:{temp_port}"},
            headers={"Content-Type": "application/json"},
            timeout=10,
        )
        assert r.status_code == 200
    finally:
        with contextlib.suppress(Exception):
            os.killpg(os.getpgid(temp_proc.pid), signal.SIGTERM)
        with contextlib.suppress(Exception):
            temp_proc.kill()
        _kill_port(temp_port)


def test_usage_cached_tokens(proxy_server):
    """Proxy reports prefiller cached_tokens in the decode response usage field."""
    usage_chunk = (
        b'data: {"id":"x","choices":[],"usage":{"prompt_tokens":10,'
        b'"completion_tokens":2,"prompt_tokens_details":{}}}\n\n'
    )
    _set_script(
        MOCK_DECODE_PORT,
        [{"kind": "stream", "status": 200, "chunks": [_sse("hi"), usage_chunk, DONE]}],
    )
    st, body = _post(proxy_server, _chat_req(stream=True), stream=True)
    assert st == 200
    # The proxy should have added cached_tokens to the usage chunk
    assert b"cached_tokens" in body


def test_stream_frame_split_across_chunks(proxy_server):
    """A SSE frame split across two network chunks is still parsed.

    The decode backend's JSON can be split across TCP segments. Parsing each
    network chunk on its own drops the frame, so the proxy misses the
    stop_reason it needs to drive its retry logic.
    """
    frame = _sse("split-token")
    cut = len(frame) // 2
    _set_script(
        MOCK_DECODE_PORT,
        [{"kind": "stream", "status": 200, "chunks": [frame[:cut], frame[cut:], DONE]}],
    )
    st, body = _post(proxy_server, _chat_req(stream=True), stream=True)
    assert st == 200
    assert b"split-token" in body


def test_stream_frames_coalesced_in_one_chunk(proxy_server):
    """Two SSE frames delivered in one network chunk are both parsed.

    The network can coalesce a frame and the [DONE] sentinel into a single
    read. Parsing the whole chunk at once fails, so the proxy loses the frame
    it needs to drive its retry logic.
    """
    frame = _sse("coalesced", stop_reason="recomputed")
    _set_script(
        MOCK_DECODE_PORT,
        [
            {"kind": "stream", "status": 200, "chunks": [frame + DONE]},
            {"kind": "stream", "status": 200, "chunks": [_sse("done"), DONE]},
        ],
    )
    st, body = _post(proxy_server, _chat_req(stream=True), stream=True)
    assert st == 200
    # The recompute signal survived the coalesced read, so the proxy retried.
    assert b"done" in body
    assert b"recomputed" not in body


def test_nonstream_frames_coalesced_in_one_chunk(proxy_server):
    """A non-streaming reply coalesced with [DONE] still returns the content."""
    _set_script(
        MOCK_DECODE_PORT,
        [{"kind": "stream", "status": 200, "chunks": [_sse("coalesced-json") + DONE]}],
    )
    st, body = _post(proxy_server, _chat_req(stream=False))
    assert st == 200
    assert b"coalesced-json" in body


def test_stream_byte_at_a_time(proxy_server):
    """A SSE frame delivered one byte per chunk is reassembled correctly."""
    frame = _sse("dripped")
    _set_script(
        MOCK_DECODE_PORT,
        [{"kind": "stream", "status": 200, "chunks": [bytes([b]) for b in frame] + [DONE]}],
    )
    st, body = _post(proxy_server, _chat_req(stream=True), stream=True)
    assert st == 200
    assert b"dripped" in body


def test_recompute_detected_when_frame_split_across_chunks(proxy_server):
    """A split frame still triggers the recompute retry.

    The recompute signal travels in the frame's stop_reason. When the frame is
    split across network chunks, a per-chunk parse misses it, so the proxy
    never retries and the client receives the aborted generation.
    """
    frame = _sse("gen", stop_reason="recomputed")
    cut = len(frame) // 2
    _set_script(
        MOCK_DECODE_PORT,
        [
            {
                "kind": "stream",
                "status": 200,
                "chunks": [frame[:cut], frame[cut:]],
            },
            {"kind": "stream", "status": 200, "chunks": [_sse("done"), DONE]},
        ],
    )
    st, body = _post(proxy_server, _chat_req(stream=True), stream=True)
    assert st == 200
    # The retry happened, so the retried content is what the client sees.
    assert b"done" in body
    assert b"recomputed" not in body


# --------------------------------------------------------------------------- #
# Tests — non-streaming response shape (Content-Length, single JSON)
# --------------------------------------------------------------------------- #
def test_nonstream_response_has_content_length(proxy_server):
    """Non-streaming returns a complete Response with Content-Length (not chunked).

    Regression test: previously non-streaming used StreamingResponse, sending the
    body with chunked transfer-encoding and no Content-Length. The fix collects all
    chunks and returns a single ``Response`` with ``application/json`` + Content-Length.
    """
    _set_script(MOCK_DECODE_PORT, [{"kind": "json", "status": 200, "body": _completion_body("hello")}])
    with httpx.Client(timeout=30) as c:
        r = c.post(
            f"{proxy_server}/v1/chat/completions",
            json=_chat_req(stream=False),
            headers={"Content-Type": "application/json"},
        )
    assert r.status_code == 200
    assert "content-length" in r.headers
    assert "chunked" not in r.headers.get("transfer-encoding", "").lower()
    assert r.headers["content-type"].startswith("application/json")
    body = r.json()
    assert "choices" in body
    assert body["choices"][0]["message"]["content"] == "hello"


def test_nonstream_assembles_fragmented_json(proxy_server):
    """Non-streaming reassembles a JSON body split across chunks before parsing.

    Regression test for the bug where each byte chunk was parsed individually: a
    partial JSON chunk raised ``json.JSONDecodeError`` and was dropped, skipping the
    recomputed/retry logic. The proxy now holds the bytes until a complete document
    is available (or decodes the tail at EOF).
    """
    full = json.dumps(_completion_body("hi")).encode("utf-8")
    half = len(full) // 2
    _set_script(
        MOCK_DECODE_PORT,
        [{"kind": "stream", "status": 200, "chunks": [full[:half], full[half:]], "chunk_delay": 0.05}],
    )
    with httpx.Client(timeout=30) as c:
        r = c.post(
            f"{proxy_server}/v1/chat/completions",
            json=_chat_req(stream=False),
            headers={"Content-Type": "application/json"},
        )
    assert r.status_code == 200
    assert "content-length" in r.headers
    body = r.json()
    assert "choices" in body
    assert body["choices"][0]["message"]["content"] == "hi"


def test_nonstream_4xx_has_content_length(proxy_server):
    """Non-streaming decode 4xx returns the real status with Content-Length.

    Validates the pre-open error-propagation path is preserved: an initial decode
    4xx is returned with the real status code (not an empty 200) in a Response that
    carries Content-Length.
    """
    _set_script(
        MOCK_DECODE_PORT,
        [
            {
                "kind": "json",
                "status": 400,
                "body": {"error": {"message": "max_tokens too long", "type": "BadRequestError"}},
            }
        ],
    )
    with httpx.Client(timeout=30) as c:
        r = c.post(
            f"{proxy_server}/v1/chat/completions",
            json=_chat_req(max_tokens=99999, stream=False),
            headers={"Content-Type": "application/json"},
        )
    assert r.status_code == 400
    assert "content-length" in r.headers
    assert b"max_tokens" in r.content


def test_nonstream_handles_sse_frames_with_done(proxy_server):
    """Non-streaming tolerates SSE framing + a trailing ``[DONE]`` sentinel.

    Even when the upstream decoder happens to speak SSE (``data: ...\\n\\n`` +
    ``data: [DONE]\\n\\n``), the proxy must still return the single JSON document as
    a clean ``application/json`` body, without the ``data:`` prefix leaking through.
    """
    raw = json.dumps(_completion_body("hi")).encode("utf-8")
    _set_script(
        MOCK_DECODE_PORT,
        [
            {
                "kind": "stream",
                "status": 200,
                "chunks": [b"data: " + raw + b"\n\n", b"data: [DONE]\n\n"],
            }
        ],
    )
    with httpx.Client(timeout=30) as c:
        r = c.post(
            f"{proxy_server}/v1/chat/completions",
            json=_chat_req(stream=False),
            headers={"Content-Type": "application/json"},
        )
    assert r.status_code == 200
    assert "content-length" in r.headers
    body = r.json()
    assert body["choices"][0]["message"]["content"] == "hi"


def test_nonstream_handles_merged_sse_frames(proxy_server):
    """Non-streaming extracts one JSON document even when two SSE frames merge.

    A single network read can concatenate ``data: {...}\\n\\ndata: [DONE]\\n\\n``
    into one chunk. Parsing the whole chunk at once fails on the ``[DONE]`` suffix
    and produces an empty 200; the frame parser splits the two frames instead.
    """
    raw = json.dumps(_completion_body("hi")).encode("utf-8")
    merged = b"data: " + raw + b"\n\ndata: [DONE]\n\n"
    _set_script(
        MOCK_DECODE_PORT,
        [{"kind": "stream", "status": 200, "chunks": [merged]}],
    )
    with httpx.Client(timeout=30) as c:
        r = c.post(
            f"{proxy_server}/v1/chat/completions",
            json=_chat_req(stream=False),
            headers={"Content-Type": "application/json"},
        )
    assert r.status_code == 200
    assert "content-length" in r.headers
    body = r.json()
    assert body["choices"][0]["message"]["content"] == "hi"


def test_nonstream_retry_clears_sentinel_buffer(proxy_server):
    """A recomputed attempt must not poison the retry attempt's body.

    The failed attempt returns ``data: {...recomputed...}\\n\\ndata: [DONE]\\n\\n``
    merged into one chunk. The parser consumes both frames, and the parser is reset
    at every attempt boundary, so the next decoder's body stays intact. Both proxy
    variants share this logic.
    """
    recomputed_merged = _sse("gen", stop_reason="recomputed") + DONE
    _set_script(
        MOCK_DECODE_PORT,
        [
            {"kind": "stream", "status": 200, "chunks": [recomputed_merged]},
            {"kind": "json", "status": 200, "body": _completion_body("ok")},
        ],
    )
    with httpx.Client(timeout=30) as c:
        r = c.post(
            f"{proxy_server}/v1/chat/completions",
            json=_chat_req(stream=False),
            headers={"Content-Type": "application/json"},
        )
    assert r.status_code == 200
    assert "content-length" in r.headers
    body = r.json()
    # The retry attempt's body must be emitted (not discarded by leftover state).
    # Its content "ok" is appended to the accumulated tokens ("gen" from the
    # recomputed attempt), which is the established retry semantics.
    assert body["choices"][0]["message"]["content"] == "genok"


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
    proxy_server: str,
    chunks: list[bytes],
    expected_status: int,
) -> None:
    """Non-streaming decode that ends without a document returns 502, not 200.

    Regression test: a decoder that closes a 200 response without a parseable
    body (partial JSON, empty, whitespace-only, or a lone ``[DONE]`` sentinel)
    must not be surfaced to the client as an empty 200. The proxy tracks whether
    a complete document was received and returns an upstream error when EOF
    arrives first. A valid document followed by ``[DONE]`` still succeeds.
    The layerwise variant runs the same matrix in its own test module.
    """
    _set_script(
        MOCK_DECODE_PORT,
        [{"kind": "stream", "status": 200, "chunks": chunks}],
    )
    with httpx.Client(timeout=30) as c:
        r = c.post(
            f"{proxy_server}/v1/chat/completions",
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


def test_nonstream_recompute_in_unterminated_tail_retries(proxy_server: str) -> None:
    """A recompute signal in the unterminated trailing frame still retries.

    A backend may close right after its last frame without sending the
    terminating blank line, so that frame is only decoded by ``flush()`` at EOF.
    When it carries ``stop_reason="recomputed"`` the proxy must retry instead of
    returning the partial answer. The layerwise variant covers this in its own
    test module.
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
            f"{proxy_server}/v1/chat/completions",
            json=_chat_req(stream=False),
            headers={"Content-Type": "application/json"},
        )
    assert r.status_code == 200
    body = r.json()
    # The retry ran and its content is appended to the aborted attempt.
    assert body["choices"][0]["message"]["content"] == "genok"


def test_nonstream_unterminated_usage_frame_merges(proxy_server: str) -> None:
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
            f"{proxy_server}/v1/chat/completions",
            json=_chat_req(stream=False),
            headers={"Content-Type": "application/json"},
        )
    assert r.status_code == 200
    body = r.json()
    assert body["choices"][0]["message"]["content"] == "hi"
    assert body["usage"]["completion_tokens"] == 9


def test_nonstream_unterminated_usage_after_metadata_frame(proxy_server: str) -> None:
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
            f"{proxy_server}/v1/chat/completions",
            json=_chat_req(stream=False),
            headers={"Content-Type": "application/json"},
        )
    assert r.status_code == 200
    body = r.json()
    assert body["choices"][0]["message"]["content"] == "hi"
    assert body["usage"]["completion_tokens"] == 9
