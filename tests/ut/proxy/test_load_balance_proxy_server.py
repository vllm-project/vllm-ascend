"""System-level tests for load_balance_proxy_server_example.

Tests the proxy's HTTP behavior and worker listener lifecycle using mock
backends and real subprocesses.
"""

import base64
import contextlib
import http.client
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import httpx
import psutil
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
@pytest.fixture(scope="module", params=[1, 4], ids=["single-worker", "multi-worker"])
def proxy_server(request):
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
                    "--workers",
                    str(request.param),
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


@pytest.mark.parametrize("shared_listener", [False, True], ids=["auto", "shared-fallback"])
def test_worker_connections_restart_and_shutdown(tmp_path, shared_listener):
    from examples.disaggregated_prefill_v1 import load_balance_proxy_server_example as proxy

    script = tmp_path / "probe.py"
    script.write_text(
        "import os, sys, signal, threading, time\n"
        "from pathlib import Path\n"
        "from fastapi import FastAPI\n"
        "import load_balance_proxy_server_example as proxy\n"
        "app = FastAPI()\n"
        "@app.get('/')\n"
        "def pid(): return {'pid': os.getpid()}\n"
        "def create_app(): return app\n"
        "def stop_on_file():\n"
        "    while not Path(sys.argv[2]).exists(): time.sleep(0.05)\n"
        "    signal.raise_signal(signal.SIGTERM)\n"
        "if __name__ == '__main__':\n"
        "    threading.Thread(target=stop_on_file, daemon=True).start()\n"
        "    if sys.argv[3] == 'shared': proxy.supports_worker_listeners = lambda host: False\n"
        "    proxy.run_worker_listeners('probe:create_app', host='127.0.0.1', port=int(sys.argv[1]), workers=4)\n"
    )
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(PROXY_SCRIPT.parent), os.environ.get("PYTHONPATH", "")]))
    with socket.socket() as available:
        available.bind(("127.0.0.1", 0))
        port = available.getsockname()[1]
    stop_file = tmp_path / "stop"
    command = [sys.executable, str(script), str(port), str(stop_file), "shared" if shared_listener else "auto"]

    def request_pid():
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
        try:
            conn.request("GET", "/", headers={"Connection": "close"})
            response = conn.getresponse()
            assert response.status == 200
            return json.loads(response.read())["pid"]
        finally:
            conn.close()

    log_path = tmp_path / "workers.log"
    with log_path.open("w") as log:
        proc = subprocess.Popen(
            command,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=log,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        children = []
        try:
            deadline = time.monotonic() + 30
            workers = set()
            while len(workers) != 4 and time.monotonic() < deadline:
                assert proc.poll() is None, log_path.read_text()
                with contextlib.suppress(OSError, http.client.HTTPException):
                    workers.add(request_pid())
                time.sleep(0.05)
            assert len(workers) == 4, log_path.read_text()
            served_workers = {request_pid() for _ in range(200)}
            assert served_workers <= workers
            if not shared_listener and proxy.supports_worker_listeners("127.0.0.1"):
                assert served_workers == workers
            killed = next(iter(workers))
            psutil.Process(killed).kill()
            deadline = time.monotonic() + 30
            replacement = None
            while time.monotonic() < deadline:
                with contextlib.suppress(OSError, http.client.HTTPException):
                    pid = request_pid()
                    if pid not in workers:
                        replacement = pid
                        break
                time.sleep(0.1)
            assert replacement is not None, log_path.read_text()
            children = psutil.Process(proc.pid).children(recursive=True)
            stop_file.touch()
            proc.wait(timeout=15)
            assert proc.returncode == 0, log_path.read_text()
            with pytest.raises(OSError), socket.create_connection(("127.0.0.1", port), timeout=1):
                pass
        finally:
            if proc.poll() is None:
                children = psutil.Process(proc.pid).children(recursive=True)
                proc.kill()
                proc.wait(timeout=15)
            for child in children:
                with contextlib.suppress(psutil.NoSuchProcess):
                    child.kill()
            psutil.wait_procs(children, timeout=5)

    # An occupied port must fail before workers join an unrelated reuse group.
    stop_file.unlink()
    with socket.socket() as occupied:
        occupied.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        if proxy.supports_worker_listeners("127.0.0.1"):
            occupied.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        occupied.bind(("127.0.0.1", port))
        occupied.listen()
        result = subprocess.run(
            command,
            env=env,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=15,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        assert result.returncode != 0
