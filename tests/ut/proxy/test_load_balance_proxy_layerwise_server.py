"""Tests for the layerwise PD load balance proxy's response reassembly.

The decode backend answers with SSE, but the network does not respect frame
boundaries: a frame can arrive split across two reads, and two frames can be
coalesced into a single read. The proxy must reassemble frames before parsing
them, otherwise it misses the frame's ``stop_reason`` and never runs its
recompute retry.
"""

import asyncio
import contextlib
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlsplit

import httpx
import psutil
import pytest
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

EXAMPLES_DIR = Path(__file__).resolve().parents[3] / "examples" / "disaggregated_prefill_v1"
sys.path.insert(0, str(EXAMPLES_DIR))

from examples.disaggregated_prefill_v1 import (  # noqa: E402
    load_balance_proxy_layerwise_server_example as proxy,
)


def _frame(content: str, stop_reason: str | None = None) -> bytes:
    payload = {
        "id": "x",
        "choices": [
            {
                "index": 0,
                "delta": {"content": content},
                "finish_reason": None,
                "stop_reason": stop_reason,
            },
        ],
    }
    return b"data: " + json.dumps(payload).encode("utf-8") + b"\n\n"


DONE = b"data: [DONE]\n\n"


class _FakeResponse:
    """Minimal stand-in for the Starlette request object."""

    def __init__(self, body: dict) -> None:
        self._body = body

    async def json(self) -> dict:
        return self._body

    async def body(self) -> bytes:
        return json.dumps(self._body).encode("utf-8")


class _FakeProxyState:
    """Minimal stand-in for the proxy's scheduling state."""

    class _FakeDecoder:
        class client:  # noqa: N801 - mimics the httpx client attribute
            pass

        url = "http://decode"

    def __init__(self) -> None:
        self.decoders = [self._FakeDecoder()]
        self.req_to_prefiller: dict[str, str] = {}

    async def next_req_id(self) -> str:
        return "req-0"

    def register_request_batch(self, *args, **kwargs) -> None:
        pass

    def calculate_decode_scores(self, *args, **kwargs) -> float:
        return 0.0

    def select_decoder(self, *args, **kwargs) -> int:
        return 0

    def release_decoder(self, *args, **kwargs) -> None:
        pass

    async def cleanup_request_batch(self, *args, **kwargs) -> None:
        pass


async def _run_proxy(chunks_per_attempt: list[list[bytes]]) -> list[bytes]:
    """Drive the layerwise proxy handler and return the forwarded chunks.

    ``chunks_per_attempt`` holds the chunks each decoder attempt emits; the
    second attempt is only reached when the proxy detects a recompute.
    """
    attempts = iter(chunks_per_attempt)

    async def fake_stream(*args, **kwargs):
        for chunk in next(attempts):
            yield chunk

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(proxy, "proxy_state", _FakeProxyState())
    monkeypatch.setattr(proxy, "stream_service_response_with_retry", fake_stream)
    monkeypatch.setattr(
        proxy,
        "global_args",
        proxy.argparse.Namespace(max_retries=0, retry_delay=0.0, host="127.0.0.1", port=8000),
        raising=False,
    )

    request = _FakeResponse(
        {
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 20,
            "stream": True,
        }
    )

    try:
        response = await proxy._handle_completions("/v1/chat/completions", request)
        return [chunk async for chunk in response.body_iterator]
    finally:
        monkeypatch.undo()


def test_frame_split_across_chunks_is_reassembled():
    """A frame split across two reads still drives the recompute retry.

    The recompute signal travels in the frame's stop_reason. When the frame is
    split across reads, a per-chunk parse misses it and the proxy never
    retries, so the aborted generation reaches the client.
    """
    frame = _frame("gen", stop_reason="recomputed")
    cut = len(frame) // 2

    chunks = asyncio.run(
        _run_proxy(
            [
                [frame[:cut], frame[cut:]],
                [_frame("retried"), DONE],
            ]
        )
    )

    body = b"".join(chunks)
    assert b"retried" in body
    assert b"recomputed" not in body


def test_coalesced_frames_are_reassembled():
    """A frame coalesced with [DONE] into one read still drives the retry."""
    chunks = asyncio.run(
        _run_proxy(
            [
                [_frame("gen", stop_reason="recomputed") + DONE],
                [_frame("retried"), DONE],
            ]
        )
    )

    body = b"".join(chunks)
    assert b"retried" in body
    assert b"recomputed" not in body


def test_plain_stream_is_forwarded_unchanged():
    """A frame that arrives in one read is forwarded as it is."""
    chunks = asyncio.run(_run_proxy([[_frame("plain-token"), DONE]]))

    body = b"".join(chunks)
    assert b"plain-token" in body


def test_embedded_parser_handles_split_utf8_and_malformed_frames():
    parser = proxy.SSEFrameParser()
    payload = {"choices": [], "text": "你好"}
    frame = b"data: " + json.dumps(payload, ensure_ascii=False).encode() + b"\n\n"
    stream = b"data: invalid\n\ndata: []\n\n" + frame + DONE + frame
    # Deliver UTF-8 characters across network reads; DONE drops trailing data.
    cut = frame.index("你".encode()) + len(b"data: invalid\n\ndata: []\n\n") + 1
    assert parser.feed(stream[:cut]) == []
    assert parser.feed(stream[cut:]) == [payload]


def test_scheduler_keeps_kv_pressure_until_release():
    scheduler = proxy.LayerwiseScheduler([("p", 1), ("p", 2)], [("d", 1)])
    for _ in range(32):
        workers = [scheduler.register_worker() for _ in range(3)]
        first = scheduler.reserve(workers[0], "prefill", 100)
        second = scheduler.reserve(workers[1], "prefill", 1)
        assert first != second
        scheduler.release(workers[0], "prefill", first, 100)
        # P tokens are finished but its KV reservation still contributes load.
        assert scheduler.reserve(workers[2], "prefill", 1) == second
        for worker_token in workers:
            scheduler.release_worker(worker_token)
    assert scheduler.reserve(scheduler.register_worker(), "prefill", 1) == first


@pytest.mark.parametrize("delayed_role", ["prefill", "decode"])
def test_shared_load_and_dead_worker_cleanup(delayed_role):
    scheduler = proxy.LayerwiseScheduler([("p", 1), ("p", 2)], [("d", 1), ("d", 2)])
    workers = [scheduler.register_worker() for _ in range(2)]
    old_worker = workers[0]
    p0 = scheduler.reserve(old_worker, "prefill", 100)
    p1 = scheduler.reserve(workers[1], "prefill", 100)
    d0 = scheduler.reserve(old_worker, "decode", 100)
    d1 = scheduler.reserve(workers[1], "decode", 100)
    assert p0 != p1 and d0 != d1
    scheduler.release(old_worker, "prefill", p0, 100)
    started = threading.Event()

    def delayed_reserve():
        started.set()
        return scheduler.reserve(old_worker, delayed_role, 100)

    with ThreadPoolExecutor(max_workers=1) as pool:
        # Model a Manager RPC received before exit but waiting for the lock.
        with scheduler._lock:
            pending = pool.submit(delayed_reserve)
            assert started.wait(timeout=5)
            scheduler.release_worker(old_worker)
            assert scheduler.prefillers[p0].active_tokens == 0
            assert scheduler.prefillers[p0].active_kv_cache == 0
            assert scheduler.decoders[d0].active_tokens == 0
            assert scheduler.prefillers[p1].active_tokens == 100
            # The replacement occupies the same port slot with a new identity.
            workers[0] = scheduler.register_worker()
            assert workers[0] != old_worker
            assert scheduler.reserve(workers[0], "prefill", 100) == p0
            assert scheduler.reserve(workers[0], "decode", 100) == d0
        with pytest.raises(RuntimeError, match="no longer active"):
            pending.result(timeout=5)

    # Late release/cleanup from the old incarnation cannot debit its replacement.
    scheduler.release(old_worker, "prefill", p0, 100)
    scheduler.release(old_worker, "prefill", p0, 100, kv_cache=True)
    scheduler.release(old_worker, "decode", d0, 100)
    scheduler.release_worker(old_worker)
    assert old_worker not in scheduler.worker_loads
    assert scheduler.prefillers[p0].active_tokens == scheduler.prefillers[p0].active_kv_cache == 100
    assert scheduler.decoders[d0].active_tokens == 100
    scheduler.release(workers[1], "prefill", p1, 100)
    scheduler.release(workers[1], "prefill", p1, 100, kv_cache=True)
    scheduler.release(workers[1], "decode", d1, 100)
    for worker_token in workers:
        scheduler.release_worker(worker_token)
    assert all(server.active_tokens == server.active_kv_cache == 0 for server in scheduler.prefillers.values())
    assert all(server.active_tokens == 0 for server in scheduler.decoders.values())


def _reserve_from_process(scheduler, worker_token, result):
    result.put(scheduler.reserve(worker_token, "decode", 100))


def test_load_is_shared_across_processes():
    ctx = proxy.multiprocessing.get_context("spawn")
    with proxy.LayerwiseManager(ctx=ctx) as manager:
        scheduler = manager.Scheduler([("p", 1)], [("d", 1), ("d", 2)])
        result = ctx.Queue()
        chosen = []
        workers = [scheduler.register_worker() for _ in range(2)]
        try:
            for worker_token in workers:
                process = ctx.Process(target=_reserve_from_process, args=(scheduler, worker_token, result))
                process.start()
                try:
                    chosen.append(result.get(timeout=10))
                    process.join(timeout=10)
                    assert process.exitcode == 0
                finally:
                    proxy.stop_workers([process])
            assert chosen[0] != chosen[1]
            scheduler.release_worker(workers[0])
            replacement = scheduler.register_worker()
            with pytest.raises(RuntimeError, match="no longer active"):
                scheduler.reserve(workers[0], "decode", 100)
            assert scheduler.reserve(replacement, "decode", 100) == chosen[0]
        finally:
            result.close()
            result.join_thread()


def test_cancelled_callback_does_not_duplicate_prefill(monkeypatch):
    async def run():
        state = proxy.ProxyState([("p", 1)], [("d", 1)])
        monkeypatch.setattr(proxy, "proxy_state", state)
        monkeypatch.setattr(proxy, "global_args", proxy.argparse.Namespace(max_retries=1, retry_delay=0), raising=False)
        state.register_request_batch("parent", ["child"], {"prompt": "hi"}, 10, "/completions")
        started, finish = asyncio.Event(), asyncio.Event()
        calls = []

        async def prefill(*args, **kwargs):
            calls.append(args)
            started.set()
            await finish.wait()

        monkeypatch.setattr(proxy, "send_request_to_service", prefill)
        first = asyncio.create_task(proxy.metaserver(_FakeResponse({"request_id": "child"})))
        await asyncio.wait_for(started.wait(), timeout=2)
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        second = asyncio.create_task(proxy.metaserver(_FakeResponse({"request_id": "child"})))
        finish.set()
        assert await asyncio.wait_for(second, timeout=2) == {"status": "ok"}
        assert len(calls) == 1
        server = next(iter(state.scheduler.prefillers.values()))
        assert server.active_tokens == server.active_kv_cache == 0
        await state.cleanup_request_batch("parent")
        assert not state.req_data_dict and not state.metaserver_dispatch_tasks
        for backend in state.prefillers + state.decoders:
            await backend.client.aclose()

    asyncio.run(run())


@pytest.mark.parametrize("port,workers", [(8000, 0), (65535, 2), (0, 2)])
def test_invalid_worker_ports(monkeypatch, port, workers):
    monkeypatch.setattr(sys, "argv", ["proxy", "--port", str(port), "--workers", str(workers)])
    with pytest.raises(SystemExit):
        proxy.parse_args()


def test_ipv6_metaserver_url():
    args = proxy.argparse.Namespace(host="::1", port=19000, metaserver_port=19003)
    assert proxy.metaserver_url(args) == "http://[::1]:19003/v1/metaserver"


def _available_ports(count=1):
    # Reserve a consecutive range while checking it; the test child will bind it.
    for _ in range(100):
        with contextlib.ExitStack() as stack:
            first = stack.enter_context(socket.socket())
            first.bind(("127.0.0.1", 0))
            port = first.getsockname()[1]
            try:
                for offset in range(1, count):
                    sock = stack.enter_context(socket.socket())
                    sock.bind(("127.0.0.1", port + offset))
            except (OSError, OverflowError):
                continue
            return port
    raise RuntimeError("No free test port range")


def _wait_http(port, process):
    deadline = time.monotonic() + 30
    with httpx.Client(timeout=1, trust_env=False) as client:
        while time.monotonic() < deadline:
            assert process.poll() is None
            try:
                if client.get(f"http://127.0.0.1:{port}/healthcheck").status_code == 200:
                    return
            except httpx.RequestError:
                pass
            time.sleep(0.1)
    raise AssertionError(f"Listener {port} did not start")


def _stop_process(process):
    if process.poll() is None:
        children = psutil.Process(process.pid).children(recursive=True)
        process.terminate()
        try:
            process.wait(timeout=20)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
            for child in children:
                with contextlib.suppress(psutil.NoSuchProcess):
                    child.kill()
            psutil.wait_procs(children, timeout=5)


@pytest.fixture(scope="module", params=[1, 4], ids=["single-worker", "multi-worker"])
def running_proxy(request, tmp_path_factory):
    workers = request.param
    if workers > 1 and not hasattr(socket, "SO_REUSEPORT"):
        pytest.skip("Independent TCP listeners require SO_REUSEPORT")
    directory = tmp_path_factory.mktemp("layerwise")
    script = directory / "standalone_proxy.py"
    shutil.copyfile(EXAMPLES_DIR / "load_balance_proxy_layerwise_server_example.py", script)
    backend_port = _available_ports()
    port = _available_ports(workers + 1)
    backend_env = dict(
        os.environ, PYTHONPATH=os.pathsep.join([str(EXAMPLES_DIR.parents[1]), os.environ.get("PYTHONPATH", "")])
    )
    # Keep the selected Uvicorn version, but exclude the checkout and sibling scripts.
    env = dict(os.environ, PYTHONPATH=str(Path(uvicorn.__file__).resolve().parents[1]))
    command = [
        sys.executable,
        str(script),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--workers",
        str(workers),
        "--prefiller-hosts",
        "127.0.0.1",
        "--prefiller-ports",
        str(backend_port),
        "--decoder-hosts",
        "127.0.0.1",
        "--decoder-ports",
        str(backend_port),
        "--max-retries",
        "1",
    ]
    with (directory / "backend.log").open("w") as backend_log, (directory / "proxy.log").open("w") as proxy_log:
        backend = subprocess.Popen(
            [sys.executable, __file__, "--backend", str(backend_port)],
            env=backend_env,
            stdout=backend_log,
            stderr=backend_log,
        )
        process = None
        try:
            _wait_http(backend_port, backend)
            process = subprocess.Popen(command, cwd=directory, env=env, stdout=proxy_log, stderr=proxy_log)
            ports = [port] if workers == 1 else list(range(port + 1, port + workers + 1))
            for listen_port in [port, *ports]:
                _wait_http(listen_port, process)
            yield port, workers, process, command, env, directory
        finally:
            if process is not None:
                _stop_process(process)
            _stop_process(backend)


def _completion(port, *, batch=False, stream=False, model="normal"):
    body = {"model": model, "stream": stream, "max_tokens": 2}
    if batch:
        body["prompt"] = ["first", "second"]
        endpoint = "completions"
    else:
        body["messages"] = [{"role": "user", "content": "hello"}]
        endpoint = "chat/completions"
    with httpx.Client(timeout=15, trust_env=False) as client:
        response = client.post(f"http://127.0.0.1:{port}/v1/{endpoint}", json=body)
        response.raise_for_status()
        return response


@pytest.mark.parametrize("batch,stream", [(False, False), (True, False), (False, True)])
def test_callbacks_reach_request_owner(running_proxy, batch, stream):
    port, workers, *_ = running_proxy
    response = _completion(port, batch=batch, stream=stream, model="duplicate")
    payload = json.loads(response.text.split("data: ")[1].split("\n\n")[0]) if stream else response.json()
    callback_port = urlsplit(payload["metaserver"]).port
    assert callback_port == port if workers == 1 else port < callback_port <= port + workers
    assert payload["callback_statuses"] == [200] * (4 if batch else 2)
    assert payload["prefill_calls"] == 1
    assert payload["uvicorn_version"] == uvicorn.__version__
    ids = payload["prefill_params"].get(proxy.BATCH_KV_TRANSFER_PARAMS, {})
    if batch:
        assert len(ids) == 2
        assert sorted(params["remote_block_ids"] for params in ids.values()) == [[[0]], [[1]]]


def test_metaserver_failure_and_unknown_request(running_proxy):
    port, workers, *_ = running_proxy
    response = _completion(port, model="fail-prefill").json()
    assert response["callback_statuses"] == [502]
    callback_port = port if workers == 1 else port + 1
    response = httpx.post(f"http://127.0.0.1:{callback_port}/v1/metaserver", json={"request_id": "missing"})
    assert response.status_code == 404


def test_independent_listeners_restart_and_port_conflicts(running_proxy):
    port, workers, process, command, env, directory = running_proxy
    if workers == 1:
        pytest.skip("Worker replacement is a multi-process scenario")
    connections = [c for c in psutil.net_connections(kind="tcp") if c.status == "LISTEN" and c.laddr.port == port]
    assert len({c.pid for c in connections}) == workers
    if sys.platform == "linux":
        assert len({os.readlink(f"/proc/{c.pid}/fd/{c.fd}") for c in connections}) == workers
    with ThreadPoolExecutor(max_workers=16) as pool:
        responses = list(pool.map(lambda _: _completion(port).json(), range(80)))
    counts = Counter(urlsplit(response["metaserver"]).port for response in responses)
    assert set(counts) == set(range(port + 1, port + workers + 1))
    (directory / "connection-counts.json").write_text(json.dumps(counts))
    callback = next(c for c in psutil.net_connections(kind="tcp") if c.status == "LISTEN" and c.laddr.port == port + 1)
    psutil.Process(callback.pid).kill()
    deadline = time.monotonic() + 30
    replacement = None
    while time.monotonic() < deadline:
        replacement = next(
            (
                c.pid
                for c in psutil.net_connections(kind="tcp")
                if c.status == "LISTEN" and c.laddr.port == port + 1 and c.pid != callback.pid
            ),
            None,
        )
        if replacement is not None:
            break
        time.sleep(0.1)
    assert replacement is not None
    _wait_http(port + 1, process)
    assert _completion(port + 1).json()["callback_statuses"] == [200]
    process.send_signal(signal.SIGTERM)
    assert process.wait(timeout=20) == 0
    for listen_port in range(port, port + workers + 1):
        with pytest.raises(OSError), socket.create_connection(("127.0.0.1", listen_port), timeout=1):
            pass
    # Occupied public or callback ports must fail, including another reuse group.
    for listen_port in (port, port + 2):
        with proxy.bind_listener("127.0.0.1", listen_port, reuse_port=True):
            result = subprocess.run(command, cwd=directory, env=env, capture_output=True, timeout=15)
            assert result.returncode != 0


def _serve_mock_backend(port):
    backend = FastAPI()
    calls = Counter()
    params = {}

    @backend.get("/healthcheck")
    async def health():
        return {"status": "ok"}

    @backend.post("/v1/completions")
    @backend.post("/v1/chat/completions")
    async def complete(request: Request):
        body = await request.json()
        parent = request.headers["X-Request-Id"]
        kv = body["kv_transfer_params"]
        if kv.get("do_remote_decode"):
            calls[parent] += 1
            params[parent] = kv
            if body["model"] == "fail-prefill":
                return JSONResponse({"error": "prefill failed"}, status_code=503)
            await asyncio.sleep(0.05)
            return {"kv_transfer_params": {}}
        api = "/completions" if "prompt" in body else "/chat/completions"
        ids = proxy.get_api_request_ids(api, parent, body)
        async with httpx.AsyncClient(timeout=10, trust_env=False) as client:
            posts = []
            for index, request_id in enumerate(ids):
                metadata = {"request_id": request_id, "do_remote_decode": True, "remote_block_ids": [[index]]}
                for _ in range(2 if body["model"] == "duplicate" else 1):
                    posts.append(client.post(kv["metaserver"], json=metadata))
            callbacks = await asyncio.gather(*posts)
        result = {
            "uvicorn_version": uvicorn.__version__,
            "choices": [],
            "metaserver": kv["metaserver"],
            "callback_statuses": [response.status_code for response in callbacks],
            "prefill_calls": calls[parent],
            "prefill_params": params.get(parent),
        }
        if body.get("stream"):
            return StreamingResponse(
                iter([f"data: {json.dumps(result)}\n\ndata: [DONE]\n\n"]), media_type="text/event-stream"
            )
        return result

    uvicorn.run(backend, host="127.0.0.1", port=port)


if __name__ == "__main__":
    _serve_mock_backend(int(sys.argv[2]))
