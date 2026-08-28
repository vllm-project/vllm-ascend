# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end tests for the fault-tolerance framework on Ascend NPU.

Requires 4 NPUs (DP=4); gated behind ``has_npu_ft_capability()``.
"""

import contextlib
import http.server
import os
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import psutil
import pytest
import requests
import torch

from tests.e2e.conftest import RemoteOpenAIServer

MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen3-30B-A3B")
DP_SIZE = 4

# Fault-detection timeout budget:
# - CPU: Gloo DP allreduce timeout detects the dead peer.
# - NPU: HCSP operator timeout detects the dead peer.
# - Deadline: slowest fallback + margin.
CPU_DISTRIBUTED_TIMEOUT_S = 15
FT_COMMUNICATION_ABORT_TIMEOUT_S = 10
FAULT_DETECTION_DEADLINE_S = 45

BENCHMARK_HOME = "./benchmark"
_GSM8K_CASE = {
    "case_type": "accuracy",
    "dataset_path": "vllm-ascend/gsm8k-lite",
    "request_conf": "vllm_api_general_chat",
    "dataset_conf": "gsm8k/gsm8k_gen_0_shot_cot_chat_prompt",
    "max_out_len": 32768,
    "batch_size": 32,
    "baseline": 95,
    "threshold": 5,
}


# ---------------------------------------------------------------------------
# Fault-injection via sitecustomize.py
# ---------------------------------------------------------------------------
# Patches ``dp_utils.sync_cudagraph_and_dp_padding`` to raise on ``rank`` after
# the DP all_reduce at a chosen step. Gated on VLLM_FT_TEST_INJECT_FAULT.
#
# The import hook waits for ``vllm.v1.worker.gpu.dp_utils`` to land in
# sys.modules and for ``sync_cudagraph_and_dp_padding`` to be defined, then
# wraps the function with a step-counting wrapper. The wrapper calls the
# original (so the all_reduce completes) and raises after it returns.
_FAULT_INJECT_SITECUSTOMIZE = """\
import builtins
import os
import sys

_SPEC = os.environ.get("VLLM_FT_TEST_INJECT_FAULT")
_MODULE = "vllm.v1.worker.gpu.dp_utils"
_FUNC = "sync_cudagraph_and_dp_padding"

if _SPEC:
    _f = dict(kv.split("=", 1) for kv in _SPEC.split(","))
    _RANK, _STEP = int(_f["rank"]), int(_f["step"])
    _steps = [0]

    def _patch(m):
        import inspect

        _orig = getattr(m, _FUNC)
        _sig = inspect.signature(_orig)

        def _wrapped(*args, **kwargs):
            result = _orig(*args, **kwargs)
            bound = _sig.bind(*args, **kwargs)
            bound.apply_defaults()
            dp_rank = bound.arguments.get("dp_rank")
            if dp_rank == _RANK:
                _steps[0] += 1
                if _steps[0] == _STEP:
                    raise RuntimeError(
                        "FT test fault injection (rank=%d step=%d)"
                        % (_RANK, _STEP)
                    )
            return result

        setattr(m, _FUNC, _wrapped)

    _real_import = builtins.__import__

    def _hook(name, *a, **k):
        module = _real_import(name, *a, **k)
        m = sys.modules.get(_MODULE)
        if (
            m is not None
            and hasattr(m, _FUNC)
            and not getattr(m, "_ft_patched", False)
        ):
            m._ft_patched = True
            _patch(m)
        return module

    builtins.__import__ = _hook
"""


def _install_fault_injection(monkeypatch, tmp_path, rank: int, step: int) -> None:
    """Arrange for the DP all_reduce sync to raise on ``rank`` at serving ``step``.

    Writes a ``sitecustomize.py`` and prepends its dir to PYTHONPATH so every
    vLLM subprocess picks it up; the fault spec is read from the environment.
    """
    site_dir = tmp_path / "ft_inject"
    site_dir.mkdir()
    (site_dir / "sitecustomize.py").write_text(_FAULT_INJECT_SITECUSTOMIZE)
    existing = os.environ.get("PYTHONPATH", "")
    monkeypatch.setenv(
        "PYTHONPATH",
        str(site_dir) + (os.pathsep + existing if existing else ""),
    )
    monkeypatch.setenv("VLLM_FT_TEST_INJECT_FAULT", f"rank={rank},step={step}")


# ---------------------------------------------------------------------------
# Server management
# ---------------------------------------------------------------------------


def _ft_server_args() -> list[str]:
    return [
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "37364",
        "--max-num-seqs",
        "128",
        "--enable-expert-parallel",
        "--enable-fault-tolerance",
        "--cpu-distributed-timeout-seconds",
        str(CPU_DISTRIBUTED_TIMEOUT_S),
        "--fault-tolerance-config",
        '{"engine_recovery_timeout_sec": 120}',
        "--additional-config",
        f'{{"ft_communication_abort_timeout": {FT_COMMUNICATION_ABORT_TIMEOUT_S}}}',
    ]


class FTServerManager:
    """Manages DP=4 vLLM server instances for fault-tolerance testing.

    Starts one process per DP rank with fixed ports (8000 + rank).
    """

    def __init__(
        self,
        model_name: str,
        dp_size: int,
        base_server_args: list[str],
        tp_size: int = 1,
    ):
        self.model_name = model_name
        self.dp_size = dp_size
        self.tp_size = tp_size
        self.base_server_args = base_server_args
        self.servers: list[tuple[RemoteOpenAIServer, list[str]]] = []
        self.server_threads: list[threading.Thread] = []

    def __enter__(self) -> list[tuple[RemoteOpenAIServer, list[str]]]:
        for rank in range(self.dp_size):
            server_args = self.base_server_args.copy()
            server_args.extend(
                [
                    "--data-parallel-size",
                    str(self.dp_size),
                    "--data-parallel-rank",
                    str(rank),
                    "--data-parallel-size-local",
                    "1",
                    "--tensor-parallel-size",
                    str(self.tp_size),
                    "--port",
                    str(8000 + rank),
                    "--api-server-count",
                    "1",
                ]
            )

            def start_server(r: int, sargs: list[str]) -> None:
                try:
                    server = RemoteOpenAIServer(
                        self.model_name,
                        sargs,
                        server_host="localhost",
                        server_port=8000 + r,
                        auto_port=False,
                        env_dict={
                            "ASCEND_RT_VISIBLE_DEVICES": str(r),
                            "VLLM_USE_V2_MODEL_RUNNER": "1",
                        },
                    )
                    self.servers.append((server, sargs))
                except Exception:
                    print(f"Failed to start server rank {r}")
                    raise

            thread = threading.Thread(target=start_server, args=(rank, server_args))
            thread.start()
            self.server_threads.append(thread)

        for thread in self.server_threads:
            thread.join()

        if len(self.servers) != self.dp_size:
            raise RuntimeError(f"Only {len(self.servers)}/{self.dp_size} servers started")

        return self.servers

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        for server, _ in reversed(self.servers):
            with contextlib.suppress(Exception):
                server.__exit__(None, None, None)
        self.servers.clear()


def _ft_manager() -> FTServerManager:
    return FTServerManager(
        MODEL_NAME,
        DP_SIZE,
        base_server_args=_ft_server_args(),
        tp_size=1,
    )


def _server_for_rank(servers: list[tuple[RemoteOpenAIServer, list[str]]], rank: int) -> RemoteOpenAIServer:
    """Locate the server for a DP rank."""
    for server, sargs in servers:
        if "--data-parallel-rank" in sargs:
            idx = sargs.index("--data-parallel-rank")
            if int(sargs[idx + 1]) == rank:
                return server
    raise AssertionError(f"no server found for DP rank {rank}")


# ---------------------------------------------------------------------------
# Round-robin proxy
# ---------------------------------------------------------------------------


class RoundRobinProxy:
    """Tiny standard-library HTTP proxy that round-robins requests across
    the DP rank server ports.

    Listens on ``listen_port`` and forwards every request (method, path,
    headers, body) verbatim to one of ``backend_ports``, chosen by a
    thread-safe round-robin counter, so requests are spread evenly across
    the ranks. Responses are passed back unchanged.

    Usage::

        with _ft_manager() as servers:
            backend_ports = [server.port for server, _ in servers]
            with RoundRobinProxy(backend_ports=backend_ports, listen_port=8100) as proxy:
                resp = requests.post(f"{proxy.url}/v1/completions", json={...})
    """

    def __init__(
        self,
        backend_ports: list[int],
        listen_port: int,
        host: str = "127.0.0.1",
    ):
        self._backend_ports = list(backend_ports)
        self._host = host
        self._listen_port = listen_port
        self._next = 0
        self._lock = threading.Lock()
        self._httpd: http.server.ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "RoundRobinProxy":
        self._httpd = http.server.ThreadingHTTPServer((self._host, self._listen_port), self._make_handler())
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None

    @property
    def url(self) -> str:
        return f"http://{self._host}:{self._listen_port}"

    @property
    def port(self) -> int:
        """Proxy listen port; makes ``RoundRobinProxy`` interchangeable with a
        ``RemoteOpenAIServer`` for helpers that only need ``.port``."""
        return self._listen_port

    def _pick_backend(self) -> int:
        with self._lock:
            port = self._backend_ports[self._next % len(self._backend_ports)]
            self._next += 1
            return port

    def _make_handler(self):
        proxy = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def _forward(self) -> None:
                length = int(self.headers.get("Content-Length", 0) or 0)
                body = self.rfile.read(length) if length else b""
                port = proxy._pick_backend()
                url = f"http://127.0.0.1:{port}{self.path}"
                req = urllib.request.Request(url, data=body, method=self.command)
                for key, value in self.headers.items():
                    if key.lower() not in ("host", "content-length", "accept-encoding"):
                        req.add_header(key, value)
                try:
                    with urllib.request.urlopen(req, timeout=300) as resp:
                        status, resp_headers, resp_body = resp.status, dict(resp.headers), resp.read()
                except urllib.error.HTTPError as exc:
                    status, resp_headers, resp_body = exc.code, dict(exc.headers), exc.read()
                self.send_response(status)
                for key, value in resp_headers.items():
                    if key.lower() not in ("transfer-encoding", "connection", "content-length"):
                        self.send_header(key, value)
                self.send_header("Content-Length", str(len(resp_body)))
                self.end_headers()
                self.wfile.write(resp_body)

            do_GET = do_POST = do_PUT = do_DELETE = do_PATCH = _forward

            def log_message(self, *args):  # silence request logging
                pass

        return Handler


# ---------------------------------------------------------------------------
# Test primitives
# ---------------------------------------------------------------------------


def _complete(client) -> Any:
    """Issue one completion request; used to drive the serving loop."""
    return client.completions.create(
        model=MODEL_NAME,
        prompt="Hello, my name is",
        max_tokens=5,
        temperature=0.0,
    )


def _in_parallel(fn, servers) -> list[Any]:
    """Run ``fn(server)`` for all servers concurrently; return in order."""
    with ThreadPoolExecutor(max_workers=len(servers)) as ex:
        return list(ex.map(fn, servers))


def _get_ft_status(server: RemoteOpenAIServer) -> dict:
    resp = requests.get(server.url_for("fault_tolerance/status"), timeout=10)
    resp.raise_for_status()
    return resp.json()


def _apply_ft(server: RemoteOpenAIServer, instruction: str, params: dict | None = None) -> dict:
    """POST an FT instruction; assert it is accepted (202) and return body."""
    resp = requests.post(
        server.url_for("fault_tolerance/apply"),
        json={"instruction": instruction, "params": params or {}},
        timeout=10,
    )
    assert resp.status_code == 202, resp.text
    return resp.json()


def _assert_serving_and_healthy(
    servers: tuple[RemoteOpenAIServer, ...],
) -> None:
    """Wait until every engine is healthy, then serve one request per server."""
    healthy = _wait_for_engines(list(servers), match_key="status", match_values={"healthy"})
    assert all(healthy), healthy
    _in_parallel(lambda s: _complete(s.get_client()), servers)


def _run_gsm8k_eval(server: Any, stage: str) -> float:
    """Run GSM8K accuracy evaluation using aisbench.

    ``server`` is any object exposing ``.port`` — either a
    ``RemoteOpenAIServer`` or a ``RoundRobinProxy``.

    Returns the measured accuracy percentage.
    """

    os.environ.setdefault("BENCHMARK_HOME", BENCHMARK_HOME)
    from tools.aisbench import AisbenchRunner

    with AisbenchRunner(
        model=MODEL_NAME,
        port=server.port,
        aisbench_config=_GSM8K_CASE,
        verify=True,
    ) as aisbench:
        accuracy = aisbench.result
        print(f"[{stage}] GSM8K accuracy: {accuracy:.2f}")
        return accuracy


def _kill_worker_process(server: RemoteOpenAIServer) -> None:
    """SIGKILL only the worker proc, leaving EngineCore and API server alive."""
    workers = [p for p in psutil.Process(server.proc.pid).children(recursive=True) if "Worker" in " ".join(p.cmdline())]
    assert len(workers) == 1, f"expected 1 worker proc, found: {workers}"
    workers[0].kill()


def _wait_for_engines(
    servers: list[RemoteOpenAIServer],
    match_key: str,
    match_values: set[str],
    deadline_s: int = FAULT_DETECTION_DEADLINE_S,
) -> list[dict[str, Any] | None]:
    """Poll ``/fault_tolerance/status`` until each server's engine status matches.

    A server matches when its engine-status dict has ``match_key`` equal to
    one of ``match_values``. Returns one engine-status dict per server.
    Servers still unmatched after ``deadline_s`` get None.
    """
    results: dict[int, dict[str, Any]] = {}
    pending = dict(enumerate(servers))
    start = time.time()
    while pending and time.time() - start < deadline_s:
        for i, server in list(pending.items()):
            with contextlib.suppress(Exception):
                for engine_status in _get_ft_status(server)["engines"]:
                    if engine_status.get(match_key) in match_values:
                        results[i] = engine_status
                        del pending[i]
                        break
        if pending:
            time.sleep(1.0)
    return [results.get(i) for i in range(len(servers))]


@contextlib.contextmanager
def _driving(*servers: RemoteOpenAIServer):
    """Pump completions at each server in the background for the block's duration.

    Keeps every engine stepping into its failed component so a fault surfaces.
    Errors are expected once faulted and are ignored.
    """
    stop = threading.Event()

    def _drive(server):
        client = server.get_client()
        while not stop.is_set():
            with contextlib.suppress(Exception):
                _complete(client)
            time.sleep(0.2)

    threads = [threading.Thread(target=_drive, args=(s,), daemon=True) for s in servers]
    for t in threads:
        t.start()
    try:
        yield
    finally:
        stop.set()
        for t in threads:
            t.join(timeout=2)


def _wait_for_ft_apply_outcome(server: RemoteOpenAIServer, request_id: str, deadline_s: int) -> str | None:
    """Wait until ``/fault_tolerance/status`` records the FT apply outcome."""
    engine_status = _wait_for_engines(
        [server],
        match_key="last_ft_request_id",
        match_values={request_id},
        deadline_s=deadline_s,
    )[0]
    return engine_status.get("ft_error") if engine_status else None


# ---------------------------------------------------------------------------
# Feature guard
# ---------------------------------------------------------------------------


def has_npu_ft_capability() -> bool:
    """Require at least 4 visible NPUs for DP=4 fault-tolerance tests."""
    if not torch.npu.is_available():
        return False
    try:
        return torch.npu.device_count() >= DP_SIZE
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not has_npu_ft_capability(),
    reason="Requires at least 4 NPUs for DP=4 fault-tolerance testing",
)
def test_injected_fault_retry_recovers_all_ranks(monkeypatch, tmp_path):
    """An exception injected after the DP allreduce drives full
    retry recovery on all 4 DP ranks.

    Inject a fault at a chosen step on rank 3:

    - Rank 3 raises after allreduce and goes UNHEALTHY.
    - Ranks 0, 1, 2 detect the now-absent peer via the Gloo DP allreduce
      timeout and also go UNHEALTHY.

    All 4 being UNHEALTHY is the precondition for ``retry``.  The fault
    is patched via a generated ``sitecustomize.py``.

    After recovery, the serving cluster must still meet the GSM8K accuracy
    standard used by tests/e2e/weekly/single_node/models/test_qwen3_30b_acc.py
    (baseline=95, threshold=5, i.e. accuracy >= 90%).
    """
    fault_step = int(os.getenv("FT_FAULT_STEP", "100"))
    _install_fault_injection(monkeypatch, tmp_path, rank=3, step=fault_step)

    with _ft_manager() as servers:
        assert len(servers) == DP_SIZE
        all_ranks = tuple(_server_for_rank(servers, r) for r in range(DP_SIZE))
        backend_ports = [server.port for server, _ in servers]

        # Requests through the round-robin proxy are spread evenly across
        # the DP rank ports; the proxy stays alive for the whole test so the
        # GSM8K accuracy evaluation is also driven through it.
        with RoundRobinProxy(backend_ports=backend_ports, listen_port=8100) as proxy:
            # 1. All engines healthy and serving.
            _assert_serving_and_healthy(all_ranks)

            # 2. Drive all ranks so rank 3 accumulates steps and trips the
            #    injected fault; ranks 0,1,2 then time out on DP allreduce.
            with _driving(*all_ranks):
                faulted = _wait_for_engines(
                    list(all_ranks),
                    match_key="status",
                    match_values={"unhealthy"},
                )

            for rank, engine_status in enumerate(faulted):
                assert engine_status is not None, (
                    f"rank {rank} did not report UNHEALTHY within {FAULT_DETECTION_DEADLINE_S}s -- it likely hung"
                )

            # The rank that raised carries the fault info from its own exception.
            assert faulted[3] is not None
            assert faulted[3].get("fault_info"), faulted[3]

            # 3. retry all engines.
            for server in all_ranks:
                _apply_ft(server, "retry")

            # 4. Recovery completes: all engines return to healthy and serve again.
            _assert_serving_and_healthy(all_ranks)

            # 5. The recovered cluster must still meet the GSM8K accuracy
            #    standard (>= 90%); AisbenchRunner raises otherwise. The injected
            #    fault is a one-shot step guard, so this dataset run does not
            #    re-trigger it. The evaluation runs through the proxy, which
            #    spreads the requests across all DP ranks.
            _run_gsm8k_eval(proxy, "post-retry")


@pytest.mark.skipif(
    not has_npu_ft_capability(),
    reason="Requires at least 4 NPUs for DP=4 fault-tolerance testing",
)
def test_worker_kill_survivor_unhealthy_and_dead_rejects_retry():
    """SIGKILL one Worker; survivors go UNHEALTHY, victim goes DEAD.

    Killing only rank 3's worker leaves all EngineCores alive, so the same
    fault is seen two ways:

    - Survivors (ranks 0, 1, 2): detect the dead peer via Gloo DP allreduce
      / HCSP timeout. Their own executor is fine, so ``on_fault`` marks them
      UNHEALTHY with a ``fault_info``.
    - Victim (rank 3): detects its own executor failure and marks itself DEAD.

    Recovery is gated on UNHEALTHY: the DEAD engine accepts ``retry`` at the
    HTTP layer (202 = background dispatch) but rejects it in the engine,
    recording the reason as ``ft_error``.
    """
    with _ft_manager() as servers:
        assert len(servers) == DP_SIZE
        survivor0 = _server_for_rank(servers, 0)
        survivor1 = _server_for_rank(servers, 1)
        survivor2 = _server_for_rank(servers, 2)
        victim = _server_for_rank(servers, 3)
        all_ranks = (survivor0, survivor1, survivor2, victim)

        # 1. Confirm all engines are healthy and serving.
        _assert_serving_and_healthy(all_ranks)

        # 2. Kill only the victim's worker; all EngineCores stay alive.
        _kill_worker_process(victim)

        # 3. Drive all engines so each keeps stepping into the failed component.
        with _driving(*all_ranks):
            faulted_results = _wait_for_engines(
                list(all_ranks),
                match_key="status",
                match_values={"dead", "unhealthy"},
            )

        s0, s1, s2, victim_faulted = faulted_results

        # Survivors must report the peer fault as UNHEALTHY.
        for label, result in [("rank 0", s0), ("rank 1", s1), ("rank 2", s2)]:
            assert result is not None, (
                f"{label} did not report the peer fault within {FAULT_DETECTION_DEADLINE_S}s -- it likely hung"
            )
            assert result["status"] == "unhealthy", result
            assert result.get("fault_info"), result

        # Victim must report DEAD (its own worker is gone).
        assert victim_faulted is not None, (
            f"victim (rank 3) did not report its worker's death within {FAULT_DETECTION_DEADLINE_S}s"
        )
        assert victim_faulted["status"] == "dead", victim_faulted

        # 4. retry is accepted at the HTTP layer (202 = background dispatch)...
        request_id = _apply_ft(victim, "retry")["request_id"]

        # 5. ...but the DEAD engine must reject it: recovery requires UNHEALTHY.
        ft_error = _wait_for_ft_apply_outcome(victim, request_id, FAULT_DETECTION_DEADLINE_S)
        assert ft_error is not None, "rejection was never recorded in /fault_tolerance/status"
        assert "status is DEAD" in ft_error, ft_error
