# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end tests for the fault-tolerance framework on Ascend NPU.

Requires 4 NPUs (DP=4). Retry is gated behind ``has_npu_ft_capability()``;
scale-down additionally behind ``has_npu_scale_down_capability()`` (CANN V3+).
"""

import contextlib
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import psutil
import pytest
import regex as re
import requests
import torch

from tests.e2e.conftest import RemoteOpenAIServer

MODEL_NAME = os.getenv("MODEL_NAME", "vllm-ascend/Qwen3-30B-A3B-W8A8")
DP_SIZE = 4

# Fault-detection timeout budget:
# - CPU: Gloo DP allreduce timeout detects the dead peer.
# - NPU: HCSP operator timeout detects the dead peer.
# - Deadline: slowest fallback + margin.
CPU_DISTRIBUTED_TIMEOUT_S = 15
FT_COMMUNICATION_ABORT_TIMEOUT_S = 10
FAULT_DETECTION_DEADLINE_S = 45

# scale_down re-hosts the dead rank's experts on the survivors and reloads
# the reassigned weights from disk; give the redistribution + dummy-batch
# check more headroom than fault detection (still under
# engine_recovery_timeout_sec=120, the busy-loop's own give-up point).
SCALE_DOWN_DEADLINE_S = 90

# Qwen3-30B-A3B has 128 routed experts; on EP=4, 48 redundant experts give
# 44 physical slots per rank, so the 3 surviving ranks still have
# 3 * 44 = 132 >= 128 slots after one rank is removed (the strict minimum
# ``check_redundancy_sufficient`` accepts for a 4 -> 3 shrink is 44).
NUM_REDUNDANT_EXPERTS = 48

# Post-recovery accuracy check: after retry recovery, every DP rank must
# answer these factual prompts correctly. Each expected answer is a single
# word, matched case-insensitively on word boundaries anywhere in the
# completion, so harmless phrasing variations do not fail the check.
_ANSWER_CASES = [
    ("The capital of France is", ("Paris",)),
    ("The largest planet in our solar system is", ("Jupiter",)),
    ("The first month of the year is", ("January",)),
]
_ANSWER_MAX_TOKENS = 16


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


def _ft_server_args(extra_args: list[str] | None = None) -> list[str]:
    # Quantized path end to end: --quantization ascend (W8A8); no --dtype,
    # the checkpoint's own config decides. MODEL_NAME must point at a W8A8
    # checkpoint (e.g. vllm-ascend/Qwen3-30B-A3B-W8A8).
    return [
        "--quantization",
        "ascend",
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
        *(extra_args or []),
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


def _ft_manager(extra_args: list[str] | None = None) -> FTServerManager:
    return FTServerManager(
        MODEL_NAME,
        DP_SIZE,
        base_server_args=_ft_server_args(extra_args),
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
    resp = requests.get(server.url_for("v1/fault_tolerance/status"), timeout=10)
    resp.raise_for_status()
    return resp.json()


def _apply_ft(server: RemoteOpenAIServer, instruction: str, params: dict | None = None) -> dict:
    """POST an FT instruction; assert it is accepted (202) and return body."""
    resp = requests.post(
        server.url_for("v1/fault_tolerance/apply"),
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


def _assert_correct_answers(servers: tuple[RemoteOpenAIServer, ...]) -> None:
    """Send factual prompts to every DP rank and assert the answer appears.

    Must only be called after FT recovery (retry or scale_down) has fully
    completed (see ``_assert_serving_and_healthy``), so the requests are not
    sent into a still-faulting cluster.
    """
    for server in servers:
        client = server.get_client()
        for prompt, answers in _ANSWER_CASES:
            resp = client.completions.create(
                model=MODEL_NAME,
                prompt=prompt,
                max_tokens=_ANSWER_MAX_TOKENS,
                temperature=0.0,
            )
            completion = resp.choices[0].text
            matched = any(re.search(rf"\b{re.escape(answer)}\b", completion, re.IGNORECASE) for answer in answers)
            print(f"[accuracy] rank {server.port} prompt {prompt!r}: {completion!r}")
            assert matched, (
                f"[post-recovery] rank {server.port} answered {prompt!r} incorrectly: "
                f"expected one of {answers!r}, got {completion!r}"
            )


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
    """Poll ``/v1/fault_tolerance/status`` until each server's engine status matches.

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
    """Wait until ``/v1/fault_tolerance/status`` records the FT apply outcome."""
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


def has_npu_scale_down_capability() -> bool:
    """scale_down additionally requires the V3 dispatch op (CANN V3+).

    Mirrors the engine-side precondition in ``WorkerSentinel
    ._validate_scale_down_preconditions`` so the test skips cleanly on
    older CANN/torch_npu stacks instead of timing out mid-recovery.
    """
    if not has_npu_ft_capability():
        return False
    try:
        import torch_npu

        return hasattr(torch_npu, "npu_moe_distribute_dispatch_v2")
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

    After retry recovery completes, every rank must answer factual prompts
    correctly (greedy decoding; case-insensitive whole-word match, not an
    exact-match golden, so the check is independent of environment numerics).
    """
    fault_step = int(os.getenv("FT_FAULT_STEP", "100"))
    _install_fault_injection(monkeypatch, tmp_path, rank=3, step=fault_step)

    with _ft_manager() as servers:
        assert len(servers) == DP_SIZE
        all_ranks = tuple(_server_for_rank(servers, r) for r in range(DP_SIZE))

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

        # 5. Only now (after retry fully completed) send the accuracy prompts
        #    to every DP rank directly. The injected fault is a one-shot step
        #    guard, so these requests do not re-trigger it.
        _assert_correct_answers(all_ranks)


@pytest.mark.skipif(
    not has_npu_scale_down_capability(),
    reason=("Requires at least 4 NPUs and npu_moe_distribute_dispatch_v2 (CANN V3+) for DP=4 scale-down testing"),
)
def test_scale_down_removes_dead_rank_and_recovers():
    """scale_down removes the dead DP rank; survivors keep serving.

    SIGKILL rank 3's worker: survivors go UNHEALTHY, the victim goes DEAD
    and rejects ``retry`` (recovery requires UNHEALTHY). ``scale_down``
    with ``removed_dp_ranks=[3]`` masks the dead rank, redistributes its
    EPLB experts onto the survivors and reloads the reassigned weights.
    Rank 3 (not rank 0, the DP store master) is removed so no
    ``dp_master_ip`` / ``dp_store_port`` params are needed. After recovery
    every survivor must answer factual prompts correctly.
    """
    victim_rank = DP_SIZE - 1
    extra_args = [
        "--compilation-config",
        json.dumps(
            {
                "cudagraph_mode": "FULL_AND_PIECEWISE",
                "cudagraph_capture_sizes": [4, 8, 12, 16, 20, 24, 28, 32],
            }
        ),
        "--enable-eplb",
        "--eplb-config.num_redundant_experts",
        str(NUM_REDUNDANT_EXPERTS),
    ]
    with _ft_manager(extra_args) as servers:
        assert len(servers) == DP_SIZE
        servers_by_rank = {r: _server_for_rank(servers, r) for r in range(DP_SIZE)}
        victim = servers_by_rank[victim_rank]
        survivor_ranks = [r for r in range(DP_SIZE) if r != victim_rank]
        survivors = tuple(servers_by_rank[r] for r in survivor_ranks)
        all_ranks = tuple(servers_by_rank[r] for r in range(DP_SIZE))

        # 1. All engines healthy and serving.
        _assert_serving_and_healthy(all_ranks)

        # 2. Kill the victim's worker; survivors detect the peer fault.
        _kill_worker_process(victim)
        with _driving(*all_ranks):
            faulted_results = _wait_for_engines(
                list(all_ranks),
                match_key="status",
                match_values={"dead", "unhealthy"},
            )

        for rank, engine_status in enumerate(faulted_results):
            assert engine_status is not None, (
                f"rank {rank} did not report the fault within {FAULT_DETECTION_DEADLINE_S}s -- it likely hung"
            )

        # Survivors must report the peer fault as UNHEALTHY with fault info
        # (UNHEALTHY is the precondition for accepting scale_down)...
        for rank in survivor_ranks:
            assert faulted_results[rank]["status"] == "unhealthy", faulted_results[rank]
            assert faulted_results[rank].get("fault_info"), faulted_results[rank]
        # ...while the victim is DEAD (its own worker is gone).
        assert faulted_results[victim_rank]["status"] == "dead", faulted_results[victim_rank]

        # 3. The DEAD victim rejects retry: recovery requires UNHEALTHY.
        #    retry is accepted at the HTTP layer (202 = background dispatch)...
        request_id = _apply_ft(victim, "retry")["request_id"]
        # 4. ...but the rejection is recorded in /v1/fault_tolerance/status.
        ft_error = _wait_for_ft_apply_outcome(victim, request_id, FAULT_DETECTION_DEADLINE_S)
        assert ft_error is not None, "rejection was never recorded in /v1/fault_tolerance/status"
        assert "status is DEAD" in ft_error, ft_error

        # 5. scale_down to every survivor: remove the dead rank.
        for server in survivors:
            _apply_ft(server, "scale_down", {"removed_dp_ranks": [victim_rank]})

        # 6. Recovery completes: survivors return to healthy and serve again.
        recovered = _wait_for_engines(
            list(survivors),
            match_key="status",
            match_values={"healthy"},
            deadline_s=SCALE_DOWN_DEADLINE_S,
        )
        for rank, engine_status in zip(survivor_ranks, recovered):
            assert engine_status is not None, (
                f"survivor {rank} did not recover within {SCALE_DOWN_DEADLINE_S}s -- "
                "expert redistribution or weight reload likely failed"
            )
        _in_parallel(lambda s: _complete(s.get_client()), survivors)

        # 7. Factual answers on the survivors: the re-hosted experts must
        #    produce correct completions on the shrunken DP group.
        _assert_correct_answers(survivors)
