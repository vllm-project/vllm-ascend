# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""A5 MRV2 regression; named worker RPCs avoid callable serialization."""

import gc
import json
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import torch
from vllm import LLM, SamplingParams
from vllm.compilation.counter import compilation_counter
from vllm.sampling_params import RequestOutputKind

from vllm_ascend.device.hardware_profile import HardwareCapability, get_current_hardware_profile
from vllm_ascend.device_allocator.camem import ACL_GRAPH_TAG, CaMemAllocator

pytestmark = pytest.mark.skipif(
    not get_current_hardware_profile().supports(HardwareCapability.SLEEP_GRAPH_REUSE),
    reason="Graph reuse regression requires an A5 NPU",
)


class SleepGraphReuseProbe:
    # vLLM mixes this extension into the actual worker in spawned processes.
    model_runner: Any
    sleep_wakeup_manager: Any
    execute_model: Callable[..., Any]

    def sleep_graph_reuse_snapshot(self):
        manager = self.model_runner.cudagraph_manager
        reuse = self.sleep_wakeup_manager.graph_reuse
        graph_allocations = [
            data for data in CaMemAllocator.get_instance().pointer_to_data.values() if data.tag == ACL_GRAPH_TAG
        ]
        return {
            "graphs": [(desc.num_tokens, id(graph)) for desc, graph in manager.graphs.items()],
            "capture_calls": self.model_runner.sleep_capture_model_calls,
            "lower_captures": compilation_counter.num_cudagraph_captured,
            "replays": manager.sleep_graph_replay_count,
            "state": reuse.state if reuse else None,
            "reuses": reuse.reuses if reuse else 0,
            "recaptures": reuse.recaptures if reuse else 0,
            "pool_bytes": sum(data.handle[1] for data in graph_allocations),
            "pool_addresses": sorted(data.handle[2] for data in graph_allocations),
            "mapped_bytes": sum(data.handle[1] for data in graph_allocations if not data.is_asleep),
            "backup_bytes": sum(
                data.cpu_backup_tensor.nbytes for data in graph_allocations if data.cpu_backup_tensor is not None
            ),
            "free_bytes": torch.npu.mem_get_info()[0],
        }

    def sleep_graph_reuse_probe_inference(self):
        try:
            self.execute_model(None)
        except RuntimeError as error:
            assert "Inference is unavailable during sleep graph state" in str(error)
            return True
        raise AssertionError("Partly restored worker accepted inference")

    def sleep_graph_reuse_inject_fault(self, kind):
        """Inject only recoverable incompatibilities between staged wake calls."""
        from vllm_ascend.device_allocator import camem

        if kind == "parameter":
            for _, value in self.model_runner.model.named_parameters():
                if 0 < value.numel() <= 8192:
                    old = value.data_ptr()
                    with torch.no_grad():
                        value.data = value.detach().clone()
                    assert value.data_ptr() != old
                    return True
        elif kind == "workspace":
            for graph in self.model_runner.cudagraph_manager.graphs.values():
                for key, value in graph.capture_resources.items():
                    if isinstance(value, torch.Tensor):
                        replacement = torch.zeros_like(value)
                        assert replacement.data_ptr() != value.data_ptr()
                        graph.capture_resources[key] = replacement
                        return True
        elif kind == "event":
            for graph in self.model_runner.cudagraph_manager.graphs.values():
                for task in graph.tasks:
                    task.event = torch.npu.ExternalEvent()
                    return True
        elif kind == "graph_restore":
            handles = {
                data.handle[2]
                for data in CaMemAllocator.get_instance().pointer_to_data.values()
                if data.tag == ACL_GRAPH_TAG
            }
            assert handles
            original = camem.create_and_map

            def fail_once(handle):
                if handle[2] in handles:
                    camem.create_and_map = original
                    raise RuntimeError("injected graph-only restore failure")
                return original(handle)

            camem.create_and_map = fail_once
            return True
        raise AssertionError(f"No resource found for {kind}")


def _generate(model: LLM, size: int, sampling: SamplingParams):
    results = model.generate(["Explain how a computer stores information."] * size, sampling, use_tqdm=False)
    return [tuple(result.outputs[0].token_ids) for result in results]


def _make_model(monkeypatch, reuse_enabled=True):
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "1")
    # Device finalization can outlast the default best-effort exit window.
    # Bound worker teardown and explicitly wait for the EngineCore below.
    monkeypatch.setenv("VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS", "35")
    # Spawned workers must resolve this module both in the repo and standalone.
    monkeypatch.setenv("PYTHONPATH", str(Path(__file__).parent) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    return LLM(
        model=os.environ.get("VLLM_TEST_MODEL", "Qwen/Qwen2.5-0.5B-Instruct"),
        dtype="bfloat16",
        enable_sleep_mode=True,
        enable_prefix_caching=False,
        max_model_len=4096,
        max_num_seqs=8,
        max_num_batched_tokens=2048,
        kv_cache_memory_bytes=512 << 20,
        seed=42,
        worker_extension_cls=f"{__name__}.SleepGraphReuseProbe",
        compilation_config={"mode": 3, "cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [1, 2, 4, 8]},
        additional_config={
            "enable_cpu_binding": False,
            "rl_config": {"enabled": True, "sleep_mode_extra_cleanup": True, "sleep_mode_graph_reuse": reuse_enabled},
        },
    )


def _shutdown(model):
    core = model.llm_engine.engine_core
    processes = list(core.resources.engine_manager.processes)
    core.shutdown(timeout=45)
    print("SLEEP_GRAPH_SHUTDOWN", json.dumps({"engine_core_exitcodes": [process.exitcode for process in processes]}))
    assert processes, "Expected real EngineCore processes in this spawned-worker regression"
    assert all(process.exitcode == 0 for process in processes)


def _check_request_boundary(model, sampling, baseline):
    engine = model.llm_engine
    prompt = "Explain how a computer stores information."
    expected = _generate(model, 1, SamplingParams(temperature=0, max_tokens=256, ignore_eos=True))[0]
    # Offline generate changes its params to FINAL_ONLY. Fresh cumulative
    # params are required to observe a genuinely unfinished engine request.
    engine.add_request(
        "sleep-inflight",
        prompt,
        SamplingParams(temperature=0, max_tokens=256, ignore_eos=True, output_kind=RequestOutputKind.CUMULATIVE),
    )
    first: list[Any] = []
    deadline = time.monotonic() + 120
    while not first:
        assert time.monotonic() < deadline, "Timed out waiting for an in-flight decode before sleep"
        first = [out for out in engine.step() if out.request_id == "sleep-inflight" and out.outputs[0].token_ids]
    assert not first[-1].finished
    before = model.collective_rpc("sleep_graph_reuse_snapshot")[0]
    model.sleep(level=1, mode="wait")
    asleep = model.collective_rpc("sleep_graph_reuse_snapshot")[0]
    engine.add_request(
        "sleep-queued",
        prompt,
        SamplingParams(
            temperature=0, max_tokens=sampling.max_tokens, ignore_eos=True, output_kind=RequestOutputKind.CUMULATIVE
        ),
    )
    queued = model.collective_rpc("sleep_graph_reuse_snapshot")[0]
    assert queued["replays"] == asleep["replays"]
    model.wake_up()
    finished: dict[str, tuple[int, ...]] = {}
    deadline = time.monotonic() + 120
    while engine.has_unfinished_requests():
        assert time.monotonic() < deadline, f"Timed out draining post-wake requests: {sorted(finished)}"
        for out in engine.step():
            if out.finished:
                finished[out.request_id] = tuple(out.outputs[0].token_ids)
    assert finished == {"sleep-inflight": expected, "sleep-queued": baseline[1][0]}
    after = model.collective_rpc("sleep_graph_reuse_snapshot")[0]
    assert after["graphs"] == before["graphs"]
    assert after["lower_captures"] == before["lower_captures"]


@pytest.mark.parametrize("level", [1, 2])
def test_mrv2_reuses_graphs_after_sleep(monkeypatch, level):
    model = _make_model(monkeypatch)
    sampling = SamplingParams(temperature=0, max_tokens=16, ignore_eos=True)

    try:
        baseline = {size: _generate(model, size, sampling) for size in (1, 2, 4, 8)}
        for cycle in range(10):
            before = model.collective_rpc("sleep_graph_reuse_snapshot")[0]
            assert sorted(size for size, _ in before["graphs"]) == [1, 2, 4, 8]
            assert before["pool_bytes"] > 0
            model.sleep(level=level)
            asleep = model.collective_rpc("sleep_graph_reuse_snapshot")[0]
            assert asleep["mapped_bytes"] == 0 and asleep["backup_bytes"] == before["pool_bytes"]
            assert asleep["free_bytes"] - before["free_bytes"] >= before["pool_bytes"]
            assert model.collective_rpc("sleep_graph_reuse_probe_inference") == [True]
            if level == 2:
                model.wake_up(tags=["weights"])
                assert model.collective_rpc("sleep_graph_reuse_probe_inference") == [True]
                model.collective_rpc("reload_weights")
                model.wake_up(tags=["kv_cache"])
            else:
                model.wake_up()
            assert {size: _generate(model, size, sampling) for size in baseline} == baseline
            after = model.collective_rpc("sleep_graph_reuse_snapshot")[0]
            assert after["graphs"] == before["graphs"]
            assert after["pool_addresses"] == before["pool_addresses"]
            assert after["capture_calls"] == before["capture_calls"]
            assert after["lower_captures"] == before["lower_captures"]
            assert after["replays"] > before["replays"]
            assert after["state"] == "READY" and after["reuses"] == before["reuses"] + 1
            print(
                "SLEEP_GRAPH_REGRESSION", json.dumps({"level": level, "cycle": cycle, "before": before, "after": after})
            )
        if level == 1:
            _check_request_boundary(model, sampling, baseline)
            for kind in ("parameter", "workspace", "event", "graph_restore"):
                before = model.collective_rpc("sleep_graph_reuse_snapshot")[0]
                model.sleep(level=1)
                model.wake_up(tags=["weights"])
                assert model.collective_rpc("sleep_graph_reuse_inject_fault", args=(kind,)) == [True]
                model.wake_up(tags=["kv_cache"])
                # Observe capture before generation to exclude lazy first-request capture.
                restored = model.collective_rpc("sleep_graph_reuse_snapshot")[0]
                assert restored["state"] == "READY"
                assert restored["recaptures"] == before["recaptures"] + 1
                assert restored["capture_calls"] > before["capture_calls"]
                assert restored["lower_captures"] > before["lower_captures"]
                assert {size: _generate(model, size, sampling) for size in baseline} == baseline
                after = model.collective_rpc("sleep_graph_reuse_snapshot")[0]
                assert after["lower_captures"] == restored["lower_captures"]
                assert after["replays"] > restored["replays"]
                print(
                    "SLEEP_GRAPH_FALLBACK",
                    json.dumps({"fault": kind, "before": before, "restored": restored, "after": after}),
                )
    finally:
        _shutdown(model)
        del model
        gc.collect()


def test_mrv2_reuse_disabled_recaptures(monkeypatch):
    model = _make_model(monkeypatch, reuse_enabled=False)
    sampling = SamplingParams(temperature=0, max_tokens=16, ignore_eos=True)
    try:
        baseline = _generate(model, 4, sampling)
        for level in (1, 2):
            before = model.collective_rpc("sleep_graph_reuse_snapshot")[0]
            assert before["state"] is None
            model.sleep(level=level)
            if level == 2:
                model.wake_up(tags=["weights"])
                model.collective_rpc("reload_weights")
                model.wake_up(tags=["kv_cache"])
            else:
                model.wake_up()
            restored = model.collective_rpc("sleep_graph_reuse_snapshot")[0]
            assert restored["capture_calls"] > before["capture_calls"]
            assert restored["lower_captures"] > before["lower_captures"]
            assert _generate(model, 4, sampling) == baseline
    finally:
        _shutdown(model)
        del model
        gc.collect()
