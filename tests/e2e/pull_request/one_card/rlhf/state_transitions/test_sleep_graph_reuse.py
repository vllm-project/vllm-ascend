# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""A5 MRV2 regression; named worker RPCs avoid callable serialization."""

import gc
import os
from pathlib import Path

import pytest
import torch
from vllm import LLM, SamplingParams
from vllm.compilation.counter import compilation_counter

from vllm_ascend.device.hardware_profile import HardwareCapability, get_current_hardware_profile
from vllm_ascend.device_allocator.camem import ACL_GRAPH_TAG, CaMemAllocator

pytestmark = pytest.mark.skipif(
    not get_current_hardware_profile().supports(HardwareCapability.SLEEP_GRAPH_REUSE),
    reason="Graph reuse regression requires an A5 NPU",
)


class SleepGraphReuseProbe:
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
            "state": reuse.state,
            "reuses": reuse.reuses,
            "recaptures": reuse.recaptures,
            "pool_bytes": sum(data.handle[1] for data in graph_allocations),
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


def _generate(model: LLM, size: int, sampling: SamplingParams):
    results = model.generate(["Explain how a computer stores information."] * size, sampling, use_tqdm=False)
    return [tuple(result.outputs[0].token_ids) for result in results]


@pytest.mark.parametrize("level", [1, 2])
def test_mrv2_reuses_graphs_after_sleep(monkeypatch, level):
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "1")
    # Device finalization can outlast the default best-effort exit window.
    # Bound worker teardown and explicitly wait for the EngineCore below.
    monkeypatch.setenv("VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS", "35")
    # Spawned workers must resolve this module both in the repo and standalone.
    monkeypatch.setenv("PYTHONPATH", str(Path(__file__).parent) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    model = LLM(
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
            "rl_config": {"enabled": True, "sleep_mode_extra_cleanup": True, "sleep_mode_graph_reuse": True},
        },
    )
    sampling = SamplingParams(temperature=0, max_tokens=16, ignore_eos=True)

    try:
        baseline = {size: _generate(model, size, sampling) for size in (1, 2, 4, 8)}
        for _ in range(2):
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
            assert after["capture_calls"] == before["capture_calls"]
            assert after["lower_captures"] == before["lower_captures"]
            assert after["replays"] > before["replays"]
            assert after["state"] == "READY" and after["reuses"] == before["reuses"] + 1
    finally:
        core = model.llm_engine.engine_core
        processes = list(core.resources.engine_manager.processes)
        core.shutdown(timeout=45)
        assert all(process.exitcode == 0 for process in processes)
        del model
        gc.collect()
