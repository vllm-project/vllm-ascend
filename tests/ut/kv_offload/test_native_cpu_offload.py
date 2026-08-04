# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from vllm.v1.kv_offload.config import (
    OffloadingCacheConfig,
    OffloadingConfig,
    OffloadingGroupConfig,
    OffloadingModelConfig,
    OffloadingParallelConfig,
)
from vllm.v1.kv_offload.cpu.gpu_worker import CPUOffloadingWorker
from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager
from vllm.v1.kv_offload.factory import OffloadingSpecFactory

from vllm_ascend.kv_offload.native.cpu_npu import NPUOffloadingWorker
from vllm_ascend.kv_offload.native.npu import NPUOffloadingSpec


def _make_config(extra_config: dict[str, object]) -> OffloadingConfig:
    return OffloadingConfig(
        groups=(
            OffloadingGroupConfig(
                tokens_per_block=16,
                layer_names=("model.layers.0.self_attn",),
            ),
        ),
        worker_kv_bytes_per_block=64,
        enable_kv_cache_events=False,
        extra_config=extra_config,
        engine_id="test-engine",
        model=OffloadingModelConfig(
            name="test-model",
            dtype="bfloat16",
        ),
        cache=OffloadingCacheConfig(
            tokens_per_hash=16,
            blocks_per_chunk=2,
        ),
        parallel=OffloadingParallelConfig(
            rank=0,
            world_size=2,
            tp_size=2,
            pp_size=1,
            pcp_size=1,
            dcp_size=1,
            data_parallel_index=0,
            is_parallelism_agnostic=True,
        ),
    )


def test_npu_offloading_spec_uses_upstream_cpu_manager() -> None:
    bytes_per_chunk = 64 * 2 * 2
    spec = NPUOffloadingSpec(_make_config({"cpu_bytes_to_use": 10 * bytes_per_chunk}))

    assert spec.num_blocks == 10
    assert isinstance(spec.get_manager(), CPUOffloadingManager)


def test_npu_offloading_spec_supports_legacy_num_cpu_blocks() -> None:
    extra_config: dict[str, object] = {"num_cpu_blocks": 10}
    spec = NPUOffloadingSpec(_make_config(extra_config))

    assert spec.num_blocks == 10
    assert extra_config["cpu_bytes_to_use"] == 10 * 64 * 2 * 2


def test_npu_offloading_spec_loads_through_vllm_factory() -> None:
    spec_cls = OffloadingSpecFactory.get_spec_cls(
        {
            "spec_name": "NPUOffloadingSpec",
            "spec_module_path": "vllm_ascend.kv_offload.native.npu",
        }
    )

    assert spec_cls is NPUOffloadingSpec


def test_npu_worker_reuses_upstream_worker_protocol() -> None:
    assert issubclass(NPUOffloadingWorker, CPUOffloadingWorker)


def test_npu_spec_caches_worker_without_upstream_platform_gate(monkeypatch) -> None:
    spec = NPUOffloadingSpec(_make_config({"cpu_bytes_to_use": 1024}))
    worker = object()
    create_calls = 0

    def create_worker(kv_caches):
        nonlocal create_calls
        create_calls += 1
        return worker

    monkeypatch.setattr(spec, "create_worker", create_worker)
    kv_caches = object()

    assert spec.get_worker(kv_caches) is worker
    assert spec.get_worker(kv_caches) is worker
    assert create_calls == 1
