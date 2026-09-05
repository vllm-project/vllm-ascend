# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project


from unittest.mock import MagicMock

import pytest
from vllm.utils.math_utils import round_up
from vllm.v1.kv_offload.config import (
    OffloadingCacheConfig,
    OffloadingConfig,
    OffloadingGroupConfig,
    OffloadingModelConfig,
    OffloadingParallelConfig,
)
from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager
from vllm.v1.kv_offload.factory import OffloadingSpecFactory

from vllm_ascend.distributed.kv_transfer.kv_pool.kv_offload.native import npu as module
from vllm_ascend.distributed.kv_transfer.kv_pool.kv_offload.native.npu import NPUOffloadingSpec
from vllm_ascend.utils import vllm_version_is


@pytest.mark.parametrize("blocks", [0, -1, "-2"])
def test_legacy_block_capacity_must_be_positive(blocks):
    with pytest.raises(ValueError, match="greater than 0"):
        module._normalize_legacy_num_blocks(_make_config({"num_cpu_blocks": blocks}), 64)


def test_without_legacy_capacity_keeps_original_config():
    config = _make_config({})
    result, legacy = module._normalize_legacy_num_blocks(config, 64)
    assert result is config and legacy is None


def test_create_worker_passes_cache_and_capacity(monkeypatch):
    constructor = MagicMock()
    monkeypatch.setattr(module, "NPUOffloadingWorker", constructor)
    spec = NPUOffloadingSpec(_make_config({"num_cpu_blocks": 3}))
    caches = object()
    assert spec.create_worker(caches) is constructor.return_value
    constructor.assert_called_once_with(kv_caches=caches, blocks_per_chunk=2, num_cpu_blocks=3)


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
            **(
                {}
                if vllm_version_is("0.27.1")
                else {
                    "data_parallel_size": 1,
                    "data_parallel_rank_local": None,
                }
            ),
        ),
    )


def test_npu_offloading_spec_uses_upstream_cpu_manager() -> None:
    bytes_per_chunk = 64 * 2 * 2
    aligned_bytes_per_chunk = round_up(
        bytes_per_chunk,
        NPUOffloadingSpec.BLOCK_SIZE_ALIGNMENT,
    )
    spec = NPUOffloadingSpec(_make_config({"cpu_bytes_to_use": 10 * aligned_bytes_per_chunk}))

    assert spec.num_blocks == 10
    assert isinstance(spec.get_manager(), CPUOffloadingManager)


def test_npu_offloading_spec_supports_legacy_num_cpu_blocks() -> None:
    extra_config: dict[str, object] = {"num_cpu_blocks": 10}
    spec = NPUOffloadingSpec(_make_config(extra_config))
    aligned_bytes_per_chunk = round_up(
        64 * 2 * 2,
        NPUOffloadingSpec.BLOCK_SIZE_ALIGNMENT,
    )

    assert spec.num_blocks == 10
    assert spec.extra_config["cpu_bytes_to_use"] == 10 * aligned_bytes_per_chunk
    assert "cpu_bytes_to_use" not in extra_config


def test_legacy_num_cpu_blocks_is_preserved_on_scheduler() -> None:
    config = _make_config({"num_cpu_blocks": 10})
    object.__setattr__(config, "worker_kv_bytes_per_block", 0)

    spec = NPUOffloadingSpec(config)

    assert spec.num_blocks == 10


def test_npu_offloading_spec_loads_through_vllm_factory() -> None:
    spec_cls = OffloadingSpecFactory.get_spec_cls(
        {
            "spec_name": "NPUOffloadingSpec",
            "spec_module_path": "vllm_ascend.distributed.kv_transfer.kv_pool.kv_offload.native.npu",
        }
    )

    assert spec_cls is NPUOffloadingSpec


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
