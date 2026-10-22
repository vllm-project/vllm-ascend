# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import ctypes
from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch
from vllm.config import CacheConfig, KVTransferConfig
from vllm.distributed.kv_transfer.kv_connector.v1 import KVConnectorRole
from vllm.utils.math_utils import round_up
from vllm.v1.core.kv_cache_utils import generate_scheduler_kv_cache_config
from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheGroupSpec, KVCacheTensor
from vllm.v1.kv_offload.base import GPULoadStoreSpec
from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec
from vllm.v1.kv_offload.cpu.shared_offload_region import SharedOffloadRegion

from vllm_ascend.distributed.kv_transfer.kv_pool.kv_offload.native import cpu_npu
from vllm_ascend.distributed.kv_transfer.kv_pool.kv_offload.native.offloading_connector import AscendOffloadingConnector
from vllm_ascend.models.glm5next.cache_config import get_glm5_next_kv_cache_config, get_glm5_next_kv_cache_groups


def _contract_layout(layout):
    names = {
        "main": "model.layers.3.attn",
        "indexer": "model.layers.3.indexer.k_cache",
        "tail": "model.layers.3.indexer.tail_cache",
        "mamba": "model.layers.0.linear_attn",
    }
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            model="test-glm",
            dtype=torch.bfloat16,
            use_mla=True,
            max_model_len=2048,
            get_total_num_kv_heads=lambda: 1,
        ),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=2,
            pipeline_parallel_size=1,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
            cp_kv_cache_interleave_size=1,
            world_size=2,
            rank=0,
            distributed_executor_backend="mp",
            nnodes_within_dp=1,
            data_parallel_index=0,
            data_parallel_size=1,
            data_parallel_rank_local=None,
        ),
        scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=False),
        cache_config=CacheConfig(block_size=40, enable_prefix_caching=True, mamba_cache_mode="align"),
        speculative_config=SimpleNamespace(num_speculative_tokens=5, use_eagle_block_drop=lambda: False),
        kv_events_config=None,
        use_v2_model_runner=True,
    )
    specs = {names[name]: spec for name, spec in layout.specs.items()}
    specs[names["mamba"]] = replace(specs[names["mamba"]], mamba_cache_mode="align")
    groups = get_glm5_next_kv_cache_groups(config, specs)
    # Keep scratch ahead of selected groups and reverse their registration order.
    groups = [groups[1], groups[2], groups[0]]
    pool_bytes = sum(page for _, _, page in layout.backings)
    plan = get_glm5_next_kv_cache_config(config, groups, 3 * pool_bytes)
    ignored = FullAttentionSpec(block_size=40, num_kv_heads=1, head_size=4, dtype=torch.bfloat16)
    plan.kv_cache_groups.append(KVCacheGroupSpec(["ignored"], ignored, enable_kv_transfer=False))
    plan.kv_cache_tensors.append(
        KVCacheTensor(
            size=3 * ignored.page_size_bytes, layers=["ignored"], layer_stride=0, block_stride=ignored.page_size_bytes
        )
    )
    chunk_bytes = round_up(pool_bytes * 8 * 2, SharedOffloadRegion.BLOCK_SIZE_ALIGNMENT)
    config.kv_transfer_config = KVTransferConfig(
        kv_connector="OffloadingConnector",
        kv_role="kv_both",
        kv_connector_extra_config={
            "spec_name": "NPUOffloadingSpec",
            "spec_module_path": "vllm_ascend.distributed.kv_transfer.kv_pool.kv_offload.native.npu",
            "cpu_bytes_to_use": 3 * chunk_bytes,
            "blocks_per_chunk": 8,
        },
    )
    caches = {names[name]: cache for name, cache in layout.caches.items()}
    return config, plan, caches, pool_bytes, chunk_bytes


def test_native_selected_groups_budget_and_transfer_contract(monkeypatch, glm_shared_physical_pages):
    register_all_kvcache_specs(None)
    config, plan, caches, pool_bytes, chunk_bytes = _contract_layout(glm_shared_physical_pages)
    calls = []

    class CPUBackedNPUTensor(torch.Tensor):
        @property
        def device(self):
            return SimpleNamespace(type="npu")

    handler_init = cpu_npu.SingleDirectionNPUOffloadingHandler.__init__

    def init_cpu_backed_handler(self, npu_tensors, **kwargs):
        handler_init(self, [tensor.as_subclass(CPUBackedNPUTensor) for tensor in npu_tensors], **kwargs)

    def copy_bytes(src, dst, sizes, direction):
        calls.append((src.tolist(), dst.tolist(), sizes.tolist(), direction))
        for source, target, size in zip(*calls[-1][:3]):
            ctypes.memmove(target, source, size)

    monkeypatch.setattr(cpu_npu, "is_pin_memory_available", lambda: False)
    monkeypatch.setattr(cpu_npu.SingleDirectionNPUOffloadingHandler, "__init__", init_cpu_backed_handler)
    monkeypatch.setattr(cpu_npu.torch.npu, "Stream", MagicMock())
    monkeypatch.setattr(cpu_npu.torch.npu, "Event", MagicMock())
    monkeypatch.setattr(cpu_npu.torch.npu, "current_stream", MagicMock())
    monkeypatch.setattr(cpu_npu.torch.npu, "stream", lambda _: nullcontext())
    monkeypatch.setattr(cpu_npu.torch.ops._C_ascend, "swap_blocks_batch", copy_bytes, raising=False)
    scheduler_plan = generate_scheduler_kv_cache_config([plan, plan])
    scheduler = AscendOffloadingConnector(config, KVConnectorRole.SCHEDULER, scheduler_plan)
    connector = AscendOffloadingConnector(config, KVConnectorRole.WORKER, plan)
    try:
        worker = connector.connector_worker
        spec = worker.spec
        assert tuple(group.group_id for group in spec.config.groups) == (1, 2)
        assert tuple(group.group_idx for group in scheduler.scheduler.config.kv_group_configs) == (1, 2)
        assert spec.config.worker_kv_bytes_per_block == pool_bytes
        assert scheduler.scheduler.manager._num_chunks == 3
        assert spec.kv_bytes_per_chunk == chunk_bytes and spec.num_chunks == 3
        assert scheduler.scheduler.manager._num_chunks == spec.num_chunks
        worker.register_kv_caches(caches)
        handler = worker.worker._store_handler
        assert [[ref.tensor_idx for ref in group] for group in handler.kv_cache_groups_data_refs] == [[0], [0, 1]]
        assert sum(tensor.numel() for tensor in handler.dst_tensors) == spec.num_chunks * pool_bytes * 8
        gpu = GPULoadStoreSpec([1, 2, 0], group_sizes=[1, 2], block_indices=[0, 0])
        assert worker.worker.submit_store(7, gpu, CPULoadStoreSpec([2, 0]))
        main, indexer = handler.src_tensors
        cpu_main, cpu_indexer = handler.dst_tensors
        main_bytes, indexer_bytes = main.shape[1], indexer.shape[1]
        assert calls == [
            (
                [
                    main[1].data_ptr(),
                    main[2].data_ptr(),
                    main[0].data_ptr(),
                    indexer[2].data_ptr(),
                    indexer[0].data_ptr(),
                ],
                [
                    cpu_main[2].data_ptr(),
                    cpu_main[0].data_ptr(),
                    cpu_main[0].data_ptr() + main_bytes,
                    cpu_indexer[0].data_ptr(),
                    cpu_indexer[0].data_ptr() + indexer_bytes,
                ],
                [main_bytes] * 3 + [indexer_bytes] * 2,
                cpu_npu.DIRECTION_D2H,
            )
        ]
        torch.testing.assert_close(cpu_main[2, :main_bytes], main[1].as_subclass(torch.Tensor), rtol=0, atol=0)
        torch.testing.assert_close(cpu_indexer[0, :indexer_bytes], indexer[2].as_subclass(torch.Tensor), rtol=0, atol=0)
    finally:
        connector.shutdown()
        scheduler.shutdown()


def test_native_packed_backing_retains_upstream_budget(glm_shared_physical_pages):
    config, plan, _, pool_bytes, _ = _contract_layout(glm_shared_physical_pages)
    for tensor in plan.kv_cache_tensors:
        tensor.size = plan.num_blocks * pool_bytes
    plan.kv_cache_tensors[1].offset = plan.kv_cache_tensors[0].block_stride * plan.num_blocks
    connector = AscendOffloadingConnector(config, KVConnectorRole.WORKER, plan)
    try:
        assert connector.connector_worker.spec.config.worker_kv_bytes_per_block == pool_bytes
    finally:
        connector.shutdown()
