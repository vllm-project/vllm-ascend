# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Store initialization against the real CANN-lane CacheConfig schema."""

from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.config import CacheConfig
from vllm.v1.core.kv_cache_utils import resolve_kv_cache_block_sizes
from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec, SlidingWindowSpec

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store import pool_scheduler, pool_worker


@pytest.mark.parametrize("prefix_match_unit", [None, 32])
@pytest.mark.parametrize("schema", ["legacy", "modern", "modern_unset"])
def test_store_worker_and_scheduler_use_matching_hash_units(schema, prefix_match_unit):
    register_all_kvcache_specs(None)
    if schema == "legacy":
        if "prefix_match_unit" not in CacheConfig.__dataclass_fields__:
            pytest.skip("Requires the vLLM 0.30 CANN CacheConfig schema")
        cache_config = CacheConfig(block_size=128, prefix_match_unit=prefix_match_unit, enable_prefix_caching=True)
        assert not hasattr(cache_config, "hash_block_size")
    else:
        cache_config = SimpleNamespace(
            block_size=128,
            enable_prefix_caching=True,
            hash_block_size=prefix_match_unit if schema == "modern" else None,
            prefix_match_unit=64 if schema == "modern" else prefix_match_unit,
        )
    model = MagicMock()
    model.model = "test/store-cache"
    model.use_mla = False
    model.hf_config = model.hf_text_config = SimpleNamespace()
    model.get_num_layers.return_value = 2
    model.get_total_num_kv_heads.return_value = 8
    model.max_model_len = 4096
    config = SimpleNamespace(
        cache_config=cache_config,
        model_config=model,
        parallel_config=SimpleNamespace(
            rank=0,
            data_parallel_rank=0,
            world_size=2,
            pipeline_parallel_size=1,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
        ),
        scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=False),
        kv_transfer_config=SimpleNamespace(
            kv_role="kv_consumer",
            kv_connector_extra_config={"backend": "mooncake"},
            get_from_extra_config=lambda key, default: default,
        ),
        speculative_config=None,
        kv_events_config=None,
    )
    full = FullAttentionSpec(block_size=128, num_kv_heads=1, head_size=16, dtype=torch.bfloat16)
    swa = SlidingWindowSpec(block_size=256, num_kv_heads=1, head_size=16, dtype=torch.bfloat16, sliding_window=128)
    plan = KVCacheConfig(
        num_blocks=16,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(["full"], full), KVCacheGroupSpec(["swa"], swa)],
    )
    with ExitStack() as stack:
        stack.enter_context(patch.object(pool_scheduler, "LookupKeyClient"))
        stack.enter_context(patch.object(pool_worker, "get_tensor_model_parallel_rank", return_value=0))
        stack.enter_context(patch.object(pool_worker, "get_tensor_model_parallel_world_size", return_value=2))
        stack.enter_context(
            patch.object(pool_worker, "get_pcp_group", return_value=SimpleNamespace(world_size=1, rank_in_group=0))
        )
        stack.enter_context(patch.object(pool_worker, "get_decode_context_model_parallel_world_size", return_value=1))
        stack.enter_context(patch.object(pool_worker, "get_decode_context_model_parallel_rank", return_value=0))
        backend_module = SimpleNamespace(MooncakeBackend=MagicMock())
        stack.enter_context(patch.object(pool_worker.importlib, "import_module", return_value=backend_module))
        scheduler = pool_scheduler.KVPoolScheduler(config, use_layerwise=False, kv_cache_config=plan)
        worker = pool_worker.KVPoolWorker(config, use_layerwize=False, kv_cache_config=plan)

    expected = prefix_match_unit or (64 if schema == "modern" else 128)
    assert scheduler.hash_block_size == worker.hash_block_size == expected
    assert worker.token_database.hash_block_size == expected
    assert worker.cache_coordinator.hash_block_size == expected
    if schema == "legacy":
        assert expected == resolve_kv_cache_block_sizes(plan, config)[1]
