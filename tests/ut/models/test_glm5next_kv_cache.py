# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-Next cache specs and compressed-cache addressing."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from vllm.v1.core.single_type_kv_cache_manager import SlidingWindowManager
from vllm.v1.kv_cache_spec_registry import KVCacheSpecRegistry

from vllm_ascend.core.kv_cache_interface import (
    AscendMLAAttentionSpec,
    register_ascend_kv_cache_specs,
)
from vllm_ascend.models.glm5next.kv_cache import (
    AscendIndexerKPoolStateSpec,
    Glm5NextIndexerCache,
    Glm5NextStateCache,
    format_indexer_kpool_slot_mapping,
)


def test_state_uses_sliding_pages_and_full_precision():
    register_ascend_kv_cache_specs()
    spec = AscendIndexerKPoolStateSpec(
        block_size=4,
        sliding_window=4,
        num_kv_heads=1,
        head_size=256,
        dtype=torch.float32,
    )
    assert spec.page_size_bytes == 4096
    assert KVCacheSpecRegistry.get_manager_class(spec) is SlidingWindowManager
    assert spec.max_admission_blocks_per_request(16, 1024) > 1
    context_parallel_config = SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=1024),
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=2,
            prefill_context_parallel_size=2,
        ),
    )
    assert (
        spec.max_memory_usage_bytes(context_parallel_config)
        == spec.page_size_bytes
    )


@pytest.mark.parametrize(
    ("dtype", "block_size", "sliding_window"),
    [
        (torch.bfloat16, 4, 4),
        (torch.float32, 8, 4),
    ],
)
def test_invalid_state_layout_is_rejected(dtype, block_size, sliding_window):
    with pytest.raises(ValueError):
        AscendIndexerKPoolStateSpec(
            block_size=block_size,
            sliding_window=sliding_window,
            num_kv_heads=1,
            head_size=256,
            dtype=dtype,
        )


def test_completed_pool_slots_preserve_logical_block_padding():
    slots = torch.tensor([0, 14, 15, 16, 127, 128, 143, -1])
    positions = torch.tensor([0, 14, 15, 16, 127, 128, 143, 15])
    actual = format_indexer_kpool_slot_mapping(slots, positions, 128, 16)
    assert actual.tolist() == [-1, -1, 0, -1, 7, -1, 8, -1]


@pytest.mark.parametrize("ratio", [0, 1, 3])
def test_invalid_pool_geometry_is_rejected(ratio):
    with pytest.raises(ValueError):
        format_indexer_kpool_slot_mapping(
            torch.tensor([0]), torch.tensor([0]), 128, ratio
        )


def test_model_cache_layers_publish_source_compatible_specs():
    current_config = SimpleNamespace(
        parallel_config=SimpleNamespace(pipeline_parallel_size=2),
        compilation_config=SimpleNamespace(static_forward_context={}),
    )
    cache_config = SimpleNamespace(block_size=256)
    with patch(
        "vllm_ascend.models.glm5next.kv_cache.get_current_vllm_config",
        return_value=current_config,
    ):
        indexer = Glm5NextIndexerCache(
            head_dim=128,
            dtype=torch.bfloat16,
            cache_role="indexer",
            cache_config=cache_config,
            prefix="model.layers.0.indexer.k_cache",
            compress_ratio=16,
        )
        state = Glm5NextStateCache(
            state_dim=256,
            dtype=torch.float32,
            compress_ratio=16,
            cache_config=cache_config,
            prefix="model.layers.0.indexer.state_cache",
        )

    indexer_spec = indexer.get_kv_cache_spec(None)
    state_spec = state.get_kv_cache_spec(None)
    assert len(indexer.kv_cache) == len(state.kv_cache) == 2
    assert isinstance(indexer_spec, AscendMLAAttentionSpec)
    assert indexer_spec.block_size == 256
    assert indexer_spec.storage_block_size == 16
    assert indexer_spec.compress_ratio == 16
    assert indexer_spec.model_version == "glm5_next"
    assert indexer_spec.indexes_kv_by_block_stride
    assert state_spec.block_size == state_spec.sliding_window == 16
    assert state_spec.head_size == 256
    assert state_spec.dtype == torch.float32
    assert state_spec.indexes_kv_by_block_stride
    assert set(current_config.compilation_config.static_forward_context) == {
        indexer.prefix,
        state.prefix,
    }
