# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Offload keys must use the actual PP stage when PCP is also enabled."""

from types import SimpleNamespace

import pytest
import torch
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.layerwise_cache_layout import (
    LayerwiseReuseLayout,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.pool_worker import LayerwiseKVPoolWorker


@pytest.mark.parametrize("pp_rank,start,end", [(0, 0, 24), (1, 24, 45)])
@pytest.mark.parametrize("pcp_rank", range(4))
def test_offload_uses_cache_stage_not_global_rank(pp_rank, start, end, pcp_rank):
    names = [f"model.layers.{i}.self_attn.attn" for i in range(start, end)]
    spec = FullAttentionSpec(block_size=16, num_kv_heads=1, head_size=8, dtype=torch.bfloat16)
    worker = LayerwiseKVPoolWorker.__new__(LayerwiseKVPoolWorker)
    worker.kv_cache_config = KVCacheConfig(
        num_blocks=2,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(layer_names=names, kv_cache_spec=spec)],
    )
    worker._layerwise_reuse_layout = LayerwiseReuseLayout(
        layers=tuple(range(start, end)),
        cache_layer_indices=dict(zip(names, range(start, end))),
        kv_cache_tensors=(),
        cache_view_sizes={},
        prefetch_layer_map={start + 4: (start + 1,)},
        independent_layers=[start],
        num_prefetch_layers=3,
        reused_cache_names=frozenset(names[1:]),
        draft_entry_names=frozenset(),
    )
    worker.pp_size = 2
    worker.pp_rank = pp_rank
    worker.pcp_size = 4
    worker.pcp_rank = pcp_rank
    # Reproduce the old ModelConfig rank formula: rank // TP % PP.
    guessed_stage = (pp_rank * 4 + pcp_rank) % 2
    worker.pp_layer_offset = 0 if guessed_stage == 0 else 24
    worker.num_layers = 24 if guessed_stage == 0 else 21
    worker.total_layers = 45
    worker.hf_config = SimpleNamespace(num_hidden_layers=45)
    worker.num_kv_cache_groups = 1
    worker.cacheable_group_ids = [0]
    worker.use_layerwise_transfer = worker.use_layerwise = True
    worker._extra_config = {}

    worker._init_layerwise_config()

    assert worker.layerwise_key_layer_offset == start
    assert worker.layerwise_key_layers == worker.num_layers == end - start
    assert worker.prefetch_layer_map == {4: (1,)}
    assert worker.independent_layers == [0]
    assert worker._reused_layers_by_group == {0: set(range(1, end - start))}
