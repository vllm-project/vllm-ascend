# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared cache construction for Aurora worker and native-operator tests."""

from types import SimpleNamespace

import torch
from vllm.v1.kv_cache_interface import KVCacheConfig

from tests.deepseek_v41_reference import build_v41_cache_specs
from vllm_ascend.core.deepseek_v41 import (
    DeepseekV41DraftSWASpec,
    allocate_cache_config,
    cache_slots_from_groups,
    group_cache_specs,
    make_cache_groups,
    pool_bytes_per_block,
    reshape_cache,
)


def make_cache_config(num_blocks, *, block_size=128, head_size=512, index_size=128, draft_layers=0):
    config = dict(
        num_hidden_layers=40,
        compress_ratios=[0, 0] + [2] * 18 + [1] * 20,
        kv_source_layers=[2, 8, 14, 20],
        index_source_layers=[2, 8, 14, 20, 24, 28, 32, 36],
        candidate_source_layer=20,
        candidate_topk_blocks=64,
        candidate_block_size=8,
        index_topk=512,
        engram_layer_ids=[1, 14],
        sliding_window=128,
        head_dim=head_size,
        index_head_dim=index_size,
    )
    runtime = SimpleNamespace(cache_config=SimpleNamespace(block_size=block_size, num_gpu_blocks_override=None))
    specs = build_v41_cache_specs(config, runtime)
    for stage in range(draft_layers):
        specs[f"mtp.{stage}.self_attn.swa_cache"] = DeepseekV41DraftSWASpec(
            block_size=block_size,
            num_kv_heads=1,
            head_size=head_size,
            dtype=torch.bfloat16,
            sliding_window=config["sliding_window"],
            cache_dtype_str="bfloat16",
            model_version="deepseek_v4",
        )
    groups = make_cache_groups(group_cache_specs(specs))
    blocks, tensors = allocate_cache_config(runtime, groups, num_blocks * pool_bytes_per_block(groups))
    return KVCacheConfig(num_blocks=blocks, kv_cache_tensors=tensors, kv_cache_groups=groups)


def allocate_cache_views(config, device="cpu"):
    specs = {n: s for g in config.kv_cache_groups for n, s in g.kv_cache_spec.kv_cache_specs.items()}
    backings, caches = [], {}
    for allocation, slot in zip(config.kv_cache_tensors, cache_slots_from_groups(config.kv_cache_groups)):
        raw = torch.zeros(allocation.size, dtype=torch.uint8, device=device)
        backings.append(raw)
        for placement in slot.placements:
            caches[placement.name] = reshape_cache(
                raw,
                specs[placement.name],
                num_blocks=config.num_blocks,
                offset=placement.offset,
                block_stride=slot.page_size_bytes,
            )
    return backings, caches
