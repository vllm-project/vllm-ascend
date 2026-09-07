# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass, replace
from types import SimpleNamespace

import pytest
import torch
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheGroupSpec, MambaSpec, UniformTypeKVCacheSpecs

from vllm_ascend.core.kv_cache_capacity import (
    replicated_draft_max_memory_usage_bytes,
    replicated_draft_pool_bytes_per_block,
)

TARGET_PAGE_BYTES = 940032
DRAFT_PAGE_BYTES = TARGET_PAGE_BYTES * 8


@dataclass(frozen=True, kw_only=True)
class ReplicatedDraftSpec(FullAttentionSpec):
    dcp_replication_size: int = 8


def make_groups():
    target = FullAttentionSpec(
        block_size=768, num_kv_heads=1, head_size=32, dtype=torch.bfloat16, page_size_padded=TARGET_PAGE_BYTES
    )
    draft = ReplicatedDraftSpec(
        block_size=768, num_kv_heads=1, head_size=32, dtype=torch.bfloat16, page_size_padded=DRAFT_PAGE_BYTES
    )
    state = MambaSpec(
        block_size=768,
        shapes=((1,),),
        dtypes=(torch.bfloat16,),
        page_size_padded=TARGET_PAGE_BYTES,
        mamba_cache_mode="align",
        num_speculative_blocks=3,
    )
    layers = {f"target.{i}": target for i in range(24)}
    layers.update({f"draft.{i}": draft for i in range(5)})
    groups = [KVCacheGroupSpec(list(layers), UniformTypeKVCacheSpecs(block_size=768, kv_cache_specs=layers))]
    for group_id in range(3):
        layers = {f"state.{group_id}.{i}": state for i in range(23)}
        groups.append(KVCacheGroupSpec(list(layers), UniformTypeKVCacheSpecs(block_size=768, kv_cache_specs=layers)))
    return groups


def test_pool_bytes_match_allocated_target_and_independent_draft_tensors():
    groups = make_groups()
    # Measured physical allocation: 24 target tensors and five 8-lane drafts.
    assert replicated_draft_pool_bytes_per_block(groups) == 60162048
    assert replicated_draft_pool_bytes_per_block(groups) * 295 == 17747804160


@pytest.mark.parametrize("length,blocks", [(65536, 26), (200000, 48), (1048576, 186)])
def test_admission_uses_shared_pool_slot_bytes(length, blocks):
    config = SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=length),
        parallel_config=SimpleNamespace(decode_context_parallel_size=8),
        cache_config=SimpleNamespace(mamba_cache_mode="align"),
    )
    assert replicated_draft_max_memory_usage_bytes(config, make_groups()) == blocks * 60162048


def test_non_replicated_layout_uses_existing_planner():
    groups = make_groups()
    specs = groups[0].kv_cache_spec.kv_cache_specs
    for name, spec in list(specs.items()):
        if isinstance(spec, ReplicatedDraftSpec):
            specs[name] = replace(spec, dcp_replication_size=1)
    assert replicated_draft_pool_bytes_per_block(groups) is None


def test_incompatible_shared_state_page_uses_existing_planner():
    groups = make_groups()
    specs = groups[1].kv_cache_spec.kv_cache_specs
    name = groups[1].layer_names[0]
    specs[name] = replace(specs[name], page_size_padded=2 * TARGET_PAGE_BYTES)
    assert replicated_draft_pool_bytes_per_block(groups) is None


def test_empty_or_single_group_uses_existing_planner():
    assert replicated_draft_pool_bytes_per_block([]) is None
    assert replicated_draft_pool_bytes_per_block(make_groups()[:1]) is None
