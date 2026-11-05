# SPDX-License-Identifier: Apache-2.0
"""Admission counts must distinguish locally computed and transferred state."""

from types import SimpleNamespace

import pytest
import torch
from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec, MambaSpec

from vllm_ascend.core.kv_cache_interface import register_ascend_kv_cache_specs
from vllm_ascend.patch.platform.patch_kv_cache_coordinator import AscendHybridKVCacheCoordinator
from vllm_ascend.patch.platform.patch_mamba_manager import AscendMambaManager


def make_coordinator():
    register_all_kvcache_specs(None)
    register_ascend_kv_cache_specs()
    specs = [
        FullAttentionSpec(block_size=b, num_kv_heads=1, head_size=8, dtype=torch.bfloat16) for b in (251904, 50688)
    ] + [
        MambaSpec(block_size=1536, shapes=((1,),), dtypes=(torch.float32,), mamba_cache_mode="align") for _ in range(3)
    ]
    config = KVCacheConfig(
        num_blocks=12,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec([str(i)], spec) for i, spec in enumerate(specs)],
    )
    return AscendHybridKVCacheCoordinator(
        config,
        max_model_len=16384,
        use_eagle=False,
        enable_caching=True,
        enable_kv_cache_events=False,
        dcp_world_size=1,
        pcp_world_size=1,
        hash_block_size=768,
        scheduler_block_size=8312832,
        max_in_flight_tokens=4096,
    )


def test_partial_boundary_decode_fits_available_blocks():
    """Regression for a real 768-token request preempting with six free pages."""
    coordinator = make_coordinator()
    request = SimpleNamespace(
        request_id="boundary",
        block_hashes=[b"x" * 32],
        num_prompt_tokens=768,
        shared_prefix_boundary=0,
    )
    coordinator.allocate_new_blocks(request.request_id, 768, 768)
    coordinator.cache_blocks(request, 768)
    coordinator.new_step_starts()
    available = coordinator.block_pool.get_num_free_blocks()
    needed = coordinator.get_num_blocks_to_allocate(
        request_id=request.request_id,
        num_tokens=769,
        new_computed_blocks=tuple(() for _ in range(5)),
        num_encoder_tokens=0,
        total_computed_tokens=768,
        num_local_computed_tokens=768,
        num_tokens_main_model=769,
    )
    assert available == 6
    assert needed <= available
    coordinator.allocate_new_blocks(request.request_id, 769, 769)
    assert coordinator.block_pool.get_num_free_blocks() >= 0


@pytest.mark.parametrize(
    "local,total,main,extra",
    [(768, 768, 769, 0), (1536, 1536, 1537, 0), (0, 768, 769, 1), (768, 1024, 1025, 1), (768, 1024, 1024, 0)],
)
def test_external_state_reservation_uses_exact_local_token_count(local, total, main, extra):
    coordinator = make_coordinator()
    manager = coordinator.single_type_managers[2]
    assert isinstance(manager, AscendMambaManager)
    manager.allocate_new_blocks("request", local or 1, local or 1)
    kwargs = dict(
        request_id="request",
        num_tokens=main,
        new_computed_blocks=(),
        total_computed_tokens=total,
        num_local_computed_tokens=local,
        num_tokens_main_model=main,
    )
    base = AscendMambaManager.__base__.get_num_blocks_to_allocate(manager, **kwargs)
    assert manager.get_num_blocks_to_allocate(**kwargs) == base + extra


def test_external_extension_of_partial_local_hit_keeps_extra_reservation():
    manager = make_coordinator().single_type_managers[2]
    manager.allocate_new_blocks("seed", 768, 768)
    kwargs = dict(
        request_id="new",
        num_tokens=1025,
        new_computed_blocks=manager.req_to_blocks["seed"],
        total_computed_tokens=1024,
        num_local_computed_tokens=768,
        num_tokens_main_model=1025,
    )
    base = AscendMambaManager.__base__.get_num_blocks_to_allocate(manager, **kwargs)
    assert manager.get_num_blocks_to_allocate(**kwargs) == base + 1


def test_dcp8_prefix_reuses_checkpoint_and_copies_partial_attention_tail():
    register_all_kvcache_specs(None)
    register_ascend_kv_cache_specs()
    specs = [FullAttentionSpec(block_size=1536, num_kv_heads=1, head_size=8, dtype=torch.bfloat16)] + [
        MambaSpec(block_size=1536, shapes=((1,),), dtypes=(torch.float32,), mamba_cache_mode="align") for _ in range(3)
    ]
    config = KVCacheConfig(
        num_blocks=64,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec([str(i)], spec) for i, spec in enumerate(specs)],
    )
    coordinator = AscendHybridKVCacheCoordinator(
        config,
        max_model_len=262144,
        use_eagle=False,
        enable_caching=True,
        enable_kv_cache_events=False,
        dcp_world_size=8,
        pcp_world_size=1,
        hash_block_size=1536,
        scheduler_block_size=12288,
        max_in_flight_tokens=4096,
    )
    assert coordinator.enable_partial_hash_hits
    assert coordinator._cache_hit_alignment_tokens == 1536
    assert [m.block_size for m in coordinator.single_type_managers] == [12288, 1536, 1536, 1536]
    request = SimpleNamespace(
        request_id="seed", block_hashes=[b"s" * 32], num_prompt_tokens=1537, shared_prefix_boundary=0
    )
    coordinator.allocate_new_blocks("seed", 1536, 1536)
    coordinator.cache_blocks(request, 1536)
    coordinator.new_step_starts()
    hits, length, *_ = coordinator.find_longest_cache_hit(request.block_hashes, 1536)
    assert length == 1536
    assert all(len(blocks) == 1 for blocks in hits)
    coordinator.allocate_new_computed_blocks("reuse", hits, 1536, 0)
    coordinator.allocate_new_blocks("reuse", 1537, 1537)
    attention = coordinator.single_type_managers[0]
    assert attention.req_to_blocks["reuse"][0] is not attention.req_to_blocks["seed"][0]
    for state in coordinator.single_type_managers[1:]:
        assert state.req_to_blocks["reuse"][-1] is not state.req_to_blocks["seed"][-1]
