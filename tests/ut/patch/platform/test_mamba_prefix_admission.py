# SPDX-License-Identifier: Apache-2.0
"""Admission counts must distinguish locally computed and transferred state."""

from types import SimpleNamespace

import pytest
import torch
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec, MambaSpec

from vllm_ascend.core.kv_cache_interface import register_ascend_kv_cache_specs
from vllm_ascend.patch.platform.patch_kv_cache_coordinator import AscendHybridKVCacheCoordinator
from vllm_ascend.patch.platform.patch_mamba_block_aligned_split import _elide_first_tail_checkpoint
from vllm_ascend.patch.platform.patch_mamba_manager import AscendMambaManager, GDNCheckpointCandidates


def make_coordinator(mamba_type=MambaAttentionBackendEnum.MAMBA2, *, transfer=None, speculative=False, fine=False):
    register_all_kvcache_specs(None)
    register_ascend_kv_cache_specs()
    specs = [
        FullAttentionSpec(block_size=b, num_kv_heads=1, head_size=8, dtype=torch.bfloat16)
        for b in ((1536, 1536) if fine else (251904, 50688))
    ] + [
        MambaSpec(
            block_size=1536,
            shapes=((1,),),
            dtypes=(torch.float32,),
            mamba_cache_mode="align",
            mamba_type=mamba_type,
        )
        for _ in range(3)
    ]
    config = KVCacheConfig(
        num_blocks=12,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec([str(i)], spec) for i, spec in enumerate(specs)],
    )
    config.kv_transfer_config = transfer
    config.has_speculative_decoding = speculative
    return AscendHybridKVCacheCoordinator(
        config,
        max_model_len=16384,
        use_eagle=False,
        enable_caching=True,
        enable_kv_cache_events=False,
        dcp_world_size=1,
        pcp_world_size=1,
        hash_block_size=128 if fine else 768,
        scheduler_block_size=1536 if fine else 8312832,
        max_in_flight_tokens=4096,
    )


def _observe_checkpoint(manager, request_id, key, *, prompt_length=1024, junction=0, finish=False):
    boundary = 768
    request = SimpleNamespace(
        request_id=request_id,
        block_hashes=[key, b"z" * 32],
        num_prompt_tokens=prompt_length,
        num_computed_tokens=0,
        shared_prefix_boundary=junction,
    )
    end = prompt_length if finish else boundary
    if not manager.req_to_blocks[request_id]:
        manager.allocate_new_blocks(request_id, end, end)
    return manager._cache_partial_tail_block(request, end, None)


def test_gdn_partial_checkpoint_is_cached_on_second_request():
    manager = make_coordinator(MambaAttentionBackendEnum.GDN_ATTN).single_type_managers[2]
    key = b"x" * 32
    group_ids = [manager.kv_cache_group_id]
    assert _observe_checkpoint(manager, "first", key, finish=True) is None
    assert _observe_checkpoint(manager, "first", key) is None
    assert manager.block_pool.get_cached_block(key, group_ids) is None

    assert _observe_checkpoint(manager, "second", key) is not None
    assert manager.block_pool.get_cached_block(key, group_ids) == [manager.req_to_blocks["second"][0]]


@pytest.mark.parametrize("role", ["kv_producer", "kv_consumer", "kv_both", "speculative", "junction"])
def test_gdn_preserves_first_checkpoint_on_protected_paths(role):
    transfer = (
        SimpleNamespace(is_kv_producer=role != "kv_consumer", is_kv_consumer=role != "kv_producer")
        if role.startswith("kv_")
        else None
    )
    manager = make_coordinator(
        MambaAttentionBackendEnum.GDN_ATTN, transfer=transfer, speculative=role == "speculative"
    ).single_type_managers[2]
    key = b"x" * 32
    if role == "junction":
        manager.shared_prefix_checkpoint = True
        published = _observe_checkpoint(manager, "first", key, prompt_length=1792, junction=768)
    else:
        published = _observe_checkpoint(manager, "first", key)
        assert "first" in manager._producer_partial_tail_reqs
    assert published is not None
    assert manager.checkpoint_candidates is None or not manager.checkpoint_candidates.chains


def test_gdn_candidate_reset_follows_successful_prefix_cache_reset():
    coordinator = make_coordinator(MambaAttentionBackendEnum.GDN_ATTN)
    manager = coordinator.single_type_managers[2]
    key = b"x" * 32
    assert _observe_checkpoint(manager, "before", key, finish=True) is None
    assert not coordinator.reset_prefix_cache()
    assert manager.checkpoint_candidates.chains["before"] == (key,)
    coordinator.free("before")
    assert coordinator.reset_prefix_cache()
    assert not manager.checkpoint_candidates.chains
    assert _observe_checkpoint(manager, "after", key, finish=True) is None


def test_gdn_partial_checkpoint_candidate_tracking_is_bounded(monkeypatch):
    monkeypatch.setattr(GDNCheckpointCandidates, "MAX_CHAINS", 2)
    manager = make_coordinator(MambaAttentionBackendEnum.GDN_ATTN).single_type_managers[2]
    keys = [bytes([index]) * 32 for index in range(3)]
    for index, key in enumerate(keys):
        assert _observe_checkpoint(manager, str(index), key, finish=True) is None
    assert list(manager.checkpoint_candidates.counts) == keys[1:]
    assert _observe_checkpoint(manager, "evicted", keys[0], finish=True) is None
    assert _observe_checkpoint(manager, "retained", keys[2]) is not None


def test_gdn_first_tail_split_is_elided_until_a_distinct_admitted_request():
    manager = make_coordinator(MambaAttentionBackendEnum.GDN_ATTN).single_type_managers[2]
    scheduler = SimpleNamespace(
        mamba_partial_cache_hit=True,
        mamba_has_prefill_checkpoint_blocks=False,
        mamba_shared_prefix_checkpoint=False,
        vllm_config=SimpleNamespace(kv_transfer_config=None, speculative_config=None),
        cache_config=SimpleNamespace(block_size=1536),
        hash_block_size=768,
        num_hash_lookahead_tokens=0,
        kv_cache_manager=SimpleNamespace(coordinator=SimpleNamespace(single_type_managers=[manager])),
    )
    request = SimpleNamespace(
        request_id="first",
        block_hashes=[b"x" * 32],
        num_prompt_tokens=1024,
        num_tokens=1024,
        num_computed_tokens=0,
        shared_prefix_boundary=0,
    )
    assert _elide_first_tail_checkpoint(scheduler, request, 0, 960, 768) == 768
    scheduler.mamba_shared_prefix_checkpoint = True
    request.shared_prefix_boundary = 768
    assert _elide_first_tail_checkpoint(scheduler, request, 0, 1024, 768) == 768
    request.shared_prefix_boundary = 0
    assert _elide_first_tail_checkpoint(scheduler, request, 0, 1024, 768) == 1024
    assert not manager.checkpoint_candidates.chains

    manager.allocate_new_blocks("first", 1024, 1024)
    assert manager._cache_partial_tail_block(request, 1024, None) is None
    assert manager.block_pool.get_cached_block(request.block_hashes[0], [manager.kv_cache_group_id]) is None
    assert _elide_first_tail_checkpoint(scheduler, request, 0, 1024, 768) == 1024
    repeated = SimpleNamespace(**vars(request))
    repeated.request_id = "repeated"
    assert _elide_first_tail_checkpoint(scheduler, repeated, 0, 1024, 768) == 768
    manager.allocate_new_blocks("repeated", 768, 768)
    assert manager._cache_partial_tail_block(repeated, 768, None) is not None


def test_gdn_hash_chain_promotes_an_earlier_prefix_of_a_short_request():
    coordinator = make_coordinator(MambaAttentionBackendEnum.GDN_ATTN, fine=True)
    scheduler = SimpleNamespace(
        mamba_partial_cache_hit=True,
        mamba_has_prefill_checkpoint_blocks=False,
        mamba_shared_prefix_checkpoint=False,
        cache_config=SimpleNamespace(block_size=1536),
        hash_block_size=128,
        num_hash_lookahead_tokens=0,
        kv_cache_manager=SimpleNamespace(coordinator=coordinator),
    )

    def request(request_id, suffix, length):
        return SimpleNamespace(
            request_id=request_id,
            block_hashes=[bytes([i]) * 32 for i in (1, 2, 3, *suffix)],
            num_prompt_tokens=length,
            num_tokens=length,
            num_computed_tokens=0,
            shared_prefix_boundary=0,
        )

    first = request("first", (4, 5, 6, 7), 1000)
    assert _elide_first_tail_checkpoint(scheduler, first, 0, 1000, 896) == 1000
    coordinator.allocate_new_blocks("first", 1000, 1000)
    coordinator.cache_blocks(first, 1000)
    assert coordinator.gdn_checkpoint_candidates.chains["first"] == tuple(first.block_hashes)
    coordinator.free("first")
    coordinator.new_step_starts()

    second = request("second", (8, 9, 10, 11, 12), 1100)
    assert _elide_first_tail_checkpoint(scheduler, second, 0, 1100, 1024) == 384
    coordinator.allocate_new_blocks("second", 384, 384)
    coordinator.cache_blocks(second, 384)
    coordinator.new_step_starts()
    second.num_computed_tokens = 384
    assert _elide_first_tail_checkpoint(scheduler, second, 384, 716, 640) == 716
    coordinator.allocate_new_blocks("second", 1100, 1100)
    coordinator.cache_blocks(second, 1100)
    coordinator.free("second")
    coordinator.new_step_starts()

    third = request("third", (13, 14, 15, 16, 17), 1050)
    _, actual_hit, _ = coordinator.find_longest_cache_hit(third.block_hashes, third.num_prompt_tokens - 1)
    assert actual_hit == 384


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
