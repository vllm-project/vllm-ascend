# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise sparse partial checkpoints with real upstream cache managers."""

import inspect
from copy import copy
from types import SimpleNamespace

import pytest
import torch
from vllm import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec, MambaSpec
from vllm.v1.request import Request

from vllm_ascend.core.kv_cache_interface import (
    AscendIndexerKPoolTailSpec,
    AscendMLAAttentionSpec,
    register_ascend_kv_cache_specs,
)
from vllm_ascend.patch.platform.patch_mamba_block_aligned_split import (
    _mamba_block_aligned_split,
    _original_mamba_block_aligned_split,
)
from vllm_ascend.utils import vllm_version_is


@pytest.mark.parametrize("skip_tail_zeroing", [False, True])
def test_external_prefix_allocation_preserves_private_tail_zeroing_policy(skip_tail_zeroing):
    init_none_hash(sha256)
    manager = _manager(64, 4)
    tail = manager.coordinator.single_type_managers[3]
    # Exercise the worker-zeroing bookkeeping even when this cache spec does
    # not require zeroing by default.
    tail._record_new_block_ids = True
    free_before = manager.block_pool.get_num_free_blocks()
    request = _request("external", list(range(193)), 64)
    zeroing_kwargs: dict = {}
    if "skip_zeroing_group_ids" in inspect.signature(manager.allocate_slots).parameters:
        zeroing_kwargs["skip_zeroing_group_ids"] = frozenset({3}) if skip_tail_zeroing else frozenset()
    elif skip_tail_zeroing:
        pytest.skip("requires upstream per-group external-load zeroing control")
    blocks = manager.allocate_slots(request, 1, num_external_computed_tokens=192, **zeroing_kwargs)
    assert blocks is not None
    private_blocks = tail.req_to_blocks[request.request_id]
    assert len(private_blocks) == 1
    block = private_blocks[0]
    assert block.block_hash is None
    assert tail.new_block_ids == ([] if skip_tail_zeroing else [block.block_id])
    request.num_computed_tokens = 192
    assert manager.allocate_slots(request, 1) is not None
    assert tail.req_to_blocks[request.request_id] == private_blocks
    assert tail.new_block_ids == ([] if skip_tail_zeroing else [block.block_id])
    manager.free(request)
    assert manager.block_pool.get_num_free_blocks() == free_before


def _request(request_id, tokens, hash_size):
    params = SamplingParams(max_tokens=8)
    params.update_from_generation_config({}, eos_token_id=100)
    return Request(
        request_id=request_id,
        prompt_token_ids=tokens,
        sampling_params=params,
        pooling_params=None,
        block_hasher=get_request_block_hasher(hash_size, sha256),
    )


def _manager(hash_size, ratio, *, role="kv_producer", use_eagle=False):
    # Match core initialization followed by the NPUPlatform registration hook.
    register_all_kvcache_specs(None)
    register_ascend_kv_cache_specs()
    page_size = 4 * hash_size
    groups = [
        KVCacheGroupSpec(
            ["attention"],
            FullAttentionSpec(
                block_size=page_size,
                num_kv_heads=1,
                head_size=1,
                dtype=torch.float32,
            ),
        ),
        KVCacheGroupSpec(
            ["mamba"],
            MambaSpec(
                block_size=page_size,
                shapes=((1,),),
                dtypes=(torch.float32,),
                mamba_cache_mode="align",
                num_speculative_blocks=2 if use_eagle else 0,
            ),
        ),
        KVCacheGroupSpec(
            ["indexer"],
            AscendMLAAttentionSpec(
                block_size=page_size,
                num_kv_heads=1,
                head_size=1,
                dtype=torch.float32,
                tokens_per_state=ratio,
                model_version="glm5_next",
                indexes_kv_by_block_stride=True,
            ),
        ),
        KVCacheGroupSpec(
            ["tail"],
            AscendIndexerKPoolTailSpec(
                block_size=ratio,
                num_kv_heads=1,
                head_size=1,
                dtype=torch.float32,
                sliding_window=ratio,
                compress_ratio=ratio,
            ),
        ),
    ]
    config = KVCacheConfig(num_blocks=128, kv_cache_tensors=[], kv_cache_groups=groups)
    config.kv_transfer_config = (
        None
        if role is None
        else SimpleNamespace(
            is_kv_producer=role in ("kv_producer", "kv_both"),
            is_kv_consumer=role in ("kv_consumer", "kv_both"),
        )
    )
    return KVCacheManager(
        config,
        max_model_len=8192,
        enable_caching=True,
        hash_block_size=hash_size,
        scheduler_block_size=page_size,
        use_eagle=use_eagle,
    )


@pytest.mark.parametrize("hash_size", [32, 64, 128])
@pytest.mark.parametrize("ratio", [2, 4])
def test_sparse_checkpoint_survives_owner_continue_and_request_release(hash_size, ratio):
    init_none_hash(sha256)
    manager = _manager(hash_size, ratio)
    page_size = 4 * hash_size
    tokens = [7] * (3 * hash_size + 1)
    owner = _request("owner", tokens, hash_size)
    free_before = manager.block_pool.get_num_free_blocks()
    coordinator = manager.coordinator
    assert coordinator.enable_partial_hash_hits
    scheduler = SimpleNamespace(
        vllm_config=SimpleNamespace(
            kv_transfer_config=SimpleNamespace(is_kv_producer=True, is_kv_consumer=False),
            model_config=SimpleNamespace(
                hf_config=SimpleNamespace(), hf_text_config=SimpleNamespace(index_topk=2048, index_kpool=ratio)
            ),
        ),
        cache_config=SimpleNamespace(block_size=page_size),
        block_size=page_size,
        hash_block_size=hash_size,
        mamba_partial_cache_hit=True,
        mamba_has_prefill_checkpoint_blocks=False,
        mamba_prefill_checkpoint_alignment=None,
        mamba_shared_prefix_checkpoint=False,
        use_eagle=False,
        use_eagle_block_drop=False,
        max_num_scheduled_tokens=8192,
        scheduler_config=SimpleNamespace(long_prefill_token_threshold=0),
    )
    boundary = _mamba_block_aligned_split(scheduler, owner, len(tokens))
    assert boundary == 3 * hash_size
    assert boundary % coordinator._cache_hit_alignment_tokens == 0
    assert manager.allocate_slots(owner, boundary) is not None
    tail_hash = owner.block_hashes[boundary // hash_size - 1]
    checkpoint = manager.block_pool.get_cached_block(tail_hash, [1])
    assert checkpoint is not None
    source_id = checkpoint[0].block_id

    owner.num_computed_tokens = boundary
    assert manager.allocate_slots(owner, 1) is not None
    copies, retained = manager.take_kv_cache_block_copies()
    preserved = next(copy for copy in copies if copy.src_block_id == source_id)
    assert preserved.dst_block_id != source_id
    offloads = manager.take_boundary_state_offloads()
    assert (1, preserved.dst_block_id, boundary) in offloads["owner"]
    assert manager.take_boundary_state_offloads() == {}
    manager.block_pool.free_blocks(retained)
    manager.free(owner)
    manager.new_step_starts()

    sibling = _request("sibling", tokens, hash_size)
    blocks, hit, _ = manager.get_computed_blocks(sibling)
    assert hit == boundary
    assert blocks.blocks[3] == []
    assert manager.allocate_slots(sibling, 1, hit, blocks) is not None
    assert manager.get_blocks("sibling").get_block_ids()[3]
    assert manager.block_pool.get_cached_block(tail_hash, [3]) is None
    copies, retained = manager.take_kv_cache_block_copies()
    manager.block_pool.free_blocks(retained)
    manager.free(sibling)
    assert manager.block_pool.get_num_free_blocks() == free_before


@pytest.mark.parametrize("hash_size", [32, 64, 128])
@pytest.mark.parametrize("role", [None, "kv_producer", "kv_both"])
@pytest.mark.parametrize("use_eagle", [False, True])
@pytest.mark.parametrize("prompt_offset", [-1, 0, 1])
@pytest.mark.parametrize("prompt_hash_offset", [0, 1])
def test_sparse_prefill_replay_boundary(hash_size, role, use_eagle, prompt_offset, prompt_hash_offset):
    init_none_hash(sha256)
    manager = _manager(hash_size, 4, role=role, use_eagle=use_eagle)
    free_before = manager.block_pool.get_num_free_blocks()
    page_size = 4 * hash_size
    tokens = [7] * (page_size + prompt_hash_offset * hash_size + prompt_offset)
    owner = _request("owner", tokens, hash_size)
    transfer = (
        None
        if role is None
        else SimpleNamespace(
            is_kv_producer=True,
            is_kv_consumer=role == "kv_both",
        )
    )
    scheduler = SimpleNamespace(
        vllm_config=SimpleNamespace(
            kv_transfer_config=transfer,
            model_config=SimpleNamespace(
                hf_config=SimpleNamespace(), hf_text_config=SimpleNamespace(index_topk=2048, index_kpool=4)
            ),
        ),
        cache_config=SimpleNamespace(block_size=page_size),
        block_size=page_size,
        hash_block_size=hash_size,
        mamba_partial_cache_hit=True,
        mamba_has_prefill_checkpoint_blocks=False,
        mamba_prefill_checkpoint_alignment=None,
        mamba_shared_prefix_checkpoint=False,
        use_eagle=use_eagle,
        use_eagle_block_drop=use_eagle,
        max_num_scheduled_tokens=8192,
        scheduler_config=SimpleNamespace(long_prefill_token_threshold=0),
    )
    offloaded_boundaries: set[int] = set()
    while owner.num_computed_tokens < len(tokens):
        manager.new_step_starts()
        scheduled = _mamba_block_aligned_split(scheduler, owner, len(tokens) - owner.num_computed_tokens)
        assert scheduled > 0
        assert manager.allocate_slots(owner, scheduled) is not None
        owner.num_computed_tokens += scheduled
        offloaded_boundaries.update(entry[2] for entry in manager.take_boundary_state_offloads().get("owner", ()))
        _, retained = manager.take_kv_cache_block_copies()
        manager.block_pool.free_blocks(retained)
    manager.free(owner)
    manager.new_step_starts()
    sibling = _request("sibling", tokens, hash_size)
    blocks, hit, _ = manager.get_computed_blocks(sibling)
    drop_eagle = use_eagle and role == "kv_both"
    expected = (len(tokens) // hash_size - 1) * hash_size if drop_eagle else (len(tokens) - 1) // hash_size * hash_size
    if vllm_version_is("0.31.0") and len(tokens) == page_size and not drop_eagle:
        # Release schedules an aligned prompt in one step and caches its full
        # state, but cannot reuse that state when the last token must replay.
        # Keep that upstream behavior rather than backport its scheduler and
        # Mamba-manager checkpoint changes as part of this adapter.
        native_scheduler = copy(scheduler)
        native_scheduler.use_eagle = native_scheduler.use_eagle_block_drop = False
        native_request = _request("native", tokens, hash_size)
        assert _original_mamba_block_aligned_split(native_scheduler, native_request, len(tokens)) == len(tokens)
        assert expected not in offloaded_boundaries
        expected = 0
    assert hit == expected
    if role in ("kv_producer", "kv_both") and not use_eagle and expected % page_size:
        assert expected in offloaded_boundaries
    assert not blocks.blocks[3]
    assert manager.block_pool.get_num_free_blocks() == free_before


@pytest.mark.parametrize("in_flight", [0, 1])
def test_sparse_finish_handoff_requires_completed_checkpoint(in_flight):
    init_none_hash(sha256)
    hash_size = 64
    manager = _manager(hash_size, 4)
    owner = _request("owner", [7] * 193, hash_size)
    assert manager.allocate_slots(owner, 192) is not None
    owner.num_computed_tokens = 192
    owner.num_in_flight_tokens = in_flight
    blocks = manager.get_blocks("owner").get_block_ids()
    offloads = manager.finalize_partial_tail_offloads(owner)
    if in_flight:
        assert offloads == []
    else:
        assert offloads == [(1, blocks[1][-1], 192)]
        assert manager.finalize_partial_tail_offloads(owner) == []
    manager.free(owner)


@pytest.mark.parametrize("allowed", [False, True])
def test_sparse_partial_hits_follow_core_capability_switch(allowed):
    manager = _manager(64, 4)
    coordinator = type(manager.coordinator)(
        kv_cache_config=manager.kv_cache_config,
        max_model_len=8192,
        scheduler_block_size=256,
        hash_block_size=64,
        use_eagle=False,
        enable_caching=True,
        enable_kv_cache_events=False,
        dcp_world_size=1,
        pcp_world_size=1,
        allow_partial_hash_hits=allowed,
    )
    assert coordinator.enable_partial_hash_hits is allowed
    assert coordinator._cache_hit_alignment_tokens == (64 if allowed else 256)


@pytest.mark.parametrize("external_hashes", [3, 5])
@pytest.mark.parametrize("ratio", [2, 4])
def test_mixed_local_and_external_hit_gets_private_partial_attention_pages(external_hashes, ratio):
    init_none_hash(sha256)
    hash_size = 64
    local_tokens = (external_hashes - 2) * hash_size
    manager = _manager(hash_size, ratio, role="kv_both")
    free_before = manager.block_pool.get_num_free_blocks()
    owner = _request("local-owner", [7] * (local_tokens + 1), hash_size)
    assert manager.allocate_slots(owner, local_tokens) is not None
    owner.num_computed_tokens = local_tokens
    assert manager.allocate_slots(owner, 1) is not None
    _, retained = manager.take_kv_cache_block_copies()
    manager.block_pool.free_blocks(retained)
    manager.take_boundary_state_offloads()
    manager.free(owner)
    manager.new_step_starts()

    reader = _request("mixed-reader", [7] * (external_hashes * hash_size + 1), hash_size)
    local_blocks, local_hit, _ = manager.get_computed_blocks(reader)
    assert local_hit == local_tokens
    shared_ids = {group: local_blocks.blocks[group][0].block_id for group in (0, 2)}
    assert (
        manager.allocate_slots(
            reader,
            1,
            num_new_computed_tokens=local_hit,
            new_computed_blocks=local_blocks,
            num_external_computed_tokens=external_hashes * hash_size - local_hit,
            delay_cache_blocks=True,
        )
        is not None
    )
    destination = manager.get_blocks(reader.request_id).get_block_ids()
    # Store reloads the physical page containing the local partial prefix.
    # It must not overwrite the shared cached attention/indexer source page.
    copies, retained = manager.take_kv_cache_block_copies()
    for group, source in shared_ids.items():
        assert destination[group][0] != source
        assert any(copy.src_block_id == source and copy.dst_block_id == destination[group][0] for copy in copies)
        # Core retains both CoW endpoints until the worker executes the copy.
        assert manager.block_pool.blocks[destination[group][0]].ref_cnt == 2
        assert manager.block_pool.blocks[destination[group][0]].block_hash is None
    manager.block_pool.free_blocks(retained)
    for group in shared_ids:
        assert manager.block_pool.blocks[destination[group][0]].ref_cnt == 1
    manager.free(reader)
    assert manager.block_pool.get_num_free_blocks() == free_before
