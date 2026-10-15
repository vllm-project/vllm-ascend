# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Role-specific sparse cache writes preserve manager speculation metadata."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.single_type_kv_cache_manager import (
    FullAttentionManager,
    SlidingWindowManager,
)
from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec, SlidingWindowSpec

from vllm_ascend.patch.platform.patch_mamba_manager import AscendMambaManager
from vllm_ascend.patch.platform.patch_prefix_cache_write_policy import (
    configure_prefix_cache_write_policy,
)


@pytest.mark.parametrize("retention_interval", [0, 512, None])
@pytest.mark.parametrize("effective_drop", [False, True])
def test_swa_mask_matches_lookup_policy(effective_drop, retention_interval):
    manager = SlidingWindowManager.__new__(SlidingWindowManager)
    manager.use_eagle = True
    spec = SlidingWindowSpec(
        block_size=128,
        num_kv_heads=1,
        head_size=8,
        dtype=torch.float32,
        sliding_window=256,
    )
    kwargs = dict(
        start_block=0,
        end_block=8,
        alignment_tokens=512,
        kv_cache_spec=spec,
        use_eagle=True,
        retention_interval=retention_interval,
        reachable_boundaries=(512,),
    )
    upstream_mask = SlidingWindowManager.reachable_block_mask(**kwargs)
    configure_prefix_cache_write_policy(manager, effective_drop)
    expected = SlidingWindowManager.reachable_block_mask(**dict(kwargs, use_eagle=effective_drop))
    assert manager.reachable_block_mask(**kwargs) == expected
    assert manager.use_eagle is True
    assert SlidingWindowManager.reachable_block_mask(**kwargs) == upstream_mask
    if retention_interval == 0:
        kept = [i for i, keep in enumerate(expected) if keep]
        assert kept == ([2, 3] if not effective_drop else [2, 3, 4])


def test_non_eagle_swa_keeps_original_mask():
    manager = SlidingWindowManager.__new__(SlidingWindowManager)
    manager.use_eagle = False
    original = manager.reachable_block_mask
    configure_prefix_cache_write_policy(manager, effective_drop=False)
    assert manager.reachable_block_mask == original
    assert "reachable_block_mask" not in manager.__dict__


@pytest.mark.parametrize(
    ("prompt_tokens", "finalized_tokens", "expected"),
    [
        (1001, 1001, [896]),
        (896, 896, [896, 768]),
        (896, 800, [768]),
        (1024, 1024, [896]),
        (128, 128, [128]),
        (1, 1, []),
    ],
)
def test_fa_publishes_legal_resend_and_extension_tail_keys(prompt_tokens, finalized_tokens, expected):
    manager = FullAttentionManager.__new__(FullAttentionManager)
    manager.use_eagle = True
    manager.block_size = 512
    manager.kv_cache_group_id = 2
    manager.block_pool = SimpleNamespace(hash_block_size=128, cache_partial_block=Mock())
    manager.req_to_blocks = {"request": [object(), object()]}
    request = SimpleNamespace(num_prompt_tokens=prompt_tokens, request_id="request")
    configure_prefix_cache_write_policy(manager, effective_drop=False)
    manager._cache_partial_tail_block(request, finalized_tokens)
    calls = manager.block_pool.cache_partial_block.call_args_list
    assert [call.kwargs["num_tokens"] for call in calls] == expected
    for call in calls:
        boundary = call.kwargs["num_tokens"]
        assert call.kwargs["block"] is manager.req_to_blocks["request"][boundary // manager.block_size]
        assert call.kwargs["kv_cache_group_id"] == 2
    assert manager.use_eagle is True


def test_fa_drop_policy_preserves_upstream_tail_publication():
    manager = FullAttentionManager.__new__(FullAttentionManager)
    manager.use_eagle = True
    manager.block_size = 512
    manager.kv_cache_group_id = 0
    manager.block_pool = SimpleNamespace(hash_block_size=128, cache_partial_block=Mock())
    manager.req_to_blocks = {"request": [object(), object()]}
    request = SimpleNamespace(num_prompt_tokens=896, request_id="request")
    configure_prefix_cache_write_policy(manager, effective_drop=True)
    manager._cache_partial_tail_block(request, 896)
    assert [call.kwargs["num_tokens"] for call in manager.block_pool.cache_partial_block.call_args_list] == [896]
    assert "_cache_partial_tail_block" not in manager.__dict__


def test_fa_physical_hash_blocks_need_no_partial_override():
    manager = FullAttentionManager.__new__(FullAttentionManager)
    manager.block_size = 128
    manager.block_pool = SimpleNamespace(hash_block_size=128)
    configure_prefix_cache_write_policy(manager, effective_drop=False)
    assert "_cache_partial_tail_block" not in manager.__dict__


@pytest.mark.parametrize("prompt_tokens", [1152, 1536])
def test_fa_retains_resend_alias_after_partial_or_full_primary_promotion(prompt_tokens):
    pool = BlockPool(num_gpu_blocks=8, enable_caching=True, hash_block_size=384)
    manager = FullAttentionManager(
        kv_cache_spec=FullAttentionSpec(
            block_size=1536,
            num_kv_heads=1,
            head_size=8,
            dtype=torch.float32,
        ),
        block_pool=pool,
        enable_caching=True,
        kv_cache_group_id=0,
        scheduler_block_size=1536,
    )
    manager.use_eagle = True
    source = pool.get_new_blocks(1)[0]
    manager.req_to_blocks["request"] = [source]
    request = SimpleNamespace(
        request_id="request",
        num_prompt_tokens=prompt_tokens,
        block_hashes=[i.to_bytes(32, "little") for i in range(1, prompt_tokens // 384 + 1)],
        shared_prefix_boundary=0,
    )
    configure_prefix_cache_write_policy(manager, effective_drop=False)
    resend_boundary = prompt_tokens - 384
    manager.cache_blocks(request, resend_boundary, replay_boundaries=(prompt_tokens - 1, prompt_tokens))
    assert pool.get_cached_block(request.block_hashes[resend_boundary // 384 - 1], [0]) == [source]
    assert source.block_hash_num_tokens == resend_boundary

    # A later chunk promotes the same append-only KV block to P. Both keys
    # must survive, whether P is interior to the page or fills the page.
    manager.cache_blocks(request, prompt_tokens, replay_boundaries=(prompt_tokens - 1, prompt_tokens))
    assert source.block_hash_num_tokens == prompt_tokens
    assert pool.get_cached_block(request.block_hashes[-1], [0]) == [source]
    assert pool.get_cached_block(request.block_hashes[resend_boundary // 384 - 1], [0]) == [source]


def _mamba_manager_and_request(prompt_tokens, *, skips_drop=True):
    pool = BlockPool(num_gpu_blocks=16, enable_caching=True, hash_block_size=128)
    manager = AscendMambaManager(
        kv_cache_spec=MambaSpec(
            block_size=512,
            shapes=((1,),),
            dtypes=(torch.float32,),
            mamba_cache_mode="align",
        ),
        block_pool=pool,
        enable_caching=True,
        kv_cache_group_id=0,
        scheduler_block_size=512,
    )
    manager.cache_hit_alignment_tokens = 128
    manager.skips_eagle_block_drop = skips_drop
    manager.drop_eagle_checkpoint_block = not skips_drop
    manager.req_to_blocks["request"] = pool.get_new_blocks(2)
    request = SimpleNamespace(
        request_id="request",
        num_prompt_tokens=prompt_tokens,
        num_computed_tokens=512,
        shared_prefix_boundary=0,
        block_hashes=[i.to_bytes(32, "little") for i in range(1, prompt_tokens // 128 + 1)],
    )
    return manager, request


@pytest.mark.parametrize("scheduled_end", [768, 896])
def test_mamba_registers_actual_aligned_prompt_chunk_state(scheduled_end):
    manager, request = _mamba_manager_and_request(896)
    partial_hash = manager._cache_partial_tail_block(request, scheduled_end, retention_interval=0)
    source = manager.req_to_blocks["request"][1]
    assert partial_hash is not None
    assert manager.block_pool.get_cached_block(request.block_hashes[scheduled_end // 128 - 1], [0]) == [source]
    assert manager._partial_hit_reqs["request"] == (1, source)
    assert manager._producer_partial_tail_reqs["request"] == (source, scheduled_end)
    assert manager.num_cached_block["request"] == 1
    # The state at one chunk end must not be published at the other boundary.
    other_end = 896 if scheduled_end == 768 else 768
    assert manager.block_pool.get_cached_block(request.block_hashes[other_end // 128 - 1], [0]) is None


def test_mamba_does_not_register_an_intermediate_running_state():
    manager, request = _mamba_manager_and_request(896)
    assert manager._cache_partial_tail_block(request, 640, retention_interval=0) is None
    assert manager.block_pool.get_cached_block(request.block_hashes[4], [0]) is None
    assert not manager._partial_hit_reqs


@pytest.mark.parametrize("option", ["shared_prefix_checkpoint", "fine_grained_prefix_cache"])
@pytest.mark.parametrize("enabled", [False, True])
def test_mamba_shared_junction_honors_installed_option_name(option, enabled):
    manager, request = _mamba_manager_and_request(896)
    for name in ("shared_prefix_checkpoint", "fine_grained_prefix_cache"):
        if hasattr(manager, name):
            delattr(manager, name)
    setattr(manager, option, enabled)
    request.shared_prefix_boundary = 640
    actual = manager._cache_partial_tail_block(request, 640, retention_interval=0)
    assert (actual is not None) is enabled
    cached = manager.block_pool.get_cached_block(request.block_hashes[4], [0])
    assert (cached is not None) is enabled


def test_mamba_keeps_exported_prompt_checkpoint_with_rounded_cache_end():
    manager, request = _mamba_manager_and_request(1001)
    manager._checkpoints["request"] = (896, 0)
    checkpoint_block = manager.req_to_blocks["request"][0]
    partial_hash = manager._cache_partial_tail_block(request, 512, retention_interval=0)
    assert partial_hash is not None
    assert checkpoint_block.block_hash_num_tokens == 896
    assert manager.block_pool.get_cached_block(request.block_hashes[6], [0]) == [checkpoint_block]
    # The reserved checkpoint is durable; running-tail CoW tracking is unused.
    assert not manager._partial_hit_reqs


def test_mamba_keeps_hash_aligned_prompt_resend_checkpoint():
    manager, request = _mamba_manager_and_request(1024)
    manager._checkpoints["request"] = (896, 0)
    assert manager._cache_partial_tail_block(request, 1024, retention_interval=0) is not None
    assert manager.req_to_blocks["request"][0].block_hash_num_tokens == 896


def test_mamba_transient_checkpoint_stays_request_local():
    manager, request = _mamba_manager_and_request(1001)
    manager._checkpoints["request"] = (768, 0)
    checkpoint_block = manager.req_to_blocks["request"][0]
    manager.block_pool.cache_partial_block(
        request=request,
        block=checkpoint_block,
        num_tokens=768,
        kv_cache_group_id=0,
        block_size=512,
        replace_existing_hashes=True,
    )
    assert manager._cache_partial_tail_block(request, 768, retention_interval=0) is None
    assert checkpoint_block.block_hash is None
    assert manager.block_pool.get_cached_block(request.block_hashes[5], [0]) is None


def test_mamba_consumer_preserves_upstream_drop_registration():
    manager, request = _mamba_manager_and_request(896, skips_drop=False)
    reference, reference_request = _mamba_manager_and_request(896, skips_drop=False)
    upstream_manager = AscendMambaManager.__bases__[0]
    # Release and main use different flags for the original tail rule. The
    # consumer must delegate to the installed upstream behavior in either case.
    for end in (896, 768):
        expected = upstream_manager._cache_partial_tail_block(reference, reference_request, end, retention_interval=0)
        actual = manager._cache_partial_tail_block(request, end, retention_interval=0)
        assert actual == expected
        assert manager.req_to_blocks["request"][1].block_hash_num_tokens == (
            reference.req_to_blocks["request"][1].block_hash_num_tokens
        )
