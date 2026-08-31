# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Regression test for vllm-ascend #14283.

DeepSeek-V4 uses a hybrid KV cache with three attention groups: C4 (full MLA,
compress_ratio=4), C128 (full MLA, compress_ratio=128) and SWA (sliding-window
MLA). ``AscendHybridKVCacheCoordinator.find_longest_cache_hit`` takes the MIN
``hit_length`` across all groups. On the decode node SWA has no cached prefix,
so it forces the global ``hit_length`` to 0.

Previously the post-loop truncation only trimmed
``self.attention_groups[0]`` (C4) down to ``hit_length`` while C128 (and any
other full-attn group) kept its already-found hit blocks. With ``hit_length``
shrunk to 0, C128 still carried N hit blocks while
``num_local_computed_tokens = hit_length = 0``, so
``allocate_external_computed_blocks`` computed
``get_new_blocks(cdiv((0 + ext // cr), bs) - N) < 0`` -> ``popleft_n(negative)``
which silently inflated ``num_free_blocks`` (list untouched, counter += |n|),
surfacing later as ``assert curr_block is not None``.

The fix truncates EVERY full-attn group to ``hit_length`` so no group's
hit_blocks outlive the agreed-upon hit_length. This test asserts that behavior
directly and confirms the free-list invariant stays intact afterwards.
"""

import pytest
import torch
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import (
    get_request_block_hasher,
    init_none_hash,
)
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheGroupSpec,
)
from vllm.v1.request import Request

from vllm_ascend.core.kv_cache_interface import (
    AscendMLAAttentionSpec,
    AscendSlidingWindowMLASpec,
)
from vllm_ascend.patch.platform.patch_kv_cache_coordinator import (
    AscendHybridKVCacheCoordinator,
)

pytestmark = pytest.mark.cpu_test


@pytest.fixture(autouse=True)
def _init_hash_seed():
    init_none_hash(sha256)


BLOCK_SIZE = 32
C4_CR = 4
C128_CR = 128
SWA_WINDOW = 4096


def _make_request(req_id, token_ids, hash_block_size=BLOCK_SIZE):
    sp = SamplingParams(max_tokens=1)
    sp.update_from_generation_config({}, eos_token_id=100)
    return Request(
        request_id=req_id,
        prompt_token_ids=token_ids,
        sampling_params=sp,
        pooling_params=None,
        block_hasher=get_request_block_hasher(hash_block_size, sha256),
    )


def _make_dsv4_coordinator(num_blocks=10000):
    """Build the real DSV4 3-group hybrid coordinator: C4 + C128 + SWA."""
    c4_spec = AscendMLAAttentionSpec(
        block_size=BLOCK_SIZE,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
        compress_ratio=C4_CR,
        model_version="deepseek_v4",
    )
    c128_spec = AscendMLAAttentionSpec(
        block_size=BLOCK_SIZE,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
        compress_ratio=C128_CR,
        model_version="deepseek_v4",
    )
    swa_spec = AscendSlidingWindowMLASpec(
        block_size=BLOCK_SIZE,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
        sliding_window=SWA_WINDOW,
        model_version="deepseek_v4",
    )
    # All groups share block_size=32 here, so the scheduler block lcm is 32.
    return AscendHybridKVCacheCoordinator(
        kv_cache_config=KVCacheConfig(
            num_blocks=num_blocks,
            kv_cache_tensors=[],
            kv_cache_groups=[
                KVCacheGroupSpec(["c4"], c4_spec),
                KVCacheGroupSpec(["c128"], c128_spec),
                KVCacheGroupSpec(["swa"], swa_spec),
            ],
        ),
        max_model_len=BLOCK_SIZE * C128_CR * 64,
        use_eagle=False,
        enable_caching=True,
        enable_kv_cache_events=False,
        dcp_world_size=1,
        pcp_world_size=1,
        hash_block_size=BLOCK_SIZE,
        max_num_batched_tokens=BLOCK_SIZE * C128_CR * 64,
        scheduler_block_size=BLOCK_SIZE,
    )


def _establish_prefix_cache(coord, req, num_tokens, skip_groups=()):
    """Allocate + cache blocks for a request (writes hashes, ref_cnt=1)."""
    for i, mgr in enumerate(coord.single_type_managers):
        mgr.allocate_new_blocks(req.request_id, num_tokens, num_tokens)
        if i not in skip_groups:
            mgr.cache_blocks(req, num_tokens=num_tokens)


def _free_request(coord, req):
    for mgr in coord.single_type_managers:
        mgr.free(req.request_id)


def _mgr_indices(coord):
    """Return (c4_idx, c128_idx, swa_idx) by compress_ratio."""
    c4 = c128 = swa = None
    for i, m in enumerate(coord.single_type_managers):
        cr = getattr(m, "compress_ratio", None)
        if cr == C4_CR:
            c4 = i
        elif cr == C128_CR:
            c128 = i
        else:
            swa = i
    return c4, c128, swa


def test_full_attn_groups_truncated_to_global_hit_length():
    """Regression for vllm-ascend #14283.

    Setup mirrors the decode-node reality: C4 + C128 prefixes are cached for
    request A (and freed, keeping their hashes for prefix-cache lookup), while
    SWA is never cached. A second request B sharing A's prefix then triggers
    ``find_longest_cache_hit``: C4 and C128 each hit N blocks, but SWA returns
    0, forcing the global ``hit_length`` to 0.

    Before the fix only ``attention_groups[0]`` (C4) was truncated, so C128
    kept its N hit blocks despite ``hit_length == 0``. The fix truncates every
    full-attn group, so C128 must end up with zero hit blocks too, and the
    subsequent external allocation must not break the free-list invariant.
    """
    coord = _make_dsv4_coordinator(num_blocks=10000)
    bp = coord.block_pool
    _, c128_idx, swa_idx = _mgr_indices(coord)

    # C128 logical block = 32 * 128 = 4096 tokens. Prompt spans 10 C128 blocks.
    logical_c128 = BLOCK_SIZE * C128_CR
    prompt_len = 10 * logical_c128 + 10
    tokens_a = list(range(prompt_len))
    req_a = _make_request("a", tokens_a)
    # Cache C4 + C128 prefixes but SKIP SWA (decode node has no SWA prefix).
    _establish_prefix_cache(coord, req_a, 10 * logical_c128, skip_groups=(swa_idx,) if swa_idx is not None else ())
    _free_request(coord, req_a)

    # Request B reuses A's prefix and asks for extra external (connector) KV.
    tokens_b = list(range(prompt_len)) + [9999999] * 1000
    req_b = _make_request("b", tokens_b)

    cache_hit_blocks, hit_length = coord.find_longest_cache_hit(req_b.block_hashes, max_cache_hit_length=prompt_len - 1)

    # SWA forces the global hit_length to 0.
    assert hit_length == 0, f"Precondition: SWA must force global hit_length to 0, got {hit_length}"

    # Core regression assertion: EVERY full-attn group must be truncated to
    # hit_length. Before the fix C128 kept 10 untruncated blocks here.
    assert len(cache_hit_blocks[c128_idx]) == 0, (
        "C128 hit blocks must be truncated to hit_length=0; #14283 regression: "
        f"got {len(cache_hit_blocks[c128_idx])} untruncated blocks"
    )

    # The downstream crash path must stay intact: with no stale C128 hit
    # blocks, allocate_new_computed_blocks cannot drive a negative
    # get_new_blocks/popleft_n, so the free-list counter must match the
    # walked list length.
    coord.allocate_new_computed_blocks(
        request_id=req_b.request_id,
        new_computed_blocks=cache_hit_blocks,
        num_local_computed_tokens=hit_length,
        num_external_computed_tokens=36864,
    )
    walked = len(bp.free_block_queue.get_all_free_blocks())
    counter = bp.get_num_free_blocks()
    assert walked == counter, (
        "Free-list DESYNC after allocate_new_computed_blocks: "
        f"walked={walked} counter={counter} (counter inflated by "
        f"{counter - walked}); #14283 root cause: full-attn group hit_blocks "
        "outlived hit_length -> negative popleft_n."
    )
