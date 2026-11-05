"""Tests for the shared mooncake v1 block-selection helpers.

The helpers are pure functions with explicit parameters, so these cases test
them directly instead of building scheduler/worker objects. Mutation-based
truncation helpers use a lightweight fake request; KV-cache specs are real vLLM
spec objects so the ``isinstance`` branches stay covered.
"""

import types

import pytest
import torch
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    MambaSpec,
    SlidingWindowSpec,
    UniformTypeKVCacheSpecs,
)

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_v1.common.block_selection import (
    GroupTransferInfo,
    build_group_transfer_info,
    group_unique_specs,
    has_attn_mamba_hybrid_cache,
    hybrid_prefill_token_count,
    iter_kv_cache_specs,
    state_prefill_token_count,
    swa_transfer_block_ids,
    transfer_block_ids,
    trim_hybrid_remote_block_ids,
    truncate_request_for_hybrid_prefill,
    truncate_request_for_prefill,
)


def _group_info(tokens_per_block=16, blocks_per_window=0, is_state_group=False):
    return GroupTransferInfo(
        tokens_per_block=tokens_per_block,
        blocks_per_window=blocks_per_window,
        is_state_group=is_state_group,
    )


def _full_attention_spec(block_size=16):
    return FullAttentionSpec(block_size=block_size, num_kv_heads=1, head_size=2, dtype=torch.float16)


def _make_group(spec, layer_names=None):
    return types.SimpleNamespace(kv_cache_spec=spec, layer_names=layer_names or [])


def _fake_kv_cache_config(specs):
    return types.SimpleNamespace(kv_cache_groups=[_make_group(spec) for spec in specs])


class FakeRequest:
    """Minimal stand-in for ``vllm.v1.request.Request`` used by truncation tests."""

    def __init__(
        self,
        *,
        kv_transfer_params=None,
        prompt_token_ids=None,
        prompt_embeds=None,
        num_prompt_tokens=None,
        all_token_ids=None,
    ):
        self.kv_transfer_params = kv_transfer_params
        self.prompt_token_ids = prompt_token_ids
        self.prompt_embeds = prompt_embeds
        self._all_token_ids = list(prompt_token_ids if all_token_ids is None else all_token_ids)
        self.num_prompt_tokens = len(self._all_token_ids) if num_prompt_tokens is None else num_prompt_tokens
        self.max_tokens = 16
        self.block_hashes = ["old-prompt-hash"]

    def update_block_hashes(self):
        self.block_hashes.append("short-prompt-hash")


# --------------------------------------------------------------------- pull


def test_transfer_block_ids_trims_attention_mtp_blocks():
    block_ids = transfer_block_ids(
        ([10, 11, 12, 13, 14],),
        33,
        group_transfer_info=[_group_info()],
        mamba_cache_mode=None,
        dcp_size=1,
    )
    assert block_ids == ([10, 11, 12],)


def test_transfer_block_ids_selects_aligned_prompt_state_block():
    block_ids = transfer_block_ids(
        ([20, 21, 22, 23, 24],),
        33,
        group_transfer_info=[_group_info(is_state_group=True)],
        mamba_cache_mode="align",
        dcp_size=1,
    )
    assert block_ids == ([22],)


def test_transfer_block_ids_keeps_non_aligned_state_group():
    block_ids = transfer_block_ids(
        ([20, 21, 22, 23],),
        16,
        group_transfer_info=[_group_info(is_state_group=True)],
        mamba_cache_mode="none",
        dcp_size=1,
    )
    assert block_ids == ([20, 21, 22, 23],)


def test_transfer_block_ids_rejects_short_aligned_state_metadata():
    with pytest.raises(RuntimeError) as excinfo:
        transfer_block_ids(
            ([20, 21],),
            48,
            group_transfer_info=[_group_info(is_state_group=True)],
            mamba_cache_mode="align",
            dcp_size=1,
        )
    assert str(excinfo.value) == (
        "Invalid aligned Mamba state block metadata: prompt_len=48, tokens_per_block=16, "
        "required_block_count=3, available_block_count=2."
    )


def test_transfer_block_ids_uses_compressed_prompt_len():
    block_ids = transfer_block_ids(
        ([30, 31, 32, 33],),
        64,
        group_transfer_info=[_group_info(tokens_per_block=32)],
        mamba_cache_mode=None,
        dcp_size=1,
    )
    assert block_ids == ([30, 31],)


def test_transfer_block_ids_applies_dcp_grouping():
    block_ids = transfer_block_ids(
        ([50, 51, 52],),
        32,
        group_transfer_info=[_group_info()],
        mamba_cache_mode=None,
        dcp_size=2,
    )
    assert block_ids == ([50],)


def test_transfer_block_ids_trims_sliding_window_mtp_blocks():
    block_ids = transfer_block_ids(
        ([40, 41, 42, 43, 44],),
        48,
        group_transfer_info=[_group_info(blocks_per_window=3)],
        mamba_cache_mode=None,
        dcp_size=1,
    )
    assert block_ids == ([40, 41, 42],)


def test_swa_transfer_block_ids_clips_sliding_window_group():
    block_ids = swa_transfer_block_ids(
        ([40, 41, 42, 43, 44],),
        group_transfer_info=[_group_info(blocks_per_window=3)],
    )
    assert block_ids == ([42, 43, 44],)


def test_swa_transfer_block_ids_drops_zero_from_sliding_window_tail():
    block_ids = swa_transfer_block_ids(
        ([0, 10],),
        group_transfer_info=[_group_info(blocks_per_window=2)],
    )
    assert block_ids == ([10],)


def test_transfer_then_swa_trims_mtp_before_zero_filter():
    group_transfer_info = [_group_info(blocks_per_window=3)]
    block_ids = transfer_block_ids(
        ([0, 10, 11, 12, 13],),
        32,
        group_transfer_info=group_transfer_info,
        mamba_cache_mode=None,
        dcp_size=1,
    )
    block_ids = swa_transfer_block_ids(block_ids, group_transfer_info=group_transfer_info)
    assert block_ids == ([10],)


def test_transfer_and_swa_reject_group_count_mismatch():
    group_transfer_info = [_group_info(), _group_info()]
    with pytest.raises(AssertionError, match="Number of KV cache groups must match"):
        transfer_block_ids(
            ([1],),
            16,
            group_transfer_info=group_transfer_info,
            mamba_cache_mode=None,
            dcp_size=1,
        )
    with pytest.raises(AssertionError, match="Number of KV cache groups must match"):
        swa_transfer_block_ids(([1],), group_transfer_info=group_transfer_info)


@pytest.mark.parametrize("need_truncate", [False, True])
@pytest.mark.parametrize("num_prompt_tokens", [0, 1, 2, 32])
def test_state_prefill_token_count_boundary(need_truncate, num_prompt_tokens):
    expected = num_prompt_tokens - 1 if need_truncate and num_prompt_tokens > 1 else num_prompt_tokens
    assert state_prefill_token_count(num_prompt_tokens, need_truncate=need_truncate) == expected


def test_truncate_request_for_prefill_truncates_once():
    request = FakeRequest(kv_transfer_params={"do_remote_decode": True}, prompt_token_ids=[1, 2, 3, 4])
    truncate_request_for_prefill(request)
    assert request.prompt_token_ids == [1, 2, 3]
    assert request._all_token_ids == [1, 2, 3]
    assert request.num_prompt_tokens == 3
    assert request.max_tokens == 1
    assert request.block_hashes == ["short-prompt-hash"]
    assert request.kv_transfer_params == {"do_remote_decode": True, "_p_side_truncated": True}

    truncate_request_for_prefill(request)
    assert request.prompt_token_ids == [1, 2, 3]
    assert request._all_token_ids == [1, 2, 3]
    assert request.num_prompt_tokens == 3


def test_truncate_request_for_prefill_handles_prompt_embeds():
    request = FakeRequest(kv_transfer_params={}, prompt_embeds=[0.1, 0.2, 0.3, 0.4], all_token_ids=[1, 2, 3, 4])
    truncate_request_for_prefill(request)
    assert request.prompt_embeds == [0.1, 0.2, 0.3]
    assert request.prompt_token_ids is None
    assert request.num_prompt_tokens == 3
    assert request.max_tokens == 1


@pytest.mark.parametrize(
    "params,prompt_token_ids,num_prompt_tokens",
    [
        (None, [1, 2, 3], 3),
        ({"_p_side_truncated": True}, [1, 2, 3], 3),
        ({}, [1], 1),
        ({}, None, 3),
    ],
    ids=["no-params", "already-truncated", "single-token", "no-payload"],
)
def test_truncate_request_for_prefill_guards(params, prompt_token_ids, num_prompt_tokens):
    request = FakeRequest(
        kv_transfer_params=params,
        prompt_token_ids=prompt_token_ids,
        num_prompt_tokens=num_prompt_tokens,
    )
    truncate_request_for_prefill(request)
    assert request._all_token_ids == list(prompt_token_ids or [])
    assert request.num_prompt_tokens == num_prompt_tokens
    assert request.max_tokens == 16
    assert request.block_hashes == ["old-prompt-hash"]
    if params is not None:
        assert "_p_side_truncated" not in params


# ---------------------------------------------------------------- layerwise


@pytest.mark.parametrize("need_truncate", [False, True])
@pytest.mark.parametrize("num_prompt_tokens", [0, 1, 2, 32])
def test_hybrid_prefill_token_count_boundary(need_truncate, num_prompt_tokens):
    expected = num_prompt_tokens - 1 if need_truncate and num_prompt_tokens > 1 else num_prompt_tokens
    assert hybrid_prefill_token_count(num_prompt_tokens, need_truncate=need_truncate) == expected


def test_truncate_request_for_hybrid_prefill_need_truncate_guard():
    request = FakeRequest(kv_transfer_params={}, prompt_token_ids=[1, 2, 3])
    truncate_request_for_hybrid_prefill(request, need_truncate=False)
    assert request.prompt_token_ids == [1, 2, 3]
    assert request.num_prompt_tokens == 3
    assert request.max_tokens == 16


def test_truncate_request_for_hybrid_prefill_already_truncated_guard():
    request = FakeRequest(kv_transfer_params={"_p_side_truncated": True}, prompt_token_ids=[1, 2, 3])
    truncate_request_for_hybrid_prefill(request, need_truncate=True)
    assert request.prompt_token_ids == [1, 2, 3]
    assert request.num_prompt_tokens == 3


def test_truncate_request_for_hybrid_prefill_prompt_embeds_path():
    request = FakeRequest(kv_transfer_params={}, prompt_embeds=[0.1, 0.2, 0.3], all_token_ids=[1, 2, 3])
    truncate_request_for_hybrid_prefill(request, need_truncate=True)
    assert request.prompt_embeds == [0.1, 0.2]
    assert request.num_prompt_tokens == 2
    assert request.max_tokens == 1
    assert request.kv_transfer_params == {"_p_side_truncated": True}


@pytest.mark.parametrize(
    "prompt_len,expected",
    [
        (17, ([1, 2], [4])),
        (32, ([1, 2, 3], [4, 5])),
        (1, ([1, 2, 3], [4, 5])),
    ],
    ids=["trims-when-remainder-1", "keeps-multiple-of-block-size", "keeps-single-token"],
)
def test_trim_hybrid_remote_block_ids(prompt_len, expected):
    block_ids = ([1, 2, 3], [4, 5])
    trimmed = trim_hybrid_remote_block_ids(
        block_ids,
        prompt_len,
        need_truncate=True,
        block_sizes=[16, 16],
    )
    assert trimmed == expected
    # The input is never mutated.
    assert block_ids == ([1, 2, 3], [4, 5])


def test_trim_hybrid_remote_block_ids_keeps_input_when_disabled():
    block_ids = ([1, 2, 3], [4, 5])
    assert trim_hybrid_remote_block_ids(block_ids, 17, need_truncate=False, block_sizes=[16, 16]) is block_ids


# ------------------------------------------------------- real-spec coverage


def test_build_group_transfer_info_from_attention_spec():
    group = _make_group(_full_attention_spec(block_size=16))
    info = build_group_transfer_info(group, default_block_size=8)
    assert info == GroupTransferInfo(tokens_per_block=16, blocks_per_window=0, is_state_group=False)


def test_build_group_transfer_info_from_sliding_window_spec():
    spec = SlidingWindowSpec(block_size=16, num_kv_heads=1, head_size=2, dtype=torch.float16, sliding_window=32)
    info = build_group_transfer_info(_make_group(spec), default_block_size=8)
    assert info == GroupTransferInfo(tokens_per_block=16, blocks_per_window=3, is_state_group=False)


def test_build_group_transfer_info_from_mamba_spec():
    spec = MambaSpec(block_size=16, shapes=((16,), (16,)), dtypes=(torch.uint8, torch.uint8))
    info = build_group_transfer_info(_make_group(spec), default_block_size=8)
    assert info == GroupTransferInfo(tokens_per_block=16, blocks_per_window=0, is_state_group=True)


def test_build_group_transfer_info_from_uniform_hybrid_specs():
    mamba = MambaSpec(block_size=16, shapes=((16,), (16,)), dtypes=(torch.uint8, torch.uint8))
    attention = _full_attention_spec(block_size=16)
    uniform = UniformTypeKVCacheSpecs(
        block_size=16,
        kv_cache_specs={"attn": attention, "mamba": mamba},
    )
    group = _make_group(uniform, layer_names=["attn", "mamba"])
    info = build_group_transfer_info(group, default_block_size=8)
    assert info.is_state_group is True
    assert info.tokens_per_block == 16


def test_group_unique_specs_deduplicates_uniform_specs():
    attention = _full_attention_spec(block_size=16)
    sliding = SlidingWindowSpec(block_size=16, num_kv_heads=1, head_size=2, dtype=torch.float16, sliding_window=16)
    uniform = UniformTypeKVCacheSpecs(
        block_size=16,
        kv_cache_specs={"a": attention, "b": attention, "c": sliding},
    )
    group = _make_group(uniform, layer_names=["a", "b", "c"])
    assert group_unique_specs(group) == [attention, sliding]


def test_iter_kv_cache_specs_flattens_uniform_and_detects_hybrid_cache():
    attention = _full_attention_spec(block_size=16)
    mamba = MambaSpec(block_size=16, shapes=((16,), (16,)), dtypes=(torch.uint8, torch.uint8))
    uniform = UniformTypeKVCacheSpecs(block_size=16, kv_cache_specs={"attn": attention, "mamba": mamba})
    kv_cache_config = _fake_kv_cache_config([attention, uniform, mamba])
    assert list(iter_kv_cache_specs(kv_cache_config)) == [attention, attention, mamba, mamba]
    assert has_attn_mamba_hybrid_cache(kv_cache_config) is True

    attention_only = _fake_kv_cache_config([attention, attention])
    assert has_attn_mamba_hybrid_cache(attention_only) is False

    mamba_only = _fake_kv_cache_config([mamba])
    assert has_attn_mamba_hybrid_cache(mamba_only) is False
