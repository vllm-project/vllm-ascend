# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm_ascend.ops.triton.gdn import recoverssm_gdn as recovery


def make_layer(dim_first, *, num_heads=2, value_dim=4, key_dim=8):
    num_key_heads = 1
    channels = 2 * num_key_heads * key_dim + num_heads * value_dim
    # Keep a wider physical row so tests also cover noncompact channel strides.
    conv = torch.zeros(4, channels + 8, 7, dtype=torch.bfloat16)[:, :channels]
    state = torch.empty(4, num_heads, value_dim, key_dim)
    replay = torch.empty(4, num_heads, 5, value_dim + key_dim + 1)
    return SimpleNamespace(
        kv_cache=(conv if dim_first else conv.transpose(-1, -2), state, replay),
        num_k_heads=num_key_heads,
        tp_size=1,
    )


@pytest.mark.parametrize("dim_first", [False, True])
def test_context_keeps_physical_page_strides_and_conv_orientation(monkeypatch, dim_first):
    monkeypatch.setattr(recovery, "is_conv_state_dim_first", lambda: dim_first)
    layers = [make_layer(dim_first), make_layer(dim_first)]
    context = recovery.GDNRecoverSSMCommitContext.create(layers, spec_query_len=5, max_num_reqs=4)
    assert context.conv_history_len == 3
    assert len(context.layer_caches) == len(layers)
    assert context._conv_cache_address_stride_table.shape == (4, len(layers))
    assert context._conv_cache_address_stride_table.is_contiguous()
    assert context._state_commit_cache_address_stride_table.shape == (2, 2, len(layers))
    assert context._state_commit_cache_address_stride_table.is_contiguous()
    for index, layer in enumerate(layers):
        cache = context.layer_caches[index]
        assert cache.conv.shape == (4, 24, 7)
        assert cache.conv.stride(0) == 32 * 7
        assert cache.conv_base_addr == cache.conv.data_ptr()
        assert cache.conv_block_stride == cache.conv.stride(0)
        assert cache.conv_dim_stride == cache.conv.stride(1)
        assert cache.conv_token_stride == cache.conv.stride(2)
        assert context._conv_cache_address_stride_table[0, index] == cache.conv_base_addr
        assert context._conv_cache_address_stride_table[1, index] == cache.conv_block_stride
        assert context._conv_cache_address_stride_table[2, index] == cache.conv_dim_stride
        assert context._conv_cache_address_stride_table[3, index] == cache.conv_token_stride
        assert cache.checkpoint is layer.kv_cache[1]
        assert cache.replay is layer.kv_cache[2]
        assert cache.checkpoint_base_addr == cache.checkpoint.data_ptr()
        assert cache.checkpoint_block_stride == cache.checkpoint.stride(0)
        assert cache.replay_base_addr == cache.replay.data_ptr()
        assert cache.replay_block_stride == cache.replay.stride(0)
        layout_pairs = (
            (cache.checkpoint_base_addr, cache.checkpoint_block_stride),
            (cache.replay_base_addr, cache.replay_block_stride),
        )
        for kind, (base_addr, block_stride) in enumerate(layout_pairs):
            assert context._state_commit_cache_address_stride_table[kind, 0, index] == base_addr
            assert context._state_commit_cache_address_stride_table[kind, 1, index] == block_stride


def test_context_accepts_bf16_checkpoint(monkeypatch):
    monkeypatch.setattr(recovery, "is_conv_state_dim_first", lambda: True)
    layer = make_layer(True)
    layer.kv_cache = (layer.kv_cache[0], layer.kv_cache[1].bfloat16(), *layer.kv_cache[2:])
    context = recovery.GDNRecoverSSMCommitContext.create([layer], spec_query_len=5, max_num_reqs=4)
    assert context.layer_caches[0].checkpoint.dtype == torch.bfloat16


def test_context_preserves_transposed_inner_state(monkeypatch):
    monkeypatch.setattr(recovery, "is_conv_state_dim_first", lambda: True)
    layer = make_layer(True)
    state = torch.empty(4, 2, 8, 4).transpose(-1, -2)
    layer.kv_cache = (layer.kv_cache[0], state, *layer.kv_cache[2:])
    context = recovery.GDNRecoverSSMCommitContext.create([layer], spec_query_len=5, max_num_reqs=4)
    assert context.layer_caches[0].checkpoint.stride() == state.stride()


def test_empty_commit_does_not_launch_and_incomplete_align_is_rejected(monkeypatch):
    monkeypatch.setattr(recovery, "is_conv_state_dim_first", lambda: True)
    context = recovery.GDNRecoverSSMCommitContext.create([make_layer(True)], spec_query_len=5, max_num_reqs=4)
    kernel = Mock(side_effect=AssertionError("empty commit launched a kernel"))
    monkeypatch.setattr(recovery, "_commit_gdn_state_kernel", kernel)
    monkeypatch.setattr(recovery, "_commit_gdn_state_grouped_kernel", kernel)
    monkeypatch.setattr(recovery, "_prepare_commit_plan_kernel", kernel)
    monkeypatch.setattr(recovery, "_compact_conv_state_kernel", kernel)
    empty = torch.empty(0, dtype=torch.int32)
    starts = torch.zeros(1, dtype=torch.int32)
    context.commit(empty, empty, starts)
    context.commit(empty, empty, starts, request_indices=torch.ones(1, dtype=torch.int32))
    with pytest.raises(ValueError, match="align metadata is incomplete"):
        context.commit(empty, empty, starts, block_table=torch.empty(0, 0))
    with pytest.raises(ValueError, match="acceptance counts"):
        context.commit(empty, torch.ones(1, dtype=torch.int32), torch.tensor([0, 1]))
    with pytest.raises(ValueError, match="sampler mapping"):
        context.commit(
            torch.ones(1, dtype=torch.int32),
            torch.ones(1, dtype=torch.int32),
            torch.tensor([0, 1]),
            request_indices=torch.empty(0, dtype=torch.int32),
        )
    with pytest.raises(ValueError, match="sampler mapping"):
        context.commit(
            torch.ones(1, dtype=torch.int32),
            torch.ones(1, dtype=torch.int32),
            torch.tensor([0, 1]),
            request_indices=torch.ones(1, 1, dtype=torch.int32),
        )


@pytest.mark.parametrize("align", [False, True])
@pytest.mark.parametrize("commit_conv", [False, True])
def test_commit_launches_plan_conv_state_in_order(monkeypatch, align, commit_conv):
    monkeypatch.setattr(recovery, "is_conv_state_dim_first", lambda: True)
    monkeypatch.setattr(recovery, "is_950", lambda: False)
    monkeypatch.setattr(recovery, "get_ub_size_bytes", lambda: 64 * 1024)
    layer = make_layer(True, value_dim=128, key_dim=128)
    context = recovery.GDNRecoverSSMCommitContext.create([layer], spec_query_len=5, max_num_reqs=4)
    launches = []

    class Kernel:
        def __init__(self, name):
            self.name = name

        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                launches.append((self.name, grid, args, kwargs))

            return launch

    for name in ("_prepare_commit_plan_kernel", "_compact_conv_state_kernel", "_commit_gdn_state_kernel"):
        monkeypatch.setattr(recovery, name, Kernel(name))
    request_indices = torch.tensor([0, 3], dtype=torch.int32)
    block_table = None
    num_computed_tokens = None
    mamba_block_size = None
    if align:
        block_table = torch.tensor([[1, 2]], dtype=torch.int32)
        num_computed_tokens = torch.tensor([6], dtype=torch.int32)
        mamba_block_size = 8
    context.commit(
        torch.tensor([4], dtype=torch.int32),
        torch.tensor([1], dtype=torch.int32),
        torch.tensor([0, 5], dtype=torch.int32),
        request_indices,
        block_table,
        num_computed_tokens,
        mamba_block_size,
        commit_conv,
    )
    expected_launches = ["_prepare_commit_plan_kernel"]
    if commit_conv:
        expected_launches.append("_compact_conv_state_kernel")
    expected_launches.append("_commit_gdn_state_kernel")
    assert [entry[0] for entry in launches] == expected_launches
    assert launches[0][2][6] is context.commit_plan.commit_lens
    assert launches[0][2][1] is request_indices
    if commit_conv:
        assert launches[1][2][6] is context.commit_plan.commit_lens
    assert launches[-1][3]["commit_lens_ptr"] is context.commit_plan.commit_lens
    assert launches[-1][3]["replay_ref_ptr"] is context.layer_caches[0].replay
    # A 64-KB UB permits BV=32 for the ordinary path. The aligned GDN
    # single-launch path carries boundary accumulators too, reducing BV to 16.
    expected_block_v = 16 if align else 32
    assert launches[-1][1] == ((128 // expected_block_v) * 2,)
    assert launches[-1][3]["BK"] == 128
    assert launches[-1][3]["BV"] == expected_block_v
    if commit_conv:
        assert launches[1][3]["ALIGN_MODE"] is align
    assert launches[-1][3]["ALIGN_MODE"] is align
    assert context._conv_cache_address_stride_table.shape == (4, len(context.layer_caches))
    assert context._conv_cache_address_stride_table.is_contiguous()
    cache = context.layer_caches[0]
    assert cache.checkpoint_base_addr == cache.checkpoint.data_ptr()
    assert cache.checkpoint_block_stride == cache.checkpoint.stride(0)
    assert cache.replay_base_addr == cache.replay.data_ptr()
    assert cache.replay_block_stride == cache.replay.stride(0)


def test_align_window_and_metadata_constraints(monkeypatch):
    monkeypatch.setattr(recovery, "is_conv_state_dim_first", lambda: True)
    context = recovery.GDNRecoverSSMCommitContext.create([make_layer(True)], spec_query_len=5, max_num_reqs=4)
    args = (torch.tensor([2]), torch.tensor([1]), torch.tensor([0, 5]))
    align = dict(block_table=torch.tensor([[1, 2]]), num_computed_tokens=torch.tensor([6]), mamba_block_size=8)
    with pytest.raises(ValueError, match="cover one speculative window"):
        context.commit(*args, **{**align, "mamba_block_size": 4})
    with pytest.raises(ValueError, match="two-dimensional"):
        context.commit(*args, **{**align, "block_table": torch.tensor([1, 2])})
    with pytest.raises(ValueError, match="integer indices"):
        context.commit(*args, **{**align, "block_table": align["block_table"].float()})
    with pytest.raises(ValueError, match="cover the sampler rows"):
        context.commit(*args, **{**align, "num_computed_tokens": torch.empty(0, dtype=torch.int32)})


def test_context_rejects_non_fp32_replay(monkeypatch):
    monkeypatch.setattr(recovery, "is_conv_state_dim_first", lambda: True)
    layer = make_layer(True)
    layer.kv_cache = (*layer.kv_cache[:2], layer.kv_cache[2].bfloat16())
    with pytest.raises(ValueError, match="replay must be FP32"):
        recovery.GDNRecoverSSMCommitContext.create([layer], spec_query_len=5, max_num_reqs=4)
