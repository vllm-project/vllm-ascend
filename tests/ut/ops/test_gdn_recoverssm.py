# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm_ascend.ops.triton import gdn_recoverssm as recovery


def make_layer(dim_first):
    conv = torch.zeros(4, 32, 7, dtype=torch.bfloat16)
    state = torch.empty(4, 2, 4, 8)
    records = (torch.empty(4, 2, 5, 4), torch.empty(4, 1, 5, 8), torch.empty(4, 2, 5))
    # Two key projections (8 channels each) plus 2*4 value channels.
    conv = conv[:, :24]
    return SimpleNamespace(kv_cache=(conv if dim_first else conv.transpose(-1, -2), state, *records))


@pytest.mark.parametrize("dim_first", [False, True])
def test_context_keeps_physical_page_strides_and_conv_orientation(monkeypatch, dim_first):
    monkeypatch.setattr(recovery, "is_conv_state_dim_first", lambda: dim_first)
    layers = [make_layer(dim_first), make_layer(dim_first)]
    context = recovery.GDNRecoverSSMCommitContext.create(layers, spec_query_len=5, max_num_reqs=4)
    assert context.conv_history_len == 3
    for index, layer in enumerate(layers):
        conv = context.pages[index][0]
        assert conv.shape == (4, 24, 7)
        assert conv.stride(0) == 32 * 7
        for offset, tensor in enumerate(context.pages[index]):
            desc = context.descriptors[index * 5 + offset]
            assert desc[0] == tensor.data_ptr()
            assert desc[1] == tensor.stride(0)
        assert context.pages[index][1] is layer.kv_cache[1]


def test_context_rejects_bf16_checkpoint_before_launch(monkeypatch):
    monkeypatch.setattr(recovery, "is_conv_state_dim_first", lambda: True)
    layer = make_layer(True)
    layer.kv_cache = (layer.kv_cache[0], layer.kv_cache[1].bfloat16(), *layer.kv_cache[2:])
    with pytest.raises(ValueError, match="FP32 page"):
        recovery.GDNRecoverSSMCommitContext.create([layer], spec_query_len=5, max_num_reqs=4)


def test_context_rejects_transposed_inner_state(monkeypatch):
    monkeypatch.setattr(recovery, "is_conv_state_dim_first", lambda: True)
    layer = make_layer(True)
    state = torch.empty(4, 2, 8, 4).transpose(-1, -2)
    layer.kv_cache = (layer.kv_cache[0], state, *layer.kv_cache[2:])
    with pytest.raises(ValueError, match="inner dimensions"):
        recovery.GDNRecoverSSMCommitContext.create([layer], spec_query_len=5, max_num_reqs=4)


def test_empty_commit_does_not_launch_and_align_is_rejected(monkeypatch):
    monkeypatch.setattr(recovery, "is_conv_state_dim_first", lambda: True)
    context = recovery.GDNRecoverSSMCommitContext.create([make_layer(True)], spec_query_len=5, max_num_reqs=4)
    kernel = Mock(side_effect=AssertionError("empty commit launched a kernel"))
    monkeypatch.setattr(recovery, "_commit_state_kernel", kernel)
    monkeypatch.setattr(recovery, "_commit_conv_kernel", kernel)
    empty = torch.empty(0, dtype=torch.int32)
    starts = torch.zeros(1, dtype=torch.int32)
    context.commit(empty, empty, starts)
    with pytest.raises(ValueError, match="mode none"):
        context.commit(empty, empty, starts, block_table=torch.empty(0, 0))
    with pytest.raises(ValueError, match="acceptance counts"):
        context.commit(empty, torch.ones(1, dtype=torch.int32), torch.tensor([0, 1]))
    with pytest.raises(ValueError, match="sampler mapping"):
        context.commit(empty, empty, starts, request_indices=torch.ones(1, dtype=torch.int32))
