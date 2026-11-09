# SPDX-License-Identifier: Apache-2.0


import pytest
import torch

from vllm_ascend.models.qwen4_exp.topk_buffer import share_target_topk_buffers, share_topk_buffer


def layer(width=4):
    module = torch.nn.Module()
    module.register_buffer("topk_indices_buffer", torch.empty(8, width, dtype=torch.int32), persistent=False)
    share_topk_buffer(module)
    return module


def test_target_storage_and_draft_isolation():
    with share_target_topk_buffers(True):
        first, second = layer(), layer()
        different = layer(5)
    draft = layer()
    assert first.topk_indices_buffer is second.topk_indices_buffer
    assert different.topk_indices_buffer is not first.topk_indices_buffer
    assert draft.topk_indices_buffer is not first.topk_indices_buffer
    assert not first.state_dict()
    first.topk_indices_buffer.fill_(7)
    assert torch.all(second.topk_indices_buffer == 7)


def test_disabled_and_exception_cleanup():
    with share_target_topk_buffers(False):
        first, second = layer(), layer()
    assert first.topk_indices_buffer is not second.topk_indices_buffer
    with pytest.raises(RuntimeError), share_target_topk_buffers(True):
        old = layer()
        raise RuntimeError("construction failed")
    assert layer().topk_indices_buffer is not old.topk_indices_buffer


def test_nested_model_scopes():
    with share_target_topk_buffers(True):
        outer = layer()
        with share_target_topk_buffers(True):
            inner = layer()
        assert layer().topk_indices_buffer is outer.topk_indices_buffer
        assert inner.topk_indices_buffer is not outer.topk_indices_buffer
