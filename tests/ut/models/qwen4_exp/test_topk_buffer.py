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


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_distinct_model_scopes_and_dtype(dtype):
    with share_target_topk_buffers(True):
        first = layer()
    with share_target_topk_buffers(True):
        second = layer()
        other = torch.nn.Module()
        other.register_buffer("topk_indices_buffer", torch.empty(8, 4, dtype=dtype), persistent=False)
        share_topk_buffer(other)
    assert first.topk_indices_buffer is not second.topk_indices_buffer
    assert (other.topk_indices_buffer is second.topk_indices_buffer) == (dtype == torch.int32)


def test_different_strides_do_not_share():
    with share_target_topk_buffers(True):
        first = layer()
        other = torch.nn.Module()
        other.register_buffer("topk_indices_buffer", torch.empty(4, 8, dtype=torch.int32).t(), persistent=False)
        share_topk_buffer(other)
    assert first.topk_indices_buffer.shape == other.topk_indices_buffer.shape
    assert first.topk_indices_buffer is not other.topk_indices_buffer


def test_disabled_nested_scope_restores_target_buffers():
    with share_target_topk_buffers(True):
        target = layer()
        with share_target_topk_buffers(False):
            first_draft, second_draft = layer(), layer()
        assert layer().topk_indices_buffer is target.topk_indices_buffer
    assert first_draft.topk_indices_buffer is not second_draft.topk_indices_buffer
    assert first_draft.topk_indices_buffer is not target.topk_indices_buffer


def test_different_devices_do_not_share():
    with share_target_topk_buffers(True):
        cpu_layer = layer()
        meta_layer = torch.nn.Module()
        meta_layer.register_buffer(
            "topk_indices_buffer", torch.empty(8, 4, dtype=torch.int32, device="meta"), persistent=False
        )
        share_topk_buffer(meta_layer)
    assert cpu_layer.topk_indices_buffer.device.type == "cpu"
    assert meta_layer.topk_indices_buffer.device.type == "meta"
