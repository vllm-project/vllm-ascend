# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import torch

from vllm_ascend.ascend_forward_context import MoECommType
from vllm_ascend.ops import register_custom_ops as custom_ops


def _patch_sp_ep_context(monkeypatch):
    monkeypatch.setattr(
        custom_ops,
        "_EXTRA_CTX",
        SimpleNamespace(flash_comm_v1_enabled=False, moe_comm_type=MoECommType.ALLGATHER),
    )
    monkeypatch.setattr(custom_ops, "enable_sp_by_pass", lambda: True)
    monkeypatch.setattr(custom_ops, "get_forward_context", lambda: SimpleNamespace(dp_metadata=object()))
    monkeypatch.setattr(custom_ops, "get_ep_group", lambda: SimpleNamespace(world_size=4))
    monkeypatch.setattr(custom_ops, "get_tensor_model_parallel_world_size", lambda: 2)


def test_fake_sp_ep_all_gather_uses_ep_group_size(monkeypatch):
    _patch_sp_ep_context(monkeypatch)

    result = custom_ops._maybe_all_gather_and_maybe_unpad_fake(torch.empty(2, 8), True, True)

    assert result.shape == (8, 8)


def test_fake_sp_ep_reduce_scatter_uses_ep_group_size(monkeypatch):
    _patch_sp_ep_context(monkeypatch)

    result = custom_ops._maybe_pad_and_reduce_fake(torch.empty(8, 8), True)

    assert result.shape == (2, 8)


def test_sp_ep_all_gather_unpads_dp_chunks(monkeypatch):
    context = SimpleNamespace(dp_metadata=SimpleNamespace(get_chunk_sizes_across_dp_rank=lambda: [1, 1, 3, 3]))
    monkeypatch.setattr(custom_ops, "_EXTRA_CTX", SimpleNamespace(flash_comm_v1_enabled=False))
    monkeypatch.setattr(custom_ops, "get_forward_context", lambda: context)
    monkeypatch.setattr(custom_ops, "enable_sp_by_pass", lambda: True)

    class Group:
        def all_gather(self, x, dim):
            return torch.arange(48, dtype=x.dtype).view(12, 4)

    monkeypatch.setattr(custom_ops, "get_ep_group", lambda: Group())
    result = custom_ops._maybe_all_gather_and_maybe_unpad_impl(torch.empty(3, 4), True, True)

    assert result.shape == (8, 4)
    assert torch.equal(result[:, 0], torch.tensor([0, 12, 24, 28, 32, 36, 40, 44], dtype=result.dtype))


def test_sp_ep_reduce_scatter_pads_dp_chunks(monkeypatch):
    context = SimpleNamespace(
        dp_metadata=SimpleNamespace(get_chunk_sizes_across_dp_rank=lambda: [1, 1, 3, 3]),
        is_draft_model=False,
    )
    monkeypatch.setattr(custom_ops, "_EXTRA_CTX", SimpleNamespace(flash_comm_v1_enabled=False))
    monkeypatch.setattr(custom_ops, "get_forward_context", lambda: context)
    monkeypatch.setattr(custom_ops, "enable_sp_by_pass", lambda: True)

    class Group:
        def reduce_scatter(self, x, dim):
            assert x.shape == (12, 4)
            assert torch.equal(x[:, 0], torch.tensor([0, 0, 0, 4, 0, 0, 8, 12, 16, 20, 24, 28], dtype=x.dtype))
            return x[:3]

    monkeypatch.setattr(custom_ops, "get_ep_group", lambda: Group())
    result = custom_ops._maybe_pad_and_reduce_impl(torch.arange(32).view(8, 4), True)

    assert result.shape == (3, 4)
