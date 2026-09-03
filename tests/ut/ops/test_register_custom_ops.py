# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import torch

from vllm_ascend.ops import register_custom_ops as custom_ops


class _EpGroup:
    world_size = 4
    rank_in_group = 2

    def all_gather(self, x: torch.Tensor, dim: int) -> torch.Tensor:
        assert dim == 0
        assert x.shape == (3, 4)
        return torch.arange(48, dtype=x.dtype).view(12, 4)

    def reduce_scatter(self, x: torch.Tensor, dim: int) -> torch.Tensor:
        assert dim == 0
        assert x.shape == (12, 4)
        assert torch.equal(
            x[:, 0],
            torch.tensor([0, 0, 0, 4, 0, 0, 8, 12, 16, 20, 24, 28], dtype=x.dtype),
        )
        return x[:3]


class _EpGroupRank0(_EpGroup):
    rank_in_group = 0


def _patch_sp_ep_context(monkeypatch):
    context = SimpleNamespace(
        dp_metadata=SimpleNamespace(
            get_chunk_sizes_across_dp_rank=lambda: [1, 1, 3, 3],
        ),
        is_draft_model=False,
    )
    monkeypatch.setattr(custom_ops, "_EXTRA_CTX", context)
    monkeypatch.setattr(custom_ops, "get_forward_context", lambda: context)
    monkeypatch.setattr(custom_ops, "get_ep_group", _EpGroup)


def test_sp_ep_all_gather_pads_and_unpads_local_chunks(monkeypatch):
    _patch_sp_ep_context(monkeypatch)

    result = custom_ops._maybe_all_gather_and_maybe_unpad_impl(torch.empty(1, 4))

    assert result.shape == (8, 4)
    assert torch.equal(
        result[:, 0],
        torch.tensor([0, 12, 24, 28, 32, 36, 40, 44], dtype=result.dtype),
    )


def test_sp_ep_reduce_scatter_pads_local_chunks(monkeypatch):
    _patch_sp_ep_context(monkeypatch)

    result = custom_ops._maybe_pad_and_reduce_impl(torch.arange(32).view(8, 4))

    assert result.shape == (3, 4)


def test_sp_ep_reduce_scatter_unpads_local_chunk(monkeypatch):
    _patch_sp_ep_context(monkeypatch)
    monkeypatch.setattr(custom_ops, "get_ep_group", _EpGroupRank0)

    result = custom_ops._maybe_pad_and_reduce_impl(torch.arange(32).view(8, 4))

    assert result.shape == (1, 4)


def test_sp_ep_fake_shapes_follow_uneven_local_chunks(monkeypatch):
    _patch_sp_ep_context(monkeypatch)

    gathered = custom_ops._maybe_all_gather_and_maybe_unpad_fake(torch.empty(1, 4))
    reduced = custom_ops._maybe_pad_and_reduce_fake(torch.empty(8, 4))

    assert gathered.shape == (8, 4)
    assert reduced.shape == (3, 4)


def test_rope_fake_uses_requested_output_dtype():
    positions = torch.arange(2)
    query = torch.empty(2, 128, dtype=torch.bfloat16)
    key = torch.empty(2, 64, dtype=torch.bfloat16)

    query_out, key_out = custom_ops._rope_forward_oot_impl_fake(
        positions,
        query,
        key,
        torch.empty(16, 64, dtype=torch.bfloat16),
        64,
        64,
        out_dtype=torch.float8_e4m3fn,
    )

    assert query_out.shape == query.shape
    assert key_out.shape == key.shape
    assert query_out.dtype == torch.float8_e4m3fn
    assert key_out.dtype == torch.float8_e4m3fn
