# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest
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


def test_sp_ep_reduce_scatter_draft_model_keeps_ep_layout(monkeypatch):
    _patch_sp_ep_context(monkeypatch)
    custom_ops._EXTRA_CTX.is_draft_model = True

    def unexpected_tp_all_reduce(_x):
        raise AssertionError("EP/SP finalize must not use TP AllReduce")

    monkeypatch.setattr(
        custom_ops,
        "tensor_model_parallel_all_reduce",
        unexpected_tp_all_reduce,
        raising=False,
    )

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


@pytest.mark.parametrize("dp_size,pcp_size,sp_size", [(1, 2, 2), (2, 1, 2), (2, 2, 2), (2, 2, 4)])
def test_sp_ep_pcp_token_order_and_round_trip(monkeypatch, dp_size, pcp_size, sp_size):
    # Uneven DP batches, including TP padding, must preserve DP/PCP/TP order.
    dp_tokens = [1, 9][:dp_size]
    sp_sizes = [(tokens + sp_size - 1) // sp_size for tokens in dp_tokens]
    local_sizes = [size for size in sp_sizes for _ in range(sp_size)]
    ep_sizes = [size for size in sp_sizes for _ in range(pcp_size * sp_size)]
    world_size = dp_size * pcp_size * sp_size
    max_size = max(ep_sizes)
    chunks = [torch.full((size, 4), float(rank + 1)) for rank, size in enumerate(ep_sizes)]
    padded = torch.stack([torch.nn.functional.pad(chunk, (0, 0, 0, max_size - len(chunk))) for chunk in chunks])
    expected = torch.cat(chunks)
    context = SimpleNamespace(
        dp_metadata=SimpleNamespace(
            get_chunk_sizes_across_dp_rank=lambda: local_sizes, num_tokens_across_dp_cpu=torch.tensor(dp_tokens)
        )
    )
    monkeypatch.setattr(custom_ops, "get_forward_context", lambda: context)
    monkeypatch.setattr(
        custom_ops, "_EXTRA_CTX", SimpleNamespace(is_draft_model=False, padded_length=max_size * sp_size)
    )
    monkeypatch.setattr(custom_ops, "get_dp_group", lambda: SimpleNamespace(world_size=dp_size))
    monkeypatch.setattr(custom_ops, "get_pcp_group", lambda: SimpleNamespace(world_size=pcp_size), raising=False)

    for rank in range(world_size):

        def gather(x, dim, rank=rank):
            assert dim == 0
            assert torch.equal(x, padded[rank])
            return padded.flatten(0, 1)

        def reduce(x, dim, rank=rank):
            assert dim == 0
            assert torch.equal(x, padded.flatten(0, 1))
            # Every EP rank contributes the same tensor to this reference sum.
            return padded[rank] * world_size

        group = SimpleNamespace(world_size=world_size, rank_in_group=rank, all_gather=gather, reduce_scatter=reduce)
        monkeypatch.setattr(custom_ops, "get_ep_group", lambda group=group: group)
        gathered = custom_ops._maybe_all_gather_and_maybe_unpad_impl(chunks[rank])
        assert torch.equal(gathered, expected)
        assert custom_ops._maybe_all_gather_and_maybe_unpad_fake(chunks[rank]).shape == expected.shape
        reduced = custom_ops._maybe_pad_and_reduce_impl(gathered)
        assert torch.equal(reduced, chunks[rank] * world_size)
        assert custom_ops._maybe_pad_and_reduce_fake(gathered).shape == chunks[rank].shape


def test_sp_ep_returns_none_for_inconsistent_topology(monkeypatch):
    metadata = SimpleNamespace(get_chunk_sizes_across_dp_rank=lambda: [2, 2, 2])
    monkeypatch.setattr(custom_ops, "get_dp_group", lambda: SimpleNamespace(world_size=2))
    monkeypatch.setattr(custom_ops, "get_pcp_group", lambda: SimpleNamespace(world_size=2), raising=False)
    assert custom_ops._get_ep_local_sizes(metadata, SimpleNamespace(world_size=8)) is None
