# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from vllm_ascend.attention.context_parallel.sfa_cp import AscendSFADCPImpl

try:
    import torch_npu  # noqa: F401

    HAS_NPU = torch.npu.is_available()
except ImportError:
    HAS_NPU = False


def _make_impl(
    rank: int,
    dcp_size: int = 2,
    interleave_size: int = 2,
    index_topk: int = 8,
    device: str = "cpu",
) -> AscendSFADCPImpl:
    impl = AscendSFADCPImpl.__new__(AscendSFADCPImpl)
    impl.dcp_size = dcp_size
    impl.dcp_rank = rank
    impl._dcp_interleave_size = interleave_size
    impl._dcp_index_topk = index_topk
    impl._remap_order = torch.arange(index_topk, dtype=torch.float32, device=device)
    impl._remap_invalid_index = torch.tensor(-1.0, device=device)
    return impl


def _reference_remap(
    topk_indices: torch.Tensor,
    dcp_size: int,
    dcp_rank: int,
    interleave_size: int,
) -> torch.Tensor:
    """Independent torch reference of the remap + compact semantics."""
    topk_count = topk_indices.shape[-1]
    indices_fp32 = topk_indices.to(torch.float32)
    block_indices = torch.floor(indices_fp32 / interleave_size)
    owner = block_indices - torch.floor(block_indices / dcp_size) * dcp_size
    owner_mask = (indices_fp32 >= 0) & (owner == dcp_rank)
    if interleave_size == 1:
        remapped = torch.floor(indices_fp32 / dcp_size)
    else:
        local_offsets = indices_fp32 - block_indices * interleave_size
        remapped = torch.floor(indices_fp32 / (dcp_size * interleave_size)) * interleave_size + local_offsets
    remapped = torch.where(owner_mask, remapped, torch.full_like(indices_fp32, -1.0))
    order = torch.arange(topk_count, dtype=torch.float32, device=topk_indices.device).expand_as(topk_indices)
    pack_keys = order + (~owner_mask).to(torch.float32) * topk_count
    _, pack_order = torch.sort(pack_keys, dim=-1)
    return torch.gather(remapped, dim=-1, index=pack_order.to(torch.int64)).to(topk_indices.dtype)


@pytest.mark.skipif(not HAS_NPU, reason="NPU is not available")
@pytest.mark.parametrize("dcp_size", [2, 4])
@pytest.mark.parametrize("interleave_size", [1, 2, 4])
@pytest.mark.parametrize("topk", [1, 8, 48, 256])
def test_sfa_dcp_sparse_indices_triton_matches_reference(dcp_size: int, interleave_size: int, topk: int) -> None:
    device = torch.device("npu")
    gen = torch.Generator(device=device).manual_seed(dcp_size * 100 + interleave_size * 10 + topk)
    max_global_idx = topk * dcp_size * interleave_size
    indices = torch.randint(
        -1,
        max_global_idx + 1,
        (5, topk),
        dtype=torch.int32,
        device=device,
        generator=gen,
    )
    for rank in range(dcp_size):
        impl = _make_impl(rank, dcp_size, interleave_size, topk, device)
        got = impl._remap_sparse_indices(indices)
        torch.testing.assert_close(
            got,
            _reference_remap(indices, dcp_size, rank, interleave_size),
            msg=f"mismatch for dcp_size={dcp_size}, interleave_size={interleave_size}, topk={topk}, rank={rank}",
        )


@pytest.mark.skipif(not HAS_NPU, reason="NPU is not available")
@pytest.mark.parametrize("interleave_size", [1, 2, 4])
def test_sfa_dcp_sparse_indices_3d_input(interleave_size: int) -> None:
    # Real DCP case: input is [dcp_size, 1, topk] after dcp_group.all_gather(dim=0).
    device = torch.device("npu")
    dcp_size, topk = 6, 2048
    gen = torch.Generator(device=device).manual_seed(interleave_size)
    indices = torch.randint(
        -1,
        topk * dcp_size * interleave_size + 1,
        (dcp_size, 1, topk),
        dtype=torch.int32,
        device=device,
        generator=gen,
    )
    for rank in range(dcp_size):
        impl = _make_impl(rank, dcp_size, interleave_size, topk, device)
        got = impl._remap_sparse_indices(indices)
        torch.testing.assert_close(
            got,
            _reference_remap(indices, dcp_size, rank, interleave_size),
            msg=f"mismatch for interleave_size={interleave_size}, rank={rank}",
        )
