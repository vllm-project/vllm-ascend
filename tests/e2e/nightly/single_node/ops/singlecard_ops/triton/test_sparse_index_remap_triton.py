# SPDX-License-Identifier: Apache-2.0
# Compare vllm_ascend.ops.triton.sparse_index_remap.remap_sparse_indices_triton
# (Triton-Ascend) with the fp32 torch remap used by the SFA DCP fallback.

import gc

import pytest
import torch

from vllm_ascend.ops.triton.sparse_index_remap import remap_sparse_indices_triton


def _torch_reference(topk_indices: torch.Tensor, dcp_size: int, dcp_rank: int, interleave_size: int) -> torch.Tensor:
    """Independent reference: same fp32 math as sfa_cp._remap_sparse_indices."""
    topk_count = topk_indices.shape[-1]
    fp32 = topk_indices.to(torch.float32)
    block_idx = torch.floor(fp32 / interleave_size)
    owner = block_idx - torch.floor(block_idx / dcp_size) * dcp_size
    owner_mask = (fp32 >= 0) & (owner == dcp_rank)
    if interleave_size == 1:
        remapped = torch.floor(fp32 / dcp_size)
    else:
        local_offsets = fp32 - block_idx * interleave_size
        remapped = torch.floor(fp32 / (dcp_size * interleave_size)) * interleave_size + local_offsets
    remapped = torch.where(owner_mask, remapped, torch.full_like(fp32, -1.0))
    order = torch.arange(topk_count, dtype=torch.float32, device=topk_indices.device).expand_as(fp32)
    pack_keys = order + (~owner_mask).to(torch.float32) * topk_count
    _, pack_order = torch.sort(pack_keys, dim=-1)
    return torch.gather(remapped, dim=-1, index=pack_order.to(torch.int64)).to(topk_indices.dtype)


@pytest.mark.parametrize("dcp_size", [2, 4, 6])
@pytest.mark.parametrize("interleave_size", [1, 2, 4])
@pytest.mark.parametrize("topk", [1, 8, 48, 256, 2048])
@torch.inference_mode()
def test_sparse_index_remap_triton_matches_reference(dcp_size: int, interleave_size: int, topk: int) -> None:
    device = "npu:0"
    gen = torch.Generator(device="npu").manual_seed(dcp_size * 100 + interleave_size * 10 + topk)
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
        got = remap_sparse_indices_triton(indices, dcp_size, rank, interleave_size)
        torch.testing.assert_close(
            got,
            _torch_reference(indices, dcp_size, rank, interleave_size),
            rtol=0,
            atol=0,
            msg=f"mismatch for dcp_size={dcp_size}, interleave_size={interleave_size}, topk={topk}, rank={rank}",
        )
    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()


@pytest.mark.parametrize("interleave_size", [1, 2, 4])
@torch.inference_mode()
def test_sparse_index_remap_triton_3d_input(interleave_size: int) -> None:
    # Real DCP case: input is [dcp_size, 1, topk] after dcp_group.all_gather(dim=0).
    device = "npu:0"
    dcp_size, topk = 6, 2048
    gen = torch.Generator(device="npu").manual_seed(interleave_size)
    indices = torch.randint(
        -1,
        topk * dcp_size * interleave_size + 1,
        (dcp_size, 1, topk),
        dtype=torch.int32,
        device=device,
        generator=gen,
    )
    for rank in range(dcp_size):
        got = remap_sparse_indices_triton(indices, dcp_size, rank, interleave_size)
        torch.testing.assert_close(
            got,
            _torch_reference(indices, dcp_size, rank, interleave_size),
            rtol=0,
            atol=0,
            msg=f"mismatch for interleave_size={interleave_size}, rank={rank}",
        )
    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()
