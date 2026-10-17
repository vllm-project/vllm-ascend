# SPDX-License-Identifier: Apache-2.0
# Compare Triton-Ascend remapping with a rank-owned page reference.

import gc

import pytest
import torch

from vllm_ascend.ops.triton.sparse_index_remap import (
    mask_invalid_sparse_attention_rows,
    remap_sparse_indices_triton,
)


def _torch_reference(topk_indices: torch.Tensor, dcp_size: int, dcp_rank: int, interleave_size: int) -> torch.Tensor:
    """Enumerate rank-owned pages, then preserve the input's TopK order."""
    topk_count = topk_indices.shape[-1]
    indices_cpu = topk_indices.cpu()
    global_to_local = {}
    for local_page, page_start in enumerate(
        range(dcp_rank * interleave_size, int(indices_cpu.max()) + 1, dcp_size * interleave_size)
    ):
        for offset in range(interleave_size):
            global_to_local[page_start + offset] = local_page * interleave_size + offset
    expected = torch.full_like(indices_cpu, -1).reshape(-1, topk_count)
    for row_idx, row in enumerate(indices_cpu.reshape(-1, topk_count).tolist()):
        local_indices = [global_to_local[index] for index in row if index in global_to_local]
        expected[row_idx, : len(local_indices)] = torch.tensor(local_indices, dtype=topk_indices.dtype)
    return expected.reshape(topk_indices.shape).to(topk_indices.device)


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


@pytest.mark.parametrize("dcp_size", [2, 4])
@pytest.mark.parametrize("topk", [129, 2048])
@torch.inference_mode()
def test_sparse_index_remap_dsv4_page_shards_reuse_buffers(dcp_size: int, topk: int) -> None:
    # DSV4 C4 shards whole 128-row compressed pages. Exercise a partial final
    # Triton chunk as well as the model's full 2048-entry TopK.
    page_size, chunk_size, rows = 128, 128, 4
    num_chunks = (topk + chunk_size - 1) // chunk_size
    out = torch.full((rows, 1, topk), 37, dtype=torch.int32, device="npu")
    chunk_out = torch.full((rows * num_chunks, chunk_size), 37, dtype=torch.int32, device="npu")
    chunk_count = torch.full((rows, num_chunks), 37, dtype=torch.int32, device="npu")
    pointers = (out.data_ptr(), chunk_out.data_ptr(), chunk_count.data_ptr())

    for rank in range(dcp_size):
        boundaries = torch.tensor(
            [page_size - 1, page_size, dcp_size * page_size - 1, dcp_size * page_size, 0, -1],
            dtype=torch.int32,
        )
        indices = torch.stack(
            (
                boundaries.repeat((topk + len(boundaries) - 1) // len(boundaries))[:topk],
                rank * page_size + torch.arange(topk, dtype=torch.int32) % page_size,
                torch.full((topk,), ((rank + 1) % dcp_size) * page_size, dtype=torch.int32),
                torch.full((topk,), -1, dtype=torch.int32),
            )
        ).unsqueeze(1)
        for current_indices in (indices, torch.full_like(indices, -1), indices.flip(-1)):
            indices_npu = current_indices.npu()
            got = remap_sparse_indices_triton(
                indices_npu,
                dcp_size,
                rank,
                page_size,
                out=out,
                chunk_out=chunk_out,
                chunk_count=chunk_count,
            )
            assert got.data_ptr() == pointers[0]
            assert (out.data_ptr(), chunk_out.data_ptr(), chunk_count.data_ptr()) == pointers
            torch.testing.assert_close(
                got,
                _torch_reference(indices_npu, dcp_size, rank, page_size),
                rtol=0,
                atol=0,
            )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("strided", [False, True])
@torch.inference_mode()
def test_mask_invalid_sparse_attention_rows(dtype: torch.dtype, strided: bool) -> None:
    tokens, heads, head_dim = 4, 8, 512
    # A local position of zero is valid; padding only marks an empty rank.
    indices = torch.tensor([[[0, -1]], [[-1, -1]], [[127, 128]], [[-1, -1]]], dtype=torch.int32)
    expected_output = torch.arange(tokens * heads * head_dim, dtype=torch.float32).reshape(tokens, heads, head_dim)
    expected_output = (expected_output % 31 + 1).to(dtype)
    expected_lse = torch.arange(tokens * heads, dtype=torch.float32).reshape(tokens, heads, 1)
    storage_shape = (tokens, heads * 2 if strided else heads, head_dim)
    output_storage = torch.full(storage_shape, 53, dtype=dtype, device="npu")
    lse_storage = torch.full((*storage_shape[:2], 1), 53, dtype=torch.float32, device="npu")
    output = output_storage[:, ::2] if strided else output_storage
    lse = lse_storage[:, ::2] if strided else lse_storage
    output.copy_(expected_output)
    lse.copy_(expected_lse)

    mask_invalid_sparse_attention_rows(indices.npu(), output, lse)

    expected_output[[1, 3]] = 0
    expected_lse[[1, 3]] = -float("inf")
    torch.testing.assert_close(output.cpu(), expected_output, rtol=0, atol=0)
    torch.testing.assert_close(lse.cpu(), expected_lse, rtol=0, atol=0)
    if strided:
        assert torch.all(output_storage[:, 1::2].cpu() == 53)
        assert torch.all(lse_storage[:, 1::2].cpu() == 53)


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
