# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.attention.context_parallel.sfa_cp import AscendSFADCPImpl
from vllm_ascend.utils import enable_custom_op

enable_custom_op()


def _reference_remap(
    indices: torch.Tensor,
    dcp_size: int,
    rank: int,
    interleave_size: int,
) -> torch.Tensor:
    indices_cpu = indices.cpu()
    blocks = torch.div(indices_cpu, interleave_size, rounding_mode="floor")
    is_local = (indices_cpu >= 0) & (blocks.remainder(dcp_size) == rank)
    remapped = (
        torch.div(
            indices_cpu,
            dcp_size * interleave_size,
            rounding_mode="floor",
        )
        * interleave_size
        + indices_cpu.remainder(interleave_size)
    )
    remapped = torch.where(is_local, remapped, -1)
    result = torch.full_like(remapped, -1)
    for source_row, local_row, result_row in zip(
        remapped.view(-1, remapped.shape[-1]),
        is_local.view(-1, is_local.shape[-1]),
        result.view(-1, result.shape[-1]),
        strict=True,
    ):
        values = source_row[local_row]
        result_row[: values.numel()] = values
    return result.to(indices.device)


@pytest.mark.parametrize(
    ("dcp_size", "rank", "interleave_size"),
    [(2, 0, 1), (3, 1, 96), (8, 3, 64), (16, 7, 128)],
)
@pytest.mark.parametrize("num_rows", [5, 16, 64, 128])
@torch.inference_mode()
def test_sfa_remap_sparse_indices(
    dcp_size: int,
    rank: int,
    interleave_size: int,
    num_rows: int,
) -> None:
    torch.manual_seed(2026)
    indices = torch.randint(
        0,
        20_000_000,
        (num_rows, 1, 2048),
        dtype=torch.int32,
        device="npu",
    )
    indices[..., ::11] = -1

    expected = _reference_remap(indices, dcp_size, rank, interleave_size)
    actual = torch.empty_like(indices)
    torch.ops._C_ascend.sfa_remap_sparse_indices(
        indices,
        actual,
        dcp_size,
        rank,
        interleave_size,
    )

    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize("top_k", [1, 7, 512, 2048, 8192])
@torch.inference_mode()
def test_sfa_remap_sparse_indices_supports_dynamic_top_k(top_k: int) -> None:
    indices = torch.arange(
        5 * top_k,
        dtype=torch.int32,
        device="npu",
    ).view(5, 1, top_k)
    indices[..., ::11] = -1

    expected = _reference_remap(indices, 16, 7, 128)
    actual = torch.empty_like(indices)
    torch.ops._C_ascend.sfa_remap_sparse_indices(indices, actual, 16, 7, 128)

    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@torch.inference_mode()
def test_sfa_cp_backend_uses_ascendc_remap() -> None:
    impl = AscendSFADCPImpl.__new__(AscendSFADCPImpl)
    impl.dcp_size = 16
    impl.dcp_rank = 7
    impl._dcp_interleave_size = 128
    impl._dcp_index_topk = 2048
    indices = torch.randint(
        0,
        20_000_000,
        (16, 1, 2048),
        dtype=torch.int32,
        device="npu",
    )
    indices[..., ::11] = -1

    expected = _reference_remap(indices, 16, 7, 128)
    actual = impl._remap_sparse_indices(indices)

    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
