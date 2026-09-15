# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Direct NPU precision tests for the auxiliary KDA Triton kernels.

Kernel-to-test mapping:

* ``chunk_local_cumsum_scalar_kernel`` ->
  ``test_chunk_local_cumsum_scalar_kernel``
* ``chunk_local_cumsum_vector_kernel`` ->
  ``test_chunk_local_cumsum_vector_kernel``
* ``l2norm_fwd_persistent_kernel`` -> ``test_l2norm_fwd_persistent_kernel``
* ``l2norm_fwd_tiled_kernel`` -> ``test_l2norm_fwd_tiled_kernel``
* ``solve_tril_16x16_kernel_kda`` -> ``test_solve_tril_16x16_kernel_kda``
* ``merge_16x16_to_32x32_inverse_kernel_kda`` ->
  ``test_merge_16x16_to_32x32_inverse_kernel_kda``
* ``merge_16x16_to_64x64_inverse_kernel_kda`` ->
  ``test_merge_16x16_to_64x64_inverse_kernel_kda``

Every test directly launches only the kernel named in the test. References are
computed independently with PyTorch in float32 on CPU.
"""

import pytest
import torch
import torch_npu  # noqa: F401
from vllm.triton_utils import triton

from vllm_ascend.ops.triton.kda.cumsum import (
    chunk_local_cumsum_scalar_kernel,
    chunk_local_cumsum_vector_kernel,
)
from vllm_ascend.ops.triton.kda.l2norm import (
    l2norm_fwd_persistent_kernel,
    l2norm_fwd_tiled_kernel,
)
from vllm_ascend.ops.triton.kda.solve_tril import (
    merge_16x16_to_32x32_inverse_kernel_kda,
    merge_16x16_to_64x64_inverse_kernel_kda,
    solve_tril_16x16_kernel_kda,
)
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton

DEVICE = "npu"
SEED = 42
L2NORM_EPS = 1e-6
L2NORM_RTOL = 3e-4
L2NORM_ATOL = 1e-3
SOLVE_RTOL = 5e-4
SOLVE_ATOL = 5e-4


@pytest.fixture(scope="module", autouse=True)
def _initialize_triton_device() -> None:
    init_device_properties_triton()


def _chunk_cumsum_reference(
    x: torch.Tensor,
    chunk_size: int,
    *,
    reverse: bool,
    scale: float = 1.0,
) -> torch.Tensor:
    """Compute independent per-chunk cumsums on CPU in float32."""
    x_fp32 = x.to(dtype=torch.float32, device="cpu")
    result = torch.empty_like(x_fp32)
    for start in range(0, x.shape[1], chunk_size):
        chunk = x_fp32[:, start : start + chunk_size]
        if reverse:
            chunk = torch.flip(torch.cumsum(torch.flip(chunk, dims=(1,)), dim=1), dims=(1,))
        else:
            chunk = torch.cumsum(chunk, dim=1)
        result[:, start : start + chunk_size] = chunk * scale
    return result


def _l2norm_reference(x: torch.Tensor, eps: float) -> torch.Tensor:
    """Normalize the final dimension on CPU with float32 accumulation."""
    x_fp32 = x.to(dtype=torch.float32, device="cpu")
    return x_fp32 * torch.rsqrt(x_fp32.square().sum(dim=-1, keepdim=True) + eps)


def _make_block_strictly_lower_input(B: int, T: int, H: int, BT: int) -> torch.Tensor:
    """Build ``[B, T, H, BT]`` block-local strictly-lower matrices."""
    generator = torch.Generator(device="cpu").manual_seed(SEED + BT)
    A = torch.zeros((B, T, H, BT), dtype=torch.float32)
    for i_b in range(B):
        for start in range(0, T, BT):
            block_size = min(BT, T - start)
            for i_h in range(H):
                block = torch.randn((block_size, block_size), generator=generator, dtype=torch.float32)
                A[i_b, start : start + block_size, i_h, :block_size] = torch.tril(block * 0.02, diagonal=-1)
    return A


def _block_inverse_reference(A: torch.Tensor) -> torch.Tensor:
    """Invert each block-local ``I + A`` matrix independently on CPU."""
    B, T, H, BT = A.shape
    result = torch.zeros_like(A, dtype=torch.float32, device="cpu")
    A_fp32 = A.to(dtype=torch.float32, device="cpu")
    for i_b in range(B):
        for start in range(0, T, BT):
            block_size = min(BT, T - start)
            identity = torch.eye(block_size, dtype=torch.float32)
            for i_h in range(H):
                block = A_fp32[i_b, start : start + block_size, i_h, :block_size]
                result[i_b, start : start + block_size, i_h, :block_size] = torch.linalg.solve(
                    identity + block, identity
                )
    return result


def _assert_close(actual: torch.Tensor, expected: torch.Tensor, *, rtol: float, atol: float) -> None:
    actual_cpu = actual.detach().to(dtype=torch.float32, device="cpu")
    assert torch.isfinite(actual_cpu).all(), "Triton output contains NaN or Inf"
    torch.testing.assert_close(actual_cpu, expected, rtol=rtol, atol=atol)


@torch.inference_mode()
def test_chunk_local_cumsum_scalar_kernel() -> None:
    B, T, H = 2, 147, 4
    chunk_size = 16
    block_t = 128
    scale = 0.125
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    x_cpu = torch.randn((B, T, H), generator=generator, dtype=torch.float32)
    expected = _chunk_cumsum_reference(x_cpu, chunk_size, reverse=False, scale=scale)
    x = x_cpu.to(DEVICE)
    output = torch.empty_like(x)
    grid = (triton.cdiv(T, block_t), B)

    chunk_local_cumsum_scalar_kernel[grid](
        s=x,
        o=output,
        scale=scale,
        cu_seqlens=None,
        chunk_indices=None,
        T=T,
        H=H,
        BLOCK_T=block_t,
        REVERSE=False,
        HEAD_FIRST=False,
        CHUNK_SIZE=chunk_size,
        num_warps=8,
        num_stages=3,
    )
    torch.npu.synchronize()

    _assert_close(output, expected, rtol=1e-5, atol=1e-5)


@torch.inference_mode()
def test_chunk_local_cumsum_vector_kernel() -> None:
    B, T, H, S = 1, 67, 2, 33
    chunk_size = 16
    block_s = 32
    generator = torch.Generator(device="cpu").manual_seed(SEED + 1)
    x_cpu = torch.randn((B, T, H, S), generator=generator, dtype=torch.float32)
    expected = _chunk_cumsum_reference(x_cpu, chunk_size, reverse=True)
    x = x_cpu.to(DEVICE)
    output = torch.empty_like(x)
    grid = (triton.cdiv(S, block_s), triton.cdiv(T, chunk_size), B * H)

    chunk_local_cumsum_vector_kernel[grid](
        s=x,
        o=output,
        cu_seqlens=None,
        chunk_indices=None,
        T=T,
        B=B,
        H=H,
        S=S,
        BT=chunk_size,
        REVERSE=True,
        HEAD_FIRST=False,
    )
    torch.npu.synchronize()

    _assert_close(output, expected, rtol=1e-5, atol=1e-5)


@torch.inference_mode()
def test_l2norm_fwd_persistent_kernel() -> None:
    M, N = 143, 128
    mblock = 69
    num_chunks = 2
    rows_per_program = mblock * num_chunks
    generator = torch.Generator(device="cpu").manual_seed(SEED + 2)
    x_cpu = torch.randn((M, N), generator=generator, dtype=torch.float32)
    x_cpu[0].zero_()
    expected = _l2norm_reference(x_cpu, L2NORM_EPS)
    x = x_cpu.to(DEVICE)
    output = torch.empty_like(x)
    grid = (triton.cdiv(M, rows_per_program),)

    l2norm_fwd_persistent_kernel[grid](
        X=x,
        Y=output,
        eps=L2NORM_EPS,
        M=M,
        N=N,
        MBLOCK=mblock,
        NUM_CHUNKS=num_chunks,
    )
    torch.npu.synchronize()

    _assert_close(output, expected, rtol=L2NORM_RTOL, atol=L2NORM_ATOL)


@torch.inference_mode()
def test_l2norm_fwd_tiled_kernel() -> None:
    M, N = 35, 96
    block_d = 128
    mblock = 32
    generator = torch.Generator(device="cpu").manual_seed(SEED + 3)
    x_cpu = torch.randn((M, N), generator=generator, dtype=torch.float32)
    x_cpu[-1].zero_()
    expected = _l2norm_reference(x_cpu, L2NORM_EPS)
    x = x_cpu.to(DEVICE)
    output = torch.empty_like(x)
    grid = (triton.cdiv(M, mblock),)

    l2norm_fwd_tiled_kernel[grid](
        X=x,
        Y=output,
        eps=L2NORM_EPS,
        M=M,
        N=N,
        BD=block_d,
        MBLOCK=mblock,
    )
    torch.npu.synchronize()

    _assert_close(output, expected, rtol=L2NORM_RTOL, atol=L2NORM_ATOL)


@torch.inference_mode()
def test_solve_tril_16x16_kernel_kda() -> None:
    B, T, H, BT = 1, 23, 2, 16
    A_cpu = _make_block_strictly_lower_input(B, T, H, BT)
    expected = _block_inverse_reference(A_cpu)
    A = A_cpu.to(DEVICE)
    output = torch.zeros_like(A)
    grid = (triton.cdiv(T, BT), B * H)

    solve_tril_16x16_kernel_kda[grid](
        A=A,
        Ai=output,
        cu_seqlens=None,
        chunk_indices=None,
        T=T,
        H=H,
        BT=BT,
        DOT_PRECISION="ieee",
    )
    torch.npu.synchronize()

    _assert_close(output, expected, rtol=SOLVE_RTOL, atol=SOLVE_ATOL)


@torch.inference_mode()
def test_merge_16x16_to_32x32_inverse_kernel_kda() -> None:
    B, T, H, BT = 1, 51, 2, 32
    A_cpu = _make_block_strictly_lower_input(B, T, H, BT)
    expected = _block_inverse_reference(A_cpu)
    A = A_cpu.to(DEVICE)
    output = torch.zeros_like(A)
    grid = (triton.cdiv(T, BT), B * H)

    merge_16x16_to_32x32_inverse_kernel_kda[grid](
        A=A,
        Ai=output,
        cu_seqlens=None,
        chunk_indices=None,
        T=T,
        H=H,
        BT=BT,
        DOT_PRECISION="ieee",
    )
    torch.npu.synchronize()

    _assert_close(output, expected, rtol=SOLVE_RTOL, atol=SOLVE_ATOL)


@torch.inference_mode()
def test_merge_16x16_to_64x64_inverse_kernel_kda() -> None:
    B, T, H, BT = 1, 123, 1, 64
    A_cpu = _make_block_strictly_lower_input(B, T, H, BT)
    expected = _block_inverse_reference(A_cpu)
    A = A_cpu.to(DEVICE)
    output = torch.zeros_like(A)
    grid = (triton.cdiv(T, BT), B * H)

    merge_16x16_to_64x64_inverse_kernel_kda[grid](
        A=A,
        Ai=output,
        cu_seqlens=None,
        chunk_indices=None,
        T=T,
        H=H,
        BT=BT,
        DOT_PRECISION="ieee",
    )
    torch.npu.synchronize()

    _assert_close(output, expected, rtol=SOLVE_RTOL, atol=SOLVE_ATOL)
