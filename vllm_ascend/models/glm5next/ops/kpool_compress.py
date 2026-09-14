# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ascend replacements for the GLM-5.3-Flash kpool indexer's fused kernels.

The upstream implementation fuses the Hadamard-128 rotation with the block-128
ue8m0 FP8 quantization into a single Triton kernel that keeps the rotated tensor
in registers. That kernel is CUDA-only, so the NPU path expresses the same
numerics in torch: the rotation becomes a matmul against a cached normalized
Hadamard matrix, and the quantization stays elementwise. Every step is
device-side, so the sequence remains ACL-graph safe.
"""

import torch

# The indexer query is quantized against the e4m3 maximum, and the resulting
# scale is restricted to a power of two (ue8m0), matching the cached K basis.
FP8_E4M3_MAX = 448.0

# Guards rows whose rotated vector is all but zero, so log2 stays finite.
_MIN_ABSMAX = 1e-4

_HADAMARD_CACHE: dict[tuple[int, torch.device, torch.dtype], torch.Tensor] = {}


def _normalized_hadamard(dim: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Return a cached ``dim x dim`` Hadamard matrix scaled by ``dim ** -0.5``.

    Folding the normalization into the matrix keeps the rotation a single
    matmul. The matrix is symmetric, so it is used without a transpose.
    """
    key = (dim, device, dtype)
    cached = _HADAMARD_CACHE.get(key)
    if cached is not None:
        return cached

    try:
        from scipy.linalg import hadamard  # type: ignore[import-untyped]
    except ImportError as err:
        raise ImportError(
            "The GLM-5.3-Flash kpool indexer requires SciPy for the Hadamard transform. Please install scipy."
        ) from err

    matrix = torch.tensor(hadamard(dim, dtype=float), dtype=dtype, device=device) * (dim**-0.5)
    _HADAMARD_CACHE[key] = matrix
    return matrix


def fwht128_quant_fp8(q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate each 128-wide row by the Hadamard-128 transform, then FP8-quant.

    Args:
        q: ``[rows, 128]`` bf16 -- one head vector per row.

    Returns:
        (q_fp8 ``[rows, 128]`` float8_e4m3fn, scale ``[rows, 1]`` float32).
    """
    assert q.ndim == 2 and q.shape[1] == 128, q.shape

    rows, dim = q.shape
    if rows == 0:
        return (
            torch.empty((0, dim), dtype=torch.float8_e4m3fn, device=q.device),
            torch.empty((0, 1), dtype=torch.float32, device=q.device),
        )

    hadamard = _normalized_hadamard(dim, q.device, torch.float32)
    rotated = q.float() @ hadamard
    # The upstream kernel materializes bf16 between the rotation and the quant,
    # so the fp8 operand carries the same rounding on both backends.
    rotated = rotated.to(torch.bfloat16).to(torch.float32)

    absmax = rotated.abs().amax(dim=-1, keepdim=True).clamp_min(_MIN_ABSMAX)
    scale = torch.exp2(torch.ceil(torch.log2(absmax / FP8_E4M3_MAX)))
    q_fp8 = (rotated / scale).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(torch.float8_e4m3fn)
    return q_fp8, scale
