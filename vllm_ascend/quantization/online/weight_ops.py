#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Load-time weight quantization kernels for online quantization.

These functions convert a dense ``[out_features, in_features]`` checkpoint
weight into the exact layouts the tree's W8A8 production schemes consume:

* :func:`quantize_weight_int8_per_channel` produces ``int8 [out, in]`` plus one
  float32 scale per output channel, matching what ``W8A8_DYNAMIC`` loads from
  an offline-quantized checkpoint (``get_perchannel_param`` in
  ``methods/w8a8/w8a8_dynamic.py``: weight ``int8 [out, in]``, scale
  ``[out, 1]`` later flattened). Symmetric quantization has an implicit
  zero ``weight_offset``; an adapter must materialize that offset as zeros.
* :func:`quantize_weight_mx_fp8` produces ``float8_e4m3fn [out, in]`` plus one
  uint8 E8M0-encoded scale per 32-element group along the reduction dim,
  matching what ``W8A8_MXFP8`` loads (``get_pergroup_param`` in
  ``methods/w8a8/w8a8_mxfp8.py``).

Peak host/device memory is bounded by processing the matrix in row chunks
(the same pattern as ``_ROWS_PER_DEQUANT_STEP`` in
``methods/w8a8/fp8_block.py``): the transient float32 staging of one chunk is
``rows_per_step * in_features * 4`` bytes regardless of layer size, and
chunking never changes the numbers, only the schedule.

Every kernel has a pure-torch CPU reference implementation so unit tests and
NPU-less environments can verify the math; the NPU paths wrap ``torch_npu``
operators where they are numerically equivalent.
"""

import torch

try:
    from vllm.utils.math_utils import cdiv  # noqa: F401
except ImportError:  # pragma: no cover - CPU unit-test environments
    cdiv = lambda x, y: -(-x // y)  # noqa: E731

# Output rows quantized per step. Bounds the transient float32 staging to
# ``DEFAULT_ROWS_PER_STEP * in_features * 4`` bytes regardless of layer size.
DEFAULT_ROWS_PER_STEP = 1024

# Elements sharing one E8M0 shared exponent in the MX formats.
MX_GROUP_SIZE = 32

# Largest finite float8_e4m3fn magnitude.
FP8_E4M3_MAX = 448.0

# Half of the float8_e4m3fn unit-in-the-last-place at the top normal binade
# (values in [256, 448) have ulp 32, so round-to-nearest deviates by at most
# 16 scaled units). The tight per-element quantization error bound.
FP8_E4M3_HALF_ULP_AT_TOP_BINADE = 16.0

# E8M0 scale bias: a stored uint8 exponent ``e`` decodes to 2**(e - 127).
E8M0_EXPONENT_BIAS = 127
E8M0_MAX_EXPONENT = 254


def _is_npu_tensor(weight: torch.Tensor) -> bool:
    return weight.device.type == "npu"


def quantize_weight_int8_per_channel(
    weight: torch.Tensor,
    rows_per_step: int = DEFAULT_ROWS_PER_STEP,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a dense weight to symmetric per-channel int8 (W8A8 layout).

    The scale is derived per output channel (per row of ``weight``) as
    ``scale = amax / 127`` where ``amax = max(|row|)``, and the row is
    quantized as ``round(w / scale).clamp(-127, 127).to(int8)``. This is the
    weight-side counterpart of the production per-token activation quantizer
    (``scale = absmax / 127`` in ``ops/triton/activation/swiglu_quant.py``),
    applied along the output dim so the result pairs directly with the
    ``weight``/``weight_scale`` of ``W8A8_DYNAMIC`` (``int8 [out, in]`` plus
    one ``float32`` scale per row).

    A zero row quantizes to zeros with ``scale = 0`` rather than dividing by
    zero.

    Args:
        weight: Dense ``[out_features, in_features]`` float tensor
            (BF16/FP16/FP32) on CPU or NPU.
        rows_per_step: Number of output rows quantized per step. Bounds the
            transient float32 staging buffer to
            ``rows_per_step * in_features * 4`` bytes. Chunking only changes
            the schedule, never the numbers. Use a large value (>= the row
            count) to disable chunking.

    Returns:
        Tuple ``(int8_weight, scale)`` where ``int8_weight`` is
        ``int8 [out_features, in_features]`` and ``scale`` is
        ``float32 [out_features]`` (one scale per output channel).
    """
    if weight.dim() != 2:
        raise ValueError(f"Expected a 2D weight, got shape {tuple(weight.shape)}.")
    if rows_per_step < 1:
        raise ValueError(f"rows_per_step must be >= 1, got {rows_per_step}.")

    out_features, in_features = weight.shape
    int8_weight = torch.empty((out_features, in_features), dtype=torch.int8, device=weight.device)
    scale = torch.empty((out_features,), dtype=torch.float32, device=weight.device)

    for row_start in range(0, out_features, rows_per_step):
        row_end = min(row_start + rows_per_step, out_features)
        chunk = weight[row_start:row_end]
        # aminmax avoids materializing a full-size abs(); the max of |lo| and
        # |hi| is computed in float32 so a BF16 chunk is not its own rounding
        # oracle (same helper shape as upstream vllm's weight_amax).
        lo, hi = chunk.to(torch.float32).aminmax(dim=1)
        chunk_scale = torch.maximum(lo.abs(), hi.abs()) / 127.0
        # A zero row must stay all-zero rather than produce NaN.
        chunk_scale = torch.where(chunk_scale > 0, chunk_scale, torch.zeros_like(chunk_scale))
        safe_scale = torch.where(chunk_scale > 0, chunk_scale, torch.ones_like(chunk_scale))
        quantized = torch.round(chunk.to(torch.float32) / safe_scale.unsqueeze(1))
        quantized = quantized.clamp(-127, 127)
        quantized = torch.where(chunk_scale.unsqueeze(1) > 0, quantized, torch.zeros_like(quantized))
        int8_weight[row_start:row_end] = quantized.to(torch.int8)
        scale[row_start:row_end] = chunk_scale

    return int8_weight, scale


def quantize_weight_int8_per_channel_reference(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Straightforward per-element oracle for
    :func:`quantize_weight_int8_per_channel`.

    Runs the same math without chunking, dtype staging, or memory controls so
    unit tests can assert the production kernel is bit-exact against an
    obviously-correct formulation. Accepts any float dtype; returns
    ``(int8 [out, in], float32 [out])``.
    """
    if weight.dim() != 2:
        raise ValueError(f"Expected a 2D weight, got shape {tuple(weight.shape)}.")
    weight_fp32 = weight.to(torch.float32)
    amax = weight_fp32.abs().amax(dim=1)
    scale = amax / 127.0
    scale = torch.where(scale > 0, scale, torch.zeros_like(scale))
    safe_scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    quantized = torch.round(weight_fp32 / safe_scale.unsqueeze(1)).clamp(-127, 127)
    quantized = torch.where(scale.unsqueeze(1) > 0, quantized, torch.zeros_like(quantized))
    quantized = quantized.to(torch.int8)
    return quantized, scale


def dequantize_weight_int8_per_channel(
    int8_weight: torch.Tensor,
    scale: torch.Tensor,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Dequantize the output of :func:`quantize_weight_int8_per_channel`.

    Returns ``int8_weight * scale[:, None]`` — the reconstruction the cosine
    accuracy gate compares against the original weight. Multiplication
    happens in float32 before the cast so a BF16 result is not itself the
    rounding bottleneck.
    """
    return (int8_weight.to(torch.float32) * scale.to(torch.float32).unsqueeze(1)).to(dtype)


def e8m0_scale_to_float(scale: torch.Tensor) -> torch.Tensor:
    """Decode uint8 E8M0 scale bytes to their float32 values ``2**(e - 127)``.

    The encoding is the E8M0 format used by the MXFP8 production path (the
    ``uint8`` ``weight_scale`` the MXFP8 schemes feed to
    ``npu_quant_matmul`` as ``float8_e8m0fnu``): a uint8 exponent with bias
    127 and no special values.
    """
    return torch.exp2((scale.to(torch.int32) - E8M0_EXPONENT_BIAS).to(torch.float32))


def quantize_weight_mx_fp8(
    weight: torch.Tensor,
    group_size: int = MX_GROUP_SIZE,
    rows_per_step: int = DEFAULT_ROWS_PER_STEP,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a dense weight to MXFP8: E4M3 data, E8M0 per-group scales.

    Groups of ``group_size`` consecutive elements along the reduction (input)
    dim share one power-of-two scale. The NPU path wraps
    ``torch_npu.npu_dynamic_mx_quant`` (the same operator
    ``methods/w8a8/fp8_block.py`` uses at load time) and collapses the
    operator's ``[..., num_groups // 2, 2]`` scale split into the
    ``[out, num_groups]`` loader layout the MXFP8 scheme consumes. On CPU the
    pure-torch reference math runs instead.

    Args:
        weight: Dense ``[out_features, in_features]`` float tensor. The input
            dim must be a multiple of ``group_size`` (the same alignment the
            MXFP8 scheme itself requires; the driver is responsible for any
            padding at TP boundaries).
        group_size: Elements per shared-exponent group along the input dim.
        rows_per_step: Output rows quantized per step, bounding transient
            staging memory. Chunking only changes the schedule, never the
            numbers.

    Returns:
        Tuple ``(fp8_weight, e8m0_scale)`` where ``fp8_weight`` is
        ``float8_e4m3fn [out, in]`` and ``e8m0_scale`` is
        ``uint8 [out, in // group_size]`` holding the E8M0-encoded shared
        exponents.
    """
    if weight.dim() != 2:
        raise ValueError(f"Expected a 2D weight, got shape {tuple(weight.shape)}.")
    if weight.shape[1] % group_size != 0:
        raise ValueError(
            f"Reduction dim {weight.shape[1]} is not a multiple of the MX group size {group_size}; "
            "pad the weight before quantizing."
        )
    if rows_per_step < 1:
        raise ValueError(f"rows_per_step must be >= 1, got {rows_per_step}.")

    out_features, in_features = weight.shape
    num_groups = in_features // group_size
    fp8_weight = torch.empty((out_features, in_features), dtype=torch.float8_e4m3fn, device=weight.device)
    e8m0_scale = torch.empty((out_features, num_groups), dtype=torch.uint8, device=weight.device)

    use_npu = _is_npu_tensor(weight)
    for row_start in range(0, out_features, rows_per_step):
        row_end = min(row_start + rows_per_step, out_features)
        chunk = weight[row_start:row_end]
        if use_npu:
            quantized, op_scale = torch_npu.npu_dynamic_mx_quant(
                chunk,
                dst_type=torch.float8_e4m3fn,
                scale_alg=0,
            )
            # The operator emits [..., num_groups // 2, 2]; collapse to the
            # [..., num_groups] loader layout (same normalization as
            # fp8_block._mx_quantize).
            fp8_weight[row_start:row_end] = quantized
            e8m0_scale[row_start:row_end] = op_scale.view(torch.uint8).reshape(row_end - row_start, num_groups)
        else:
            quantized, chunk_scale = quantize_weight_mx_fp8_reference(chunk, group_size)
            fp8_weight[row_start:row_end] = quantized
            e8m0_scale[row_start:row_end] = chunk_scale

    return fp8_weight, e8m0_scale


def quantize_weight_mx_fp8_reference(
    weight: torch.Tensor,
    group_size: int = MX_GROUP_SIZE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pure-torch MXFP8 oracle (CPU-safe).

    Implements the no-saturation MX E4M3/E8M0 semantics used by the ue8m0
    production path (``_ceil_to_ue8m0`` in upstream vLLM's
    ``per_block_cast_to_fp8``): per group of ``group_size`` along the input
    dim, ``scale = 2**ceil(log2(amax / 448))``, the smallest power of two
    that keeps every group element at or below the largest finite E4M3
    magnitude (448). The group's amax therefore lands in the top binade
    ``[224, 448]`` and the full E4M3 mantissa range is used. Elements are
    divided by the scale and cast to ``float8_e4m3fn`` (round-to-nearest,
    never saturating), and the scale is stored as the E8M0 byte
    ``log2(scale) + 127`` clamped to ``[0, 254]``.

    ``ceil`` (rather than ``floor``) is load-bearing: ``floor`` can produce a
    scale with ``amax / scale > 448``, which saturates the largest element of
    the group and loses up to a full binade of magnitude.

    Returns:
        Tuple ``(fp8_weight, e8m0_scale)``: ``float8_e4m3fn [out, in]`` and
        ``uint8 [out, in // group_size]``.
    """
    if weight.dim() != 2:
        raise ValueError(f"Expected a 2D weight, got shape {tuple(weight.shape)}.")
    if weight.shape[1] % group_size != 0:
        raise ValueError(
            f"Reduction dim {weight.shape[1]} is not a multiple of the MX group size {group_size}; "
            "pad the weight before quantizing."
        )
    out_features, in_features = weight.shape
    num_groups = in_features // group_size
    weight_fp32 = weight.to(torch.float32)
    grouped = weight_fp32.reshape(out_features, num_groups, group_size)
    amax = grouped.abs().amax(dim=2)
    exponent = torch.ceil(torch.log2(amax / FP8_E4M3_MAX))
    encoded = (exponent + E8M0_EXPONENT_BIAS).clamp(0, E8M0_MAX_EXPONENT).to(torch.uint8)
    scale_value = e8m0_scale_to_float(encoded)
    # An all-zero group (log2(0) = -inf, exponent clamps to the E8M0 minimum
    # 2**-127) divides zero by a finite scale and stays exactly zero.
    fp8 = (grouped / scale_value.unsqueeze(2)).to(torch.float8_e4m3fn)
    return fp8.reshape(out_features, in_features), encoded


def dequantize_weight_mx_fp8(
    fp8_weight: torch.Tensor,
    e8m0_scale: torch.Tensor,
    group_size: int = MX_GROUP_SIZE,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Dequantize the output of :func:`quantize_weight_mx_fp8`.

    Returns ``fp8_weight.to(float32) * scale[:, :, None]`` reshaped back to
    ``[out, in]`` and cast to ``dtype`` — the reconstruction the cosine
    accuracy gate compares against the original weight.
    """
    out_features, in_features = fp8_weight.shape
    num_groups = in_features // group_size
    scale_value = e8m0_scale_to_float(e8m0_scale)
    reconstructed = fp8_weight.reshape(out_features, num_groups, group_size).to(torch.float32)
    reconstructed = reconstructed * scale_value.unsqueeze(2)
    return reconstructed.reshape(out_features, in_features).to(dtype)


def quantize_weight_mx_fp4(
    weight: torch.Tensor,
    group_size: int = MX_GROUP_SIZE,
    rows_per_step: int = DEFAULT_ROWS_PER_STEP,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a dense weight to MXFP4 (E2M1 data, E8M0 per-group scales).

    .. note:: Stub. The MXFP4 production format packs two E2M1 elements per
        byte in a layout owned by the W4A4 schemes, and the NPU operator that
        emits it (``dst_type=float4_e2m1fn_x2``) is not yet wired through the
        load-time path. Until then this raises ``NotImplementedError`` so a
        caller cannot silently get a wrong layout. The MXFP8 kernel above
        covers the supported online-quantization recipes.
    """
    raise NotImplementedError(
        "MXFP4 online weight quantization is not implemented yet; the packed E2M1 "
        "layout is owned by the W4A4 schemes. Use quantize_weight_mx_fp8 for now."
    )


# torch_npu is only importable on an NPU build; reference it lazily so this
# module stays importable (and testable) on CPU. The NPU branch above is the
# only consumer.
try:
    import torch_npu  # noqa: F401
except ImportError:
    torch_npu = None  # type: ignore[assignment]
