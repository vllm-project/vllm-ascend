# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""MXFP8 Dense MLP communication fusion for sequence-parallel token shards.

Construction selects bias-free group-32 MXFP8 linears on Ascend 950DT and
prepares ND weights. Runtime admission stays inside opaque custom ops so token
thresholds do not specialize a dynamic compilation graph. Both alternatives
preserve the ordinary SP collective and quantized-matmul semantics.
"""

from __future__ import annotations

import torch
import torch_npu
from vllm.distributed.parallel_state import get_tp_group
from vllm.utils.torch_utils import direct_register_custom_op

from vllm_ascend.models.common.ops.sequence_parallel import sp_all_gather, sp_reduce_scatter

MX_GROUP_SIZE = 32
MX_SCALE_PACK_SIZE = 2
MX_SCALE_K_BLOCK = MX_GROUP_SIZE * MX_SCALE_PACK_SIZE
ND_FORMAT = 2
MIN_FUSED_LOCAL_TOKENS = 128
MIN_FUSED_K = 256
MAX_FUSED_K_EXCLUSIVE = 65535
MAX_FUSED_DIM = 2147483647
MAX_REDUCE_SCATTER_BYTES = 63 * 256 * 1024 * 1024
SUPPORTED_TP_SIZES = (2, 4, 8)


def dense_mxfp8_sp_available() -> bool:
    """Check the API surface; the caller separately admits 950DT hardware."""
    return hasattr(torch_npu, "float8_e8m0fnu") and all(
        callable(getattr(torch_npu, name, None))
        for name in (
            "npu_all_gather_quant_mm",
            "npu_quant_mm_reduce_scatter",
            "npu_dynamic_mx_quant",
            "npu_quant_matmul",
            "get_npu_format",
        )
    )


def normalize_mx_scale(scale: torch.Tensor, rows: int, k: int) -> torch.Tensor:
    """Pack adjacent E8M0 bytes without converting their numerical values."""
    scale_blocks = (k + MX_SCALE_K_BLOCK - 1) // MX_SCALE_K_BLOCK
    packed_shape = (rows, scale_blocks, MX_SCALE_PACK_SIZE)
    flat_shape = (rows, scale_blocks * MX_SCALE_PACK_SIZE)
    native_e8m0 = getattr(torch, "float8_e8m0fnu", torch.uint8)
    if scale.dtype not in (torch.uint8, native_e8m0):
        raise ValueError("MXFP8 activation scales must contain E8M0 bytes.")
    if tuple(scale.shape) not in (packed_shape, flat_shape):
        raise ValueError(f"Expected MXFP8 scale shape {packed_shape} or {flat_shape}, got {tuple(scale.shape)}.")
    return scale.reshape(packed_shape).contiguous()


def _validate_inputs(x: torch.Tensor, weight: torch.Tensor, weight_scale: torch.Tensor, tp_size: int) -> None:
    # These are errors in the baseline contract, not reasons to try another
    # collective. Reject them before communication on every TP rank.
    if tp_size < 1:
        raise ValueError("The tensor-parallel group size must be positive.")
    if get_tp_group().world_size != tp_size:
        raise ValueError("Dense MXFP8 SP operator TP size does not match its communication group.")
    if x.ndim != 2 or weight.ndim != 2:
        raise ValueError("Dense MXFP8 SP operators require two-dimensional input and weight tensors.")
    if x.shape[1] != weight.shape[0]:
        raise ValueError("Dense MXFP8 SP input and weight K dimensions must match.")
    scale_blocks = (weight.shape[0] + MX_SCALE_K_BLOCK - 1) // MX_SCALE_K_BLOCK
    expected_scale_shape = (scale_blocks, weight.shape[1], MX_SCALE_PACK_SIZE)
    if tuple(weight_scale.shape) != expected_scale_shape:
        raise ValueError(f"Expected MXFP8 weight scale shape {expected_scale_shape}, got {tuple(weight_scale.shape)}.")


def _is_nd_contiguous(tensor: torch.Tensor) -> bool:
    return tensor.is_contiguous() and torch_npu.get_npu_format(tensor) == ND_FORMAT


def _can_fuse(x: torch.Tensor, weight: torch.Tensor, weight_scale: torch.Tensor, tp_size: int) -> bool:
    # Only tensor metadata and process-wide capabilities select the path.
    # Participating ranks must use the same evenly sharded Dense MLP layout.
    return (
        dense_mxfp8_sp_available()
        and tp_size in SUPPORTED_TP_SIZES
        and x.dtype == torch.bfloat16
        and weight.dtype == torch.float8_e4m3fn
        and weight_scale.dtype == torch.uint8
        and MIN_FUSED_K <= x.shape[1] < MAX_FUSED_K_EXCLUSIVE
        and x.shape[1] % MX_SCALE_K_BLOCK == 0
        and 0 < x.shape[0] <= MAX_FUSED_DIM
        and 0 < weight.shape[1] <= MAX_FUSED_DIM
        and _is_nd_contiguous(x)
        and _is_nd_contiguous(weight)
        and _is_nd_contiguous(weight_scale)
    )


def _tp_hcom(x: torch.Tensor, tp_size: int) -> str:
    group = get_tp_group()
    if group.world_size != tp_size:
        raise ValueError("Dense MXFP8 SP operator TP size does not match its communication group.")
    backend = group.device_group._get_backend(x.device)
    return backend.get_hccl_comm_name(group.rank_in_group)


def _quantize(x: torch.Tensor, scale_alg: int) -> tuple[torch.Tensor, torch.Tensor]:
    return torch_npu.npu_dynamic_mx_quant(x, dst_type=torch.float8_e4m3fn, scale_alg=scale_alg)


def _quant_matmul(x: torch.Tensor, weight: torch.Tensor, weight_scale: torch.Tensor, scale_alg: int) -> torch.Tensor:
    quantized_x, activation_scale = _quantize(x, scale_alg)
    return torch_npu.npu_quant_matmul(
        quantized_x,
        weight,
        weight_scale,
        scale_dtype=torch_npu.float8_e8m0fnu,
        pertoken_scale=activation_scale,
        pertoken_scale_dtype=torch_npu.float8_e8m0fnu,
        bias=None,
        output_dtype=x.dtype,
        group_sizes=[1, 1, MX_GROUP_SIZE],
    )


def dense_mxfp8_all_gather_mm(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    scale_alg: int,
    tp_size: int,
) -> torch.Tensor:
    """Map local token rows to the full gate/up output on this TP rank."""
    _validate_inputs(x, weight, weight_scale, tp_size)
    if not _can_fuse(x, weight, weight_scale, tp_size) or x.shape[0] < MIN_FUSED_LOCAL_TOKENS:
        return _quant_matmul(sp_all_gather(x), weight, weight_scale, scale_alg)

    quantized_x, activation_scale = _quantize(x, scale_alg)
    activation_scale = normalize_mx_scale(activation_scale, x.shape[0], x.shape[1])
    hcom = _tp_hcom(x, tp_size)
    # The kernel gathers activations AND their MX scales. The Dense MLP has
    # no other consumer of gathered activations, so do not materialize them.
    output, _, _ = torch_npu.npu_all_gather_quant_mm(
        quantized_x,
        weight,
        hcom,
        tp_size,
        x1_scale=activation_scale,
        x2_scale=weight_scale,
        x1_scale_dtype=torch_npu.float8_e8m0fnu,
        x2_scale_dtype=torch_npu.float8_e8m0fnu,
        group_sizes=[1, 1, MX_GROUP_SIZE],
        gather_output=False,
        y_dtype=x.dtype,
        comm_mode="ai_cpu",
    )
    # A collective failure must propagate: starting fallback communication
    # after a partially launched collective can desynchronize the TP ranks.
    return output


def dense_mxfp8_all_gather_mm_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    scale_alg: int,
    tp_size: int,
) -> torch.Tensor:
    return x.new_empty((x.shape[0] * tp_size, weight.shape[1]))


def dense_mxfp8_mm_reduce_scatter(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    scale_alg: int,
    tp_size: int,
) -> torch.Tensor:
    """Sum the TP down-projection partials and return this rank's rows."""
    _validate_inputs(x, weight, weight_scale, tp_size)
    if (
        not _can_fuse(x, weight, weight_scale, tp_size)
        or x.shape[0] % tp_size != 0
        or x.shape[0] // tp_size < MIN_FUSED_LOCAL_TOKENS
        or x.shape[0] * weight.shape[1] * x.element_size() > MAX_REDUCE_SCATTER_BYTES
    ):
        # sp_reduce_scatter pads a non-divisible token count after matmul.
        return sp_reduce_scatter(_quant_matmul(x, weight, weight_scale, scale_alg))

    quantized_x, activation_scale = _quantize(x, scale_alg)
    activation_scale = normalize_mx_scale(activation_scale, x.shape[0], x.shape[1])
    hcom = _tp_hcom(x, tp_size)
    output, _ = torch_npu.npu_quant_mm_reduce_scatter(
        quantized_x,
        weight,
        hcom,
        tp_size,
        reduce_op="sum",
        x1_scale=activation_scale,
        x2_scale=weight_scale,
        x1_scale_dtype=torch_npu.float8_e8m0fnu,
        x2_scale_dtype=torch_npu.float8_e8m0fnu,
        group_sizes=[1, 1, MX_GROUP_SIZE],
        y_dtype=x.dtype,
        comm_mode="ai_cpu",
    )
    return output


def dense_mxfp8_mm_reduce_scatter_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    scale_alg: int,
    tp_size: int,
) -> torch.Tensor:
    return x.new_empty(((x.shape[0] + tp_size - 1) // tp_size, weight.shape[1]))


direct_register_custom_op(
    op_name="dense_mxfp8_all_gather_mm",
    op_func=dense_mxfp8_all_gather_mm,
    fake_impl=dense_mxfp8_all_gather_mm_fake,
    mutates_args=[],
    dispatch_key="PrivateUse1",
)

direct_register_custom_op(
    op_name="dense_mxfp8_mm_reduce_scatter",
    op_func=dense_mxfp8_mm_reduce_scatter,
    fake_impl=dense_mxfp8_mm_reduce_scatter_fake,
    mutates_args=[],
    dispatch_key="PrivateUse1",
)
