# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import torch.nn.functional as F
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_reduce_scatter,
)
from vllm.utils.math_utils import cdiv
from vllm.utils.torch_utils import direct_register_custom_op

from vllm_ascend.device.device_op import DeviceOperator
from vllm_ascend.ops.linear_op import CustomColumnParallelOp, CustomRowParallelOp


def _custom_collective(name: str, x: torch.Tensor) -> torch.Tensor | None:
    device_communicator = get_tp_group().device_communicator
    if device_communicator is None:
        return None
    collective = getattr(device_communicator, name, None)
    return None if collective is None else collective(x)


def sp_all_gather(x: torch.Tensor) -> torch.Tensor:
    output = _custom_collective("custom_all_gather", x)
    if output is not None:
        return output
    return tensor_model_parallel_all_gather(x, 0)


def _ascend_sp_shard_impl(x: torch.Tensor) -> torch.Tensor:
    """Pad the token axis (dim 0) to the TP multiple, then take this rank's chunk."""
    tp_size = get_tensor_model_parallel_world_size()
    tp_rank = get_tensor_model_parallel_rank()
    sp_pad = (-x.shape[0]) % tp_size
    # Upstream counterpart: vllm/models/common/ops/sequence_parallel.py
    # sp_shard L45-48 (introduced in 38a466e7b6, #46789).
    if sp_pad > 0:
        x = F.pad(x, (0, 0) * (x.ndim - 1) + (0, sp_pad))
    chunk = x.shape[0] // tp_size
    out = x[tp_rank * chunk : (tp_rank + 1) * chunk]
    return out.clone() if sp_pad == 0 else out


def _ascend_sp_shard_fake(x: torch.Tensor) -> torch.Tensor:
    tp_size = get_tensor_model_parallel_world_size()
    shape = list(x.shape)
    shape[0] = cdiv(x.shape[0], tp_size)
    return torch.empty(shape, dtype=x.dtype, device=x.device)


direct_register_custom_op(
    op_name="ascend_sp_shard_impl",
    op_func=_ascend_sp_shard_impl,
    mutates_args=[],
    fake_impl=_ascend_sp_shard_fake,
    dispatch_key="PrivateUse1",
)


def sp_shard(x: torch.Tensor) -> torch.Tensor:
    """Shard the token axis across TP ranks for sequence parallelism."""
    return torch.ops.vllm.ascend_sp_shard_impl(x)


def _ascend_sp_reduce_scatter_impl(x: torch.Tensor) -> torch.Tensor:
    """Pad rows to the TP multiple, then reduce-scatter across TP ranks."""
    tp_size = get_tensor_model_parallel_world_size()
    sp_pad = (-x.shape[0]) % tp_size
    # Avoid copying the full input when its token count is already aligned.
    if sp_pad > 0:
        pad_shape = [sp_pad, x.shape[1]]
        x = torch.cat([x, x.new_zeros(pad_shape)], dim=0)
    output = _custom_collective("custom_reduce_scatter", x)
    if output is not None:
        return output
    return tensor_model_parallel_reduce_scatter(x, 0)


def _ascend_sp_reduce_scatter_fake(x: torch.Tensor) -> torch.Tensor:
    tp_size = get_tensor_model_parallel_world_size()
    return torch.empty(
        [cdiv(x.shape[0], tp_size), x.shape[1]],
        dtype=x.dtype,
        device=x.device,
    )


direct_register_custom_op(
    op_name="ascend_sp_reduce_scatter_impl",
    op_func=_ascend_sp_reduce_scatter_impl,
    mutates_args=[],
    fake_impl=_ascend_sp_reduce_scatter_fake,
    dispatch_key="PrivateUse1",
)


def sp_reduce_scatter(x: torch.Tensor) -> torch.Tensor:
    assert x.ndim == 2
    return torch.ops.vllm.ascend_sp_reduce_scatter_impl(x)


def _ascend_sp_padding_mask_impl(is_padding: torch.Tensor) -> torch.Tensor:
    """Pad with True rows up to the TP multiple, then take this rank's chunk.

    The output row layout matches ``sp_shard`` so the mask stays aligned with
    the sharded hidden states. Same ``if sp_pad > 0`` pad guard as the
    upstream ``sp_padding_mask``. A custom op keeps the modulo padding
    invisible to dynamo (same shape-baking hazard as ``sp_shard``), so like
    ``sp_shard`` the no-pad slice is cloned to avoid returning a view of an
    input from a functional custom op.
    """
    tp_size = get_tensor_model_parallel_world_size()
    tp_rank = get_tensor_model_parallel_rank()
    sp_pad = (-is_padding.shape[0]) % tp_size
    # Upstream counterpart: vllm/models/common/ops/sequence_parallel.py
    # sp_padding_mask L63-65.
    if sp_pad > 0:
        is_padding = F.pad(is_padding, (0, sp_pad), value=True)
    chunk = is_padding.shape[0] // tp_size
    out = is_padding[tp_rank * chunk : (tp_rank + 1) * chunk]
    return out.clone() if sp_pad == 0 else out


def _ascend_sp_padding_mask_fake(is_padding: torch.Tensor) -> torch.Tensor:
    tp_size = get_tensor_model_parallel_world_size()
    return torch.empty(
        [cdiv(is_padding.shape[0], tp_size)],
        dtype=is_padding.dtype,
        device=is_padding.device,
    )


direct_register_custom_op(
    op_name="ascend_sp_padding_mask_impl",
    op_func=_ascend_sp_padding_mask_impl,
    fake_impl=_ascend_sp_padding_mask_fake,
    tags=(torch.Tag.needs_fixed_stride_order,),
)


def sp_padding_mask(
    is_padding: torch.Tensor | None,
    hidden_states: torch.Tensor,
) -> torch.Tensor:
    num_tokens = hidden_states.shape[0]
    if is_padding is None:
        is_padding = hidden_states.new_zeros(num_tokens, dtype=torch.bool)
    assert is_padding.shape[0] == num_tokens

    return torch.ops.vllm.ascend_sp_padding_mask_impl(is_padding)


# ---------------------------------------------------------------------------
# Dense-SP fused matmul-collective custom ops
#
# Moved here from patch/worker/patch_qwen3_5.py per reviewer request so the
# fused AllGather+Matmul / Matmul+ReduceScatter ops live alongside the other
# sequence-parallel communication primitives.  The dense-SP patch installs
# these ops on eligible projections at layer-construction time; the per-forward
# branch decision (fused kernel vs unfused fallback) lives inside the custom
# op's runtime implementation so torch.compile only sees the fake shape impl.
# ---------------------------------------------------------------------------

# torch_npu's fused matmul-collective kernels are restricted to these TP
# sizes (npu_mm_reduce_scatter_base rejects larger ones on the installed
# torch_npu builds; see the tp_size >= 16 TODO in ascend_forward_context.py).
DENSE_SP_FUSED_TP_SIZES = (2, 4, 8)
_DENSE_SP_FUSED_DTYPES = (torch.bfloat16, torch.float16)
# Minimum total token count for fusion: prefill and mixed batches have large
# token counts; pure decode batches (concurrency ~8, 1 token/seq) stay below
# this threshold and fall back to the unfused GEMM + collective path.
_DENSE_SP_FUSED_MIN_TOKENS = 128


def _dense_sp_can_fuse(
    x: torch.Tensor,
    weight: torch.Tensor,
    world_size: int,
    *,
    is_shard: bool = False,
) -> bool:
    """Whether the inputs are eligible for the fused matmul-collective kernel.

    The decision is purely structural (shapes / dtypes / TP size / token
    count) so the same path is taken during compile tracing, warmup, ACL
    graph capture and replay.  The threshold is applied to the **total**
    token count: for column-parallel layers ``x`` is the sequence-parallel
    shard (``is_shard=True``), so ``x.shape[0] * world_size`` recovers the
    full token count; for row-parallel layers ``x`` is already the full
    sequence.  This keeps the threshold consistent across projections at
    every TP size.
    """
    num_tokens = x.shape[0] * world_size if is_shard else x.shape[0]
    return (
        world_size in DENSE_SP_FUSED_TP_SIZES
        and x.ndim == 2
        and weight.ndim == 2
        and x.dtype in _DENSE_SP_FUSED_DTYPES
        and weight.dtype == x.dtype
        and num_tokens >= _DENSE_SP_FUSED_MIN_TOKENS
        and x.shape[1] == weight.shape[1]
    )


# ---------------------------------------------------------------------------
# Column-parallel: AllGather + Matmul
# ---------------------------------------------------------------------------


def _get_hcom():
    """Resolve the HCCL communicator handle at runtime (not at trace time)."""
    tp_group = get_tp_group()
    backend = tp_group.device_group._get_backend(torch.device("npu"))
    return backend.get_hccl_comm_name(tp_group.rank_in_group)


def _dense_sp_all_gather_mm_impl(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    world_size: int,
) -> torch.Tensor:
    if _dense_sp_can_fuse(x, weight, world_size, is_shard=True):
        return DeviceOperator.npu_all_gather_base_mm(
            x.contiguous(),
            weight.t(),
            _get_hcom(),
            world_size,
            bias=bias,
        )
    input_parallel = sp_all_gather(x)
    return F.linear(input_parallel, weight, bias)


def _dense_sp_all_gather_mm_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    world_size: int,
) -> torch.Tensor:
    return x.new_empty((x.shape[0] * world_size, weight.shape[0]))


direct_register_custom_op(
    op_name="dense_sp_all_gather_mm",
    op_func=_dense_sp_all_gather_mm_impl,
    fake_impl=_dense_sp_all_gather_mm_fake,
    mutates_args=[],
    dispatch_key="PrivateUse1",
)


# ---------------------------------------------------------------------------
# Row-parallel: Matmul + ReduceScatter
# ---------------------------------------------------------------------------


def _dense_sp_mm_reduce_scatter_impl(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    skip_bias_add: bool,
    world_size: int,
) -> torch.Tensor:
    if _dense_sp_can_fuse(x, weight, world_size):
        sp_pad = (-x.shape[0]) % world_size
        if sp_pad:
            x = F.pad(x, (0, 0, 0, sp_pad))
        output = DeviceOperator.npu_mm_reduce_scatter_base(
            x,
            weight.t(),
            _get_hcom(),
            world_size,
            reduce_op="sum",
        )
        if bias is not None and not skip_bias_add:
            output = output + bias
        return output
    tp_rank = get_tp_group().rank_in_group
    bias_ = None if (tp_rank > 0 or skip_bias_add) else bias
    output_parallel = F.linear(x, weight, bias_)
    return sp_reduce_scatter(output_parallel)


def _dense_sp_mm_reduce_scatter_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    skip_bias_add: bool,
    world_size: int,
) -> torch.Tensor:
    return x.new_empty((cdiv(x.shape[0], world_size), weight.shape[0]))


direct_register_custom_op(
    op_name="dense_sp_mm_reduce_scatter",
    op_func=_dense_sp_mm_reduce_scatter_impl,
    fake_impl=_dense_sp_mm_reduce_scatter_fake,
    mutates_args=[],
    dispatch_key="PrivateUse1",
)


class DenseSpAllGatherMmOp(CustomColumnParallelOp):
    """Fused AllGather + Matmul for a dense-SP column projection.

    The branch decision lives inside the registered custom op's runtime
    implementation so torch.compile only sees the fake (shape) impl during
    tracing.  The structural guard (token-count threshold) directs prefill
    and mixed batches to the fused kernel and pure decode to the unfused
    path, captured and replayed in FULL_DECODE_ONLY.
    """

    def __init__(self, layer):
        super().__init__(layer)
        self.update_attrs()
        tp_group = get_tp_group()
        backend = tp_group.device_group._get_backend(torch.device("npu"))
        self.hcom = backend.get_hccl_comm_name(tp_group.rank_in_group)
        self.fused_tp_size = tp_group.world_size

    def apply_impl(self, input_):
        bias = self.bias if not self.skip_bias_add else None
        output = torch.ops.vllm.dense_sp_all_gather_mm(
            input_,
            self.layer.weight,
            bias,
            self.fused_tp_size,
        )
        output_bias = self.bias if self.skip_bias_add else None
        return output, output_bias


class DenseSpMmReduceScatterOp(CustomRowParallelOp):
    """Fused Matmul + ReduceScatter for a dense-SP row projection.

    Same custom-op dispatch pattern as the column projection: the branch
    decision lives in the runtime implementation, not in ``apply_impl``.
    The token-count threshold keeps prefill/mixed on the fused path and
    pure decode on the unfused path.
    """

    def __init__(self, layer):
        super().__init__(layer)
        self.update_attrs()
        tp_group = get_tp_group()
        backend = tp_group.device_group._get_backend(torch.device("npu"))
        self.hcom = backend.get_hccl_comm_name(tp_group.rank_in_group)
        self.fused_tp_size = tp_group.world_size

    def apply_impl(self, input_):
        input_parallel = self.get_input_parallel(input_)
        bias = self.bias
        output = torch.ops.vllm.dense_sp_mm_reduce_scatter(
            input_parallel,
            self.layer.weight,
            bias,
            self.skip_bias_add,
            self.fused_tp_size,
        )
        output_bias = self.bias if self.skip_bias_add else None
        return output, output_bias


# ---------------------------------------------------------------------------
# Row-parallel: Matmul + ReduceScatter (pre-quantized W8A8 INT8 input)
#
# Shared experts (W8A8/W4A8) produce INT8-quantized activations from
# ``dequant_situ_quant`` / ``npu_dequant_swiglu_quant`` with a per-token
# scale.  The existing ``dense_sp_mm_reduce_scatter`` path expects a bf16
# input and does its own ``npu_dynamic_quant`` internally; this variant
# accepts already-quantized INT8 + pertoken_scale so the shared expert's
# down_proj can fuse Matmul + ReduceScatter without a redundant quant.
# ---------------------------------------------------------------------------


def _dense_sp_mm_reduce_scatter_quant_prequant_impl(
    quantized_x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    pertoken_scale: torch.Tensor,
    output_dtype: torch.dtype,
    world_size: int,
) -> torch.Tensor:
    """Fused MM+RS for pre-quantized INT8 activations from SwiGLU/dequant.

    ``quantized_x`` is INT8 with shape [T, K] (already quantized by
    ``dequant_situ_quant`` or ``npu_dequant_swiglu_quant``).
    ``weight`` is INT8 with shape [K, N] (post process_weights layout).
    ``pertoken_scale`` is the per-token scale from the quant kernel.
    """
    tp_size = world_size
    num_tokens = quantized_x.shape[0]
    if (
        tp_size in DENSE_SP_FUSED_TP_SIZES
        and quantized_x.ndim == 2
        and weight.ndim == 2
        and num_tokens >= _DENSE_SP_FUSED_MIN_TOKENS
    ):
        sp_pad = (-num_tokens) % tp_size
        if sp_pad:
            quantized_x = F.pad(quantized_x, (0, 0, 0, sp_pad))
            pertoken_scale = F.pad(pertoken_scale, (0, 0, 0, sp_pad))
        output = DeviceOperator.npu_mm_reduce_scatter_base(
            quantized_x,
            weight,
            _get_hcom(),
            tp_size,
            reduce_op="sum",
            x1_scale=pertoken_scale.reshape(-1, 1).to(torch.float32),
            x2_scale=weight_scale.reshape(1, -1).to(torch.float32),
            output_dtype=output_dtype,
        )
        return output
    # Fallback: unfused npu_quant_matmul + sp_reduce_scatter
    import torch_npu

    output_parallel = torch_npu.npu_quant_matmul(
        quantized_x,
        weight,
        weight_scale,
        pertoken_scale=pertoken_scale,
        bias=None,
        output_dtype=output_dtype,
    )
    return sp_reduce_scatter(output_parallel)


def _dense_sp_mm_reduce_scatter_quant_prequant_fake(
    quantized_x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    pertoken_scale: torch.Tensor,
    output_dtype: torch.dtype,
    world_size: int,
) -> torch.Tensor:
    return quantized_x.new_empty((cdiv(quantized_x.shape[0], world_size), weight.shape[1]))


direct_register_custom_op(
    op_name="dense_sp_mm_reduce_scatter_quant_prequant",
    op_func=_dense_sp_mm_reduce_scatter_quant_prequant_impl,
    fake_impl=_dense_sp_mm_reduce_scatter_quant_prequant_fake,
    mutates_args=[],
    dispatch_key="PrivateUse1",
)


# ---------------------------------------------------------------------------
# Column-parallel: AllGather + Matmul (W8A8 INT8 quantized weights)
#
# Shared experts (W8A8/W4A8) use INT8 weights with bf16 SP-shard activations.
# The fused AllGather+Matmul quantizes the activation inside the kernel via
# x1_scale/x2_scale so we pass the raw bf16 SP shard and the INT8 weight.
# ---------------------------------------------------------------------------


def _dense_sp_all_gather_mm_quant_int8_impl(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    world_size: int,
) -> torch.Tensor:
    """Fused AG+MM for W8A8 INT8 weights with bf16 SP-shard activation.

    ``x`` is the bf16 SP shard [T/TP, H].
    ``weight`` is INT8 [K, N] (post process_weights layout).
    Dynamic quantization of x happens inside the fused kernel via x1_scale.
    """
    import torch_npu

    num_tokens = x.shape[0] * world_size
    if (
        world_size in DENSE_SP_FUSED_TP_SIZES
        and x.ndim == 2
        and weight.ndim == 2
        and x.dtype in _DENSE_SP_FUSED_DTYPES
        and num_tokens >= _DENSE_SP_FUSED_MIN_TOKENS
    ):
        quantized_x, pertoken_scale = torch_npu.npu_dynamic_quant(x, dst_type=torch.int8)
        if pertoken_scale.dim() == 2:
            quantized_x = quantized_x.squeeze(dim=1)
            pertoken_scale = pertoken_scale.squeeze(dim=1)
        output = DeviceOperator.npu_all_gather_base_mm(
            quantized_x,
            weight,
            _get_hcom(),
            world_size,
            bias=bias,
            x1_scale=pertoken_scale.reshape(-1, 1).to(torch.float32),
            x2_scale=weight_scale.reshape(1, -1).to(torch.float32),
            output_dtype=x.dtype,
        )
        return output
    # Fallback: unfused AllGather + npu_quant_matmul
    input_parallel = sp_all_gather(x)
    quantized_x, pertoken_scale = torch_npu.npu_dynamic_quant(input_parallel, dst_type=torch.int8)
    if pertoken_scale.dim() == 2:
        quantized_x = quantized_x.squeeze(dim=1)
        pertoken_scale = pertoken_scale.squeeze(dim=1)
    return torch_npu.npu_quant_matmul(
        quantized_x,
        weight,
        weight_scale,
        pertoken_scale=pertoken_scale,
        bias=bias,
        output_dtype=x.dtype,
    )


def _dense_sp_all_gather_mm_quant_int8_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    world_size: int,
) -> torch.Tensor:
    return x.new_empty((x.shape[0] * world_size, weight.shape[1]))


direct_register_custom_op(
    op_name="dense_sp_all_gather_mm_quant_int8",
    op_func=_dense_sp_all_gather_mm_quant_int8_impl,
    fake_impl=_dense_sp_all_gather_mm_quant_int8_fake,
    mutates_args=[],
    dispatch_key="PrivateUse1",
)


# ---------------------------------------------------------------------------
# MXFP8 helpers (A5 scene-4 fused quant operators)
#
# npu_all_gather_quant_mm / npu_quant_mm_reduce_scatter accept MXFP8
# (float8_e4m3fn) activations with E8M0 per-group scales (blockSize=32).
# Weights must be in ND format (NZ is rejected), so callers cache an ND copy
# after process_weights_after_loading.
# ---------------------------------------------------------------------------

MX_GROUP_SIZE = 32
MX_SCALE_PAIR_SIZE = 2 * MX_GROUP_SIZE
ACL_FORMAT_FRACTAL_ND = 2


def _ensure_nd(tensor: torch.Tensor) -> torch.Tensor:
    """Convert NZ (FRACTAL_NZ) tensors back to ND for A5 fused quant ops."""
    import torch_npu

    if tensor.device.type == "npu":
        fmt = int(torch_npu.get_npu_format(tensor))
        if fmt not in (0, ACL_FORMAT_FRACTAL_ND):
            return torch_npu.npu_format_cast(tensor, ACL_FORMAT_FRACTAL_ND)
    return tensor


def _normalize_mx_scale(scale: torch.Tensor, rows: int, k: int) -> torch.Tensor:
    """Reshape E8M0 scale to the paired (rows, K//64, 2) layout CANN expects."""
    if k % MX_SCALE_PAIR_SIZE:
        raise ValueError("MXFP8 MC2 requires K aligned to two MX groups.")
    packed_shape = (rows, k // MX_SCALE_PAIR_SIZE, 2)
    if tuple(scale.shape) == (rows, k // MX_GROUP_SIZE):
        return scale.reshape(packed_shape).contiguous()
    if tuple(scale.shape) != packed_shape:
        raise ValueError(
            f"Invalid MXFP8 activation scale shape: {tuple(scale.shape)} != {packed_shape}"
        )
    return scale.contiguous()


def _mx_quant_activations(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """MXFP8 activation quantization via npu_dynamic_mx_quant.

    Returns (quantized_x, scale) where quantized_x is float8_e4m3fn and
    scale is the E8M0 per-group scale in the paired (M, K//64, 2) layout.
    """
    import torch_npu

    quantized_x, scale = torch_npu.npu_dynamic_mx_quant(
        x,
        dst_type=torch.float8_e4m3fn,
    )
    if scale.ndim == 2 and scale.shape[-1] % 2 == 0:
        scale = scale.reshape(scale.shape[0], scale.shape[1] // 2, 2)
    return quantized_x, scale


# ---------------------------------------------------------------------------
# Column-parallel: AllGather + Matmul (MXFP8 quantized weights)
# ---------------------------------------------------------------------------


def _dense_sp_all_gather_mm_quant_mxfp8_impl(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    world_size: int,
) -> torch.Tensor:
    """Fused AG+MM for MXFP8 (float8_e4m3fn) weights with bf16 SP-shard activation.

    Uses the A5 dedicated fused quant operator npu_all_gather_quant_mm
    (scene 4: E8M0 per-group scale, blockSize=32).  The op AllGathers the
    quantized activation and its scale internally, so we quantize the local
    SP shard (not the gathered result).  Falls back to unfused AllGather +
    npu_quant_matmul when the structural guard rejects fusion.
    """
    import torch_npu

    num_tokens = x.shape[0] * world_size
    if (
        world_size in DENSE_SP_FUSED_TP_SIZES
        and x.ndim == 2
        and weight.ndim == 2
        and x.dtype in _DENSE_SP_FUSED_DTYPES
        and num_tokens >= _DENSE_SP_FUSED_MIN_TOKENS
    ):
        quantized_x, x_scale = _mx_quant_activations(x)
        if bias is not None and bias.dtype != torch.float32:
            bias = bias.to(torch.float32)
        nd_weight = _ensure_nd(weight)
        output, _, _ = torch_npu.npu_all_gather_quant_mm(
            quantized_x,
            nd_weight,
            _get_hcom(),
            world_size,
            bias=bias,
            x1_scale=x_scale,
            x2_scale=weight_scale,
            group_sizes=[1, 1, MX_GROUP_SIZE],
            y_dtype=x.dtype,
            x1_scale_dtype=torch_npu.float8_e8m0fnu,
            x2_scale_dtype=torch_npu.float8_e8m0fnu,
        )
        return output
    # Fallback: unfused AllGather + npu_quant_matmul
    input_parallel = sp_all_gather(x)
    quantized_x, pertoken_scale = _mx_quant_activations(input_parallel)
    if bias is not None and bias.dtype != torch.float32:
        bias = bias.to(torch.float32)
    return torch_npu.npu_quant_matmul(
        quantized_x,
        weight,
        weight_scale,
        scale_dtype=torch_npu.float8_e8m0fnu,
        pertoken_scale=pertoken_scale,
        pertoken_scale_dtype=torch_npu.float8_e8m0fnu,
        bias=bias,
        output_dtype=x.dtype,
        group_sizes=[1, 1, MX_GROUP_SIZE],
    )


def _dense_sp_all_gather_mm_quant_mxfp8_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    world_size: int,
) -> torch.Tensor:
    # MXFP8 weights are stored as (K, N) after process_weights transpose.
    return x.new_empty((x.shape[0] * world_size, weight.shape[1]))


direct_register_custom_op(
    op_name="dense_sp_all_gather_mm_quant_mxfp8",
    op_func=_dense_sp_all_gather_mm_quant_mxfp8_impl,
    fake_impl=_dense_sp_all_gather_mm_quant_mxfp8_fake,
    mutates_args=[],
    dispatch_key="PrivateUse1",
)


# ---------------------------------------------------------------------------
# Row-parallel: Matmul + ReduceScatter (MXFP8 pre-quantized input)
#
# Shared experts (W4A8MXFP) produce MXFP8 (float8_e4m3fn) activations from
# situ_mx_quant / npu_swiglu_group_quant with an E8M0 per-group scale.  This
# variant accepts already-quantized MXFP8 + scale so the shared expert's
# down_proj can fuse MM+RS without a redundant re-quantization.
# ---------------------------------------------------------------------------


def _dense_sp_mm_reduce_scatter_quant_mxfp8_prequant_impl(
    quantized_x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    pertoken_scale: torch.Tensor,
    output_dtype: torch.dtype,
    world_size: int,
) -> torch.Tensor:
    """Fused MM+RS for pre-quantized MXFP8 activations from SwiGLU/MX quant.

    ``quantized_x`` is float8_e4m3fn with shape [T, K].
    ``weight`` is float8_e4m3fn with shape [K, N] (post process_weights ND).
    ``pertoken_scale`` is the E8M0 per-group scale from the quant kernel,
    in the paired (T, K//64, 2) layout.
    """
    import torch_npu

    tp_size = world_size
    num_tokens = quantized_x.shape[0]
    if (
        tp_size in DENSE_SP_FUSED_TP_SIZES
        and quantized_x.ndim == 2
        and weight.ndim == 2
        and num_tokens >= _DENSE_SP_FUSED_MIN_TOKENS
    ):
        sp_pad = (-num_tokens) % tp_size
        if sp_pad:
            quantized_x = F.pad(quantized_x, (0, 0, 0, sp_pad))
            # pertoken_scale is (T, K//64, 2) — pad the token dim (dim 0).
            pertoken_scale = F.pad(pertoken_scale, (0, 0, 0, 0, 0, sp_pad))
        nd_weight = _ensure_nd(weight)
        output, _ = torch_npu.npu_quant_mm_reduce_scatter(
            quantized_x,
            nd_weight,
            _get_hcom(),
            tp_size,
            reduce_op="sum",
            x1_scale=pertoken_scale,
            x2_scale=weight_scale,
            group_sizes=[1, 1, MX_GROUP_SIZE],
            y_dtype=output_dtype,
            x1_scale_dtype=torch_npu.float8_e8m0fnu,
            x2_scale_dtype=torch_npu.float8_e8m0fnu,
        )
        return output
    # Fallback: unfused npu_quant_matmul + sp_reduce_scatter
    output_parallel = torch_npu.npu_quant_matmul(
        quantized_x,
        weight,
        weight_scale,
        scale_dtype=torch_npu.float8_e8m0fnu,
        pertoken_scale=pertoken_scale,
        pertoken_scale_dtype=torch_npu.float8_e8m0fnu,
        bias=None,
        output_dtype=output_dtype,
        group_sizes=[1, 1, MX_GROUP_SIZE],
    )
    return sp_reduce_scatter(output_parallel)


def _dense_sp_mm_reduce_scatter_quant_mxfp8_prequant_fake(
    quantized_x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    pertoken_scale: torch.Tensor,
    output_dtype: torch.dtype,
    world_size: int,
) -> torch.Tensor:
    return quantized_x.new_empty((cdiv(quantized_x.shape[0], world_size), weight.shape[1]))


direct_register_custom_op(
    op_name="dense_sp_mm_reduce_scatter_quant_mxfp8_prequant",
    op_func=_dense_sp_mm_reduce_scatter_quant_mxfp8_prequant_impl,
    fake_impl=_dense_sp_mm_reduce_scatter_quant_mxfp8_prequant_fake,
    mutates_args=[],
    dispatch_key="PrivateUse1",
)
