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
"""Ascend online (load-time) INT8 quantization.

Unlike a ModelSlim / LLM-Compressor checkpoint, an online-quantized layer has no
``weight_scale`` in the checkpoint at all: only the full-precision weight
exists, and the scale has to be derived from it while the weights are being
loaded. vLLM owns that staging -- ``OnlineLinearBase`` / ``OnlineMoEMethodBase``
allocate the checkpoint-dtype weight on the meta device and wrap the loaders --
so the methods here only override ``process_weights_after_loading``: quantize
the loaded weight, install the scale/offset parameters, and hand the layer over
to the offline ``W8A8_DYNAMIC`` scheme for the runtime layout pass.

The recipe is per-output-channel symmetric INT8 weights with dynamically
per-token INT8 activations -- the same one a ModelSlim ``W8A8_DYNAMIC``
checkpoint uses, so kernel dispatch, gmm hooks, EPLB views and MC2 weights come
from that scheme unchanged.
"""

from __future__ import annotations

import torch
from vllm.logger import logger
from vllm.model_executor.layers.linear import RowParallelLinear
from vllm.model_executor.layers.quantization.utils.quant_utils import weight_amax
from vllm.model_executor.utils import replace_parameter

from vllm_ascend.quantization.method_adapters import (
    AscendOnlineFusedMoEMethod,
    AscendOnlineLinearMethod,
)
from vllm_ascend.quantization.methods.w8a8.w8a8_dynamic import (
    AscendW8A8DynamicFusedMoEMethod,
    AscendW8A8DynamicLinearMethod,
)
from vllm_ascend.utils import mlp_tp_enable, oproj_tp_enable

__all__ = [
    "AscendInt8PerChannelOnlineLinearMethod",
    "AscendInt8PerChannelOnlineMoEMethod",
    "quantize_last_dim_to_int8",
]

# Symmetric INT8: the representable magnitudes are -127..127 and the dequant
# scale is ``amax / 127``, matching ModelSlim and llm-compressor's
# INT8 W8A8_DYNAMIC recipe.
_INT8_MAX = torch.iinfo(torch.int8).max
# Bound the float32 intermediate of the quantize pass: each element costs 4
# bytes, so 32M elements is a 128 MB temporary regardless of the weight shape.
# The amax pass needs no such bound -- ``weight_amax`` reduces with ``aminmax``
# and never materializes a full-size ``|weight|``.
_QUANT_CHUNK_ELEMS = 32 * 1024 * 1024

# Same contract as upstream online quantization: a layer's process pass is
# idempotent and may be re-entered by vLLM's post-load hook.
_PROCESSED_FLAG = "_already_called_process_weights_after_loading"


def _int8_max(device: torch.device) -> torch.Tensor:
    """0-d float32 divisor, so the scale uses the kernels' true division.

    Dividing by a Python scalar can be lowered to multiply-by-reciprocal, which
    rounds differently; upstream's FP8 scale helper uses the same trick.
    """
    return torch.tensor(_INT8_MAX, device=device, dtype=torch.float32)


def _reduce_amax(amax: torch.Tensor, reduce_group) -> torch.Tensor:
    """Take the max amax across the ranks that shard the weight."""
    if reduce_group is None or reduce_group.world_size <= 1:
        return amax
    torch.distributed.all_reduce(
        amax,
        op=torch.distributed.ReduceOp.MAX,
        group=reduce_group.device_group,
    )
    return amax


def quantize_last_dim_to_int8(
    weight: torch.Tensor,
    *,
    reduce_group=None,
    scale_dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric per-output-channel INT8 quantization of ``weight``.

    The reduction happens along the **last** dimension, which is the K
    (input/contraction) dimension for every Ascend weight layout:
    ``[out, in]`` for a linear, ``[experts, out, in]`` for a fused expert stack.

    Args:
        weight: Full-precision weight whose last dim is the contraction dim.
        reduce_group: Group coordinator spanning the ranks that shard the last
            dim. ``None`` means each rank already owns the whole reduction dim
            and therefore derives the same scale the unsharded weight would.
        scale_dtype: dtype of the returned dequant scale. Ascend's
            ``npu_quant_matmul`` consumes the INT8 W8A8 scale in the activation
            dtype, so this must match the offline scheme's parameter dtype.

    Returns:
        ``(qweight, scale)`` where ``qweight`` is ``int8`` with the same shape
        as ``weight`` and ``scale`` is ``weight.shape[:-1] + (1,)``.
    """
    if weight.dtype == torch.int8:
        raise NotImplementedError(
            f"Refusing to re-quantize an already-int8 online-quantized weight of shape {tuple(weight.shape)}. "
            "Reloading weights into an online-quantized model is not supported."
        )

    original_shape = weight.shape
    reduction_shape = (*original_shape[:-1], 1)
    contraction_dim = original_shape[-1] if original_shape else 0
    if weight.numel() == 0 or contraction_dim == 0:
        return (
            torch.empty(original_shape, dtype=torch.int8, device=weight.device),
            torch.zeros(reduction_shape, dtype=scale_dtype, device=weight.device),
        )

    flat = weight.reshape(-1, contraction_dim)
    rows = flat.shape[0]

    # ``weight_amax`` reduces with ``aminmax``, so no full-size ``|weight|`` is
    # ever materialized and the amax pass needs no chunking.
    amax = weight_amax(flat, dim=-1, keepdim=True).float()
    amax = _reduce_amax(amax, reduce_group)

    # An all-zero row would make the scale zero and the division undefined.
    # Quantize with a neutral scale and zero the stored scale afterwards; the
    # reconstruction ``q * scale`` is still exactly zero.
    nonzero = amax > 0
    safe_amax = torch.where(nonzero, amax, torch.ones_like(amax))
    scale = (safe_amax / _int8_max(amax.device)).to(scale_dtype)

    # Quantize with the *stored* scale so ``q * scale`` is the best
    # reconstruction the round-tripped scale allows.
    scale_fp32 = scale.float()
    row_step = max(1, _QUANT_CHUNK_ELEMS // contraction_dim)
    qweight = torch.empty(flat.shape, dtype=torch.int8, device=flat.device)
    for start in range(0, rows, row_step):
        end = min(start + row_step, rows)
        # The divide allocates the only float32 temporary; round/clamp on it.
        codes = flat[start:end].float() / scale_fp32[start:end]
        qweight[start:end] = codes.round_().clamp_(-_INT8_MAX, _INT8_MAX).to(torch.int8)

    scale = torch.where(nonzero, scale, torch.zeros_like(scale))
    return qweight.view(original_shape), scale.view(reduction_shape)


def _linear_reduce_group(layer: torch.nn.Module):
    """Group to all-reduce the weight amax over, or ``None``.

    A per-output-channel scale reduces over the input dim, so it only needs a
    cross-rank reduction when the input dim is the one being sharded -- i.e. a
    row-parallel projection. Ascend shards ``o_proj`` and ``down_proj`` with
    their own fine-grained tensor-parallel groups when those are enabled, so
    the reduction has to follow the group that actually owns the shard.
    """
    if getattr(layer, "tp_size", 1) <= 1:
        return None
    input_size = getattr(layer, "input_size", None)
    input_size_per_partition = getattr(layer, "input_size_per_partition", None)
    if input_size is None or input_size_per_partition is None:
        return None
    if input_size == input_size_per_partition:
        # Column-parallel (or replicated): every row this rank owns is complete.
        return None

    prefix = getattr(layer, "prefix", "") or ""
    if isinstance(layer, RowParallelLinear):
        from vllm_ascend.distributed.parallel_state import get_mlp_tp_group, get_otp_group

        if "o_proj" in prefix and oproj_tp_enable():
            return get_otp_group()
        if "down_proj" in prefix and mlp_tp_enable():
            return get_mlp_tp_group()

    from vllm.distributed import get_tp_group

    return get_tp_group()


def _moe_reduce_group(moe_config):
    """Group to all-reduce the expert weight amax over, or ``None``.

    ``moe_config.tp_size`` is 1 whenever expert parallelism is on (each rank
    then owns whole experts), so no reduction is needed. It is greater than 1
    only when the expert weights themselves are tensor-sharded, in which case
    ``w2``'s contraction dim is split across the TP ranks.
    """
    tp_size = getattr(moe_config, "tp_size", 1)
    if tp_size <= 1:
        return None
    dp_size = getattr(moe_config, "dp_size", 1)
    pcp_size = getattr(moe_config, "pcp_size", 1)
    if dp_size > 1 or pcp_size > 1:
        raise NotImplementedError(
            "Online int8 quantization of tensor-sharded MoE experts is only supported "
            "when the flattened expert TP group is the plain TP group, but got "
            f"tp_size={tp_size}, dp_size={dp_size}, pcp_size={pcp_size}. "
            "Enable expert parallelism (--enable-expert-parallel) or use "
            "--tensor-parallel-size 1 for the expert weights."
        )

    from vllm.distributed import get_tp_group

    return get_tp_group()


def _params_on(device: torch.device, params: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Materialize the offline scheme's parameter buffers on ``device``.

    The offline schemes build them with a bare ``torch.empty`` and rely on the
    ambient default device, which vLLM only sets to the accelerator inside its
    model-construction scope. Online quantization creates these parameters
    later, from ``process_weights_after_loading``, where the default device is
    still the CPU.
    """
    return {name: tensor.to(device) for name, tensor in params.items()}


class AscendInt8PerChannelOnlineLinearMethod(AscendOnlineLinearMethod):
    """Online INT8 per-output-channel quantization for dense linear layers.

    The weight is quantized from the loaded checkpoint dtype with one symmetric
    INT8 scale per output channel, while activations are quantized dynamically
    per token at runtime. That is the ModelSlim ``W8A8_DYNAMIC`` recipe, so
    :class:`AscendW8A8DynamicLinearMethod` supplies the runtime layout, the
    kernel call and the scale parameters unchanged.
    """

    scheme_cls = AscendW8A8DynamicLinearMethod

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if getattr(layer, _PROCESSED_FLAG, False):
            return

        weight = layer.weight.data
        if weight.dim() != 2:
            raise ValueError(
                f"Online int8 quantization expects a 2-D linear weight, got shape {tuple(weight.shape)} "
                f"for {getattr(layer, 'prefix', '<unknown>')}."
            )

        params_dtype = weight.dtype
        qweight, scale = quantize_last_dim_to_int8(
            weight,
            reduce_group=_linear_reduce_group(layer),
            scale_dtype=params_dtype,
        )

        # Allocate the scale/offset tensors through the offline scheme so their
        # shape, dtype and device are exactly what the runtime kernel expects.
        buffers = _params_on(weight.device, self.scheme.get_perchannel_param(weight.shape[0], params_dtype))
        scale_buffer = buffers["weight_scale"]
        offset_buffer = buffers["weight_offset"]
        scale_buffer.copy_(scale.reshape(scale_buffer.shape))
        offset_buffer.zero_()

        replace_parameter(layer, "weight", qweight)
        replace_parameter(layer, "weight_scale", scale_buffer)
        replace_parameter(layer, "weight_offset", offset_buffer)

        # Transpose to K-major, cast to FRACTAL_NZ and flatten the scales.
        super().process_weights_after_loading(layer)

        setattr(layer, _PROCESSED_FLAG, True)
        logger.debug(
            "Online int8 quantized linear %s to %s.",
            getattr(layer, "prefix", "<unknown>"),
            tuple(layer.weight.data.shape),
        )


def _zero_expert_padding(moe_config, layer: torch.nn.Module) -> None:
    """Zero the rounded-up rows/columns of the expert weights.

    ``RoutedExperts`` may allocate the expert weights padded to a hardware
    friendly size. Those elements are never written by the checkpoint, so they
    hold whatever ``torch.empty`` returned and would otherwise dominate the
    per-channel amax. The padded lanes are masked out of the matmul anyway, so
    zeroing them is safe; the quantized values there are never read.
    """
    hidden_size = getattr(moe_config, "hidden_dim_unpadded", None)
    intermediate_size = getattr(moe_config, "intermediate_size_per_partition_unpadded", None)
    if hidden_size is None or intermediate_size is None:
        return

    num_shards = getattr(moe_config, "w13_num_shards", 2)
    w13_weight = layer.w13_weight.data
    w2_weight = layer.w2_weight.data

    shard_size = w13_weight.shape[1] // num_shards
    if shard_size > intermediate_size:
        for shard in range(num_shards):
            start = shard * shard_size + intermediate_size
            w13_weight[:, start : (shard + 1) * shard_size, :] = 0
    if w13_weight.shape[2] > hidden_size:
        w13_weight[:, :, hidden_size:] = 0
    if w2_weight.shape[1] > hidden_size:
        w2_weight[:, hidden_size:, :] = 0
    if w2_weight.shape[2] > intermediate_size:
        w2_weight[:, :, intermediate_size:] = 0


class AscendInt8PerChannelOnlineMoEMethod(AscendOnlineFusedMoEMethod):
    """Online INT8 per-output-channel quantization for routed experts.

    Per-expert, per-output-channel weights plus dynamically per-token quantized
    INT8 activations; the layout pass, kernel dispatch, EPLB views and MC2
    weights come from :class:`AscendW8A8DynamicFusedMoEMethod` unchanged.
    """

    scheme_cls = AscendW8A8DynamicFusedMoEMethod

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if getattr(layer, _PROCESSED_FLAG, False):
            return

        w13_weight = layer.w13_weight.data
        w2_weight = layer.w2_weight.data
        if w13_weight.dim() != 3 or w2_weight.dim() != 3:
            raise ValueError(
                "Online int8 quantization expects 3-D expert weights, got "
                f"w13={tuple(w13_weight.shape)} and w2={tuple(w2_weight.shape)} "
                f"for {getattr(layer, 'prefix', '<unknown>')}."
            )

        params_dtype = w13_weight.dtype
        num_experts, w13_out, hidden_size = w13_weight.shape
        if w13_out % 2 != 0:
            raise ValueError(
                "Online int8 quantization expects a fused gate/up expert weight with an even output dim, "
                f"got {w13_out} for {getattr(layer, 'prefix', '<unknown>')}."
            )
        intermediate_size = w13_out // 2

        _zero_expert_padding(layer.moe_config, layer)

        reduce_group = _moe_reduce_group(layer.moe_config)
        # w13 contracts over hidden_size, which expert parallelism never shards.
        # w2 contracts over intermediate_size, which is sharded when the experts
        # themselves are tensor-parallel.
        w13_qweight, w13_scale = quantize_last_dim_to_int8(w13_weight, reduce_group=None, scale_dtype=params_dtype)
        w2_qweight, w2_scale = quantize_last_dim_to_int8(w2_weight, reduce_group=reduce_group, scale_dtype=params_dtype)

        buffers = _params_on(
            w13_weight.device,
            self.quant_method.get_dynamic_quant_param(num_experts, intermediate_size, hidden_size, params_dtype),
        )

        buffers["w13_weight_scale"].copy_(w13_scale.reshape(buffers["w13_weight_scale"].shape))
        buffers["w2_weight_scale"].copy_(w2_scale.reshape(buffers["w2_weight_scale"].shape))
        buffers["w13_weight_offset"].zero_()
        buffers["w2_weight_offset"].zero_()

        replace_parameter(layer, "w13_weight", w13_qweight)
        replace_parameter(layer, "w2_weight", w2_qweight)
        for name in ("w13_weight_scale", "w13_weight_offset", "w2_weight_scale", "w2_weight_offset"):
            replace_parameter(layer, name, buffers[name])

        # Expert transpose, FRACTAL_NZ cast, scale flattening, fused-MC2 scale
        # packing and the EPLB per-expert views.
        super().process_weights_after_loading(layer)

        setattr(layer, _PROCESSED_FLAG, True)
        logger.debug(
            "Online int8 quantized MoE %s: %d experts, w13=%s, w2=%s.",
            getattr(layer, "prefix", "<unknown>"),
            num_experts,
            tuple(w13_qweight.shape),
            tuple(w2_qweight.shape),
        )
