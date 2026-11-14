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
"""Schemes for native block-wise FP8 checkpoints (``quant_method: "fp8"``).

Official FP8 releases such as ``Qwen/Qwen3.8-27B-FP8`` store each quantized
matrix as a ``float8_e4m3fn`` tensor plus a ``weight_scale_inv`` tensor holding
one ``float32`` scale per ``weight_block_size`` tile. ``fp8_mode`` decides how
that layout reaches the hardware:

* ``"native"``: the float8 weight and its 128x128 block scales are kept and
  consumed by the G-B (group-blockwise) matmul kernels, with dynamically
  block-quantized activations.
* ``"mxfp8"`` (default): the tiles are resolved after loading and re-quantized
  to MXFP8 (one E8M0 scale per 32 elements along the reduction dim).

A few matrices stay dense in the model dtype either way: the ones the attention
backend absorbs (``kv_b_proj``) and, under ``"mxfp8"``, the ones whose reduction
dim is not a multiple of the MXFP8 group size.

Ascend generations without an FP8 matmul are rejected instead of served: rewriting
such a checkpoint into the model dtype doubles the weight footprint to run an
unquantized GEMM, which is silent dequantization rather than a fallback.
"""

from typing import Any

import torch
import torch_npu
from vllm.config import get_current_vllm_config
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.logger import logger
from vllm.model_executor.utils import replace_parameter
from vllm.utils.math_utils import cdiv

from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.ops.fused_moe.dataclass.fused_experts import build_fused_experts_input
from vllm_ascend.ops.fused_moe.dataclass.moe_mlp import MoEMlpComputeInput
from vllm_ascend.quantization.utils import get_dynamic_mx_quant_scale_alg
from vllm_ascend.utils import FP8_METHOD, dispose_tensor, maybe_trans_nz

from ..base import (
    AscendLinearScheme,
    AscendMoEScheme,
    PreparedLinearInput,
    QuantType,
    WeightSwitchConfig,
    WeightSwitchGatherSpec,
    WeightSwitchRepeatSpec,
)
from ..registry import register_scheme
from .w8a8_mxfp8 import AscendW8A8MXFP8DynamicFusedMoEMethod, AscendW8A8MXFP8DynamicLinearMethod

BLOCK_FP8_WEIGHT_DTYPE = torch.float8_e4m3fn

# Only this block layout is accepted by the G-B matmul kernels.
NATIVE_FP8_BLOCK_SIZE = 128

# ``group_sizes`` of the G-B ``npu_quant_matmul``: one row per group, then one
# 128-wide group along the reduction dim and one along the output dim.
_NATIVE_BLOCK_GROUP_SIZES = [1, NATIVE_FP8_BLOCK_SIZE, NATIVE_FP8_BLOCK_SIZE]

# Output rows resolved per step. Bounds the float32 staging buffer to
# ``_ROWS_PER_DEQUANT_STEP * in_features * 4`` bytes regardless of layer size.
_ROWS_PER_DEQUANT_STEP = 1024


def _swap_block_axes(tensor: torch.Tensor) -> torch.Tensor:
    """Swap the last two axes of tensor into canonical contiguous strides.

    ``.contiguous()`` keeps the strides a size-one axis arrived with, and the
    G-B kernels read the weight/scale orientation from them: a reduction dim
    holding a single 128-element block (a typical TP shard) would otherwise read
    as a transposed scale. Copying into a fresh tensor fixes the strides.
    """
    transposed = tensor.transpose(-1, -2)
    swapped = torch.empty(transposed.shape, dtype=tensor.dtype, device=tensor.device)
    swapped.copy_(transposed)
    return swapped


def _block_quantize_activation(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize activations to FP8 per 128x128 block along the reduction dim.

    Returns the ``float8_e4m3fn`` activation and its ``float32`` per-token,
    per-block scale, the exact pair the G-B kernels consume.
    """
    return torch_npu.npu_dynamic_block_quant(x, dst_type=BLOCK_FP8_WEIGHT_DTYPE)


def resolve_block_scales(
    weight: torch.Tensor,
    scale_inv: torch.Tensor,
    block_n: int,
    block_k: int,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """Expand a block-wise FP8 matrix into a dense ``out_dtype`` matrix.

    ``weight`` is ``[out_features, in_features]`` and ``scale_inv`` carries one
    scale per ``block_n x block_k`` tile. The scales stay in float32 during the
    multiply so a block's shared scale is not itself rounded to bfloat16.
    """
    if weight.dim() != 2:
        raise ValueError(f"Expected a 2D block-quantized weight, got shape {tuple(weight.shape)}.")

    out_features, in_features = weight.shape
    expected_scale_shape = (cdiv(out_features, block_n), cdiv(in_features, block_k))
    if tuple(scale_inv.shape) != expected_scale_shape:
        raise ValueError(
            f"Block-wise FP8 weight of shape {tuple(weight.shape)} with block size "
            f"({block_n}, {block_k}) needs a scale of shape {expected_scale_shape}, but the "
            f"checkpoint provided {tuple(scale_inv.shape)}. The weight and its "
            "`weight_scale_inv` are mismatched or unpaired."
        )

    resolved = torch.empty((out_features, in_features), dtype=out_dtype, device=weight.device)
    rows_per_step = max(block_n, _ROWS_PER_DEQUANT_STEP // block_n * block_n)
    for row_start in range(0, out_features, rows_per_step):
        row_end = min(row_start + rows_per_step, out_features)
        # Keep dtype conversion on the weight device so a CPU-resident scale
        # cannot multiply an NPU weight. Same-device `.to()` is a no-op.
        row_scales = scale_inv[row_start // block_n : cdiv(row_end, block_n)].to(
            device=weight.device, dtype=torch.float32
        )
        row_scales = row_scales.repeat_interleave(block_n, dim=0)[: row_end - row_start]
        row_scales = row_scales.repeat_interleave(block_k, dim=1)[:, :in_features]
        resolved[row_start:row_end] = weight[row_start:row_end].to(torch.float32) * row_scales
    return resolved


def _mx_quantize(resolved: torch.Tensor, scale_alg: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Re-quantize a dense matrix to MXFP8, returning the weight and uint8 scale.

    ``npu_dynamic_mx_quant`` emits the scale with its group axis already split into
    ``[..., num_groups // 2, 2]`` pairs, whereas both MXFP8 schemes consume the loader
    layout ``[..., num_groups]`` and pair the groups up themselves. Collapse the trailing
    axis so a requantized checkpoint reaches them in the same layout as a native one.
    """
    quantized, scale = torch_npu.npu_dynamic_mx_quant(
        resolved,
        dst_type=BLOCK_FP8_WEIGHT_DTYPE,
        scale_alg=scale_alg,
    )
    return quantized, scale.flatten(-2)


def _supports_mx_regroup(in_features: int, group_size: int) -> bool:
    return in_features % group_size == 0


def _is_absorbed_by_attention(layer: torch.nn.Module) -> bool:
    """True for projections that MLA/SFA folds into its own weights.

    ``kv_b_proj`` is split into ``W_UK``/``W_UV`` and the layer is disposed right
    after, so it never runs a matmul. Quantizing it buys nothing and destroys the
    dense matrix the attention backend has to absorb.
    """
    return getattr(layer, "prefix", "").endswith("kv_b_proj")


def _get_layer_tp_size(layer: torch.nn.Module) -> int:
    tp_size = getattr(layer, "tp_size", None)
    return get_tensor_model_parallel_world_size() if tp_size is None else tp_size


@register_scheme(FP8_METHOD, "linear")
class AscendFp8BlockLinearMethod(AscendLinearScheme):
    """Linear method for native block-wise FP8 checkpoints.

    Weights arrive as ``float8_e4m3fn`` plus a float32 ``weight_scale_inv`` with
    one entry per ``block_n x block_k`` tile. ``process_weights_after_loading``
    resolves those tiles and re-quantizes them to MXFP8. Building the scheme on a
    device without an FP8 matmul fails, so the dense path is reserved for matrices
    the attention backend absorbs or for shapes MXFP8 cannot regroup.
    """

    supports_weight_switch = True

    # Dense post-processing keeps weight in [output, input] layout.
    weight_switch_gather_specs = (WeightSwitchGatherSpec("weight", gather_dim=1),)
    weight_switch_output_gather_specs = (WeightSwitchGatherSpec("weight"),)

    def _get_weight_switch_specs(
        self,
        layer: torch.nn.Module,
        config: WeightSwitchConfig,
    ) -> tuple[
        tuple[WeightSwitchGatherSpec, ...],
        tuple[WeightSwitchRepeatSpec, ...],
        str,
    ]:
        gather_specs, repeat_specs, shard_axis = super()._get_weight_switch_specs(layer, config)
        if not self.use_mxfp8:
            return gather_specs, repeat_specs, shard_axis
        if shard_axis == "input":
            return (
                self.mxfp8_method.weight_switch_gather_specs,
                self.mxfp8_method.weight_switch_repeat_specs,
                shard_axis,
            )
        return (
            self.mxfp8_method.weight_switch_output_gather_specs,
            self.mxfp8_method.weight_switch_output_repeat_specs,
            shard_axis,
        )

    def __init__(self, weight_block_size: tuple[int, int]):
        self.block_n, self.block_k = weight_block_size
        self.model_dtype = get_current_vllm_config().model_config.dtype
        self.mxfp8_method = AscendW8A8MXFP8DynamicLinearMethod()
        self.use_mxfp8 = True

    def validate_weight_shapes(
        self,
        layer: torch.nn.Module,
        input_size: int,
        output_size: int,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
    ) -> None:
        # Upstream explicitly permits some packed projections with padded
        # block scales. Preserve the loader's policy for those layers.
        if getattr(layer, "allow_fp8_block_shape_mismatch", False):
            return

        tp_size = _get_layer_tp_size(layer)
        if tp_size > 1 and input_size == input_size_per_partition * tp_size and input_size_per_partition % self.block_k:
            raise ValueError(
                f"Weight input_size_per_partition = {input_size_per_partition} is not divisible by "
                f"weight quantization block_k = {self.block_k}."
            )

        # Internal packed boundaries must align even for replicated weights.
        # A replicated final projection may own a partial tile at its tail.
        # QKV layers can replicate K/V independently of Q, so the packed
        # output size alone cannot establish whether the last slice is sharded.
        logical_sizes = getattr(layer, "output_sizes", [output_size])
        last_size = output_partition_sizes[-1]
        kv_replicas = getattr(layer, "num_kv_head_replicas", 1)
        last_is_sharded = tp_size > kv_replicas and logical_sizes[-1] == last_size * tp_size
        sizes_to_check = output_partition_sizes if last_is_sharded else output_partition_sizes[:-1]
        for size in sizes_to_check:
            if size % self.block_n:
                raise ValueError(
                    f"Weight output_partition_size = {size} is not divisible by "
                    f"weight quantization block_n = {self.block_n}."
                )

    def get_weight(self, input_size: int, output_size: int, params_dtype: torch.dtype) -> dict[str, Any]:
        return {"weight": torch.empty(output_size, input_size, dtype=BLOCK_FP8_WEIGHT_DTYPE)}

    def get_pergroup_param(
        self, input_size: int, output_size: int, params_dtype: torch.dtype, layer_type: str | None = None
    ) -> dict[str, Any]:
        return {
            "weight_scale_inv": torch.empty(
                cdiv(output_size, self.block_n), cdiv(input_size, self.block_k), dtype=torch.float32
            ),
            # Loaders offset shards in weight elements and must convert to scale
            # entries. Rounding that conversion up matters: a shard whose height
            # is not a multiple of block_n still owns the tile covering its tail.
            "_block_quant_scale": (self.block_n, self.block_k),
        }

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        resolved = resolve_block_scales(
            layer.weight.data,
            layer.weight_scale_inv.data,
            self.block_n,
            self.block_k,
            self.model_dtype,
        )
        del layer.weight_scale_inv

        if _is_absorbed_by_attention(layer):
            # The attention backend splits this layer and disposes of it, so
            # apply() is never reached. The matrix it absorbs is dense, so neither
            # the matmul nor the weight switch may describe this layer as MXFP8;
            # and since each layer owns its own scheme instance, this stays local.
            self.use_mxfp8 = False
            layer.weight = torch.nn.Parameter(maybe_trans_nz(resolved), requires_grad=False)
            return

        self.use_mxfp8 = _supports_mx_regroup(resolved.shape[1], self.mxfp8_method.group_size)
        if not self.use_mxfp8:
            logger.warning_once(
                "Reduction dim %d of %s is not a multiple of the MXFP8 group size %d; serving this "
                "layer in %s instead of MXFP8.",
                resolved.shape[1],
                getattr(layer, "prefix", "the linear layer"),
                self.mxfp8_method.group_size,
                self.model_dtype,
            )
            layer.weight = torch.nn.Parameter(maybe_trans_nz(resolved), requires_grad=False)
            return

        quantized, mx_scale = _mx_quantize(resolved, self.mxfp8_method.dynamic_mx_quant_scale_alg)
        # mx_scale_pairs: CANN 9.1 npu_dynamic_mx_quant packs two E8M0 scales
        # per entry on a 2D weight, so the native result is [out, in // 64, 2]
        # while the MXFP8 method unpacks [out, in // 32]. view() keeps the
        # bytes; reshape() restores the layout the method expects.
        mx_scale = mx_scale.view(torch.uint8).reshape(resolved.shape[0], -1)
        layer.weight = torch.nn.Parameter(quantized, requires_grad=False)
        layer.weight_scale = torch.nn.Parameter(mx_scale, requires_grad=False)
        self.mxfp8_method.process_weights_after_loading(layer)

    def prepare_input_for_overlap(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
    ) -> PreparedLinearInput | None:
        if not self.use_mxfp8:
            return None
        return self.mxfp8_method.prepare_input_for_overlap(layer, x)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor | PreparedLinearInput,
        bias: torch.Tensor | None = None,
        tp_rank: int | None = 0,
    ) -> torch.Tensor:
        if self.use_mxfp8:
            return self.mxfp8_method.apply(layer, x, bias, tp_rank)
        return torch.ops.vllm.unquantized_gemm(x, layer.weight, bias)


@register_scheme(FP8_METHOD, "moe")
class AscendFp8BlockFusedMoEMethod(AscendMoEScheme):
    """FusedMoE method for native block-wise FP8 checkpoints.

    Experts are resolved one at a time so the staging buffer stays proportional
    to a single expert rather than to the whole expert stack.
    """

    quant_type: QuantType = QuantType.NONE

    def __init__(self, weight_block_size: tuple[int, int], moe_config):
        self.block_n, self.block_k = weight_block_size
        # AscendFusedMoEMethod tags a scale as group-wise when the scheme reports
        # a group size, which is how the upstream loader learns to narrow
        # `*_weight_scale_inv` per expert shard.
        self.group_size = self.block_k
        self.model_dtype = get_current_vllm_config().model_config.dtype
        self.moe_config = moe_config
        self.mxfp8_method = AscendW8A8MXFP8DynamicFusedMoEMethod()
        # MoE MXFP8 has group_size only. Snapshot the Linear helper here so
        # requantize does not call get_current_vllm_config() at load time.
        self._mx_scale_alg = get_dynamic_mx_quant_scale_alg(get_current_vllm_config())
        self.use_mxfp8 = True
        self._bf16_method: Any = None

    def validate_weight_shapes(self, layer: torch.nn.Module, intermediate_size_per_partition: int) -> None:
        if intermediate_size_per_partition % self.block_n:
            raise ValueError(
                f"The per-partition intermediate size = {intermediate_size_per_partition} is not "
                f"divisible by weight quantization block_n = {self.block_n}."
            )
        if _get_layer_tp_size(layer) > 1 and intermediate_size_per_partition % self.block_k:
            raise ValueError(
                f"The per-partition intermediate size = {intermediate_size_per_partition} is not "
                f"divisible by weight quantization block_k = {self.block_k}."
            )

    @property
    def bf16_method(self):
        """The unquantized MoE method that serves resolved bfloat16 experts."""
        if self._bf16_method is None:
            # Delayed import: routed_experts pulls in the MoE communication stack.
            from vllm_ascend.ops.fused_moe.routed_experts import AscendUnquantizedFusedMoEMethod

            self._bf16_method = AscendUnquantizedFusedMoEMethod(self.moe_config)
        return self._bf16_method

    def get_weight(
        self, num_experts: int, intermediate_size_per_partition: int, hidden_sizes: int, params_dtype: torch.dtype
    ) -> dict[str, Any]:
        return {
            "w13_weight": torch.empty(
                num_experts, 2 * intermediate_size_per_partition, hidden_sizes, dtype=BLOCK_FP8_WEIGHT_DTYPE
            ),
            "w2_weight": torch.empty(
                num_experts, hidden_sizes, intermediate_size_per_partition, dtype=BLOCK_FP8_WEIGHT_DTYPE
            ),
        }

    def get_dynamic_quant_param(
        self, num_experts: int, intermediate_size_per_partition: int, hidden_sizes: int, params_dtype: torch.dtype
    ) -> dict[str, Any]:
        return {
            "w13_weight_scale_inv": torch.empty(
                num_experts,
                cdiv(2 * intermediate_size_per_partition, self.block_n),
                cdiv(hidden_sizes, self.block_k),
                dtype=torch.float32,
            ),
            "w2_weight_scale_inv": torch.empty(
                num_experts,
                cdiv(hidden_sizes, self.block_n),
                cdiv(intermediate_size_per_partition, self.block_k),
                dtype=torch.float32,
            ),
        }

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        weight_names = ("w13_weight", "w2_weight")
        self.use_mxfp8 = all(
            _supports_mx_regroup(getattr(layer, name).shape[-1], self.mxfp8_method.group_size) for name in weight_names
        )
        if not self.use_mxfp8:
            logger.warning_once(
                "Expert reduction dims of %s are not all multiples of the MXFP8 group size %d; "
                "serving this layer in %s instead of MXFP8.",
                getattr(layer, "prefix", "the fused MoE layer"),
                self.mxfp8_method.group_size,
                self.model_dtype,
            )

        for weight_name in weight_names:
            scale_name = f"{weight_name}_scale_inv"
            if self.use_mxfp8:
                mx_scale = self._requantize_experts_inplace(layer, weight_name, scale_name)
                layer.register_parameter(f"{weight_name}_scale", torch.nn.Parameter(mx_scale, requires_grad=False))
            else:
                replace_parameter(layer, weight_name, self._resolve_experts(layer, weight_name, scale_name))
            delattr(layer, scale_name)

        if self.use_mxfp8:
            self.quant_type = self.mxfp8_method.quant_type
            self.mxfp8_method.process_weights_after_loading(layer)
        else:
            self.bf16_method.process_weights_after_loading(layer)

    def _resolve_experts(self, layer: torch.nn.Module, weight_name: str, scale_name: str) -> torch.Tensor:
        """Expand every expert of one MoE weight into the model dtype."""
        weight = getattr(layer, weight_name).data
        scale_inv = getattr(layer, scale_name).data
        resolved = torch.empty(weight.shape, dtype=self.model_dtype, device=weight.device)
        for expert in range(weight.shape[0]):
            resolved[expert] = resolve_block_scales(
                weight[expert], scale_inv[expert], self.block_n, self.block_k, self.model_dtype
            )
        return resolved

    def _requantize_experts_inplace(self, layer: torch.nn.Module, weight_name: str, scale_name: str) -> torch.Tensor:
        """Rewrite every expert of one MoE weight from block-wise FP8 to MXFP8.

        The FP8 weight buffer is reused, so peak memory is one expert of staging
        rather than a dense copy of the whole expert stack.
        """
        scale_alg = self._mx_scale_alg
        weight = getattr(layer, weight_name).data
        scale_inv = getattr(layer, scale_name).data
        num_experts, _, in_features = weight.shape
        group_size = self.mxfp8_method.group_size
        mx_scale = torch.empty(
            (num_experts, weight.shape[1], in_features // group_size),
            dtype=torch.uint8,
            device=weight.device,
        )
        for expert in range(num_experts):
            resolved = resolve_block_scales(
                weight[expert], scale_inv[expert], self.block_n, self.block_k, self.model_dtype
            )
            quantized, expert_scale = _mx_quantize(resolved, scale_alg)
            weight[expert].copy_(quantized)
            mx_scale[expert].copy_(expert_scale.view(torch.uint8).reshape(mx_scale[expert].shape))
        return mx_scale

    def get_eplb_weight_views(self, layer: torch.nn.Module) -> list[torch.Tensor]:
        active = self.mxfp8_method if self.use_mxfp8 else self.bf16_method
        return active.get_eplb_weight_views(layer)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts: Any | None,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        active = self.mxfp8_method if self.use_mxfp8 else self.bf16_method
        return active.apply(
            layer=layer,
            x=x,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            shared_experts=shared_experts,
            shared_experts_input=shared_experts_input,
        )


@register_scheme(FP8_METHOD, "native_linear")
class AscendNativeFp8BlockLinearMethod(AscendLinearScheme):
    """Linear method that runs block-wise FP8 weights on the G-B matmul.

    The ``fp8_mode: "native"`` counterpart of
    :class:`AscendFp8BlockLinearMethod`: nothing is requantized, the
    checkpoint's own float8 weight and 128x128 block scales are re-laid-out for
    the kernel, and the activations are block-quantized at run time.

    NOTE: the parameter builders mirror the fallback scheme because both read
    the same checkpoint. They must stay in sync until the two schemes share a
    common base.
    """

    # A transposed weight no longer follows the [output, input] axis that
    # weight switching rebuilds.
    supports_weight_switch = False

    def __init__(self, weight_block_size: tuple[int, int]):
        self.block_n, self.block_k = weight_block_size
        self.model_dtype = get_current_vllm_config().model_config.dtype

    def get_weight(self, input_size: int, output_size: int, params_dtype: torch.dtype) -> dict[str, Any]:
        return {"weight": torch.empty(output_size, input_size, dtype=BLOCK_FP8_WEIGHT_DTYPE)}

    def get_pergroup_param(
        self, input_size: int, output_size: int, params_dtype: torch.dtype, layer_type: str | None = None
    ) -> dict[str, Any]:
        return {
            "weight_scale_inv": torch.empty(
                cdiv(output_size, self.block_n), cdiv(input_size, self.block_k), dtype=torch.float32
            ),
            # Loaders convert a shard's weight extent into scale entries and
            # round up: a sub-tile shard still owns the tile over its tail.
            "_block_quant_scale": (self.block_n, self.block_k),
        }

    def validate_weight_shapes(
        self,
        layer: torch.nn.Module,
        input_size: int,
        output_size: int,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
    ) -> None:
        """Reject layouts whose per-rank tiles the G-B kernels cannot describe.

        A block scale cannot describe a rank that owns only part of a
        ``[128, 128]`` tile, and this path has no dense matrix to fall back to.
        Sharded dimensions are therefore checked per rank, replicated ones
        against the full extent the checkpoint's scale covers.
        """
        # Upstream lets fused projections with sub-matrices that do not span
        # whole tiles load with padded block scales, but G-B addresses one scale
        # per tile of real rows and would silently misread those layers.
        if getattr(layer, "allow_fp8_block_shape_mismatch", False):
            raise ValueError(
                f"Native block-wise FP8 cannot serve {getattr(layer, 'prefix', 'linear')}: its fused output "
                f"partitions {output_partition_sizes} do not span whole {self.block_n}-wide blocks, so the "
                "checkpoint's per-sub-matrix scales cannot be mapped onto the G-B kernel. Set fp8_mode to "
                '"mxfp8" to serve this checkpoint through the dequantize-and-requantize path.'
            )

        tp_size = _get_layer_tp_size(layer)
        logical_sizes = getattr(layer, "output_sizes", [output_size])
        if tp_size > 1 and input_size == input_size_per_partition * tp_size:
            reduction_dim = input_size_per_partition
        else:
            reduction_dim = input_size
        if reduction_dim % self.block_k:
            raise ValueError(
                f"Native block-wise FP8 cannot describe a {self.block_k}-wide tile on a rank that owns "
                f"only {reduction_dim} elements along the reduction dim of "
                f"{getattr(layer, 'prefix', 'linear')}. Use a smaller --tensor-parallel-size, or "
                f'fp8_mode: "mxfp8" to requantize instead.'
            )
        for index, packed_size in enumerate(output_partition_sizes):
            logical_size = logical_sizes[index] if index < len(logical_sizes) else packed_size
            sharded = tp_size > 1 and logical_size == packed_size * tp_size
            output_dim = packed_size if sharded else logical_size
            if output_dim % self.block_n:
                raise ValueError(
                    f"Native block-wise FP8 cannot describe a {self.block_n}-wide tile on a rank that owns "
                    f"only {output_dim} elements along the output dim of "
                    f"{getattr(layer, 'prefix', 'linear')}. Use a smaller --tensor-parallel-size, or "
                    f'fp8_mode: "mxfp8" to requantize instead.'
                )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # kv_b_proj is split into W_UK/W_UV by the attention backend instead of
        # matmul'ing, so it always stays dense in the model dtype:
        # resolve_block_scales folds the block scales into the weight, which is
        # why no scale survives this branch and why this single layer doubles.
        if _is_absorbed_by_attention(layer):
            resolved = resolve_block_scales(
                layer.weight.data,
                layer.weight_scale_inv.data,
                self.block_n,
                self.block_k,
                self.model_dtype,
            )
            delattr(layer, "weight_scale_inv")
            layer.weight = torch.nn.Parameter(maybe_trans_nz(resolved), requires_grad=False)
            return

        # G-B ``npu_quant_matmul`` consumes a [in, out] weight with an
        # [in / block, out / block] scale, so transpose both. The weight is
        # stored in NZ when ``weight_nz_mode`` allows it; the scale stays ND.
        layer.weight.data = maybe_trans_nz(_swap_block_axes(layer.weight.data))
        layer.register_parameter(
            "weight_scale",
            torch.nn.Parameter(_swap_block_axes(layer.weight_scale_inv.data), requires_grad=False),
        )
        delattr(layer, "weight_scale_inv")

    def prepare_input_for_overlap(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
    ) -> PreparedLinearInput | None:
        if x.dim() != 2:
            return None
        quantized_x, block_scale = _block_quantize_activation(x)
        return PreparedLinearInput(quantized_x, block_scale, x.dtype)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor | PreparedLinearInput,
        bias: torch.Tensor | None = None,
        tp_rank: int | None = 0,
    ) -> torch.Tensor:
        if isinstance(x, PreparedLinearInput):
            assert x.scale is not None
            quantized_x, block_scale, output_dtype = x.quantized, x.scale, x.output_dtype
            leading_shape = quantized_x.shape[:-1]
        else:
            leading_shape = x.shape[:-1]
            output_dtype = x.dtype
            quantized_x, block_scale = _block_quantize_activation(x.reshape(-1, x.shape[-1]))

        output = torch_npu.npu_quant_matmul(
            quantized_x,
            layer.weight,
            layer.weight_scale,
            pertoken_scale=block_scale,
            group_sizes=_NATIVE_BLOCK_GROUP_SIZES,
            output_dtype=output_dtype,
        )
        # G-B has no bias epilogue on this kernel, so the bias is applied here.
        if bias is not None:
            output = output + bias
        return output.reshape(*leading_shape, -1)


@register_scheme(FP8_METHOD, "native_moe")
class AscendNativeFp8BlockFusedMoEMethod(AscendMoEScheme):
    """FusedMoE method that runs block-wise FP8 experts on the grouped matmul.

    The ``fp8_mode: "native"`` counterpart of
    :class:`AscendFp8BlockFusedMoEMethod`: the experts stay exactly the values
    the checkpoint shipped.

    NOTE: like the native linear scheme, the parameter builders mirror the
    fallback scheme and must stay in sync until both share a common base.
    """

    quant_type: QuantType = QuantType.W8A8FP8_BLOCK
    act_quant_type: torch.dtype = BLOCK_FP8_WEIGHT_DTYPE

    def __init__(self, weight_block_size: tuple[int, int], moe_config):
        self.block_n, self.block_k = weight_block_size
        # Reporting a group size is how the loader learns to shard the
        # ``*_weight_scale_inv`` tensors along the reduction dim.
        self.group_size = self.block_k
        self.model_dtype = get_current_vllm_config().model_config.dtype
        self.moe_config = moe_config

    def validate_weight_shapes(self, layer: torch.nn.Module, intermediate_size_per_partition: int) -> None:
        if intermediate_size_per_partition % self.block_n:
            raise ValueError(
                f"The per-partition intermediate size = {intermediate_size_per_partition} is not "
                f"divisible by weight quantization block_n = {self.block_n}."
            )
        if _get_layer_tp_size(layer) > 1 and intermediate_size_per_partition % self.block_k:
            raise ValueError(
                f"The per-partition intermediate size = {intermediate_size_per_partition} is not "
                f"divisible by weight quantization block_k = {self.block_k}."
            )

    def get_weight(
        self, num_experts: int, intermediate_size_per_partition: int, hidden_sizes: int, params_dtype: torch.dtype
    ) -> dict[str, Any]:
        return {
            "w13_weight": torch.empty(
                num_experts, 2 * intermediate_size_per_partition, hidden_sizes, dtype=BLOCK_FP8_WEIGHT_DTYPE
            ),
            "w2_weight": torch.empty(
                num_experts, hidden_sizes, intermediate_size_per_partition, dtype=BLOCK_FP8_WEIGHT_DTYPE
            ),
        }

    def get_dynamic_quant_param(
        self, num_experts: int, intermediate_size_per_partition: int, hidden_sizes: int, params_dtype: torch.dtype
    ) -> dict[str, Any]:
        return {
            "w13_weight_scale_inv": torch.empty(
                num_experts,
                cdiv(2 * intermediate_size_per_partition, self.block_n),
                cdiv(hidden_sizes, self.block_k),
                dtype=torch.float32,
            ),
            "w2_weight_scale_inv": torch.empty(
                num_experts,
                cdiv(hidden_sizes, self.block_n),
                cdiv(intermediate_size_per_partition, self.block_k),
                dtype=torch.float32,
            ),
        }

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Keep the checkpoint's float8 experts and lay them out for G-B.

        ``npu_grouped_matmul`` consumes [E, K, N] weights with a
        [E, K / block, N / block] scale, the checkpoint stores [E, N, K] with a
        [E, N / block, K / block] scale, so both are transposed. No data is
        requantized, and the grouped matmul takes no NZ weights alongside the
        block scales, so they stay ND.
        """
        for weight_name in ("w13_weight", "w2_weight"):
            scale_name = f"{weight_name}_scale_inv"
            weight = getattr(layer, weight_name)
            scale = getattr(layer, scale_name).data
            weight.data = _swap_block_axes(weight.data)
            layer.register_parameter(
                f"{weight_name}_scale",
                torch.nn.Parameter(_swap_block_axes(scale), requires_grad=False),
            )
            delattr(layer, scale_name)

    def apply_gmm1(self, mlp_compute_input: MoEMlpComputeInput) -> torch.Tensor:
        # Block-wise FP8 is not a dispatch-quantized type, so the tokens arrive in
        # the model dtype; MC2 still reports a scale for its unquantized dispatch,
        # so quantize here rather than trusting that scale.
        hidden_states, pertoken_scale = _block_quantize_activation(mlp_compute_input.hidden_states)
        layer = mlp_compute_input.layer
        assert layer is not None
        output = torch_npu.npu_grouped_matmul(
            x=[hidden_states],
            weight=[layer.w13_weight],
            scale=[layer.w13_weight_scale],
            bias=None,
            per_token_scale=[pertoken_scale],
            split_item=2,
            group_type=0,
            group_list=mlp_compute_input.group_list,
            group_list_type=mlp_compute_input.group_list_type,
            output_dtype=self.model_dtype,
        )[0]
        dispose_tensor(mlp_compute_input.hidden_states)
        return output

    def apply_act_quant(
        self, mlp_compute_input: MoEMlpComputeInput, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return _block_quantize_activation(hidden_states)

    def apply_gmm2(
        self, mlp_compute_input: MoEMlpComputeInput, hidden_states: torch.Tensor, act_out_scale: torch.Tensor
    ) -> torch.Tensor:
        layer = mlp_compute_input.layer
        assert layer is not None
        return torch_npu.npu_grouped_matmul(
            x=[hidden_states],
            weight=[layer.w2_weight],
            scale=[layer.w2_weight_scale],
            bias=None,
            per_token_scale=[act_out_scale],
            split_item=2,
            group_list_type=mlp_compute_input.group_list_type,
            group_type=0,
            group_list=mlp_compute_input.group_list,
            output_dtype=self.model_dtype,
        )[0]

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts: Any | None,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        if topk_weights is None or topk_ids is None:
            raise RuntimeError("topk_weights and topk_ids must be set before fused MoE execution.")
        return _EXTRA_CTX.moe_comm_method.fused_experts(
            fused_experts_input=build_fused_experts_input(
                hidden_states=x,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                layer=layer,
                quant_type=self.quant_type,
                # G-B grouped matmul only accepts one stacked weight tensor.
                dynamic_eplb=False,
                expert_map=layer.ascend_expert_map,
                global_redundant_expert_num=layer.global_redundant_expert_num,
                mc2_mask=layer.ascend_mc2_mask,
                apply_router_weight_on_input=layer.apply_router_weight_on_input,
                pertoken_scale=layer.ascend_pertoken_scale,
                activation=layer.activation,
            ),
            quant_method=self,
        )
