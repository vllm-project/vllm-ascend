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
one ``float32`` scale per ``weight_block_size`` tile. No Ascend matmul consumes
that layout directly, so the tiles are resolved once, after loading, and the
result is handed to a layout the hardware does support:

* On Ascend 950 the resolved matrix is re-quantized to MXFP8 (one E8M0 scale per
  32 elements along the reduction dim). The native ``npu_quant_matmul`` path then
  runs and weights stay at one byte per element, which is what makes a 27B FP8
  checkpoint fit on a single card.
* On every other Ascend generation the resolved matrix is kept in the model dtype
  and served by the ordinary unquantized GEMM. Correct, but the footprint doubles.

Both paths read the same checkpoint, so no per-model patch or offline
re-quantization step is needed.
"""

from typing import Any

import torch
import torch_npu
from vllm.config import get_current_vllm_config
from vllm.logger import logger
from vllm.model_executor.utils import replace_parameter
from vllm.utils.math_utils import cdiv

from vllm_ascend.quantization.utils import get_dynamic_mx_quant_scale_alg
from vllm_ascend.utils import FP8_METHOD, is_950, maybe_trans_nz

from ..base import (
    AscendLinearScheme,
    AscendMoEScheme,
    QuantType,
    TPWeightGatherSpec,
    TPWeightRepeatSpec,
)
from ..registry import register_scheme
from .w8a8_mxfp8 import AscendW8A8MXFP8DynamicFusedMoEMethod, AscendW8A8MXFP8DynamicLinearMethod

BLOCK_FP8_WEIGHT_DTYPE = torch.float8_e4m3fn

# Output rows resolved per step. Bounds the float32 staging buffer to
# ``_ROWS_PER_DEQUANT_STEP * in_features * 4`` bytes regardless of layer size.
_ROWS_PER_DEQUANT_STEP = 1024


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


@register_scheme(FP8_METHOD, "linear")
class AscendFp8BlockLinearMethod(AscendLinearScheme):
    """Linear method for native block-wise FP8 checkpoints.

    Weights arrive as ``float8_e4m3fn`` plus a float32 ``weight_scale_inv`` with
    one entry per ``block_n x block_k`` tile. ``process_weights_after_loading``
    resolves those tiles and then either re-quantizes to MXFP8 (Ascend 950) or
    keeps the model dtype (everything else).
    """

    supports_tp_weight_switch = True

    def get_tp_weight_switch_specs(
        self,
        *,
        input_sharded: bool,
    ) -> tuple[tuple[TPWeightGatherSpec, ...], tuple[TPWeightRepeatSpec, ...]]:
        if self.mxfp8_method is not None:
            if input_sharded:
                return self.mxfp8_method.tp_weight_gather_specs, self.mxfp8_method.tp_weight_repeat_specs
            return self.mxfp8_method.tp_weight_output_gather_specs, self.mxfp8_method.tp_weight_output_repeat_specs
        gather_dim = 1 if input_sharded else 0
        return (TPWeightGatherSpec("weight", gather_dim=gather_dim),), ()

    def __init__(self, weight_block_size: tuple[int, int]):
        self.block_n, self.block_k = weight_block_size
        self.model_dtype = get_current_vllm_config().model_config.dtype
        self.mxfp8_method = AscendW8A8MXFP8DynamicLinearMethod() if is_950() else None

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
            # Decided locally rather than on the scheme: the attention backend
            # splits this layer and disposes of it, so apply() is never reached
            # and nothing about this layer should speak for any other.
            layer.weight = torch.nn.Parameter(maybe_trans_nz(resolved), requires_grad=False)
            return

        if self.mxfp8_method is not None and not _supports_mx_regroup(resolved.shape[1], self.mxfp8_method.group_size):
            logger.warning_once(
                "Reduction dim %d of %s is not a multiple of the MXFP8 group size %d; serving this "
                "layer in %s instead of MXFP8.",
                resolved.shape[1],
                getattr(layer, "prefix", "the linear layer"),
                self.mxfp8_method.group_size,
                self.model_dtype,
            )
            self.mxfp8_method = None

        if self.mxfp8_method is None:
            layer.weight = torch.nn.Parameter(maybe_trans_nz(resolved), requires_grad=False)
            return

        quantized, mx_scale = _mx_quantize(resolved, self.mxfp8_method.dynamic_mx_quant_scale_alg)
        layer.weight = torch.nn.Parameter(quantized, requires_grad=False)
        layer.weight_scale = torch.nn.Parameter(mx_scale, requires_grad=False)
        self.mxfp8_method.process_weights_after_loading(layer)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
        tp_rank: int | None = 0,
    ) -> torch.Tensor:
        if self.mxfp8_method is not None:
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
        self.mxfp8_method = AscendW8A8MXFP8DynamicFusedMoEMethod() if is_950() else None
        # MoE MXFP8 has group_size only. Snapshot the Linear helper here so
        # requantize does not call get_current_vllm_config() at load time.
        self._mx_scale_alg = (
            get_dynamic_mx_quant_scale_alg(get_current_vllm_config()) if self.mxfp8_method is not None else 0
        )
        self._bf16_method: Any = None

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
        if self.mxfp8_method is not None and not all(
            _supports_mx_regroup(getattr(layer, name).shape[-1], self.mxfp8_method.group_size) for name in weight_names
        ):
            logger.warning_once(
                "Expert reduction dims of %s are not all multiples of the MXFP8 group size %d; "
                "serving this layer in %s instead of MXFP8.",
                getattr(layer, "prefix", "the fused MoE layer"),
                self.mxfp8_method.group_size,
                self.model_dtype,
            )
            self.mxfp8_method = None

        for weight_name in weight_names:
            scale_name = f"{weight_name}_scale_inv"
            if self.mxfp8_method is None:
                replace_parameter(layer, weight_name, self._resolve_experts(layer, weight_name, scale_name))
            else:
                mx_scale = self._requantize_experts_inplace(layer, weight_name, scale_name)
                layer.register_parameter(f"{weight_name}_scale", torch.nn.Parameter(mx_scale, requires_grad=False))
            delattr(layer, scale_name)

        if self.mxfp8_method is None:
            self.bf16_method.process_weights_after_loading(layer)
        else:
            self.quant_type = self.mxfp8_method.quant_type
            self.mxfp8_method.process_weights_after_loading(layer)

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
        mxfp8_method = self.mxfp8_method
        if mxfp8_method is None:
            raise RuntimeError("Block-wise FP8 expert requantize requires the MXFP8 method.")
        scale_alg = self._mx_scale_alg
        weight = getattr(layer, weight_name).data
        scale_inv = getattr(layer, scale_name).data
        num_experts, _, in_features = weight.shape
        mx_scale = torch.empty(
            (num_experts, weight.shape[1], in_features // mxfp8_method.group_size),
            dtype=torch.uint8,
            device=weight.device,
        )
        for expert in range(num_experts):
            resolved = resolve_block_scales(
                weight[expert], scale_inv[expert], self.block_n, self.block_k, self.model_dtype
            )
            quantized, expert_scale = _mx_quantize(resolved, scale_alg)
            weight[expert].copy_(quantized)
            mx_scale[expert].copy_(expert_scale)
        return mx_scale

    def get_eplb_weight_views(self, layer: torch.nn.Module) -> list[torch.Tensor]:
        active = self.mxfp8_method if self.mxfp8_method is not None else self.bf16_method
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
        active = self.mxfp8_method if self.mxfp8_method is not None else self.bf16_method
        return active.apply(
            layer=layer,
            x=x,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            shared_experts=shared_experts,
            shared_experts_input=shared_experts_input,
        )
