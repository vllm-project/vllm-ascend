#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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

from typing import Any

import numpy as np
import torch
import torch_npu
from vllm.config import get_current_vllm_config
from vllm.distributed import get_tensor_model_parallel_world_size

from vllm_ascend.ascend_config import get_ascend_config, is_mega_moe_supported
from vllm_ascend.ascend_forward_context import _EXTRA_CTX, MoECommType
from vllm_ascend.distributed.parallel_state import get_mc2_group
from vllm_ascend.ops.fused_moe.dataclass.fused_experts import build_fused_experts_input
from vllm_ascend.ops.fused_moe.routed_experts import AscendRoutedExperts  # noqa: F401
from vllm_ascend.utils import ASCEND_QUANTIZATION_METHOD, COMPRESSED_TENSORS_METHOD, maybe_trans_nz

from ..base import AscendLinearScheme, AscendMoEScheme, QuantType
from ..registry import register_scheme


class AscendKimiK3W4A8DynamicLinearMethod(AscendLinearScheme):
    """Kimi K3 per-channel W4A8 shared-expert projection.

    Generic W4A8 linear support was removed because it was deprecated. Kimi K3
    still stores its shared-expert projections as ModelSlim per-channel W4A8
    linears, so keep the required loader and grouped-matmul execution scoped to
    those projections instead of registering a global linear scheme.
    """

    is_kimi_k3_shared_expert_w4a8 = True

    def __init__(self) -> None:
        quant_description = get_current_vllm_config().quant_config.quant_description
        group_size = quant_description.get("group_size", 0)
        if group_size != 0:
            raise ValueError("Kimi K3 W4A8 shared experts require per-channel weights (group_size=0).")
        quant_version = quant_description.get("version", "0")
        if quant_version != "1.0.0":
            raise ValueError("Kimi K3 W4A8 shared experts require quantization version 1.0.0.")
        self.tp_size = get_tensor_model_parallel_world_size()

    def get_weight(self, input_size: int, output_size: int, params_dtype: torch.dtype) -> dict[str, Any]:
        del params_dtype
        pack_factor = 2
        if output_size % pack_factor != 0:
            raise ValueError(f"Kimi K3 W4A8 output size {output_size} must be divisible by {pack_factor}.")
        return {
            "weight": torch.empty(output_size // pack_factor, input_size, dtype=torch.int8),
            "_packed_dim": 0,
            "_packed_factor": pack_factor,
        }

    def get_pergroup_param(
        self,
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        layer_type: str | None = None,
    ) -> dict[str, Any]:
        del input_size
        scale_bias_width = 16 if layer_type == "row" else 1
        return {
            "weight_scale": torch.empty(output_size, 1, dtype=params_dtype),
            "weight_offset": torch.empty(output_size, 1, dtype=params_dtype),
            "scale_bias": torch.empty(output_size, scale_bias_width, dtype=torch.float32),
        }

    def _local_scale_bias(self, layer: torch.nn.Module, tp_rank: int | None = None) -> torch.Tensor:
        scale_bias = layer.scale_bias
        if scale_bias.dim() != 2 or scale_bias.shape[1] == 1:
            return scale_bias.flatten()

        tp_size = getattr(layer, "tp_size", self.tp_size)
        rank = getattr(layer, "tp_rank", tp_rank if tp_rank is not None else 0)
        num_offline_shards = scale_bias.shape[1]
        if tp_size <= 0 or num_offline_shards % tp_size != 0:
            raise ValueError(
                f"scale_bias width {num_offline_shards} must be divisible by the projection TP size {tp_size}"
            )
        if rank < 0 or rank >= tp_size:
            raise ValueError(f"tp_rank {rank} exceeds projection TP size {tp_size}")

        shards_per_rank = num_offline_shards // tp_size
        start = rank * shards_per_rank
        return scale_bias[:, start : start + shards_per_rank].sum(dim=1)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        bias: torch.Tensor | None = None,
        tp_rank: int | None = None,
    ) -> torch.Tensor:
        if isinstance(x, tuple):
            quantized_x, pertoken_scale = x
            input_shape = quantized_x.shape
            output_dtype = torch.bfloat16
        else:
            input_shape = x.shape
            x_2d = x.reshape(-1, input_shape[-1])
            quantized_x, pertoken_scale = torch_npu.npu_dynamic_quant(x_2d)
            output_dtype = x.dtype

        # Build the single-expert token count on device so ACL Graph capture
        # does not introduce a synchronous host-to-device copy.
        group_list = torch.full(
            (1,),
            quantized_x.shape[0],
            dtype=torch.int64,
            device=quantized_x.device,
        )
        scale_bias = self._local_scale_bias(layer, tp_rank)
        output = torch_npu.npu_grouped_matmul(
            x=[quantized_x],
            weight=[layer.weight],
            scale=[layer.weight_scale.reshape(1, 1, -1)],
            bias=[scale_bias.reshape(1, -1)],
            per_token_scale=[pertoken_scale],
            split_item=2,
            group_list=group_list,
            group_type=0,
            group_list_type=1,
            output_dtype=output_dtype,
        )[0]
        if bias is not None:
            output = output + bias
        return output.reshape(*input_shape[:-1], output.shape[-1])

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        layer.weight.data = layer.weight.data.transpose(0, 1).contiguous()
        layer.weight_scale.data = layer.weight_scale.data.flatten()
        layer.weight_offset.data = layer.weight_offset.data.flatten()

        # The fused SiTU path needs the floating-point scale, while grouped
        # matmul expects each FP32 bit pattern stored in an int64 element.
        layer.weight_scale_fp32 = layer.weight_scale.data.to(torch.float32)
        scale_np = layer.weight_scale_fp32.contiguous().cpu().numpy()
        scale_np.dtype = np.uint32
        layer.weight_scale.data = torch.from_numpy(scale_np.astype(np.int64)).to(layer.weight_scale.device)

        layer.scale_bias.data = self._local_scale_bias(layer).contiguous()
        if layer.weight.data.shape[-1] % 4 != 0:
            raise ValueError(
                "the last dim of Kimi K3 W4A8 shared-expert weight must be divisible by 4, "
                f"but got shape {layer.weight.data.shape}"
            )
        # Model the shared projection as one grouped expert. Preserve the
        # packed INT4 bytes while converting to WeightNZ, then expose groups of
        # four bytes as the int32 storage required by grouped matmul.
        layer.weight.data = maybe_trans_nz(layer.weight.data.unsqueeze(0)).view(torch.int32).contiguous()


@register_scheme("W4A8_DYNAMIC", "moe")
class AscendW4A8DynamicFusedMoEMethod(AscendMoEScheme):
    """FusedMoE method for Ascend W4A8_DYNAMIC.

    This method supports two MoE weight formats: one generated by
    msModelSlim and one generated by LLM-Compressor. Only per-channel weight quantization
    is supported; per-group quantization is rejected at init time.

    The names below use ``L`` for the layer index, ``E`` for the expert
    index, ``num_experts`` for the routed expert count, ``hidden_sizes``
    for the hidden dimension, ``moe_intermediate_size`` for the expert
    intermediate dimension, and ``tp_size`` for tensor parallel size.

    Original MoE layer weights generated by msModelSlim:

    - ``model.layers.L.mlp.experts.E.gate_proj.weight``:
      ``torch.int8``, ``[moe_intermediate_size // 2, hidden_sizes]``.
      Each int8 element stores two packed 4-bit weight values along the
      output dimension.
    - ``model.layers.L.mlp.experts.E.up_proj.weight``:
      ``torch.int8``, ``[moe_intermediate_size // 2, hidden_sizes]``.
    - ``model.layers.L.mlp.experts.E.down_proj.weight``:
      ``torch.int8``, ``[hidden_sizes // 2, moe_intermediate_size]``.
    - Each linear also has ``weight_scale`` and ``weight_offset``:
      ``torch.float32``, ``[out_features, 1]``.
    - Each linear also has ``scale_bias``: ``torch.float32``,
      ``[moe_intermediate_size, 1]`` for ``gate_proj`` and ``up_proj``,
      and ``[hidden_sizes, 16 // tp_size]`` for ``down_proj``.

    Original MoE layer weights generated by LLM-Compressor:

    - ``model.layers.L.mlp.experts.E.gate_proj.weight``:
      ``torch.int8``, ``[moe_intermediate_size, hidden_sizes]``.
    - ``model.layers.L.mlp.experts.E.up_proj.weight``:
      ``torch.int8``, ``[moe_intermediate_size, hidden_sizes]``.
    - ``model.layers.L.mlp.experts.E.down_proj.weight``:
      ``torch.int8``, ``[hidden_sizes, moe_intermediate_size]``.
    - Each linear also has ``weight_scale``: ``torch.bfloat16``,
      ``[out_features, 1]``.

    During loading, ``gate_proj`` and ``up_proj`` are fused into ``w13``
    and ``down_proj`` is loaded as ``w2``. Before
    :meth:`process_weights_after_loading`, their logical shapes are:

    - msModelSlim: ``w13_weight`` ``torch.int8``,
      ``[num_experts, moe_intermediate_size, hidden_sizes]``; and
      ``w2_weight`` ``torch.int8``,
      ``[num_experts, hidden_sizes // 2, moe_intermediate_size]``.
    - LLM-Compressor: ``w13_weight`` ``torch.int8``,
      ``[num_experts, 2 * moe_intermediate_size, hidden_sizes]``; and
      ``w2_weight`` ``torch.int8``,
      ``[num_experts, hidden_sizes, moe_intermediate_size]``.

    After processing, ``apply`` passes these tensors to the fused MoE
    operator:

    - ``w13_weight``: ``torch.int32``,
      ``[num_experts, hidden_sizes, moe_intermediate_size // 4]``.
    - ``w2_weight``: ``torch.int32``,
      ``[num_experts, moe_intermediate_size, hidden_sizes // 8]``.
    - ``w13_scale_bias``: ``torch.float32``,
      ``[num_experts, 2 * moe_intermediate_size]``.
    - ``w2_scale_bias``: ``torch.float32``,
      ``[num_experts, hidden_sizes]``.
    - ``w13_weight_scale``: ``torch.int64``,
      ``[num_experts, 2 * moe_intermediate_size]``.
    - ``w2_weight_scale``: ``torch.int64``,
      ``[num_experts, 1, hidden_sizes]``.
    """

    supports_eplb = True
    # Declare the quantization type for this scheme
    quant_type: QuantType = QuantType.W4A8

    def __init__(self):
        vllm_config = get_current_vllm_config()
        group_size = vllm_config.quant_config.quant_description.get("group_size", 0)
        if group_size > 0:
            raise ValueError(
                "The current weights use Per‑Group quantization, which is no longer supported. Please "
                "switch to Per‑Channel quantized weights."
            )

        self.quant_method = vllm_config.quant_config.get_name()
        self.tp_size = (
            1 if vllm_config.parallel_config.enable_expert_parallel else get_tensor_model_parallel_world_size()
        )
        self.dynamic_eplb = False if vllm_config.use_v2_model_runner else get_ascend_config().eplb_config.dynamic_eplb
        self.use_expert_weight_list = self.dynamic_eplb or (
            vllm_config.use_v2_model_runner is True and vllm_config.parallel_config.enable_eplb is True
        )
        if self.quant_method == ASCEND_QUANTIZATION_METHOD and self.tp_size > 16:
            raise ValueError("The current weight does not support moe part tp>16.")

        try:
            device_group = get_mc2_group().device_group
            # TODO: Try local_rank = ep_group.rank_in_group
            local_rank = torch.distributed.get_rank(group=device_group)
            backend = device_group._get_backend(torch.device("npu"))
            self.moe_all_to_all_group_name = backend.get_hccl_comm_name(local_rank)
        except AttributeError:
            self.moe_all_to_all_group_name = ""

    def get_weight(
        self, num_experts: int, intermediate_size_per_partition: int, hidden_sizes: int, params_dtype: torch.dtype
    ) -> dict[str, Any]:
        param_dict = {}
        # Note: compressed‑tensors: one int4 per int8; modelslim: packs two int4 into one int8
        if self.quant_method == COMPRESSED_TENSORS_METHOD:
            w13_output_size = 2 * intermediate_size_per_partition
            w2_output_size = hidden_sizes
        else:
            w13_output_size = intermediate_size_per_partition
            w2_output_size = hidden_sizes // 2
        param_dict["w13_weight"] = torch.empty(num_experts, w13_output_size, hidden_sizes, dtype=torch.int8)
        param_dict["w2_weight"] = torch.empty(
            num_experts, w2_output_size, intermediate_size_per_partition, dtype=torch.int8
        )
        return param_dict

    def get_dynamic_quant_param(
        self, num_experts: int, intermediate_size_per_partition: int, hidden_sizes: int, params_dtype: torch.dtype
    ) -> dict[str, Any]:
        if self.quant_method == COMPRESSED_TENSORS_METHOD:
            return self.get_dynamic_quant_param_compressed_tensors(
                num_experts, intermediate_size_per_partition, hidden_sizes
            )
        else:
            return self.get_dynamic_quant_param_modelslim(num_experts, intermediate_size_per_partition, hidden_sizes)

    def get_dynamic_quant_param_compressed_tensors(
        self, num_experts: int, intermediate_size_per_partition: int, hidden_sizes: int
    ) -> dict[str, Any]:
        param_dict = {}
        param_dict["w13_weight_scale"] = torch.empty(
            num_experts, 2 * intermediate_size_per_partition, 1, dtype=torch.bfloat16
        )
        param_dict["w2_weight_scale"] = torch.empty(num_experts, hidden_sizes, 1, dtype=torch.bfloat16)
        return param_dict

    def get_dynamic_quant_param_modelslim(
        self, num_experts: int, intermediate_size_per_partition: int, hidden_sizes: int
    ) -> dict[str, Any]:
        param_dict = {}
        param_dict["w13_weight_scale"] = torch.empty(
            num_experts, 2 * intermediate_size_per_partition, 1, dtype=torch.float32
        )
        param_dict["w13_weight_offset"] = torch.empty(
            num_experts, 2 * intermediate_size_per_partition, 1, dtype=torch.float32
        )
        param_dict["w2_weight_scale"] = torch.empty(num_experts, hidden_sizes, 1, dtype=torch.float32)
        param_dict["w2_weight_offset"] = torch.empty(num_experts, hidden_sizes, 1, dtype=torch.float32)
        param_dict["w13_scale_bias"] = torch.empty(
            num_experts, 2 * intermediate_size_per_partition, 1, dtype=torch.float32
        )
        param_dict["w2_scale_bias"] = torch.empty(num_experts, hidden_sizes, 16 // self.tp_size, dtype=torch.float32)
        return param_dict

    def apply(
        self,
        layer: "AscendRoutedExperts",
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts: Any | None,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        topk_weights = topk_weights.to(x.dtype)

        use_mega_moe = (
            _EXTRA_CTX.moe_comm_type == MoECommType.FUSED_MC2
            and get_ascend_config().enable_fused_mc2 == 1
            and is_mega_moe_supported()
        )
        w1_scale_bias: list[torch.Tensor] | None
        w2_scale_bias: list[torch.Tensor] | None

        if self.use_expert_weight_list:
            if use_mega_moe:
                # EPLB rearranges these lists in place. MegaMoE must consume
                # their original INT8/NZ tensors instead of the INT32 views
                # used by the legacy dynamic-EPLB kernels.
                w1 = layer.w13_weight_list
                w1_scale = [t.reshape(-1) for t in layer.w13_weight_scale_list]
                w2 = layer.w2_weight_list
                w2_scale = [t.reshape(-1) for t in layer.w2_weight_scale_list]
                w1_scale_bias = [t.reshape(-1) for t in layer.w13_scale_bias_list]
                w2_scale_bias = [t.reshape(-1) for t in layer.w2_scale_bias_list]
            else:
                w1 = [i.view(torch.int32) for i in layer.w13_weight_list]
                w1_scale = layer.w13_weight_scale_list
                w2 = [i.view(torch.int32) for i in layer.w2_weight_list]
                w2_scale = layer.w2_weight_scale_list
                w1_scale_bias = layer.w13_scale_bias_list
                w2_scale_bias = layer.w2_scale_bias_list
        elif use_mega_moe:
            w1 = layer.cann_mega_moe_w13_weight_list
            w1_scale = layer.cann_mega_moe_w13_weight_scale_list
            w2 = layer.cann_mega_moe_w2_weight_list
            w2_scale = layer.cann_mega_moe_w2_weight_scale_list

            w1_scale_bias = layer.cann_mega_moe_w13_scale_bias_list
            w2_scale_bias = layer.cann_mega_moe_w2_scale_bias_list
        else:
            w1 = [layer.w13_weight]
            w1_scale = [layer.w13_weight_scale]
            w2 = [layer.w2_weight]
            w2_scale = [layer.w2_weight_scale]
            w1_scale_bias = [layer.w13_scale_bias.detach()] if hasattr(layer, "w13_scale_bias") else None
            w2_scale_bias = [layer.w2_scale_bias.detach()] if hasattr(layer, "w2_scale_bias") else None

        moe_comm_method = _EXTRA_CTX.moe_comm_method
        return moe_comm_method.fused_experts(
            fused_experts_input=build_fused_experts_input(
                hidden_states=x,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                w1=w1,
                w2=w2,
                quant_type=self.quant_type,
                dynamic_eplb=self.use_expert_weight_list,
                expert_map=layer.ascend_expert_map,
                global_redundant_expert_num=layer.global_redundant_expert_num,
                mc2_mask=layer.ascend_mc2_mask,
                apply_router_weight_on_input=layer.apply_router_weight_on_input,
                pertoken_scale=layer.ascend_pertoken_scale,
                activation=layer.activation,
                w1_scale=w1_scale,
                w2_scale=w2_scale,
                w1_scale_bias=w1_scale_bias,
                w2_scale_bias=w2_scale_bias,
                is_per_channel_weight=True,
            )
        )

    @staticmethod
    def get_eplb_weight_views(layer: torch.nn.Module) -> list:
        if hasattr(layer, "w13_weight_list"):
            weights = [
                layer.w13_weight_list,
                layer.w2_weight_list,
                layer.w13_weight_scale_list,
                layer.w2_weight_scale_list,
            ]
            w13_scale_bias = layer.w13_scale_bias_list
            w2_scale_bias = layer.w2_scale_bias_list
            if (w13_scale_bias is None) != (w2_scale_bias is None):
                raise RuntimeError(
                    "W4A8 EPLB requires w13_scale_bias_list and w2_scale_bias_list to be present or absent together."
                )
            if w13_scale_bias is not None:
                weights.extend([w13_scale_bias, w2_scale_bias])
            return weights

        weights = [
            layer.w13_weight,
            layer.w2_weight,
            layer.w13_weight_scale,
            layer.w2_weight_scale,
        ]
        w13_scale_bias = getattr(layer, "w13_scale_bias", None)
        w2_scale_bias = getattr(layer, "w2_scale_bias", None)
        if (w13_scale_bias is None) != (w2_scale_bias is None):
            raise RuntimeError("W4A8 EPLB requires w13_scale_bias and w2_scale_bias to be present or absent together.")
        if w13_scale_bias is not None:
            weights.extend([w13_scale_bias, w2_scale_bias])
        return weights

    @staticmethod
    def maybe_squeeze_per_channel_weight_scale(scale: torch.Tensor) -> torch.Tensor:
        if scale.dim() > 1 and scale.shape[1] == 1:
            return scale.squeeze(1)
        return scale

    def _pack_to_int32(self, weight: torch.Tensor):
        # pack 4 int8(int4*2) to int32, because in pytorch, we need to use int32 to represent int4
        assert weight.shape[-1] % 4 == 0, (
            f"the last dim of weight needs to be divided by 4 but got shape {weight.shape}"
        )
        return weight.view(torch.int32).contiguous()

    def _process_scale(self, scale):
        scale = scale.transpose(1, 2).contiguous()
        scale_np = scale.cpu().numpy()
        scale_np.dtype = np.uint32
        scale_uint64_tensor = torch.from_numpy(scale_np.astype(np.int64)).npu()
        return scale_uint64_tensor

    def _pack_int4_to_int8(self, weight: torch.Tensor) -> torch.Tensor:
        shape = weight.shape
        weight = weight.reshape(-1, 2)
        weight0 = weight[:, :1]
        weight1 = weight[:, 1:]
        weight1_4 = torch.bitwise_left_shift(weight1, 4)
        weight2_4 = weight0 & 0b00001111
        weight_add = torch.bitwise_or(weight1_4, weight2_4)
        # The clone() call is used to break the view chain
        return weight_add.reshape(shape[:-1] + (shape[-1] // 2,)).clone()

    def process_weights_after_loading(self, layer):
        if self.quant_method == COMPRESSED_TENSORS_METHOD:
            self.process_weights_after_loading_compressed_tensors(layer)
        else:
            self.process_weights_after_loading_modelslim(layer)
        layer.w13_weight.data = maybe_trans_nz(layer.w13_weight.data)
        layer.w2_weight.data = maybe_trans_nz(layer.w2_weight.data)
        tensor_names = (
            "w13_weight",
            "w2_weight",
            "w13_weight_scale",
            "w2_weight_scale",
            "w13_scale_bias",
            "w2_scale_bias",
        )
        if self.use_expert_weight_list:
            for tensor_name in tensor_names:
                tensor = getattr(layer, tensor_name)
                expert_list = [expert.clone() for expert in tensor.data.unbind(dim=0)]
                if (
                    tensor_name in ("w13_scale_bias", "w2_scale_bias")
                    and get_ascend_config().enable_fused_mc2 == 1
                    and is_mega_moe_supported()
                ):
                    expert_list = [expert.to(torch.float32) for expert in expert_list]
                setattr(layer, f"{tensor_name}_list", expert_list)
                delattr(layer, tensor_name)
        elif get_ascend_config().enable_fused_mc2 == 1 and is_mega_moe_supported():
            layer.cann_mega_moe_w13_weight_list = [weight.clone() for weight in layer.w13_weight.data.unbind(dim=0)]
            layer.cann_mega_moe_w2_weight_list = [weight.clone() for weight in layer.w2_weight.data.unbind(dim=0)]

            layer.cann_mega_moe_w13_weight_scale_list = [
                t.reshape(-1) for t in layer.w13_weight_scale.data.unbind(dim=0)
            ]
            layer.cann_mega_moe_w2_weight_scale_list = [t.reshape(-1) for t in layer.w2_weight_scale.data.unbind(dim=0)]
            layer.cann_mega_moe_w13_scale_bias_list = [
                t.reshape(-1).to(torch.float32) for t in layer.w13_scale_bias.data.unbind(dim=0)
            ]
            layer.cann_mega_moe_w2_scale_bias_list = [
                t.reshape(-1).to(torch.float32) for t in layer.w2_scale_bias.data.unbind(dim=0)
            ]
            for tensor_name in tensor_names:
                delattr(layer, tensor_name)
        else:
            layer.w13_weight.data = self._pack_to_int32(layer.w13_weight.data)
            layer.w2_weight.data = self._pack_to_int32(layer.w2_weight.data)

    def process_weights_after_loading_compressed_tensors(self, layer):
        layer.w13_weight.data = layer.w13_weight.data.transpose(1, 2).contiguous()
        layer.w2_weight.data = layer.w2_weight.data.transpose(1, 2).contiguous()

        def update_bias_compressed_tensors(weight: torch.Tensor, scale: torch.Tensor):
            group_num, k, n = weight.shape
            scale = scale.transpose(1, 2).contiguous()
            scale = scale.reshape(group_num, -1, n)
            bias = 8 * (weight.to(torch.float32) * scale).sum(axis=1)
            return bias

        w13_bias = update_bias_compressed_tensors(layer.w13_weight.data, layer.w13_weight_scale.data)
        w2_bias = update_bias_compressed_tensors(layer.w2_weight.data, layer.w2_weight_scale.data)
        w13_scale_bias = torch.nn.Parameter(w13_bias, requires_grad=False)
        layer.register_parameter("w13_scale_bias", w13_scale_bias)
        w2_scale_bias = torch.nn.Parameter(w2_bias, requires_grad=False)
        layer.register_parameter("w2_scale_bias", w2_scale_bias)

        layer.w13_weight_scale.data = self._process_scale(layer.w13_weight_scale.data.to(torch.float32))
        layer.w2_weight_scale.data = self._process_scale(layer.w2_weight_scale.data.to(torch.float32))
        # To use torch_npu.npu_grouped_matmul, keep w2_weigh_scale unsqueezed
        layer.w13_weight_scale.data = self.maybe_squeeze_per_channel_weight_scale(layer.w13_weight_scale.data)

        # Packs 2 int4 into 1 int8 on-the-fly to mirror the modelslim path
        layer.w13_weight.data = self._pack_int4_to_int8(layer.w13_weight.data)
        layer.w2_weight.data = self._pack_int4_to_int8(layer.w2_weight.data)

    def process_weights_after_loading_modelslim(self, layer):
        layer.w13_weight.data = layer.w13_weight.data.transpose(1, 2).contiguous()
        layer.w2_weight.data = layer.w2_weight.data.transpose(1, 2).contiguous()

        layer.w13_weight_scale.data = self._process_scale(layer.w13_weight_scale.data)
        layer.w2_weight_scale.data = self._process_scale(layer.w2_weight_scale.data)

        layer.w13_scale_bias.data = layer.w13_scale_bias.data.transpose(1, 2).contiguous().sum(axis=1)
        layer.w2_scale_bias.data = layer.w2_scale_bias.data.transpose(1, 2).contiguous().sum(axis=1)
        layer.w13_weight_scale.data = self.maybe_squeeze_per_channel_weight_scale(layer.w13_weight_scale.data)
