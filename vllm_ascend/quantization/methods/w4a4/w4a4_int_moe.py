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

"""ModelSlim symmetric per-channel INT4 experts with per-token INT4 activations."""

import torch
import torch_npu
from vllm.config import get_current_vllm_config

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.quantization.quant_type import QuantType

from ..registry import register_scheme
from ..w8a8.w8a8_dynamic import AscendW8A8DynamicFusedMoEMethod

_INT4_VALUES_PER_INT32 = 8


@register_scheme("W4A4_DYNAMIC", "moe")
class AscendW4A4DynamicFusedMoEMethod(AscendW8A8DynamicFusedMoEMethod):
    """Route floating point tokens, then quantize locally before each GMM.

    ModelSlim stores one signed INT4 value per INT8 for this scheme. This
    differs from the packed W4A8 checkpoint layout. No A8W4 shift bias applies.
    """

    quant_type = QuantType.W4A4
    act_quant_type = torch.quint4x2
    fused_activations = frozenset()
    supports_eplb = False

    def __init__(self):
        config = get_current_vllm_config()
        if config.parallel_config.enable_eplb:
            raise ValueError("W4A4_DYNAMIC MoE does not support EPLB.")
        if get_ascend_config().enable_fused_mc2:
            raise ValueError("W4A4_DYNAMIC MoE requires additional_config.enable_fused_mc2=0.")
        super().__init__()

    def get_dynamic_quant_param(self, num_experts, intermediate_size_per_partition, hidden_sizes, params_dtype):
        return super().get_dynamic_quant_param(
            num_experts, intermediate_size_per_partition, hidden_sizes, torch.float32
        )

    def process_weights_after_loading(self, layer):
        for prefix in ("w13", "w2"):
            weight = getattr(layer, f"{prefix}_weight")
            scale = getattr(layer, f"{prefix}_weight_scale")
            offset = getattr(layer, f"{prefix}_weight_offset")
            # Loading-only CPU validation; never synchronize in the forward path.
            if torch.any(offset.detach().cpu() != 0):
                raise ValueError("W4A4_DYNAMIC MoE requires symmetric weight offsets.")
            experts, output_size, input_size = weight.shape
            if output_size % _INT4_VALUES_PER_INT32:
                raise ValueError("W4A4_DYNAMIC expert output size must be divisible by 8.")
            # ND GMM requires non-transposed packed weights. Pack the output
            # dimension after transposing, not the checkpoint input dimension.
            unpacked = weight.data.transpose(1, 2).contiguous().reshape(-1, output_size)
            packed = torch_npu.npu_convert_weight_to_int4pack(unpacked.to(torch.int32))
            weight.data = packed.reshape(experts, input_size, output_size // _INT4_VALUES_PER_INT32)
            # CANN A4W4 GMM consumes FP32 scale bits in the low half of INT64.
            scale.data = scale.data.reshape(experts, output_size).contiguous().view(torch.int32).to(torch.int64)

    def _gmm(self, mlp_compute_input, hidden_states, token_scale, prefix):
        layer = mlp_compute_input.layer
        return torch_npu.npu_grouped_matmul(
            x=[hidden_states],
            weight=[getattr(layer, f"{prefix}_weight")],
            scale=[getattr(layer, f"{prefix}_weight_scale")],
            per_token_scale=[token_scale],
            group_list=mlp_compute_input.group_list,
            group_list_type=mlp_compute_input.group_list_type,
            group_type=0,
            split_item=2,
            output_dtype=self.in_dtype,
        )[0]

    def apply_gmm1(self, mlp_compute_input):
        # Routing can carry scale metadata even when dispatch did not quantize
        # its floating point payload. Only packed INT4 inputs may reuse scales.
        hidden_states = mlp_compute_input.hidden_states
        if hidden_states.is_floating_point():
            hidden_states, token_scale = torch_npu.npu_dynamic_quant(hidden_states, dst_type=self.act_quant_type)
        else:
            if hidden_states.dtype != torch.int32 or mlp_compute_input.dynamic_scale is None:
                raise ValueError("W4A4 MoE expects floating point or packed INT4 input with token scales.")
            token_scale = mlp_compute_input.dynamic_scale
        return self._gmm(mlp_compute_input, hidden_states, token_scale, "w13")

    def apply_act_quant(self, mlp_compute_input, hidden_states):
        return torch_npu.npu_dynamic_quant(hidden_states, dst_type=self.act_quant_type)

    def apply_gmm2(self, mlp_compute_input, hidden_states, act_out_scale):
        return self._gmm(mlp_compute_input, hidden_states, act_out_scale, "w2")
