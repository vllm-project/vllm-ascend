#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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
# This file is a part of the vllm-ascend project.
#

from typing import Optional, Tuple, Union

import torch
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.layernorm import RMSNorm


def _addrmsnorm_forward_oot(
    self,
    x: torch.Tensor,
    residual: Optional[torch.Tensor] = None,
    layer: Optional[torch.nn.Module] = None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    import torch_npu

    if layer is not None:
        x, _, residual = torch_npu.npu_add_rms_norm_quant(
            x,
            residual,
            self.weight,
            layer.aclnn_input_scale,
            layer.aclnn_input_offset,
            epsilon=self.variance_epsilon)
    else:
        from vllm_ascend.utils import is_310p
        if is_310p():
            orig_dtype = residual.dtype
            x = x + residual.to(x.dtype)
            residual = x.to(orig_dtype)
            x, _ = torch_npu.npu_rms_norm(x, self.weight,
                                            self.variance_epsilon)
        else:
            x, _, residual = torch_npu.npu_add_rms_norm(
                x, residual, self.weight, self.variance_epsilon)
    return x, residual


class AscendRMSNorm(RMSNorm):

    @property
    def next_need_quant_fusion_linear(self):
        next_linear = None
        if get_forward_context().addrmsnorm_quant_fusion_enabled:
            forward_context = get_forward_context()
            prefetch_model = forward_context.prefetch_model
            num_hidden_layers = forward_context.num_hidden_layers
            layer_idx = forward_context.layer_idx
            fusion_linear = forward_context.fusion_linear
            next_linear = None
            if fusion_linear == "qkv_dense":
                next_linear = prefetch_model.model.layers[layer_idx].self_attn.qkv_proj
                forward_context.fusion_linear = "gate_up_dense"
            elif fusion_linear == "gate_up_dense":
                next_linear = prefetch_model.model.layers[layer_idx].mlp.gate_up_proj
                forward_context.fusion_linear = "qkv_dense"
                forward_context.layer_idx += 1
            # last norm before lm_head
            if forward_context.layer_idx == num_hidden_layers:
                forward_context.addrmsnorm_quant_fusion_enabled = False
            from vllm_ascend.quantization.w8a8 import AscendW8A8LinearMethod
            if next_linear is not None and \
                not isinstance(next_linear.quant_method.quant_method, AscendW8A8LinearMethod):
                next_linear = None
        return next_linear

    def forward_oot(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        import torch_npu

        if residual is not None:
            # FIXME(rjg-lyh): This is a hacky way to chunk residuals when the flashcomm_v1 feature
            # is enabled, without interfering with the normal operation of components like torchair.
            # The final solution should be to move this check into the operator and support
            # integration with torchair.
            if x.size(0) != residual.size(0):
                residual = torch.ops.vllm.maybe_chunk_residual(x, residual)
            assert x.size(0) == residual.size(0)
            x, residual = _addrmsnorm_forward_oot(self, x, residual, self.next_need_quant_fusion_linear)
            return x, residual
        x, residual = torch_npu.npu_rms_norm(x, self.weight,
                                             self.variance_epsilon)
        return x
