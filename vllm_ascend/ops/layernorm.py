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
import vllm_ascend.envs as envs_ascend


class AddRMSNormW8A8Quant(RMSNorm):
    # Fuse AddRmsNorm and W8A8 quantization ops together

    def __init__(
        self,
        hidden_size: int,
        layer: torch.nn.Module,
        eps: float = 1e-6,
        var_hidden_size: Optional[int] = None,
        has_weight: bool = True,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__(hidden_size, eps, var_hidden_size, has_weight, dtype)
        self.layer = layer

    def forward(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        import torch_npu

        if residual is not None:
            x, _, residual = torch_npu.npu_add_rms_norm_quant(
                x,
                residual,
                self.weight,
                self.layer.aclnn_input_scale,
                self.layer.aclnn_input_offset,
                epsilon=self.variance_epsilon)
            return x, residual

        x, residual = torch_npu.npu_rms_norm(x, self.weight,
                                             self.variance_epsilon)
        return x

def _addrmsnorm_w8a8_quant_forward_oot(
    self,
    x: torch.Tensor,
    residual: torch.Tensor,
    layer: torch.nn.Module,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    import torch_npu

    x, _, residual = torch_npu.npu_add_rms_norm_quant(
        x,
        residual,
        self.weight,
        layer.aclnn_input_scale,
        layer.aclnn_input_offset,
        epsilon=self.variance_epsilon)
    return x, residual


def _addrmsnorm_forward_oot(
    self,
    x: torch.Tensor,
    residual: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    import torch_npu

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

    def forward_oot(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        import torch_npu

        from vllm_ascend.utils import is_310p
        if residual is not None:
            if envs_ascend.VLLM_ASCEND_ENABLE_ADDRMSNORM_QUANT_FUSION and \
                not is_310p():
                forward_context = get_forward_context()
                prefetch_model = forward_context.prefetch_model
                layer_idx = forward_context.layer_idx
                fusion_linear = forward_context.fusion_linear
                if fusion_linear == "gate_up":
                    fusion_linear = prefetch_model.model.layers[layer_idx].mlp.gate_up_proj
                    forward_context.fusion_linear = "qkv"
                    forward_context.layer_idx += 1
                elif fusion_linear == "qkv":
                    fusion_linear = prefetch_model.model.layers[layer_idx].self_attn.qkv_proj
                    forward_context.fusion_linear = "gate_up"
                from vllm_ascend.quantization.w8a8 import AscendW8A8LinearMethod

                # assert isinstance(quant_config, AscendQuantConfig), \
                #     "Expected quant_config to be an instance of AscendQuantConfig"
                if isinstance(fusion_linear.quant_method.quant_method,
                      AscendW8A8LinearMethod):
                    x, residual = _addrmsnorm_w8a8_quant_forward_oot(x, residual, fusion_linear)
                else:
                    x, residual = _addrmsnorm_forward_oot(x, residual)
            else:
                x, residual = _addrmsnorm_forward_oot(x, residual)
            return x, residual

        x, residual = torch_npu.npu_rms_norm(x, self.weight,
                                             self.variance_epsilon)
        return x
