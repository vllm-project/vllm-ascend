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

from typing import Optional, Tuple, Union, cast

import torch
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.layernorm import GemmaRMSNorm, RMSNorm


def _addrmsnorm_forward_oot(
    self,
    x: torch.Tensor,
    residual: torch.Tensor,
    layer: Optional[torch.nn.Module] = None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    import torch_npu

    from vllm_ascend.utils import is_310p

    if layer is not None and not is_310p():
        x, _, residual = torch_npu.npu_add_rms_norm_quant(
            x,
            residual,
            self.weight,
            layer.aclnn_input_scale,
            layer.aclnn_input_offset,
            epsilon=self.variance_epsilon)
    else:
        if is_310p():
            orig_dtype = residual.dtype
            x = x + residual.to(x.dtype)
            residual = x.to(orig_dtype)
            x, _ = torch_npu.npu_rms_norm(x, self.weight,
                                          self.variance_epsilon)
        else:
            x, _, residual = torch_npu.npu_add_rms_norm(
                x, residual, self.weight, self.variance_epsilon)
    torch.ops.vllm.maybe_wait_prefetch_done(x)
    return x, residual


class AscendRMSNorm(RMSNorm):

    def forward_oot(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        import torch_npu

        if residual is not None:
            assert x.size(0) == residual.size(0)
            x, residual = _addrmsnorm_forward_oot(
                self, x, residual, self.next_need_quant_fusion_linear)
            return x, residual
        x, residual = torch_npu.npu_rms_norm(x, self.weight,
                                             self.variance_epsilon)
        return x

    @property
    def next_need_quant_fusion_linear(self):
        try:
            forward_context = get_forward_context()
            if not forward_context.addrmsnorm_quant_fusion_enabled or \
                forward_context.layer_idx == forward_context.num_hidden_layers:
                return None
        except AssertionError:
            return None

        next_linear = None
        model_instance = forward_context.model_instance
        layer_idx = forward_context.layer_idx
        fusion_linear = forward_context.fusion_linear
        next_linear = None
        if fusion_linear == "qkv_dense":
            next_linear = model_instance.model.layers[
                layer_idx].self_attn.qkv_proj
            forward_context.fusion_linear = "gate_up_dense"
        elif fusion_linear == "gate_up_dense":
            next_linear = model_instance.model.layers[
                layer_idx].mlp.gate_up_proj
            forward_context.fusion_linear = "qkv_dense"
            # if prefetch_mlp_weight enabled, following accumulation operation
            # does not need to be repeated
            if not forward_context.prefetch_mlp_enabled:
                forward_context.layer_idx += 1
        from vllm_ascend.quantization.w8a8 import AscendW8A8LinearMethod
        if next_linear is not None and \
            not isinstance(next_linear.quant_method.quant_method, AscendW8A8LinearMethod):
            next_linear = None
        return next_linear


class AscendQuantRMSNorm(AscendRMSNorm):

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        var_hidden_size: Optional[int] = None,
        has_weight: bool = True,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__(hidden_size, eps, var_hidden_size, has_weight, dtype)
        self.bias = torch.nn.Parameter(torch.zeros(hidden_size),
                                       requires_grad=False)

    def forward_oot(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if residual is not None:
            x, residual = super().forward_oot(x, residual)
            return x.add_(self.bias), residual
        return cast(torch.Tensor, super().forward_oot(x)).add_(self.bias)


class AscendGemmaRMSNorm(GemmaRMSNorm):

    def forward_oot(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        import torch_npu

        from vllm_ascend.utils import is_310p
        if residual is not None:
            if is_310p():
                orig_dtype = residual.dtype
                x = x + residual.to(x.dtype)
                residual = x.to(orig_dtype)
                x, _ = torch_npu.npu_rms_norm(x, 1.0 + self.weight,
                                              self.variance_epsilon)
            else:
                x, _, residual = torch_npu.npu_add_rms_norm(
                    x, residual, 1.0 + self.weight, self.variance_epsilon)
            return x, residual

        x, _ = torch_npu.npu_rms_norm(x, 1.0 + self.weight,
                                      self.variance_epsilon)
        return x
