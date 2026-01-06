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
import torch.nn.functional as F
from vllm.config import get_current_vllm_config
from vllm.model_executor.layers.layernorm import GemmaRMSNorm, RMSNorm
from vllm.forward_context import get_forward_context
from vllm.distributed import (get_tensor_model_parallel_rank,
                              get_tensor_model_parallel_world_size)


class AscendRMSNorm(RMSNorm):

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        var_hidden_size: Optional[int] = None,
        has_weight: bool = True,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__(hidden_size, eps, var_hidden_size, has_weight, dtype)
        vllm_config = get_current_vllm_config()
        self.bias = None
        # quantization with anti_method m4 will generate none-zero norm bias
        if vllm_config.quant_config is not None and \
                any("norm.bias" in name for name in vllm_config.quant_config.quant_description.keys()):
            self.bias = torch.nn.Parameter(torch.zeros(hidden_size),
                                           requires_grad=False)

    def forward_oot(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        import torch_npu

        from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type
        if residual is not None:
            if x.size(0) != residual.size(0):
                sp_enabled = get_forward_context().sp_enabled
                assert sp_enabled is True, ("Currently, this situation only occurs "
                                            "when sp is enabled")
                pad_size = get_forward_context().pad_size
                if pad_size > 0:
                    residual = F.pad(residual, (0, 0, 0, pad_size))
                tp_size = get_tensor_model_parallel_world_size()
                tp_rank = get_tensor_model_parallel_rank()
                residual = torch.chunk(residual, tp_size, dim=0)[tp_rank]
            if get_ascend_device_type() == AscendDeviceType._310P:
                orig_dtype = residual.dtype
                x = x + residual.to(x.dtype)
                residual = x.to(orig_dtype)
                x, _ = torch_npu.npu_rms_norm(x, self.weight,
                                              self.variance_epsilon)
            else:
                x, _, residual = torch_npu.npu_add_rms_norm(
                    x, residual, self.weight, self.variance_epsilon)
                if self.bias is not None:
                    x.add_(self.bias)
            return x, residual

        x, residual = torch_npu.npu_rms_norm(x, self.weight,
                                             self.variance_epsilon)
        if self.bias is not None:
            x.add_(self.bias)
        return x


class AscendGemmaRMSNorm(GemmaRMSNorm):

    def forward_oot(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        import torch_npu

        from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type
        if residual is not None:
            if get_ascend_device_type() == AscendDeviceType._310P:
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
