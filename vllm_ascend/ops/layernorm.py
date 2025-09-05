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
from vllm.model_executor.layers.layernorm import RMSNorm


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
            residual = torch.ops.vllm.maybe_chunk_residual(x, residual)
            assert x.size(0) == residual.size(0)
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


class AscendRMSNorm(RMSNorm):

    def forward_oot(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        import torch_npu

        from vllm_ascend.utils import is_310p
        if residual is not None:
            residual = torch.ops.vllm.maybe_chunk_residual(x, residual)
            assert x.size(0) == residual.size(0)
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

        x, residual = torch_npu.npu_rms_norm(x, self.weight,
                                             self.variance_epsilon)
        return x
