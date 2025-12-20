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
from vllm.config import get_current_vllm_config
from vllm.model_executor.layers.layernorm import RMSNorm

_original_re_init = RMSNorm.__init__


def torchair_rmsnorm_init_(
    self,
    hidden_size: int,
    eps: float = 1e-6,
    var_hidden_size: Optional[int] = None,
    has_weight: bool = True,
    dtype: Optional[torch.dtype] = None,
) -> None:
    _original_re_init(self, hidden_size, eps, var_hidden_size, has_weight,
                      dtype)
    vllm_config = get_current_vllm_config()
    self.bias = None
    # quantization with anti_method m4 will generate none-zero norm bias
    if vllm_config.quant_config is not None and \
            any("norm.bias" in name for name in vllm_config.quant_config.quant_description.keys()):
        self.bias = torch.nn.Parameter(torch.zeros(hidden_size),
                                       requires_grad=False)


def torchair_rmsnorm_forward_oot(
    self,
    x: torch.Tensor,
    residual: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """AscendRMSNorm forward in torchair mode.

    The key difference from the original implementation is the removal of operators
    from the torch.ops.vllm class, as these operators only function in non-torchair
    modes. Adding them back would cause the graph compilation to fail.
    """

    import torch_npu

    from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type

    # DEBUG: 详细信息追踪精度问题 - TORCHAIR版本
    print(f"[DEBUG TORCHAIR LAYERNORM] torchair_rmsnorm_forward_oot called!")
    print(f"[DEBUG TORCHAIR LAYERNORM] Device: {get_ascend_device_type()}")
    print(f"[DEBUG TORCHAIR LAYERNORM] Input x shape: {x.shape}, dtype: {x.dtype}")
    print(f"[DEBUG TORCHAIR LAYERNORM] Input x stats: mean={x.float().mean():.6f}, std={x.float().std():.6f}")
    print(f"[DEBUG TORCHAIR LAYERNORM] Input x min/max: {x.min():.6f}/{x.max():.6f}")
    print(f"[DEBUG TORCHAIR LAYERNORM] Weight shape: {self.weight.shape}, dtype: {self.weight.dtype}")
    print(f"[DEBUG TORCHAIR LAYERNORM] Weight stats: mean={self.weight.float().mean():.6f}, std={self.weight.float().std():.6f}")
    print(f"[DEBUG TORCHAIR LAYERNORM] Variance epsilon: {self.variance_epsilon}")

    if residual is not None:
        print(f"[DEBUG TORCHAIR LAYERNORM] Residual shape: {residual.shape}, dtype: {residual.dtype}")
        print(f"[DEBUG TORCHAIR LAYERNORM] Residual stats: mean={residual.float().mean():.6f}, std={residual.float().std():.6f}")
        print(f"[DEBUG TORCHAIR LAYERNORM] Residual min/max: {residual.min():.6f}/{residual.max():.6f}")
    if residual is not None:
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

        # DEBUG: 输出统计信息
        print(f"[DEBUG TORCHAIR LAYERNORM] Output x stats: mean={x.float().mean():.6f}, std={x.float().std():.6f}")
        print(f"[DEBUG TORCHAIR LAYERNORM] Output x min/max: {x.min():.6f}/{x.max():.6f}")
        if residual is not None:
            print(f"[DEBUG TORCHAIR LAYERNORM] Output residual stats: mean={residual.float().mean():.6f}, std={residual.float().std():.6f}")
            print(f"[DEBUG TORCHAIR LAYERNORM] Output residual min/max: {residual.min():.6f}/{residual.max():.6f}")

        return x, residual

    x, residual = torch_npu.npu_rms_norm(x, self.weight, self.variance_epsilon)
    if self.bias is not None:
        x.add_(self.bias)

    # DEBUG: 输出统计信息
    print(f"[DEBUG TORCHAIR LAYERNORM] Output x stats: mean={x.float().mean():.6f}, std={x.float().std():.6f}")
    print(f"[DEBUG TORCHAIR LAYERNORM] Output x min/max: {x.min():.6f}/{x.max():.6f}")

    return x
