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
from vllm.config import get_current_vllm_config
from vllm.model_executor.layers.layernorm import GemmaRMSNorm, RMSNorm


class AscendRMSNorm(RMSNorm):

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        var_hidden_size: Optional[int] = None,
        has_weight: bool = True,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        # DEBUG: 追踪RMSNorm初始化
        print(f"[DEBUG RMSNorm INIT] AscendRMSNorm.__init__ called!")
        print(f"[DEBUG RMSNorm INIT] hidden_size: {hidden_size}, eps: {eps}, dtype: {dtype}")
        super().__init__(hidden_size, eps, var_hidden_size, has_weight, dtype)
        vllm_config = get_current_vllm_config()
        self.bias = None
        # quantization with anti_method m4 will generate none-zero norm bias
        if vllm_config.quant_config is not None and \
                any("norm.bias" in name for name in vllm_config.quant_config.quant_description.keys()):
            self.bias = torch.nn.Parameter(torch.zeros(hidden_size),
                                           requires_grad=False)

        # DEBUG: 检查CustomOp分发机制
        print(f"[DEBUG RMSNorm INIT] CustomOp enabled: {self.enabled()}")
        print(f"[DEBUG RMSNorm INIT] Available methods: {[m for m in dir(self) if 'forward' in m]}")

    def forward(self, *args, **kwargs):
        """通用的forward方法拦截器"""
        print(f"[DEBUG RMSNorm FORWARD] AscendRMSNorm.forward called!")
        print(f"[DEBUG RMSNorm FORWARD] Args count: {len(args)}")
        if args:
            x = args[0]
            print(f"[DEBUG RMSNorm FORWARD] Input shape: {x.shape}, dtype: {x.dtype}")
            print(f"[DEBUG RMSNorm FORWARD] Input stats: mean={x.float().mean():.6f}, std={x.float().std():.6f}")

        # 调用父类的forward方法
        result = super().forward(*args, **kwargs)

        if isinstance(result, tuple):
            print(f"[DEBUG RMSNorm FORWARD] Output shapes: {[r.shape if hasattr(r, 'shape') else type(r) for r in result]}")
        else:
            print(f"[DEBUG RMSNorm FORWARD] Output shape: {result.shape if hasattr(result, 'shape') else type(result)}")

        return result

    def forward_oot(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        import torch_npu

        # DEBUG: 详细信息追踪精度问题 - 新版本（对称老版本）
        from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type
        print(f"[DEBUG RMSNorm NEW] forward_oot called!")
        print(f"[DEBUG RMSNorm NEW] Device: {'310P' if get_ascend_device_type() == AscendDeviceType._310P else 'Other'}")
        print(f"[DEBUG RMSNorm NEW] Input x shape: {x.shape}, dtype: {x.dtype}")
        print(f"[DEBUG RMSNorm NEW] Input x stats: mean={x.float().mean():.6f}, std={x.float().std():.6f}")
        print(f"[DEBUG RMSNorm NEW] Input x min/max: {x.min():.6f}/{x.max():.6f}")
        print(f"[DEBUG RMSNorm NEW] Weight shape: {self.weight.shape}, dtype: {self.weight.dtype}")
        print(f"[DEBUG RMSNorm NEW] Weight stats: mean={self.weight.float().mean():.6f}, std={self.weight.float().std():.6f}")
        print(f"[DEBUG RMSNorm NEW] Variance epsilon: {self.variance_epsilon}")

        if residual is not None:
            print(f"[DEBUG RMSNorm NEW] Residual shape: {residual.shape}, dtype: {residual.dtype}")
            print(f"[DEBUG RMSNorm NEW] Residual stats: mean={residual.float().mean():.6f}, std={residual.float().std():.6f}")
            print(f"[DEBUG RMSNorm NEW] Residual min/max: {residual.min():.6f}/{residual.max():.6f}")
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

            # DEBUG: 输出统计信息 - 新版本（对称老版本）
            print(f"[DEBUG RMSNorm NEW] Output x stats: mean={x.float().mean():.6f}, std={x.float().std():.6f}")
            print(f"[DEBUG RMSNorm NEW] Output x min/max: {x.min():.6f}/{x.max():.6f}")
            print(f"[DEBUG RMSNorm NEW] Output residual stats: mean={residual.float().mean():.6f}, std={residual.float().std():.6f}")
            print(f"[DEBUG RMSNorm NEW] Output residual min/max: {residual.min():.6f}/{residual.max():.6f}")

            return x, residual

        x, residual = torch_npu.npu_rms_norm(x, self.weight,
                                             self.variance_epsilon)
        if self.bias is not None:
            x.add_(self.bias)

        # DEBUG: 输出统计信息 - 新版本（对称老版本）
        print(f"[DEBUG RMSNorm NEW] Output x stats: mean={x.float().mean():.6f}, std={x.float().std():.6f}")
        print(f"[DEBUG RMSNorm NEW] Output x min/max: {x.min():.6f}/{x.max():.6f}")

        return x


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

        # DEBUG: 简单检查AscendRMSNorm是否被调用
        from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type
        print(f"[DEBUG] AscendRMSNorm.forward_oot called! Device: {get_ascend_device_type()}")
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
