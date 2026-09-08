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


import torch
from torch import nn
from vllm.config import get_current_vllm_config
from vllm.model_executor.layers.layernorm import GemmaRMSNorm, RMSNorm, RMSNormGated
from vllm.third_party.flash_linear_attention.ops.kda import FusedRMSNormGated

from vllm_ascend import envs
from vllm_ascend.device.device_op import DeviceOperator
from vllm_ascend.ops.triton.kda.kda import rms_norm_gated
from vllm_ascend.ops.triton.layernorm_gated import layer_norm_fwd_npu
from vllm_ascend.utils import (
    AscendDeviceType,
    bootstrap_custom_op_env,
    enable_custom_op,
    get_ascend_device_type,
)


def _enable_a5_add_rms_norm_bias(x: torch.Tensor) -> bool:
    if not envs.VLLM_ASCEND_ENABLE_ADD_RMS_NORM_BIAS:
        return False
    if get_ascend_device_type() != AscendDeviceType.A5:
        return False
    import vllm.envs as vllm_envs

    if vllm_envs.VLLM_BATCH_INVARIANT:
        return False
    if x.shape[-1] == 0 or x.shape[-1] > 6144 or x.shape[-1] % 16:
        return False
    bootstrap_custom_op_env(include_vendor_lib=True)
    # Explicit opt-in must report a missing build instead of silently benchmarking
    # the baseline. Keep the global custom-op enablement unchanged on A5.
    import vllm_ascend.vllm_ascend_C  # noqa: F401

    return True


def _add_bias_with_axpy(x: torch.Tensor, negative_bias: torch.Tensor) -> torch.Tensor:
    """Add bias through aclnnAdd's Axpy lowering without changing numerics."""
    return x.add_(negative_bias, alpha=-1.0)


class AscendRMSNorm(RMSNorm):
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        var_hidden_size: int | None = None,
        has_weight: bool = True,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(hidden_size, eps, var_hidden_size, has_weight, dtype)
        vllm_config = get_current_vllm_config()
        self.bias = None
        self.bias_loaded = False
        self.register_buffer("_negative_bias", None, persistent=False)

        # quantization with anti_method m4 will generate none-zero norm bias
        quant_description = getattr(vllm_config.quant_config, "quant_description", None) or {}
        if any("norm.bias" in name for name in quant_description):
            self.bias = torch.nn.Parameter(torch.zeros(hidden_size), requires_grad=False)
            self.bias.weight_loader = self._bias_weight_loader

    def _bias_weight_loader(self, param: torch.nn.Parameter, loaded_weight: torch.Tensor) -> None:
        if param.numel() == 1 and loaded_weight.numel() == 1:
            # Sometimes scalar values aren't considered tensors with shapes
            # so if both param and loaded_weight are a scalar,
            # "broadcast" instead of copy
            param.data.fill_(loaded_weight.item())
        else:
            assert param.size() == loaded_weight.size(), (
                f"Attempted to load weight ({loaded_weight.size()}) into parameter ({param.size()})"
            )

            param.data.copy_(loaded_weight)
        # CANN lowers aclnnAdd with alpha=-1 to Axpy. Cache the negation at
        # weight-load time so the forward path remains exactly x + bias and
        # does not introduce a runtime Neg kernel.
        self._negative_bias = -param.data
        self.bias_loaded = True

    def forward_oot(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        import torch_npu

        if residual is not None:
            if _enable_a5_add_rms_norm_bias(x) or enable_custom_op():
                x, _, residual = torch.ops._C_ascend.npu_add_rms_norm_bias(
                    x, residual, self.weight, self.bias, self.variance_epsilon
                )
            else:
                x, _, residual = torch_npu.npu_add_rms_norm(x, residual, self.weight, self.variance_epsilon)
                if self.bias is not None:
                    assert self._negative_bias is not None
                    _add_bias_with_axpy(x, self._negative_bias)
            return x, residual

        x, residual = torch_npu.npu_rms_norm(x, self.weight, self.variance_epsilon)
        if self.bias_loaded:
            assert self._negative_bias is not None
            _add_bias_with_axpy(x, self._negative_bias)

        return x


class AscendGemmaRMSNorm(GemmaRMSNorm):
    def forward_oot(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        import torch_npu

        if residual is not None:
            if _enable_a5_add_rms_norm_bias(x) or enable_custom_op():
                x, _, residual = torch.ops._C_ascend.npu_add_rms_norm_bias(
                    x, residual, 1.0 + self.weight, None, self.variance_epsilon
                )
            else:
                x, _, residual = torch_npu.npu_add_rms_norm(x, residual, 1.0 + self.weight, self.variance_epsilon)
            return x, residual

        x = DeviceOperator.npu_gemma_rms_norm(x, self.weight, self.variance_epsilon)

        return x


class LayerNormFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x,
        weight,
        bias,
        z=None,
        eps=1e-6,
        group_size=None,
        norm_before_gate=True,
        is_rms_norm=False,
        activation: str = "swish",
    ):
        """If z is not None, we do norm(x) * silu(z) if norm_before_gate, else norm(x * silu(z))"""

        x_shape_og = x.shape
        # reshape input data into 2D tensor
        x = x.reshape(-1, x.shape[-1])
        if x.stride(-1) != 1:
            x = x.contiguous()
        if z is not None:
            assert z.shape == x_shape_og
            z = z.reshape(-1, z.shape[-1])
            if z.stride(-1) != 1:
                z = z.contiguous()
        weight = weight.contiguous()
        if bias is not None:
            bias = bias.contiguous()
        y, mean, rstd = layer_norm_fwd_npu(
            x,
            weight,
            bias,
            eps,
            z=z,
            group_size=group_size,
            norm_before_gate=norm_before_gate,
            is_rms_norm=is_rms_norm,
        )
        ctx.save_for_backward(x, weight, bias, mean, rstd, z)
        ctx.x_shape_og = x_shape_og
        ctx.eps = eps
        ctx.group_size = group_size
        ctx.norm_before_gate = norm_before_gate
        ctx.is_rms_norm = is_rms_norm
        return y.reshape(x_shape_og)


class AscendRMSNormGated(RMSNormGated):
    def __init__(
        self,
        hidden_size,
        eps: float = 1e-5,
        group_size: int | None = None,
        norm_before_gate: bool = False,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        # `activation` was added in vLLM #40245 (Qwen3-Next/GDN). Accept and
        # forward it; older vllm versions did not pass this kwarg so the
        # default keeps existing behavior.
        activation: str = "swish",
    ):
        """If group_size is not None, we do GroupNorm with each group having group_size elements.
        group_size=None is equivalent to group_size=hidden_size (i.e. there's only 1 group).
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(
            hidden_size,
            eps,
            group_size,
            norm_before_gate,
            device,
            dtype,
            activation=activation,
        )
        self.eps = eps
        self.weight = nn.Parameter(torch.empty(hidden_size, **factory_kwargs))
        self.register_parameter("bias", None)
        self.group_size = group_size
        self.norm_before_gate = norm_before_gate
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.ones_(self.weight)

    def forward_oot(self, x, z=None):
        """If z is not None, we do norm(x) * silu(z) if norm_before_gate, else norm(x * silu(z))"""
        return LayerNormFn.apply(x, self.weight, self.bias, z, self.eps, self.group_size, self.norm_before_gate, True)


class AscendFusedRMSNormGated(FusedRMSNormGated):
    """Use Ascend's fused kernel at the upstream FLA CustomOp boundary."""

    def forward_oot(self, x, g, residual=None, prenorm=False, residual_in_fp32=False):
        return rms_norm_gated(
            x,
            g,
            self.weight,
            self.bias,
            self.activation,
            residual=residual,
            eps=self.eps,
            prenorm=prenorm,
            residual_in_fp32=residual_in_fp32,
        )
