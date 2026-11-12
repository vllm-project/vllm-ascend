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
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

from vllm_ascend.device.device_op import DeviceOperator
from vllm_ascend.ops.triton.layernorm_gated import layer_norm_fwd_npu
from vllm_ascend.utils import enable_custom_op

# Scanning quant_description is O(number of quantized tensors) and every
# RMSNorm used to redo it. The answer is a property of the checkpoint, so
# cache it per quant_description object. The dict is kept in the entry so its
# id cannot be recycled by a later allocation, and the identity check makes a
# stale entry harmless either way.
_NORM_BIAS_IN_QUANT_DESCRIPTION: dict[int, tuple[dict, bool]] = {}


def _quant_description_has_norm_bias(quant_description: dict) -> bool:
    if not quant_description:
        return False
    cache_key = id(quant_description)
    cached = _NORM_BIAS_IN_QUANT_DESCRIPTION.get(cache_key)
    if cached is not None and cached[0] is quant_description:
        return cached[1]
    has_norm_bias = any("norm.bias" in name for name in quant_description)
    _NORM_BIAS_IN_QUANT_DESCRIPTION[cache_key] = (quant_description, has_norm_bias)
    return has_norm_bias


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

        # quantization with anti_method m4 will generate none-zero norm bias
        quant_description = getattr(getattr(vllm_config, "quant_config", None), "quant_description", None) or {}
        if _quant_description_has_norm_bias(quant_description):
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
        self.bias_loaded = True

    def forward_oot(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        import torch_npu

        if residual is not None:
            if enable_custom_op():
                x, _, residual = torch.ops._C_ascend.npu_add_rms_norm_bias(
                    x, residual, self.weight, self.bias, self.variance_epsilon
                )
            else:
                # Under VLLM_BATCH_INVARIANT, batch_invariant.py patches npu_add_rms_norm
                # for bitwise batch invariance; bias is added elementwise below.
                x, _, residual = torch_npu.npu_add_rms_norm(x, residual, self.weight, self.variance_epsilon)
                if self.bias is not None:
                    x.add_(self.bias)
            return x, residual

        x, residual = torch_npu.npu_rms_norm(x, self.weight, self.variance_epsilon)
        if self.bias_loaded:
            x.add_(self.bias)

        return x


class AscendGemmaRMSNorm(GemmaRMSNorm):
    _effective_weight: torch.Tensor

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__(hidden_size, eps)
        # Allocate on every rank so parameter/buffer transfer manifests agree
        # even when a receiver bypasses the checkpoint loader. Such receivers
        # keep the runtime addition until a supported loader prepares the cache.
        self.register_buffer("_effective_weight", torch.ones_like(self.weight), persistent=False)
        self._effective_weight_ready = False
        self.weight.weight_loader = self._weight_loader

    def register_buffer(self, name: str, tensor: torch.Tensor | None, persistent: bool = True) -> None:
        super().register_buffer(name, tensor, persistent=persistent)
        # Native layerwise reload restores parameters before re-registering the
        # original buffers, but skips copying derived nonpersistent buffers.
        # Refresh the restored cache in place; meta staging is not usable yet.
        if (
            name == "_effective_weight"
            and getattr(self, "_effective_weight_ready", False)
            and tensor is not None
            and not tensor.is_meta
            and not self.weight.is_meta
        ):
            self._refresh_effective_weight()

    @torch.inference_mode()
    def _refresh_effective_weight(self) -> None:
        # Add in the parameter dtype, preserving the original BF16/FP16
        # rounding. Refresh in place so captured graphs retain the same address.
        value = 1.0 + self.weight
        cached = self._effective_weight
        if (
            cached is not None
            and cached.device == value.device
            and cached.dtype == value.dtype
            and cached.shape == value.shape
        ):
            cached.copy_(value)
        else:
            self._effective_weight = value
        self._effective_weight_ready = True

    def _weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor) -> None:
        default_weight_loader(param, loaded_weight)
        self._refresh_effective_weight()

    def _load_from_state_dict(self, *args, **kwargs):
        super()._load_from_state_dict(*args, **kwargs)
        # assign=True can replace the parameter, including its loader attribute.
        self.weight.weight_loader = self._weight_loader
        self._refresh_effective_weight()

    def _apply(self, fn, recurse=True):
        prepared = self._effective_weight_ready
        result = super()._apply(fn, recurse=recurse)
        self.weight.weight_loader = self._weight_loader
        if prepared:
            # Casting a previously rounded (1 + weight) differs from adding
            # after casting weight. Recompute after device/dtype conversion.
            self._refresh_effective_weight()
        return result

    @property
    def effective_weight(self) -> torch.Tensor:
        if self._effective_weight_ready and not torch.is_grad_enabled():
            return self._effective_weight
        return 1.0 + self.weight

    def forward_oot(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        import torch_npu

        if residual is not None:
            x, _, residual = torch_npu.npu_add_rms_norm(x, residual, self.effective_weight, self.variance_epsilon)
            return x, residual

        x = DeviceOperator.npu_gemma_rms_norm(
            x,
            self.weight,
            self.variance_epsilon,
            effective_weight=(
                self._effective_weight if self._effective_weight_ready and not torch.is_grad_enabled() else None
            ),
        )

        return x


def get_gemma_rms_norm_weight(norm: GemmaRMSNorm) -> torch.Tensor:
    """Share loaded effective weights with fused Q/K normalization consumers."""
    if isinstance(norm, AscendGemmaRMSNorm):
        return norm.effective_weight
    return 1.0 + norm.weight


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
            activation=activation,
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
        return LayerNormFn.apply(
            x,
            self.weight,
            self.bias,
            z,
            self.eps,
            self.group_size,
            self.norm_before_gate,
            True,
            self.activation,
        )


def __getattr__(name: str):
    # Keep the existing class import available without loading FLA during
    # ordinary layernorm discovery. Python serializes the lazy module import.
    # TODO: Remove this lazy accessor when the class moves back here after
    # supported vLLM FLA or Triton Ascend versions fix import-time NPU init.
    if name == "AscendFusedRMSNormGated":
        from vllm_ascend.utils import register_ascend_fla_customop

        return register_ascend_fla_customop()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
