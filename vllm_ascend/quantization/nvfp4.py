# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""NVFP4 quantization support for Ascend NPU.

The serialized layout follows the ModelOpt NVFP4 convention used by
Mistral-Large-3-675B: two E2M1 values are packed into one byte, every group
of 16 values has an FP8 E4M3 scale, and every tensor has an FP32 secondary
scale.  The CPU implementation is a correctness reference; production NPU
execution is dispatched to optional vLLM Ascend custom operators.
"""

from __future__ import annotations

import re
from fnmatch import fnmatch
from typing import Any, Optional

import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from vllm.model_executor.layers.fused_moe import FusedMoEMethodBase, MoERunner
from vllm.model_executor.layers.linear import (
    LinearBase,
    LinearMethodBase,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization import (
    QUANTIZATION_METHODS,
    register_quantization_config,
)
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.parameter import ModelWeightParameter, PerTensorScaleParameter

NVFP4_METHOD = "nvfp4"
NVFP4_GROUP_SIZE = 16
NVFP4_PACK_FACTOR = 2
_NVFP4_E2M1_VALUES = (
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
)


def unpack_nvfp4(packed_weight: torch.Tensor) -> torch.Tensor:
    """Unpack uint8 E2M1 pairs to float32, low nibble first."""
    if packed_weight.dtype != torch.uint8:
        raise TypeError(
            f"NVFP4 packed weight must be uint8, got {packed_weight.dtype}."
        )

    low = packed_weight & 0x0F
    high = packed_weight >> 4
    indices = torch.stack((low, high), dim=-1).reshape(*packed_weight.shape[:-1], -1)
    value_table = torch.tensor(
        _NVFP4_E2M1_VALUES, dtype=torch.float32, device=packed_weight.device
    )
    return value_table[indices.to(torch.long)]


def dequantize_nvfp4(
    packed_weight: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_scale_2: torch.Tensor,
    group_size: int = NVFP4_GROUP_SIZE,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Reference dequantization for packed NVFP4 weights.

    ``weight_scale_2`` is the ModelOpt multiplier (amax / 2688), not its
    reciprocal.  A scalar or one value per leading tensor dimension is
    accepted.  This helper intentionally contains no host synchronization.
    """
    if group_size <= 0:
        raise ValueError(f"group_size must be positive, got {group_size}.")

    unpacked = unpack_nvfp4(packed_weight)
    if unpacked.shape[-1] % group_size != 0:
        raise ValueError(
            f"NVFP4 logical input size {unpacked.shape[-1]} must be divisible by group_size {group_size}."
        )
    expected_scale_shape = (*unpacked.shape[:-1], unpacked.shape[-1] // group_size)
    if tuple(weight_scale.shape) != expected_scale_shape:
        raise ValueError(
            f"NVFP4 weight_scale shape must be {expected_scale_shape}, got {tuple(weight_scale.shape)}."
        )

    block_scale = weight_scale.to(torch.float32).repeat_interleave(group_size, dim=-1)
    global_scale = weight_scale_2.to(torch.float32)
    if global_scale.numel() == 1:
        pass
    elif tuple(global_scale.shape) == tuple(unpacked.shape[:-1]):
        global_scale = global_scale.unsqueeze(-1)
    else:
        raise ValueError(
            "NVFP4 weight_scale_2 must be scalar or match the leading weight dimensions; "
            f"got {tuple(global_scale.shape)} for weight {tuple(unpacked.shape)}."
        )
    return (unpacked * block_scale * global_scale).to(dtype)


def _get_ascend_nvfp4_op(name: str):
    """Return an optional Ascend NVFP4 custom op without importing torch_npu."""
    namespace = getattr(torch.ops, "_C_ascend", None)
    return getattr(namespace, name, None) if namespace is not None else None


def _is_npu_tensor(tensor: torch.Tensor) -> bool:
    return tensor.device.type in ("npu", "privateuseone")


class AscendNvFp4LinearMethod(LinearMethodBase):
    """NVFP4 linear method with an Ascend-op dispatch and CPU reference path."""

    def __init__(self, quant_config: "AscendNvFp4Config") -> None:
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs: Any,
    ) -> None:
        del input_size, output_size, params_dtype
        if input_size_per_partition % self.quant_config.group_size != 0:
            raise ValueError(
                "NVFP4 input size must be divisible by "
                f"{self.quant_config.group_size}, got {input_size_per_partition}."
            )

        output_size_per_partition = sum(output_partition_sizes)
        weight_loader = extra_weight_attrs.get("weight_loader")
        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition

        weight = ModelWeightParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition // NVFP4_PACK_FACTOR,
                dtype=torch.uint8,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)
        weight_scale = ModelWeightParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition // self.quant_config.group_size,
                dtype=torch.float8_e4m3fn,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight_scale", weight_scale)
        for name in ("input_scale", "weight_scale_2"):
            scale = PerTensorScaleParameter(
                data=torch.empty(len(output_partition_sizes), dtype=torch.float32),
                weight_loader=weight_loader,
            )
            layer.register_parameter(name, scale)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # Fused projections need one scale for the initial operator skeleton.
        # Taking the maximum matches upstream ModelOpt's conservative behavior.
        layer.input_global_scale = Parameter(
            layer.input_scale.max().to(torch.float32), requires_grad=False
        )
        layer.weight_global_scale = Parameter(
            layer.weight_scale_2.max().to(torch.float32), requires_grad=False
        )
        layer.alpha = Parameter(
            layer.input_global_scale * layer.weight_global_scale,
            requires_grad=False,
        )
        layer.input_global_scale_inv = Parameter(
            (1.0 / layer.input_global_scale).to(torch.float32),
            requires_grad=False,
        )
        layer.weight.data = layer.weight.data.contiguous()
        layer.weight_scale.data = layer.weight_scale.data.contiguous()

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        npu_op = _get_ascend_nvfp4_op("nvfp4_linear")
        if npu_op is not None:
            return npu_op(
                x,
                layer.weight,
                layer.weight_scale,
                layer.weight_global_scale,
                layer.input_global_scale,
                bias,
            )
        if _is_npu_tensor(x):
            raise RuntimeError(
                "Ascend NVFP4 linear operator is unavailable. Build the extension "
                "with torch.ops._C_ascend.nvfp4_linear support."
            )

        # CPU-only correctness fallback. Never expand a 675B checkpoint on NPU.
        weight = dequantize_nvfp4(
            layer.weight,
            layer.weight_scale,
            layer.weight_global_scale,
            group_size=self.quant_config.group_size,
            dtype=x.dtype,
        )
        return F.linear(x, weight, bias)


class AscendNvFp4FusedMoEMethod(FusedMoEMethodBase):
    """Mistral-Large-3 NVFP4 MoE weight loader and NPU-op skeleton."""

    def __init__(self, quant_config: "AscendNvFp4Config", moe_config: Any) -> None:
        super().__init__(moe_config)
        self.quant_config = quant_config

    def uses_weight_scale_2_pattern(self) -> bool:
        return True

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs: Any,
    ) -> None:
        del params_dtype
        group_size = self.quant_config.group_size
        if (
            hidden_size % group_size != 0
            or intermediate_size_per_partition % group_size != 0
        ):
            raise ValueError(
                "NVFP4 MoE hidden and intermediate sizes must be divisible by 16."
            )

        weight_loader = extra_weight_attrs.get("weight_loader")

        def register_weight(
            name: str, shape: tuple[int, ...], dtype: torch.dtype
        ) -> None:
            param = ModelWeightParameter(
                data=torch.empty(shape, dtype=dtype),
                input_dim=2,
                output_dim=1,
                weight_loader=weight_loader,
            )
            layer.register_parameter(name, param)

        register_weight(
            "w13_weight",
            (
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size // NVFP4_PACK_FACTOR,
            ),
            torch.uint8,
        )
        register_weight(
            "w2_weight",
            (
                num_experts,
                hidden_size,
                intermediate_size_per_partition // NVFP4_PACK_FACTOR,
            ),
            torch.uint8,
        )
        register_weight(
            "w13_weight_scale",
            (
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size // group_size,
            ),
            torch.float8_e4m3fn,
        )
        register_weight(
            "w2_weight_scale",
            (num_experts, hidden_size, intermediate_size_per_partition // group_size),
            torch.float8_e4m3fn,
        )

        scale_shapes = {
            "w13_weight_scale_2": (num_experts, 2),
            "w2_weight_scale_2": (num_experts,),
            "w13_input_scale": (num_experts, 2),
            "w2_input_scale": (num_experts,),
        }
        for name, shape in scale_shapes.items():
            scale = PerTensorScaleParameter(
                data=torch.empty(shape, dtype=torch.float32),
                weight_loader=weight_loader,
            )
            layer.register_parameter(name, scale)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # Preserve the serialized layout until the fused NPU kernel converts it.
        for name in (
            "w13_weight",
            "w2_weight",
            "w13_weight_scale",
            "w2_weight_scale",
            "w13_weight_scale_2",
            "w2_weight_scale_2",
            "w13_input_scale",
            "w2_input_scale",
        ):
            getattr(layer, name).data = getattr(layer, name).data.contiguous()

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        **kwargs: Any,
    ) -> torch.Tensor:
        npu_op = _get_ascend_nvfp4_op("nvfp4_moe")
        if npu_op is None:
            raise RuntimeError(
                "Ascend NVFP4 MoE operator is unavailable. Build the extension "
                "with torch.ops._C_ascend.nvfp4_moe support."
            )
        return npu_op(
            x,
            router_logits,
            layer.w13_weight,
            layer.w2_weight,
            layer.w13_weight_scale,
            layer.w2_weight_scale,
            layer.w13_weight_scale_2,
            layer.w2_weight_scale_2,
            layer.w13_input_scale,
            layer.w2_input_scale,
            top_k,
            renormalize,
        )

    def get_fused_moe_quant_config(self, layer: torch.nn.Module):
        del layer
        return None


def _remove_existing_registration() -> None:
    if NVFP4_METHOD in QUANTIZATION_METHODS:
        QUANTIZATION_METHODS.remove(NVFP4_METHOD)


_remove_existing_registration()


@register_quantization_config(NVFP4_METHOD)
class AscendNvFp4Config(QuantizationConfig):
    """Configuration for serialized Mistral/ModelOpt NVFP4 checkpoints."""

    def __init__(
        self, group_size: int = NVFP4_GROUP_SIZE, ignore: list[str] | None = None
    ) -> None:
        super().__init__()
        if group_size != NVFP4_GROUP_SIZE:
            raise ValueError(
                f"Ascend NVFP4 currently requires group_size={NVFP4_GROUP_SIZE}, got {group_size}."
            )
        self.group_size = group_size
        self.ignore = ignore or []

    @classmethod
    def get_name(cls) -> str:
        return NVFP4_METHOD

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.float16, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        raise NotImplementedError(
            'Ascend hardware does not use CUDA "min capability" checks.'
        )

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        # Mistral native checkpoints embed this data in params.json.
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "AscendNvFp4Config":
        quant_config = config.get("quantization_config", config)
        quant_config = quant_config.get("quantization", quant_config)
        group_size = quant_config.get("group_size")
        if group_size is None:
            config_groups = quant_config.get("config_groups", {})
            nvfp4_group = config_groups.get("NVFP4", {})
            weight_config = nvfp4_group.get("weights", {})
            group_size = weight_config.get("group_size", NVFP4_GROUP_SIZE)
        group_size = int(group_size)
        ignore = quant_config.get(
            "exclude_modules", quant_config.get("ignore", config.get("ignore", []))
        )
        if not isinstance(ignore, list):
            raise ValueError(
                f"NVFP4 ignore/exclude_modules must be a list, got {type(ignore)}."
            )
        return cls(group_size=group_size, ignore=ignore)

    @classmethod
    def override_quantization_method(
        cls,
        hf_quant_cfg: dict[str, Any] | None,
        user_quant: str | None,
        hf_config: Any | None = None,
    ) -> Optional[str]:
        del hf_config
        if user_quant == NVFP4_METHOD:
            return NVFP4_METHOD
        if not hf_quant_cfg:
            return None
        quant_config = hf_quant_cfg.get("quantization_config", hf_quant_cfg)
        quant_config = quant_config.get("quantization", quant_config)
        quant_algo = str(quant_config.get("quant_algo", "")).upper()
        quant_method = str(quant_config.get("quant_method", "")).lower()
        quant_format = str(quant_config.get("format", "")).lower()
        config_groups = quant_config.get("config_groups", {})
        has_nvfp4_group = any(
            str(group.get("format", "")).lower() == "nvfp4-pack-quantized"
            for group in config_groups.values()
        )
        if (
            "NVFP4" in quant_algo
            or quant_method == NVFP4_METHOD
            or quant_format == "nvfp4-pack-quantized"
            or has_nvfp4_group
        ):
            return NVFP4_METHOD
        return None

    def _is_ignored(self, prefix: str) -> bool:
        for pattern in self.ignore:
            if pattern.startswith("re:"):
                if re.fullmatch(pattern.removeprefix("re:"), prefix):
                    return True
            elif fnmatch(prefix, pattern):
                return True
        return False

    def get_quant_method(
        self,
        layer: torch.nn.Module,
        prefix: str,
        tid2eid: Any | None = None,
    ) -> Optional[QuantizeMethodBase]:
        del tid2eid
        if self._is_ignored(prefix):
            return UnquantizedLinearMethod() if isinstance(layer, LinearBase) else None
        if isinstance(layer, LinearBase):
            layer.ascend_quant_method = NVFP4_METHOD
            return AscendNvFp4LinearMethod(self)
        if isinstance(layer, MoERunner):
            layer.ascend_quant_method = NVFP4_METHOD
            return AscendNvFp4FusedMoEMethod(self, layer.moe_config)
        return None


__all__ = [
    "AscendNvFp4Config",
    "AscendNvFp4FusedMoEMethod",
    "AscendNvFp4LinearMethod",
    "NVFP4_GROUP_SIZE",
    "NVFP4_METHOD",
    "dequantize_nvfp4",
    "unpack_nvfp4",
]
