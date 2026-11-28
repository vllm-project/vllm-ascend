# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Online formats for Ascend draft LM head copies."""

from __future__ import annotations

from typing import TYPE_CHECKING

from vllm.model_executor.layers.quantization.base_config import QuantizationConfig

from vllm_ascend.device.hardware_profile import DeviceAdaptorFamily, get_current_hardware_profile

if TYPE_CHECKING:
    from vllm_ascend.quantization.method_adapters import AscendEmbeddingMethod


class AscendDraftLMHeadQuantizationConfig(QuantizationConfig):
    """Select a draft copy format independently of checkpoint quantization."""

    @classmethod
    def _draft_lm_head_methods(cls) -> dict[str, str | None]:
        family = get_current_hardware_profile().device_adaptor_family
        # Claim the core names so 310P reports no supported formats.
        methods: dict[str, str | None] = {"fp8": None, "nvfp4": None}
        if family is DeviceAdaptorFamily.FP8_OPTIMIZED:
            methods["mxfp8"] = f"{__name__}.create_mxfp8_lm_head_method"
        elif family is DeviceAdaptorFamily.STANDARD:
            methods["int8"] = f"{__name__}.create_int8_lm_head_method"
        return methods


def create_mxfp8_lm_head_method() -> AscendEmbeddingMethod:
    """Quantize the draft copy into per-group MXFP8 (A5-class devices)."""
    from vllm_ascend.quantization.method_adapters import AscendEmbeddingMethod
    from vllm_ascend.quantization.methods.w8a8.w8a8_mxfp8 import AscendW8A8MXFP8DynamicLmHeadMethod

    return AscendEmbeddingMethod(AscendW8A8MXFP8DynamicLmHeadMethod())


def create_int8_lm_head_method() -> AscendEmbeddingMethod:
    """Quantize the draft copy into per-channel INT8 (A2/A3-class devices)."""
    from vllm_ascend.quantization.method_adapters import AscendEmbeddingMethod
    from vllm_ascend.quantization.methods.w8a8.w8a8_dynamic import AscendW8A8DynamicLmHeadMethod

    return AscendEmbeddingMethod(AscendW8A8DynamicLmHeadMethod())
