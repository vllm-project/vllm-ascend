#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
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
#
"""GPTQ quantization configuration for Ascend NPU.

This module provides the GPTQConfig class for parsing GPTQ quantization
configs and creating appropriate quantization schemes for Ascend hardware.

Reference: https://arxiv.org/abs/2210.17323
"""

from typing import Any

import torch
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.layers.quantization.gptq import GPTQConfig

from vllm_ascend.utils import GPTQ_METHOD

logger = init_logger(__name__)


@register_quantization_config(GPTQ_METHOD)
class AscendGPTQConfig(GPTQConfig):
    """Config class for GPTQ quantization on Ascend NPU.

    Inherits from vLLM's GPTQConfig and overrides methods specific to Ascend NPU.
    GPTQ is a post-training quantization method that quantizes model weights
    to low-bit representations (4/8 bits) to reduce model size and
    inference latency.

    Attributes:
        weight_bits: Number of bits for weight quantization (4 or 8).
        group_size: Size of quantization groups. -1 means per-channel.
        desc_act: Whether to use activation reordering (not supported on NPU).
        checkpoint_format: Format of the checkpoint ("gptq" or "gptq_v2").
    """

    def __init__(
        self,
        weight_bits: int,
        group_size: int,
        desc_act: bool,
        lm_head_quantized: bool = False,
        dynamic: dict[str, dict[str, int | bool]] | None = None,
        autoround_version: str = "",
        modules_in_block_to_quantize: list[str] | None = None,
        checkpoint_format: str = "gptq",
    ) -> None:
        """Initialize GPTQ configuration for Ascend NPU.

        Args:
            weight_bits: Number of bits for weight quantization (4 or 8).
            group_size: Size of quantization groups.
            desc_act: Whether to use activation reordering.
            lm_head_quantized: Whether the language model head is quantized.
            dynamic: Per-module quantization config (from GPTQModel).
            autoround_version: Version string for autoround quantization.
            modules_in_block_to_quantize: List of module names to quantize.
            checkpoint_format: Format of the checkpoint.

        Raises:
            ValueError: If weight_bits is not in [4, 8].
            ValueError: If desc_act is True (not supported on NPU).
        """
        # Validate Ascend NPU specific constraints before calling parent
        if weight_bits not in [4, 8]:
            raise ValueError(
                f"Currently, only 4/8-bit weight quantization is "
                f"supported for GPTQ on Ascend NPU, but got {weight_bits} bits."
            )

        if desc_act:
            raise ValueError(
                "desc_act=True is not supported for GPTQ on Ascend NPU. "
                "Please use desc_act=False in your quantization config."
            )

        # Initialize parent class with all parameters
        super().__init__(
            weight_bits=weight_bits,
            group_size=group_size,
            desc_act=desc_act,
            lm_head_quantized=lm_head_quantized,
            dynamic=dynamic or {},
            autoround_version=autoround_version,
            modules_in_block_to_quantize=modules_in_block_to_quantize,
            checkpoint_format=checkpoint_format,
        )

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        """Get supported activation data types for Ascend NPU."""
        return [torch.half, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        """Get minimum hardware capability.

        Raises:
            NotImplementedError: NPU hardware does not use capability numbers.
        """
        raise NotImplementedError('NPU hardware does not support "get_min_capability" feature.')

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "AscendGPTQConfig":
        """Create a AscendGPTQConfig from a config dictionary.

        Args:
            config: Configuration dictionary from quantize_config.json.

        Returns:
            AscendGPTQConfig instance for Ascend NPU.
        """
        # Extract all parameters from config
        weight_bits = cls.get_from_keys(config, ["bits", "weight_bits"])
        group_size = cls.get_from_keys(config, ["group_size"])
        desc_act = cls.get_from_keys(config, ["desc_act"])
        lm_head_quantized = cls.get_from_keys_or(config, ["lm_head"], default=False)
        dynamic = cls.get_from_keys_or(config, ["dynamic"], default={})
        dynamic = {} if dynamic is None else dynamic
        autoround_version = cls.get_from_keys_or(config, ["autoround_version"], default="")
        modules_in_block_to_quantize = cls.get_from_keys_or(config, ["modules_in_block_to_quantize"], default=None)
        checkpoint_format = cls.get_from_keys_or(config, ["checkpoint_format"], default="gptq")

        return cls(
            weight_bits=weight_bits,
            group_size=group_size,
            desc_act=desc_act,
            lm_head_quantized=lm_head_quantized,
            dynamic=dynamic,
            autoround_version=autoround_version,
            modules_in_block_to_quantize=modules_in_block_to_quantize,
            checkpoint_format=checkpoint_format,
        )

    def get_quant_method(self, layer: torch.nn.Module, prefix: str) -> QuantizeMethodBase | None:
        """Get the quantization method for a specific layer on Ascend NPU.

        Args:
            layer: The layer to quantize.
            prefix: The layer's prefix in the model.

        Returns:
            Quantization method instance, or None if layer is not supported.

        Raises:
            NotImplementedError: If layer is FusedMoE (not supported).
        """
        from .methods.gptq import AscendGPTQLinearMethod

        if isinstance(layer, LinearBase):
            logger.info_once("Using the vLLM Ascend GPTQ Quantization now!")
            # Return GPTQ method directly without wrapping
            return AscendGPTQLinearMethod(self)

        elif isinstance(layer, FusedMoE):
            raise NotImplementedError("GPTQ quantization does not support MoE layers on Ascend NPU.")

        return None

    def get_scaled_act_names(self) -> list[str]:
        """Get activation function names that should be post-scaled.

        Returns:
            Empty list (not used by GPTQ).
        """
        return []
