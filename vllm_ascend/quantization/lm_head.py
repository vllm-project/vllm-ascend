# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Select the target LM head method independently of checkpoint quantization."""

from typing import TYPE_CHECKING

import torch
from vllm.logger import logger
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig, QuantizeMethodBase
from vllm.model_executor.layers.vocab_parallel_embedding import UnquantizedEmbeddingMethod

from vllm_ascend.device.hardware_profile import DeviceAdaptorFamily, get_current_hardware_profile

if TYPE_CHECKING:
    from vllm.config import VllmConfig


def select_lm_head_method(
    layer: torch.nn.Module,
    prefix: str,
    quant_config: QuantizationConfig | None,
    checkpoint_method: QuantizeMethodBase | None,
    vllm_config: "VllmConfig",
) -> QuantizeMethodBase | None:
    """Preserve checkpoint dispatch unless the target output head is selected."""
    additional_config = vllm_config.additional_config or {}
    quantization = additional_config.get("lm_head_quantization")
    if quantization is None:
        return checkpoint_method
    if vllm_config.model_config.runner_type == "draft" or prefix.rsplit(".", 1)[-1] != "lm_head":
        return checkpoint_method
    if not vllm_config.use_v2_model_runner:
        raise ValueError("lm_head_quantization requires Model Runner V2.")

    device_family = get_current_hardware_profile().device_adaptor_family
    if device_family == DeviceAdaptorFamily.COMPATIBILITY:
        raise ValueError("Online LM head quantization is not yet adapted for 310P.")

    # Imports are local to avoid pulling every checkpoint format into normal
    # unquantized model initialization and creating quantization import cycles.
    from vllm_ascend.quantization.configs.modelslim_config import AscendModelSlimConfig, _find_quant_description_key
    from vllm_ascend.quantization.method_adapters import AscendEmbeddingMethod
    from vllm_ascend.quantization.methods import get_scheme_class

    quant_type = quantization.upper()
    scheme_cls = get_scheme_class(quant_type, "lm_head")
    if scheme_cls is None:
        raise ValueError(f"Unsupported LM head quantization: {quantization!r}.")
    if device_family == DeviceAdaptorFamily.FP8_OPTIMIZED:
        quant_type = getattr(scheme_cls, "a5_quant_type", quant_type)
        scheme_cls = get_scheme_class(quant_type, "lm_head")
        if scheme_cls is None:
            raise ValueError(f"Unsupported LM head quantization on A5: {quantization!r}.")

    if isinstance(quant_config, AscendModelSlimConfig):
        model_type = vllm_config.model_config.hf_config.model_type
        mapped_prefix = quant_config.quant_prefix_mapper(model_type, prefix)
        key = _find_quant_description_key(quant_config.quant_description, mapped_prefix)
        if key is None:
            raise ValueError(f"Cannot determine checkpoint quantization for LM head {prefix!r}.")
        recipe = quant_config.quant_description[key]
        if recipe in (quantization.upper(), quant_type):
            return checkpoint_method
        if recipe != "FLOAT":
            raise ValueError(f"lm_head_quantization conflicts with checkpoint LM head format {recipe!r}.")
    elif quant_config is not None and quant_config.get_name() == "compressed-tensors":
        from vllm.model_executor.layers.quantization.compressed_tensors.utils import should_ignore_layer

        if not should_ignore_layer(
            prefix, ignore=quant_config.ignore, fused_mapping=quant_config.packed_modules_mapping
        ):
            return checkpoint_method
    elif quant_config is not None:
        raise ValueError(f"Unsupported checkpoint format for lm_head_quantization: {type(quant_config).__name__}.")

    if checkpoint_method is not None and not isinstance(checkpoint_method, UnquantizedEmbeddingMethod):
        raise ValueError(f"LM head {prefix!r} already has an incompatible quantization method.")
    if getattr(vllm_config.model_config.hf_text_config, "tie_word_embeddings", False):
        raise ValueError("lm_head_quantization requires an LM head that does not share embedding weights.")

    logger.info_once("Quantizing target LM head weights online with %s.", quant_type)
    return AscendEmbeddingMethod(scheme_cls())
