# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""vLLM v0.27 model-config compatibility for DeepSeek-V4 vision."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import PretrainedConfig

_REGISTERED = False


def register_deepseek_v4_vision_config_convertor() -> None:
    """Route vision checkpoints to the Ascend multimodal wrapper.

    Keep vLLM config imports out of module scope. Global patches are imported
    while spawned engine processes unpickle their state, which can happen while
    ``model_arch_config_convertor`` itself is only partially initialized.
    """
    global _REGISTERED
    if _REGISTERED:
        return

    from vllm.transformers_utils.model_arch_config_convertor import (
        MODEL_ARCH_CONFIG_CONVERTORS,
        ModelArchConfigConvertorBase,
    )

    from vllm_ascend.utils import vllm_version_is

    class AscendDeepseekV4ModelArchConfigConvertor(ModelArchConfigConvertorBase):
        """Route vision checkpoints to the Ascend multimodal wrapper."""

        def __init__(
            self,
            hf_config: "PretrainedConfig",
            hf_text_config: "PretrainedConfig",
            revision: str | None = None,
        ) -> None:
            if getattr(hf_config, "vision_n_layers", 0) > 0:
                hf_config.architectures = ["DeepseekV4ForConditionalGeneration"]
                hf_config.mm_prefix_clamp_sliding_window = True
                hf_config.mm_prefix_span_leading_pad_modulus = 4
            if vllm_version_is("0.27.1"):
                super().__init__(hf_config, hf_text_config)
            else:
                super().__init__(hf_config, hf_text_config, revision)

        def is_mm_prefix_lm(self, supports_multimodal: bool = True) -> bool:
            return supports_multimodal and (getattr(self.hf_config, "vision_n_layers", 0) > 0)

    MODEL_ARCH_CONFIG_CONVERTORS["deepseek_v4"] = AscendDeepseekV4ModelArchConfigConvertor
    _REGISTERED = True
