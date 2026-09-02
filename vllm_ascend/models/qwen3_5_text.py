# SPDX-License-Identifier: Apache-2.0

"""Ascend hybrid-cache adapters for text-only Qwen3.5 checkpoints."""

import torch
from vllm.config import VllmConfig
from vllm.model_executor.layers.mamba.mamba_utils import MambaStateCopyFunc
from vllm.model_executor.models.qwen3_5 import (
    Qwen3_5ForCausalLM,
    Qwen3_5ForConditionalGeneration,
    Qwen3_5MoeForCausalLM,
)


class _Qwen3_5TextHybridMixin:
    """Expose the hybrid GDN state contract implemented by the VL wrapper."""

    is_hybrid = True

    @classmethod
    def get_mamba_state_dtype_from_config(
        cls,
        vllm_config: VllmConfig,
    ) -> tuple[torch.dtype, torch.dtype]:
        return Qwen3_5ForConditionalGeneration.get_mamba_state_dtype_from_config(
            vllm_config
        )

    @classmethod
    def get_mamba_state_shape_from_config(
        cls,
        vllm_config: VllmConfig,
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        return Qwen3_5ForConditionalGeneration.get_mamba_state_shape_from_config(
            vllm_config
        )

    @classmethod
    def get_mamba_state_copy_func(
        cls,
    ) -> tuple[MambaStateCopyFunc, MambaStateCopyFunc]:
        return Qwen3_5ForConditionalGeneration.get_mamba_state_copy_func()


class AscendQwen3_5ForCausalLM(_Qwen3_5TextHybridMixin, Qwen3_5ForCausalLM):
    """Text-only Qwen3.5 model with hybrid attention/GDN cache setup."""


class AscendQwen3_5MoeForCausalLM(
    _Qwen3_5TextHybridMixin,
    Qwen3_5MoeForCausalLM,
):
    """MoE text-only Qwen3.5 model with hybrid attention/GDN cache setup."""
