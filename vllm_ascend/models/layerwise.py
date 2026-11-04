# SPDX-License-Identifier: Apache-2.0
"""Select model-owned layerwise planning from the configured architecture."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from vllm.config import ModelConfig, VllmConfig
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheSpec

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.layerwise_cache_layout import (
    LayerwiseReuseLayout,
)
from vllm_ascend.models.qwen4_exp.layerwise import (
    apply_qwen4_layerwise_kv_cache_plan,
    build_qwen4_reuse_layout,
)


@dataclass(frozen=True)
class ModelLayerwiseHandler:
    apply_plan: Callable[[KVCacheConfig, VllmConfig, set[str] | None], LayerwiseReuseLayout | None]
    build_reuse_layout: Callable[[dict[str, KVCacheSpec], int, dict[str, Any]], LayerwiseReuseLayout]
    # None uses the generic logical/physical page ratio.
    memory_scale: float | None = None


def get_model_layerwise_handler(model_config: ModelConfig) -> ModelLayerwiseHandler | None:
    architecture = getattr(model_config, "architecture", None)
    if architecture in ("Qwen4ExpForCausalLM", "Qwen4ExpForConditionalGeneration"):
        # Packed backing capacity is unchanged by Qwen's owner remapping.
        return ModelLayerwiseHandler(apply_qwen4_layerwise_kv_cache_plan, build_qwen4_reuse_layout, memory_scale=1.0)
    return None
