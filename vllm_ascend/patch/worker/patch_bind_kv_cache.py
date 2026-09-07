"""Compatibility wrapper around vLLM's standardized KV-cache binder."""

from collections.abc import Sequence

import torch
from vllm.model_executor.layers.attention import Attention
from vllm.v1.kv_cache_interface import KVCacheGroupSpec
from vllm.v1.worker.utils import bind_kv_cache as _core_bind_kv_cache


def bind_kv_cache(
    kv_caches: dict[str, torch.Tensor],
    forward_context: dict[str, Attention],
    runner_kv_caches: list[torch.Tensor],
    num_attn_module: int = 1,
    kv_cache_groups: Sequence[KVCacheGroupSpec] | None = None,
) -> None:
    """Delegate binding so each layer receives its standardized cache view."""
    _core_bind_kv_cache(
        kv_caches,
        forward_context,
        runner_kv_caches,
        num_attn_module,
        kv_cache_groups,
    )
