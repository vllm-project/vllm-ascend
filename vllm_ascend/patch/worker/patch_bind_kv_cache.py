"""Compatibility wrapper for vLLM's KV-cache binding transition.

Current vLLM provides ``AttentionLayerBase.bind_kv_cache`` and its core
``bind_kv_cache`` helper handles standardized strided cache views. Older
vLLM-Ascend releases replaced that helper because the layer method used to be
abstract. Keeping the replacement on current core drops cache-group metadata
and bypasses the layer-specific binding contract, so this module intentionally
does not monkey-patch the upstream helper anymore. The exported wrapper remains
for the v2 adaptor, which imports it as an explicit dependency.
"""

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
    """Delegate to the current core binding contract without replacing it."""
    _core_bind_kv_cache(
        kv_caches,
        forward_context,
        runner_kv_caches,
        num_attn_module,
        kv_cache_groups,
    )
