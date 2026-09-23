from typing import Any

import torch
import vllm
from vllm.model_executor.models.deepseek_v2 import DeepseekV32IndexerCache

from vllm_ascend.attention.indexer import AscendSFAIndexerBackend
from vllm_ascend.patch.worker.patch_bind_kv_cache import (
    bind_kv_cache,
    bind_kv_cache_to_layers,
)
from vllm_ascend.utils import vllm_version_is
from vllm_ascend.worker.v2.attn_utils import (
    _allocate_kv_cache,
    _reshape_kv_cache_v2,
    allocate_kv_cache_main,
    get_kv_cache_spec,
)


def _get_ascend_sfa_indexer_backend(_self):
    return AscendSFAIndexerBackend


class _KVCacheComponents(tuple[Any, ...]):
    """Ascend multi-tensor KV cache with a ``device`` proxy.

    vLLM #53781 makes ``GPUModelRunner.initialize_kv_cache`` filter the
    ``init_kv_cache`` result with ``cache.device``. Ascend stores each layer as
    a ``(k_cache, v_cache)`` tuple rather than one tensor, so expose the first
    component's device while keeping the tuple layout the Ascend attention
    impls index into.
    """

    @property
    def device(self) -> torch.device:
        return self[0].device


def _with_component_device(kv_caches: dict[str, Any]) -> dict[str, Any]:
    return {name: cache if hasattr(cache, "device") else _KVCacheComponents(cache) for name, cache in kv_caches.items()}


_orig_init_kv_cache = vllm.v1.worker.gpu.model_runner.init_kv_cache


def _init_kv_cache_main(*args: Any, **kwargs: Any) -> dict[str, Any]:
    return _with_component_device(_orig_init_kv_cache(*args, **kwargs))


DeepseekV32IndexerCache.get_attn_backend = _get_ascend_sfa_indexer_backend
vllm.v1.worker.gpu.attn_utils._allocate_kv_cache = _allocate_kv_cache
vllm.v1.worker.gpu.attn_utils._reshape_kv_cache = _reshape_kv_cache_v2
# vLLM #51718 made this the live allocation symbol used by init_kv_cache.
vllm.v1.worker.gpu.attn_utils.allocate_kv_cache = allocate_kv_cache_main
vllm.v1.worker.gpu.attn_utils.bind_kv_cache = bind_kv_cache
vllm.v1.worker.gpu.model_runner.get_kv_cache_spec = get_kv_cache_spec

if not vllm_version_is("0.29.0"):
    # vLLM #53781 routes binding through ``bind_kv_cache_to_layers`` and reads
    # ``.device`` off each returned cache; both break Ascend's component layout.
    vllm.v1.worker.gpu.attn_utils.bind_kv_cache_to_layers = bind_kv_cache_to_layers
    vllm.v1.worker.gpu.model_runner.init_kv_cache = _init_kv_cache_main
