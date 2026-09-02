import torch
import vllm
from vllm.model_executor.models.deepseek_v2 import DeepseekV32IndexerCache

from vllm_ascend.attention.indexer import AscendSFAIndexerBackend
from vllm_ascend.patch.worker.patch_bind_kv_cache import bind_kv_cache
from vllm_ascend.worker.v2.attn_utils import (
    _allocate_kv_cache,
    _reshape_kv_cache_v2,
    get_kv_cache_spec,
)


def _get_ascend_sfa_indexer_backend(_self):
    return AscendSFAIndexerBackend


_upstream_allocate_kv_cache = vllm.v1.worker.gpu.attn_utils.allocate_kv_cache


def _allocate_kv_cache_for_ascend(*args, **kwargs):
    """Adapt vLLM's packed KV views to Ascend's split K/V interface."""
    kv_caches = _upstream_allocate_kv_cache(*args, **kwargs)
    for layer_name, kv_cache in kv_caches.items():
        if (
            isinstance(kv_cache, torch.Tensor)
            and kv_cache.ndim == 5
            and kv_cache.shape[0] == 2
        ):
            kv_caches[layer_name] = (kv_cache[0], kv_cache[1])
    return kv_caches


DeepseekV32IndexerCache.get_attn_backend = _get_ascend_sfa_indexer_backend
vllm.v1.worker.gpu.attn_utils._allocate_kv_cache = _allocate_kv_cache
vllm.v1.worker.gpu.attn_utils.allocate_kv_cache = _allocate_kv_cache_for_ascend
vllm.v1.worker.gpu.attn_utils._reshape_kv_cache = _reshape_kv_cache_v2
vllm.v1.worker.gpu.attn_utils.bind_kv_cache = bind_kv_cache
vllm.v1.worker.gpu.model_runner.get_kv_cache_spec = get_kv_cache_spec
