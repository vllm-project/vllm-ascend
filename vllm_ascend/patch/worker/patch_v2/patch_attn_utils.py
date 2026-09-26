import vllm
from vllm.model_executor.models.deepseek_v2 import DeepseekV32IndexerCache

from vllm_ascend.attention.indexer import AscendSFAIndexerBackend
from vllm_ascend.patch.worker.patch_bind_kv_cache import (
    bind_kv_cache,
    bind_kv_cache_to_layers,
)
from vllm_ascend.worker.v2.attn_utils import (
    _allocate_kv_cache,
    _reshape_kv_cache_v2,
    allocate_kv_cache_main,
    get_kv_cache_spec,
)


def _get_ascend_sfa_indexer_backend(_self):
    return AscendSFAIndexerBackend


DeepseekV32IndexerCache.get_attn_backend = _get_ascend_sfa_indexer_backend
vllm.v1.worker.gpu.attn_utils._allocate_kv_cache = _allocate_kv_cache
vllm.v1.worker.gpu.attn_utils._reshape_kv_cache = _reshape_kv_cache_v2
# vLLM #51718 made this the live allocation symbol used by init_kv_cache.
vllm.v1.worker.gpu.attn_utils.allocate_kv_cache = allocate_kv_cache_main
vllm.v1.worker.gpu.attn_utils.bind_kv_cache = bind_kv_cache
# vLLM main (#53781) routes init_kv_cache through bind_kv_cache_to_layers,
# which Ascend overrides with direct raw-allocation binding.
vllm.v1.worker.gpu.attn_utils.bind_kv_cache_to_layers = bind_kv_cache_to_layers

# vLLM main (#53781) also builds self.kv_caches by filtering
# cache.device, assuming single-tensor allocations; Ascend allocates
# per-layer (k, v) tuples. Expose the first tensor for that filter.
# The binding inside init_kv_cache still uses the raw allocations.
_orig_init_kv_cache = vllm.v1.worker.gpu.model_runner.init_kv_cache


def _ascend_init_kv_cache(*args, **kwargs):
    """Expose one tensor per layer to the runner, keep the complete K/V views.

    Model Runner V2 builds ``self.kv_caches`` from ``kv_caches_dict.values()``
    filtered by ``.device``, which needs a single tensor per layer, while
    Ascend allocates K and V as separate views. KV cache offloading has to
    register every one of those views, so carry the untouched mapping next to
    the runner-facing one.
    """
    kv_caches = _orig_init_kv_cache(*args, **kwargs)

    class _AscendRunnerKVCaches(dict):
        def __init__(self, runner_view, full_view) -> None:
            super().__init__(runner_view)
            self.full_view = full_view

    runner_view = {
        name: (value[0] if isinstance(value, (tuple, list)) and value else value) for name, value in kv_caches.items()
    }
    return _AscendRunnerKVCaches(runner_view, kv_caches)


def _ascend_get_kv_connector(vllm_config, kv_caches_dict, *args, **kwargs):
    """Register every K/V view with the connector, not only the runner tensor."""
    from vllm.v1.worker.gpu.kv_connector import get_kv_connector

    full_view = getattr(kv_caches_dict, "full_view", kv_caches_dict)
    return get_kv_connector(vllm_config, full_view, *args, **kwargs)


vllm.v1.worker.gpu.model_runner.init_kv_cache = _ascend_init_kv_cache
vllm.v1.worker.gpu.model_runner.get_kv_connector = _ascend_get_kv_connector


vllm.v1.worker.gpu.model_runner.get_kv_cache_spec = get_kv_cache_spec
