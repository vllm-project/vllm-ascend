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
    d = _orig_init_kv_cache(*args, **kwargs)
    return {name: (v[0] if isinstance(v, (tuple, list)) and v else v) for name, v in d.items()}


vllm.v1.worker.gpu.model_runner.init_kv_cache = _ascend_init_kv_cache
vllm.v1.worker.gpu.model_runner.get_kv_cache_spec = get_kv_cache_spec
