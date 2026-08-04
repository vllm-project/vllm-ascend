import vllm
import vllm.v1.worker.utils as worker_utils
from vllm.model_executor.models.deepseek_v2 import DeepseekV32IndexerCache

from vllm_ascend.attention.indexer import AscendSFAIndexerBackend

from vllm_ascend.worker.v2.attn_utils import (
    _allocate_kv_cache,
    _reshape_kv_cache_v2,
    get_kv_cache_spec,
)

vllm.v1.worker.gpu.attn_utils._allocate_kv_cache = _allocate_kv_cache
vllm.v1.worker.gpu.attn_utils._reshape_kv_cache = _reshape_kv_cache_v2
vllm.v1.worker.gpu.model_runner.get_kv_cache_spec = get_kv_cache_spec

# gpu.attn_utils imported bind_kv_cache by value before Ascend replaced the
# canonical worker_utils function. Keep that upstream binder for every normal
# V2 model, but use the existing Ascend binder when the split SFA indexer is
# present: the indexer and attention cache have the same layer index.
_upstream_bind_kv_cache = vllm.v1.worker.gpu.attn_utils.bind_kv_cache


def _bind_kv_cache_with_sfa_indexer(
    kv_caches,
    forward_context,
    runner_kv_caches,
    num_attn_module=1,
):
    if any(
        isinstance(layer, DeepseekV32IndexerCache)
        for layer in forward_context.values()
    ):
        return worker_utils.bind_kv_cache(
            kv_caches, forward_context, runner_kv_caches, num_attn_module
        )
    return _upstream_bind_kv_cache(
        kv_caches, forward_context, runner_kv_caches, num_attn_module
    )


vllm.v1.worker.gpu.attn_utils.bind_kv_cache = _bind_kv_cache_with_sfa_indexer


def _get_ascend_sfa_indexer_backend(_: DeepseekV32IndexerCache):
    return AscendSFAIndexerBackend


# The upstream indexer backend has a different cache layout and block-size
# contract.  V2 creates attention groups through this method, so redirect the
# split SFA indexer layer to the Ascend cache-only backend before KV caches are
# planned and reshaped.
DeepseekV32IndexerCache.get_attn_backend = _get_ascend_sfa_indexer_backend
