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


DeepseekV32IndexerCache.get_attn_backend = _get_ascend_sfa_indexer_backend
vllm.v1.worker.gpu.attn_utils._allocate_kv_cache = _allocate_kv_cache
vllm.v1.worker.gpu.attn_utils._reshape_kv_cache = _reshape_kv_cache_v2
# vLLM #51718 made this the live allocation symbol used by init_kv_cache.
vllm.v1.worker.gpu.attn_utils.allocate_kv_cache = allocate_kv_cache_main
vllm.v1.worker.gpu.attn_utils.bind_kv_cache = bind_kv_cache
if not vllm_version_is("0.28.0"):
    # Upstream #53781 introduced bind_kv_cache_to_layers which calls
    # layer.bind_kv_cache(). Ascend Mamba layers use a list of per-state
    # tensors, so skip the layer-level bind and assign the cache directly.
    vllm.v1.worker.gpu.attn_utils.bind_kv_cache_to_layers = bind_kv_cache_to_layers
vllm.v1.worker.gpu.model_runner.get_kv_cache_spec = get_kv_cache_spec

# Upstream PR #53781 introduced `self.kv_caches = [cache for cache in
# kv_caches_dict.values() if cache.device == self.device]` in
# GPUModelRunner.initialize_kv_cache.  Ascend's _reshape_kv_cache_v2 stores
# list (Mamba) / tuple (SFA) values alongside plain tensors; the filter
# crashes on non-tensor values.  Patch init_kv_cache to drop non-tensor
# values: they are already bound to their layers by bind_kv_cache_to_layers,
# and leaving them in the dict breaks both the self.kv_caches filter and
# get_kv_connector (which accesses .shape on every value).
if not vllm_version_is("0.28.0"):
    from vllm.v1.worker.gpu import attn_utils

    _orig_init_kv_cache = attn_utils.init_kv_cache

    def _ascend_init_kv_cache(*args, **kwargs):
        kv_caches = _orig_init_kv_cache(*args, **kwargs)
        # Keep the full mapping (including tuple/list values) for
        # _ascend_get_kv_connector below: connectors must register against
        # the unstripped dict, since Ascend attention layers hold (K, V)
        # tuples and the stripped dict is empty for pure-attention models.
        _ascend_init_kv_cache._full = dict(kv_caches)
        for name in [n for n, c in kv_caches.items() if not isinstance(c, torch.Tensor)]:
            del kv_caches[name]
        return kv_caches

    attn_utils.init_kv_cache = _ascend_init_kv_cache
    import sys

    _model_runner = sys.modules.get("vllm.v1.worker.gpu.model_runner")
    if _model_runner is not None:
        _model_runner.init_kv_cache = _ascend_init_kv_cache

        _orig_get_kv_connector = _model_runner.get_kv_connector

        def _ascend_get_kv_connector(vllm_config, kv_caches_dict):
            # register_kv_caches needs the unstripped mapping: Ascend
            # attention layers register (K, V) tuples which the stripped
            # dict drops, so connectors (OffloadingConnector /
            # SimpleCPUOffload) would see an empty dict and either raise
            # KeyError or never initialize their backends. The runner-side
            # self.kv_caches filter above keeps using the stripped dict.
            full = getattr(_ascend_init_kv_cache, "_full", None)
            if full is not None:
                kv_caches_dict = full
            return _orig_get_kv_connector(vllm_config, kv_caches_dict)

        _model_runner.get_kv_connector = _ascend_get_kv_connector

if not vllm_version_is("0.28.0"):
    from vllm.v1.worker.gpu import cudagraph_utils
    from vllm_ascend.worker.model_runner_v1 import graph_capture
    cudagraph_utils.graph_capture = graph_capture
