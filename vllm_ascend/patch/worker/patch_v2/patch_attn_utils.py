import vllm

from vllm_ascend.utils import vllm_version_is
from vllm_ascend.worker.v2.attn_utils import _allocate_kv_cache, _reshape_kv_cache, _reshape_kv_cache_v2

if vllm_version_is("0.20.2"):

    def _allocate_kv_cache_compat(kv_cache_config, device):
        return _allocate_kv_cache(kv_cache_config, {}, device)

    vllm.v1.worker.gpu.attn_utils._allocate_kv_cache = _allocate_kv_cache_compat
    vllm.v1.worker.gpu.attn_utils._reshape_kv_cache = _reshape_kv_cache
else:
    vllm.v1.worker.gpu.attn_utils._allocate_kv_cache = _allocate_kv_cache
    vllm.v1.worker.gpu.attn_utils._reshape_kv_cache = _reshape_kv_cache_v2
