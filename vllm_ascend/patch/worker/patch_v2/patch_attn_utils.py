import vllm
from vllm.model_executor.models.deepseek_v2 import DeepseekV32IndexerCache

from vllm_ascend.attention.indexer import AscendSFAIndexerBackend
from vllm_ascend.patch.worker.patch_bind_kv_cache import bind_kv_cache
from vllm_ascend.utils import vllm_version_is
from vllm_ascend.worker.v2.attn_utils import (
    _allocate_kv_cache,
    _reshape_kv_cache_v2,
    get_kv_cache_spec,
)


def _get_ascend_sfa_indexer_backend(_self):
    return AscendSFAIndexerBackend


DeepseekV32IndexerCache.get_attn_backend = _get_ascend_sfa_indexer_backend
vllm.v1.worker.gpu.model_runner.get_kv_cache_spec = get_kv_cache_spec
if vllm_version_is("0.27.1"):
    # Upstream v0.27.1 still routes init_kv_cache through the module-level
    # _allocate_kv_cache/_reshape_kv_cache helpers; on newer versions the
    # standardized-layout path is installed by ascend_init_kv_cache_override
    # from NPUModelRunner.initialize_kv_cache instead.
    vllm.v1.worker.gpu.attn_utils._allocate_kv_cache = _allocate_kv_cache  # type: ignore[attr-defined]
    vllm.v1.worker.gpu.attn_utils._reshape_kv_cache = _reshape_kv_cache_v2  # type: ignore[attr-defined]
    vllm.v1.worker.gpu.attn_utils.bind_kv_cache = bind_kv_cache
