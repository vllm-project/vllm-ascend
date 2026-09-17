import vllm
from vllm.config import VllmConfig
from vllm.model_executor.models.deepseek_v2 import DeepseekV32IndexerCache
from vllm.v1.worker.cp_utils import check_attention_cp_compatibility as _upstream_check_attention_cp_compatibility

from vllm_ascend import envs as ascend_envs
from vllm_ascend.attention.indexer import AscendSFAIndexerBackend
from vllm_ascend.patch.worker.patch_bind_kv_cache import bind_kv_cache
from vllm_ascend.utils import vllm_version_is
from vllm_ascend.worker.v2.attn_utils import (
    _allocate_kv_cache,
    _reshape_kv_cache_v2,
    allocate_kv_cache_main,
    get_kv_cache_spec,
    init_attn_backend,
)


def _get_ascend_sfa_indexer_backend(_self):
    return AscendSFAIndexerBackend


def check_attention_cp_compatibility(vllm_config: VllmConfig) -> None:
    # The experimental Flash path includes a DCP1 GQA draft in the shared
    # target DCP registry. Let execution report unsupported operator inputs.
    if ascend_envs.VLLM_ASCEND_ENABLE_FLASH_MLA:
        return
    _upstream_check_attention_cp_compatibility(vllm_config)


DeepseekV32IndexerCache.get_attn_backend = _get_ascend_sfa_indexer_backend
vllm.v1.worker.gpu.attn_utils._allocate_kv_cache = _allocate_kv_cache
vllm.v1.worker.gpu.attn_utils._reshape_kv_cache = _reshape_kv_cache_v2
if not vllm_version_is("0.28.0"):
    # vLLM #51718 made this the live allocation symbol used by init_kv_cache.
    vllm.v1.worker.gpu.attn_utils.allocate_kv_cache = allocate_kv_cache_main
vllm.v1.worker.gpu.attn_utils.bind_kv_cache = bind_kv_cache
vllm.v1.worker.gpu.model_runner.get_kv_cache_spec = get_kv_cache_spec
vllm.v1.worker.gpu.model_runner.check_attention_cp_compatibility = check_attention_cp_compatibility
vllm.v1.worker.gpu.attn_utils.init_attn_backend = init_attn_backend
vllm.v1.worker.gpu.model_runner.init_attn_backend = init_attn_backend
vllm.v1.worker.gpu.spec_decode.speculator.init_attn_backend = init_attn_backend
