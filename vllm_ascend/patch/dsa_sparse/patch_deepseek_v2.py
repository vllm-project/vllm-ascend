"""DeepSeek-V2-family shared model targets patched for DSA sparse cache.

``GlmMoeDsaForCausalLM`` inherits the same upstream model implementation and
uses ``DeepseekV32IndexerCache`` as DeepSeek-V3.2. Patching that shared target
once therefore covers both architectures; a second GLM-named patch module
would only wrap or verify the same class redundantly.
"""

from __future__ import annotations

from vllm.model_executor.models.deepseek_v2 import DeepseekV32IndexerCache

from vllm_ascend.core.kv_cache_interface import IndexerKVSpec
from vllm_ascend.dsa_sparse.dsa_config import (
    attach_dsa_sparse_cache_attrs,
    is_dsa_sparse_config_enabled,
)

_ORIGINAL_GET_KV_CACHE_SPEC = "_vllm_ascend_dsa_original_get_kv_cache_spec"


def is_dsa_indexer_cache_spec_patch_installed() -> bool:
    """Return whether the shared indexer-cache spec target is patched."""
    return hasattr(DeepseekV32IndexerCache, _ORIGINAL_GET_KV_CACHE_SPEC)


def _dsa_indexer_get_kv_cache_spec(self, vllm_config):
    attach_dsa_sparse_cache_attrs(vllm_config)
    cache_config = vllm_config.cache_config
    if (is_dsa_sparse_config_enabled(vllm_config) or bool(
            getattr(cache_config, "enable_dsa_split_indexer_cache", False))):
        # vLLM v0.23's native DeepseekV32IndexerCache is an FP8-naive
        # [index_head_dim + scale-bytes] cache (typically uint8/132). The
        # retained LIDU/KSC ABI consumes the pre-quantization Indexer key
        # (typically BF16/128), so the split plane must use the model's logical
        # index dimension and activation dtype rather than the upstream cache
        # object's storage description.
        index_head_dim = int(
            getattr(
                vllm_config.model_config.hf_text_config,
                "index_head_dim",
                self.head_dim,
            )
        )
        return IndexerKVSpec(
            block_size=self.cache_config.block_size,
            num_kv_heads=1,
            head_size=index_head_dim,
            dtype=vllm_config.model_config.dtype,
        )

    original = getattr(DeepseekV32IndexerCache, _ORIGINAL_GET_KV_CACHE_SPEC)
    return original(self, vllm_config)


def patch_deepseek_v2_indexer_cache_spec() -> None:
    if is_dsa_indexer_cache_spec_patch_installed():
        return

    setattr(DeepseekV32IndexerCache, _ORIGINAL_GET_KV_CACHE_SPEC,
            DeepseekV32IndexerCache.get_kv_cache_spec)
    DeepseekV32IndexerCache.get_kv_cache_spec = _dsa_indexer_get_kv_cache_spec


patch_deepseek_v2_indexer_cache_spec()
