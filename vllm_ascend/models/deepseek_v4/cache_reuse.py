# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4 eligibility for the common layerwise cache planner."""

from vllm.config import VllmConfig
from vllm.v1.kv_cache_interface import KVCacheSpec

from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec, AscendSlidingWindowMLASpec
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.layerwise_cache_layout import SpecReuseRules
from vllm_ascend.quantization.methods.kv_cache.turboquant import TURBOQUANT_CACHE_DTYPE


def get_spec_reuse_rules() -> SpecReuseRules:
    """Return eligibility rules without shared mutable model state."""
    return {
        AscendMLAAttentionSpec: lambda spec: spec.model_version == "deepseek_v4",
        # Compressor state uses the same sliding-window spec, without a model
        # version tag. The registry scopes these rules to DeepSeek-V4 only.
        AscendSlidingWindowMLASpec: True,
        KVCacheSpec: False,
    }


def validate_deepseek_v4_layerwise_config(vllm_config: VllmConfig) -> None:
    """Reject formats and commit protocols not covered by this adaptation."""
    if vllm_config.cache_config.cache_dtype == TURBOQUANT_CACHE_DTYPE:
        raise NotImplementedError("DeepSeek-V4 layerwise offload does not support TurboQuant KV transfer.")
    if vllm_config.speculative_config is not None:
        raise NotImplementedError("DeepSeek-V4 layerwise offload does not support speculative state commits yet.")
    compilation = getattr(vllm_config, "compilation_config", None)
    graph_mode = getattr(compilation, "cudagraph_mode", None)
    if graph_mode is not None and graph_mode.has_full_cudagraphs():
        # FULL replay skips the host hooks that restore and publish each layer.
        # PIECEWISE retains dsa_forward as an eager boundary between graphs.
        raise NotImplementedError(
            "DeepSeek-V4 layerwise offload requires NONE or PIECEWISE graph mode; "
            "FULL replay does not execute per-layer transfer hooks."
        )
