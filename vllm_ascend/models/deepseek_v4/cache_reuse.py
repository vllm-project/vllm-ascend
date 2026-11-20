# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4 eligibility for the common layerwise cache planner."""

from vllm.config import VllmConfig
from vllm.v1.kv_cache_interface import KVCacheSpec

from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec, AscendSlidingWindowMLASpec
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.layerwise_cache_layout import (
    SpecReuseRules,
)
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


def uses_dspark_push(config: VllmConfig) -> bool:
    """The DSV4 context writer runs on P, including chunks with no proposals."""
    spec = getattr(config, "speculative_config", None)
    transfer = getattr(config, "kv_transfer_config", None)
    if transfer is None or not transfer.is_kv_producer or spec is None or spec.method != "dspark":
        return False
    from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.connector import MooncakeConnectorV2

    if transfer.kv_connector == "MultiConnector":
        return any(
            child.get("kv_role") in ("kv_producer", "kv_both")
            and MooncakeConnectorV2.is_push_config(child["kv_connector"], child.get("kv_connector_extra_config") or {})
            for child in transfer.kv_connector_extra_config.get("connectors", ())
        )
    return MooncakeConnectorV2.is_push_config(transfer.kv_connector, transfer.kv_connector_extra_config)


def validate_rotating_config(config: VllmConfig) -> None:
    from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.cache_planner import (
        get_layerwise_gva_config,
    )

    if get_layerwise_gva_config(config.kv_transfer_config) is None:
        raise ValueError("DeepSeek-V4 rotation requires a GVA store to restore shared pages.")
    if config.cache_config.cache_dtype == TURBOQUANT_CACHE_DTYPE:
        raise NotImplementedError("DeepSeek-V4 TurboQuant KV transfer is unsupported.")
    graph = getattr(getattr(config, "compilation_config", None), "cudagraph_mode", None)
    if graph is not None and graph.has_full_cudagraphs():
        raise ValueError("DeepSeek-V4 rotating target caches require eager or PIECEWISE layer callbacks.")
    spec = getattr(config, "speculative_config", None)
    if spec is not None and spec.method == "dspark" and not (uses_dspark_push(config) and spec.enforce_eager):
        raise ValueError("DeepSeek-V4 rotating DSpark requires a Push producer and eager draft context writes.")
