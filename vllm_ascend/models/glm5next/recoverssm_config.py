# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM KDA RecoverSSM uses the upstream ReplaySSM configuration switch."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.config import KVTransferConfig, VllmConfig


def is_glm_recoverssm_model(vllm_config: VllmConfig) -> bool:
    model_config = vllm_config.model_config
    return model_config is not None and model_config.hf_text_config.model_type in ("glm5_next", "glm5_next_text")


def use_glm_recoverssm(vllm_config: VllmConfig) -> bool:
    return is_glm_recoverssm_model(vllm_config) and vllm_config.cache_config.use_replayssm


def validate_glm_recoverssm(vllm_config: VllmConfig) -> None:
    if vllm_config.speculative_config is None:
        raise ValueError("GLM RecoverSSM requires speculative decoding")
    cache = vllm_config.cache_config
    # TODO: FP16 KV cache with RecoverSSM has not been validated.
    if vllm_config.mamba_config.enable_stochastic_rounding:
        raise ValueError("GLM RecoverSSM does not support stochastic rounding")
    expected_cache_mode = "align" if cache.enable_prefix_caching else "none"
    if cache.mamba_cache_mode != expected_cache_mode:
        raise ValueError(
            f"GLM RecoverSSM requires Mamba cache mode {expected_cache_mode} with this prefix caching setting"
        )
    parallel = vllm_config.parallel_config
    if parallel.pipeline_parallel_size != 1:
        raise ValueError("GLM RecoverSSM currently requires PP=1")
    if parallel.decode_context_parallel_size != 1:
        raise ValueError("GLM RecoverSSM currently requires DCP=1")
    transfer = vllm_config.kv_transfer_config
    mooncake_connectors = ("MooncakeConnectorV2", "MooncakePullConnector")
    if parallel.prefill_context_parallel_size > 1:
        if parallel.tensor_parallel_size != 1:
            raise ValueError("GLM RecoverSSM PCP requires TP=1")
        if transfer is None or transfer.kv_connector not in mooncake_connectors or transfer.kv_role != "kv_producer":
            raise ValueError("GLM RecoverSSM PCP requires a Mooncake KV producer")
    if transfer is not None:
        if transfer.kv_connector == "AscendStoreConnector":
            _validate_recoverssm_store(vllm_config, transfer)
        elif transfer.kv_connector not in mooncake_connectors:
            raise ValueError(
                "GLM RecoverSSM KV transfer requires MooncakeConnectorV2, MooncakePullConnector "
                "or synchronous bulk AscendStoreConnector"
            )


def _validate_recoverssm_store(vllm_config: VllmConfig, transfer: KVTransferConfig) -> None:
    if not vllm_config.use_v2_model_runner:
        raise ValueError("GLM RecoverSSM with AscendStoreConnector requires Model Runner V2")
    if not vllm_config.cache_config.enable_prefix_caching:
        raise ValueError("GLM RecoverSSM with AscendStoreConnector requires prefix caching")
    extra = transfer.kv_connector_extra_config
    for option in ("use_layerwise", "load_async", "save_decode_cache"):
        if extra.get(option, False):
            raise ValueError(f"GLM RecoverSSM with AscendStoreConnector requires {option}=false")
    if extra.get("backend", "mooncake").lower() not in ("mooncake", "memcache"):
        raise ValueError("GLM RecoverSSM AscendStore requires the mooncake or memcache backend")
    tp_size = vllm_config.parallel_config.tensor_parallel_size
    for option in ("prefill_tp_size", "decode_tp_size"):
        if int(extra.get(option, tp_size)) != tp_size:
            raise ValueError("GLM RecoverSSM AscendStore requires matching TP sizes")
    if (vllm_config.additional_config or {}).get("enable_kvpp", False):
        raise ValueError("GLM RecoverSSM AscendStore does not support KVPP")
