# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Explicit, default-off GLM KDA RecoverSSM configuration."""

from vllm.config import VllmConfig


def use_glm_recoverssm(vllm_config: VllmConfig) -> bool:
    return vllm_config.additional_config.get("kda_recoverssm", False)


def validate_glm_recoverssm(vllm_config: VllmConfig) -> None:
    if vllm_config.speculative_config is None:
        raise ValueError("GLM RecoverSSM requires speculative decoding")
    cache = vllm_config.cache_config
    if cache.mamba_cache_mode != "none" or cache.enable_prefix_caching:
        raise ValueError("GLM RecoverSSM currently requires Mamba cache mode none and prefix caching disabled")
    parallel = vllm_config.parallel_config
    if parallel.pipeline_parallel_size != 1 or parallel.prefill_context_parallel_size != 1:
        raise ValueError("GLM RecoverSSM currently requires PP=1 and PCP=1")
    if parallel.decode_context_parallel_size != 1:
        raise ValueError("GLM RecoverSSM currently requires DCP=1")
    transfer = vllm_config.kv_transfer_config
    if transfer is not None and transfer.kv_connector not in ("MooncakeConnectorV2", "MooncakePullConnector"):
        raise ValueError("GLM RecoverSSM KV transfer requires MooncakeConnectorV2 or MooncakePullConnector")
