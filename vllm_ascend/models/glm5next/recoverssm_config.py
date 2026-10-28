# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM KDA RecoverSSM uses the upstream ReplaySSM configuration switch."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.config import VllmConfig


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
