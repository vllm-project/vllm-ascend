# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Select the experimental Ascend Kimi KDA RecoverSSM path."""

from __future__ import annotations

from typing import TYPE_CHECKING

from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
from vllm.transformers_utils.configs.kimi_linear import KimiLinearConfig

from vllm_ascend.device.device_config import is_950
from vllm_ascend.utils import model_uses_kimi_kda

if TYPE_CHECKING:
    from vllm.config import VllmConfig


def supports_request_indexed_kda_recoverssm(vllm_config: VllmConfig) -> bool:
    """Select the KDA implementation by its text-config contract, not its wrapper name."""
    model = vllm_config.model_config
    if model is None:
        return False
    text_config = getattr(model, "hf_text_config", None)
    linear_config = getattr(text_config, "linear_attn_config", None)
    return (
        isinstance(text_config, KimiLinearConfig)
        and isinstance(linear_config, dict)
        and bool(linear_config.get("kda_layers"))
    )


def use_request_indexed_kda_recoverssm(vllm_config: VllmConfig) -> bool:
    return vllm_config.cache_config.use_replayssm and supports_request_indexed_kda_recoverssm(vllm_config)


def use_fused_dspark_recovery(config: VllmConfig) -> bool:
    """Use the fused KDA state on an aligned A5 Mooncake decoder."""
    spec = config.speculative_config
    transfer = config.kv_transfer_config
    return (
        config.cache_config.use_kda_recoverssm
        and config.cache_config.mamba_cache_mode == "align"
        and spec is not None
        and spec.use_dspark()
        and transfer is not None
        and transfer.kv_role == "kv_consumer"
        and transfer.kv_connector == "MooncakeConnectorV2"
        and config.parallel_config.pipeline_parallel_size == 1
        and config.parallel_config.prefill_context_parallel_size == 1
        and model_uses_kimi_kda(config.model_config)
        and is_950()
    )


def validate_request_indexed_kda_recoverssm(vllm_config: VllmConfig) -> None:
    spec = vllm_config.speculative_config
    num_speculative_tokens = getattr(spec, "num_speculative_tokens", None)
    if num_speculative_tokens is None or num_speculative_tokens <= 0:
        raise ValueError("Kimi RecoverSSM requires speculative decoding")
    if spec.method != "dspark":
        raise ValueError("Kimi RecoverSSM supports DSpark speculative decoding")
    if getattr(spec, "enable_adaptive_verification", False):
        raise ValueError("Kimi RecoverSSM does not yet support adaptive verification")
    if not vllm_config.use_v2_model_runner:
        raise ValueError("Kimi RecoverSSM requires Model Runner V2")
    if vllm_config.mamba_config.enable_stochastic_rounding:
        raise ValueError("Kimi RecoverSSM does not support stochastic rounding")
    cache = vllm_config.cache_config
    expected_cache_mode = "align" if cache.enable_prefix_caching else "none"
    if cache.mamba_cache_mode != expected_cache_mode:
        raise ValueError(
            f"Kimi RecoverSSM requires Mamba cache mode {expected_cache_mode} with this prefix caching setting"
        )
    transfer = vllm_config.kv_transfer_config
    if transfer is not None and transfer.kv_connector not in ("MooncakeConnectorV2", "MooncakePullConnector"):
        raise ValueError("KDA RecoverSSM KV transfer requires MooncakeConnectorV2 or MooncakePullConnector")
    if transfer is not None and is_conv_state_dim_first():
        raise ValueError("Kimi RecoverSSM Mooncake P/D requires SD convolution cache layout")
