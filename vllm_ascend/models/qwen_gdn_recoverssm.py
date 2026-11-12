# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen GDN RecoverSSM configuration and paged record layout.

Keep this module free of worker/operator imports: platform validation runs
before NPU workers and their custom operators are initialized.
"""

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from vllm.config import VllmConfig


_GDN_MODEL_TYPES = frozenset(
    {
        "qwen3_5_moe_text",
        "qwen3_5_text",
        "qwen3_next",
        "qwen4_exp_text",
    }
)


_GDN_ARCHITECTURES = frozenset(
    {
        "Qwen3_5ForCausalLM",
        "Qwen3_5MoeForCausalLM",
        "Qwen3NextForCausalLM",
        "Qwen3_5ForConditionalGeneration",
        "Qwen3_5MoeForConditionalGeneration",
        "Qwen4ExpForConditionalGeneration",
    }
)


def is_gdn_model(config: "VllmConfig") -> bool:
    """Return whether the model has a supported Qwen GDN descriptor path.

    Keep this capability check local to the GDN path.  KDA's selector and
    validation remain untouched, so adding another GDN model type cannot
    change the KDA RecoverSSM path.
    """
    model = config.model_config
    hf = getattr(model, "hf_text_config", None) if model is not None else None
    if hf is None:
        return False
    architectures = getattr(getattr(model, "hf_config", hf), "architectures", None)
    return getattr(hf, "model_type", None) in _GDN_MODEL_TYPES and (
        not architectures or any(architecture in _GDN_ARCHITECTURES for architecture in architectures)
    )


def _uses_gdn_recoverssm(vllm_config: "VllmConfig") -> bool:
    spec = vllm_config.speculative_config
    num_spec = spec.num_speculative_tokens if spec is not None else 0
    return vllm_config.cache_config.use_replayssm and is_gdn_model(vllm_config) and (num_spec or 0) > 0


def validate_qwen_gdn_recoverssm(config: "VllmConfig") -> None:
    spec = config.speculative_config
    if spec is None or spec.num_speculative_tokens < 1:
        raise ValueError("Qwen GDN RecoverSSM requires speculative decoding with at least one speculative token")
    if not config.use_v2_model_runner:
        raise ValueError("Qwen GDN RecoverSSM requires Model Runner V2")
    cache = config.cache_config
    if config.mamba_config.enable_stochastic_rounding:
        raise ValueError("Qwen GDN RecoverSSM does not support stochastic rounding")
    if config.model_config.dtype != torch.bfloat16 or cache.mamba_ssm_cache_dtype not in (
        "auto",
        "float32",
        "bfloat16",
    ):
        raise ValueError("Qwen GDN RecoverSSM requires BF16 activations and BF16 or FP32 SSM state")
    if cache.mamba_cache_dtype not in ("auto", "bfloat16"):
        raise ValueError("Qwen GDN RecoverSSM requires BF16 convolution state")
    parallel = config.parallel_config
    if (
        parallel.pipeline_parallel_size,
        parallel.prefill_context_parallel_size,
        parallel.decode_context_parallel_size,
    ) != (1, 1, 1):
        raise ValueError("Qwen GDN RecoverSSM requires PP=PCP=DCP=1")
    transfer = config.kv_transfer_config
    if transfer is not None:
        if transfer.kv_connector not in ("MooncakeConnectorV2", "MooncakePullConnector"):
            raise ValueError("Qwen GDN RecoverSSM KV transfer requires MooncakeConnectorV2 or MooncakePullConnector")
        if cache.mamba_cache_mode != "none" or cache.enable_prefix_caching:
            raise ValueError("Qwen GDN RecoverSSM PD requires Mamba cache mode none and prefix caching disabled")
    else:
        expected_cache_mode = "align" if cache.enable_prefix_caching else "none"
        if cache.mamba_cache_mode != expected_cache_mode:
            raise ValueError(
                f"Qwen GDN RecoverSSM requires Mamba cache mode {expected_cache_mode} with this prefix caching setting"
            )
    hf = config.model_config.hf_text_config
    if (
        hf.linear_num_key_heads % parallel.tensor_parallel_size
        or hf.linear_num_value_heads % parallel.tensor_parallel_size
    ):
        raise ValueError("Qwen GDN RecoverSSM TP size must divide key and value heads")
    if hf.linear_num_key_heads < 1 or hf.linear_num_value_heads % hf.linear_num_key_heads:
        raise ValueError("Qwen GDN RecoverSSM requires integral GQA")


def gdn_mamba_state_dtype_from_config(vllm_config: "VllmConfig") -> tuple[torch.dtype, ...]:
    # Keep platform configuration validation independent of layer imports.
    from vllm.model_executor.layers.mamba.mamba_utils import MambaStateDtypeCalculator

    dtypes = MambaStateDtypeCalculator.gated_delta_net_state_dtype(
        vllm_config.model_config.dtype,
        vllm_config.cache_config.mamba_cache_dtype,
        vllm_config.cache_config.mamba_ssm_cache_dtype,
    )
    return (*dtypes, torch.float32) if _uses_gdn_recoverssm(vllm_config) else dtypes


def gdn_mamba_state_shape_from_config(vllm_config: "VllmConfig") -> tuple[tuple[int, ...], ...]:
    # Imported only when descriptors are requested, not during selector validation.
    from vllm.model_executor.layers.mamba.mamba_utils import MambaStateShapeCalculator

    hf = vllm_config.model_config.hf_text_config
    tp_size = vllm_config.parallel_config.tensor_parallel_size
    spec = vllm_config.speculative_config
    num_spec = (spec.num_speculative_tokens or 0) if spec is not None else 0
    shapes = MambaStateShapeCalculator.gated_delta_net_state_shape(
        tp_size,
        hf.linear_num_key_heads,
        hf.linear_num_value_heads,
        hf.linear_key_head_dim,
        hf.linear_value_head_dim,
        hf.linear_conv_kernel_dim,
        num_spec,
    )
    if _uses_gdn_recoverssm(vllm_config):
        shapes = (
            *shapes,
            (hf.linear_num_value_heads // tp_size, num_spec + 1, hf.linear_value_head_dim + hf.linear_key_head_dim + 1),
        )
    return shapes
