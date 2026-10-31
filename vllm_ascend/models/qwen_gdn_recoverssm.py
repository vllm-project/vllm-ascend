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


def is_qwen_gdn_recoverssm_model(config: "VllmConfig") -> bool:
    model = config.model_config
    return model is not None and model.hf_text_config.model_type == "qwen3_5_moe_text"


def use_qwen_gdn_recoverssm(config: "VllmConfig") -> bool:
    return config.cache_config.use_replayssm and is_qwen_gdn_recoverssm_model(config)


def validate_qwen_gdn_recoverssm(config: "VllmConfig") -> None:
    spec = config.speculative_config
    if spec is None or spec.method != "mtp" or spec.num_speculative_tokens < 1:
        raise ValueError("Qwen GDN RecoverSSM requires MTP with at least one speculative token")
    if not config.use_v2_model_runner:
        raise ValueError("Qwen GDN RecoverSSM requires Model Runner V2")
    cache = config.cache_config
    if cache.mamba_cache_mode != "none" or cache.enable_prefix_caching:
        raise ValueError("Qwen GDN RecoverSSM requires Mamba cache mode none and prefix caching disabled")
    if config.mamba_config.enable_stochastic_rounding:
        raise ValueError("Qwen GDN RecoverSSM does not support stochastic rounding")
    if config.model_config.dtype != torch.bfloat16 or cache.mamba_ssm_cache_dtype not in ("auto", "float32"):
        raise ValueError("Qwen GDN RecoverSSM requires BF16 activations and FP32 SSM state")
    if cache.mamba_cache_dtype not in ("auto", "bfloat16"):
        raise ValueError("Qwen GDN RecoverSSM requires BF16 convolution state")
    parallel = config.parallel_config
    if (
        parallel.pipeline_parallel_size,
        parallel.prefill_context_parallel_size,
        parallel.decode_context_parallel_size,
    ) != (1, 1, 1):
        raise ValueError("Qwen GDN RecoverSSM requires PP=PCP=DCP=1")
    if config.kv_transfer_config is not None:
        raise ValueError("Qwen GDN RecoverSSM does not yet support KV transfer")
    hf = config.model_config.hf_text_config
    if (
        hf.linear_num_key_heads,
        hf.linear_num_value_heads,
        hf.linear_key_head_dim,
        hf.linear_value_head_dim,
        hf.linear_conv_kernel_dim,
    ) != (16, 32, 128, 128, 4):
        raise ValueError("Qwen GDN RecoverSSM currently supports the Qwen3.6-35B-A3B GDN geometry")
    if 16 % parallel.tensor_parallel_size:
        raise ValueError("Qwen GDN RecoverSSM TP size must divide 16 key heads")


def append_gdn_record_shapes(shapes, num_key_heads: int, spec_query_len: int):
    """Append correction, effective key and log-decay records to conv/state."""
    _, (num_value_heads, value_dim, key_dim) = shapes
    return (
        *shapes,
        (num_value_heads, spec_query_len, value_dim),
        (num_key_heads, spec_query_len, key_dim),
        (num_value_heads, spec_query_len),
    )


def append_gdn_record_dtypes(dtypes):
    # Force the checkpoint to FP32 even when an unresolved "auto" reached us.
    return (dtypes[0], torch.float32, torch.float32, torch.float32, torch.float32)


class QwenGDNRecoverSSMModelMixin:
    @classmethod
    def get_mamba_state_shape_from_config(cls, config):
        shapes = super().get_mamba_state_shape_from_config(config)
        if not use_qwen_gdn_recoverssm(config):
            return shapes
        return append_gdn_record_shapes(
            shapes,
            config.model_config.hf_text_config.linear_num_key_heads // config.parallel_config.tensor_parallel_size,
            config.speculative_config.num_speculative_tokens + 1,
        )

    @classmethod
    def get_mamba_state_dtype_from_config(cls, config):
        dtypes = super().get_mamba_state_dtype_from_config(config)
        return append_gdn_record_dtypes(dtypes) if use_qwen_gdn_recoverssm(config) else dtypes
