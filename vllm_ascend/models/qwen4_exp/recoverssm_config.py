# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Explicit Ascend GDN RecoverSSM opt-in, independent of Kimi KDA."""


def use_qwen_gdn_recoverssm(config) -> bool:
    return config.additional_config.get("qwen_gdn_recoverssm", False) is True


def validate_qwen_gdn_recoverssm(config) -> None:
    if not use_qwen_gdn_recoverssm(config):
        return
    if config.model_config.hf_text_config.model_type not in ("qwen4_exp", "qwen4_exp_text"):
        raise ValueError("qwen_gdn_recoverssm requires Qwen4Exp")
    if config.speculative_config is None or config.speculative_config.num_speculative_tokens < 1:
        raise ValueError("Qwen GDN RecoverSSM requires speculative decoding")
    if not config.use_v2_model_runner:
        raise ValueError("Qwen GDN RecoverSSM requires the V2 model runner")
    cache = config.cache_config
    if cache.enable_prefix_caching or cache.mamba_cache_mode != "none":
        raise ValueError("Qwen GDN RecoverSSM currently requires prefix caching disabled and Mamba cache mode none")
    if cache.use_replayssm:
        raise ValueError("Select Qwen GDN RecoverSSM with additional_config, without --use-replayssm")
    parallel = config.parallel_config
    if getattr(parallel, "use_ubatching", False):
        raise ValueError("Qwen GDN RecoverSSM currently does not support microbatching")
    if (
        parallel.pipeline_parallel_size,
        parallel.decode_context_parallel_size,
        parallel.prefill_context_parallel_size,
    ) != (1, 1, 1):
        raise ValueError("Qwen GDN RecoverSSM currently requires PP=1, DCP=1 and PCP=1")
