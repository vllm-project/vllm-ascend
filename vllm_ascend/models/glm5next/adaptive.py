# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Capability bounds for GLM-5.3-Flash MTP adaptive verification."""

from vllm.config import VllmConfig


def glm_mtp_adaptive_query_len(vllm_config: VllmConfig) -> int | None:
    model = getattr(vllm_config, "model_config", None)
    spec = getattr(vllm_config, "speculative_config", None)
    if (
        model is None
        or model.hf_text_config.model_type not in ("glm5_next", "glm5_next_text")
        or spec is None
        or getattr(spec, "method", None) not in ("mtp", "deepseek_mtp")
        or not spec.enable_adaptive_verification
    ):
        return None
    return spec.num_speculative_tokens + 1


def validate_glm_mtp_adaptive(vllm_config: VllmConfig) -> None:
    if glm_mtp_adaptive_query_len(vllm_config) is None:
        return
    parallel = vllm_config.parallel_config
    if parallel.prefill_context_parallel_size != 1 or parallel.decode_context_parallel_size != 1:
        raise ValueError("GLM MTP adaptive verification requires PCP=1 and DCP=1")
    if vllm_config.cache_config.use_replayssm:
        raise ValueError("GLM MTP adaptive verification does not support RecoverSSM")
