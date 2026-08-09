# SPDX-License-Identifier: Apache-2.0
"""Ascend 310P integration helpers for DeepSeek V4."""

from __future__ import annotations

DSA_BACKEND_310P = "vllm_ascend._310p.attention.dsa_v1.AscendDSABackend310"
DSV4_OP_TIMEOUT_SECONDS = 1800


def is_deepseek_v4_model(model_config) -> bool:
    """Identify DeepSeek V4 without depending on a specific vLLM config API."""
    hf_text_config = getattr(model_config, "hf_text_config", None)
    return getattr(hf_text_config, "model_type", None) == "deepseek_v4"


def validate_dsv4_310p_topology(model_config, tensor_parallel_size: int) -> None:
    """Fail early when the 310P O-LoRA fallback cannot match TP topology."""
    if not is_deepseek_v4_model(model_config):
        return
    hf_text_config = getattr(model_config, "hf_text_config", None)
    o_groups = getattr(hf_text_config, "o_groups", None)
    if o_groups is not None and tensor_parallel_size != o_groups:
        raise ValueError(
            "Ascend 310P DeepSeek V4 currently requires tensor_parallel_size "
            f"to equal o_groups ({o_groups}), got {tensor_parallel_size}."
        )


def get_dsv4_310p_backend(
    *,
    model_config,
    tensor_parallel_size: int,
    use_mla: bool,
    use_sparse: bool,
    use_compress: bool,
) -> str | None:
    """Select the 310P DSA backend for DeepSeek V4 automatically."""
    del use_compress
    if not is_deepseek_v4_model(model_config) or not use_mla or use_sparse:
        return None
    validate_dsv4_310p_topology(model_config, tensor_parallel_size)
    return DSA_BACKEND_310P
