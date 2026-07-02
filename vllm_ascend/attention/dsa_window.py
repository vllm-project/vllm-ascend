# SPDX-License-Identifier: Apache-2.0

from typing import Any


def is_dspark_speculative_config(speculative_config: Any) -> bool:
    if speculative_config is None:
        return False
    use_dspark = getattr(speculative_config, "use_dspark", None)
    if callable(use_dspark):
        return bool(use_dspark())
    draft_model_config = getattr(speculative_config, "draft_model_config", None)
    draft_hf_config = getattr(draft_model_config, "hf_config", None)
    return bool(getattr(draft_hf_config, "dspark_block_size", 0))


def get_draft_swa_window(
    vllm_config: Any,
    common_attn_metadata: Any,
) -> tuple[int, int]:
    hf_config = vllm_config.model_config.hf_config
    window_size = int(hf_config.sliding_window)
    speculative_config = getattr(vllm_config, "speculative_config", None)
    if (
        not getattr(common_attn_metadata, "causal", True)
        and is_dspark_speculative_config(speculative_config)
    ):
        draft_model_config = getattr(speculative_config, "draft_model_config", None)
        draft_hf_config = getattr(draft_model_config, "hf_config", None)
        block_size = int(
            getattr(speculative_config, "num_speculative_tokens", 0)
            or getattr(draft_hf_config, "dspark_block_size", 0)
            or 0
        )
        if block_size > 0:
            return window_size + block_size - 1, block_size - 1
    return window_size - 1, 0
