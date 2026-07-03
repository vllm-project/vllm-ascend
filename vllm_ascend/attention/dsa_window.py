# SPDX-License-Identifier: Apache-2.0

from typing import Any


def _get_dspark_draft_hf_config(vllm_config: Any) -> Any | None:
    speculative_config = getattr(vllm_config, "speculative_config", None)
    draft_model_config = getattr(speculative_config, "draft_model_config", None)
    return getattr(draft_model_config, "hf_config", None)


def get_dspark_query_block_size(vllm_config: Any) -> int:
    speculative_config = getattr(vllm_config, "speculative_config", None)
    num_speculative_tokens = getattr(speculative_config, "num_speculative_tokens", None)
    if num_speculative_tokens:
        return int(num_speculative_tokens)

    draft_hf_config = _get_dspark_draft_hf_config(vllm_config)
    return int(getattr(draft_hf_config, "dspark_block_size", 0) or 0)


def is_dspark_noncausal_draft(vllm_config: Any, common_attn_metadata: Any) -> bool:
    if getattr(common_attn_metadata, "causal", True):
        return False

    speculative_config = getattr(vllm_config, "speculative_config", None)
    use_dspark = getattr(speculative_config, "use_dspark", None)
    if callable(use_dspark):
        return bool(use_dspark())

    draft_hf_config = _get_dspark_draft_hf_config(vllm_config)
    return bool(getattr(draft_hf_config, "dspark_block_size", 0))


def get_draft_swa_window(
    vllm_config: Any,
    common_attn_metadata: Any,
) -> tuple[int, int]:
    del common_attn_metadata
    hf_config = vllm_config.model_config.hf_config
    window_size = int(hf_config.sliding_window)
    # DSpark full-draft-block visibility is expressed by explicit sparse slot
    # indices. Do not encode it as a band window on the full paged KV cache.
    return window_size - 1, 0


def get_dspark_sparse_sas_window(
    vllm_config: Any,
    common_attn_metadata: Any,
) -> tuple[int, int]:
    """Scheduling bound for DSpark PA_ND + explicit ori sparse indices."""
    if not is_dspark_noncausal_draft(vllm_config, common_attn_metadata):
        return get_draft_swa_window(vllm_config, common_attn_metadata)

    hf_config = vllm_config.model_config.hf_config
    window_size = int(hf_config.sliding_window)
    block_size = get_dspark_query_block_size(vllm_config)
    if block_size <= 0:
        return window_size - 1, 0
    return window_size + block_size - 1, 0
