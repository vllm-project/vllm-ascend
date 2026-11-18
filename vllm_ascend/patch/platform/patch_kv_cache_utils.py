# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

# Both supported versions use the KV-cache planner API from #61183/#42309.
import vllm.v1.core.kv_cache_planning as _kv_cache_planning
from vllm.config import VllmConfig
from vllm.v1.kv_cache_interface import KVCacheGroupSpec, KVCacheSpec

_orig_annotate_eagle_groups = getattr(_kv_cache_planning, "_annotate_eagle_groups", None)


def _uses_qwen4_exp_mtp(vllm_config: VllmConfig) -> bool:
    speculative_config = vllm_config.speculative_config
    if speculative_config is None:
        return False
    draft_model_config = getattr(
        speculative_config,
        "draft_model_config",
        None,
    )
    architectures = tuple(getattr(draft_model_config, "architectures", ()) or ())
    if "Qwen4ExpMTP" in architectures:
        return True
    text_config = getattr(draft_model_config, "hf_text_config", None)
    text_architectures = tuple(getattr(text_config, "architectures", ()) or ())
    return getattr(text_config, "model_type", None) == "qwen4_exp_mtp" or ("Qwen4ExpMTP" in text_architectures)


def _ascend_annotate_eagle_groups(
    vllm_config: VllmConfig,
    kv_cache_spec: dict[str, KVCacheSpec],
    kv_cache_groups: list[KVCacheGroupSpec],
    use_trailing_layer_fallback: bool = False,
) -> None:
    """Annotate Qwen4Exp's actual MTP cache group, not every group.

    Qwen4Exp's MTP attention spec is otherwise indistinguishable from the
    target's spec. Its predictor is registered at the target layer count, so
    use that model-scoped position to find only the group containing the real
    draft layer. Leaving every group unannotated triggers vLLM's flag-all
    fallback, which incorrectly treats target Mamba groups as draft state and
    disables cross-request prefix reuse.
    """
    if _orig_annotate_eagle_groups is not None:
        _orig_annotate_eagle_groups(
            vllm_config,
            kv_cache_spec,
            kv_cache_groups,
            use_trailing_layer_fallback=use_trailing_layer_fallback,
        )
    speculative_config = vllm_config.speculative_config
    if (
        speculative_config is None
        or not speculative_config.use_eagle_block_drop()
        or not _uses_qwen4_exp_mtp(vllm_config)
    ):
        return

    text_config = vllm_config.model_config.hf_text_config
    first_draft_layer = int(text_config.num_hidden_layers)
    num_draft_layers = int(
        getattr(
            getattr(
                speculative_config.draft_model_config,
                "hf_text_config",
                None,
            ),
            "mtp_num_hidden_layers",
            1,
        )
        or 1
    )
    draft_markers = tuple(
        f".layers.{layer_idx}."
        for layer_idx in range(
            first_draft_layer,
            first_draft_layer + num_draft_layers,
        )
    )
    for group in kv_cache_groups:
        if any(marker in f".{layer_name}." for marker in draft_markers for layer_name in group.layer_names):
            group.is_eagle_group = True

    if not any(group.is_eagle_group for group in kv_cache_groups):
        raise ValueError(f"Qwen4Exp MTP is enabled, but no KV cache group contains its draft layer(s) {draft_markers}.")


if _orig_annotate_eagle_groups is not None:
    _kv_cache_planning._annotate_eagle_groups = _ascend_annotate_eagle_groups
