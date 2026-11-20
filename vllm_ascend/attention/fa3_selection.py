# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.config import VllmConfig


# These targets may use FA3 on full-attention layers when their draft uses FA3.
_FA3_TARGET_ARCHITECTURES = frozenset(
    {
        "Qwen3NextForCausalLM",
        "Qwen3_5ForCausalLM",
        "Qwen3_5MoeForCausalLM",
        "Qwen3_5ForConditionalGeneration",
        "Qwen3_5MoeForConditionalGeneration",
    }
)
MINIMAX_M3_ARCHITECTURES = frozenset({"MiniMaxM3SparseForCausalLM", "MiniMaxM3SparseForConditionalGeneration"})
# Each target family opts in only its listed DSpark drafts.
_FA3_DSPARK_DRAFT_ARCHITECTURES = {
    MINIMAX_M3_ARCHITECTURES: frozenset({"DSparkMiniMaxDraftModel", "MiniMaxM31FlashA3DSparkModel"}),
}
_FA3_DFLASH_DRAFT_ARCHITECTURES = frozenset({"DFlash2DraftModel"})
_FA3_CACHE_DTYPES = ("auto", "float16", "bfloat16")


def _supports_dflash_fa3(config: VllmConfig, *, for_target: bool) -> bool:
    from vllm.config import VllmConfig
    from vllm.model_executor.model_loader.utils import get_draft_load_config

    spec = config.speculative_config
    draft = spec.draft_model_config
    # The target is constructed first. Resolve the same quantization recipe
    # that the DFlash loader will use when it constructs the draft layers.
    load_config = get_draft_load_config(config) if for_target else config.load_config
    quant_config = VllmConfig.get_quantization_config(draft, load_config)
    if getattr(quant_config, "enable_c8_quant", False):
        return False
    if not for_target:
        # During draft construction the platform sees the actual layer's
        # sinks, KV dtype and quantization scales in its selector.
        return True

    hf = draft.hf_config
    dflash = getattr(hf, "dflash_config", None) or {}
    if dflash.get("attention_sink_bias", getattr(hf, "add_swa_attention_sink_bias", False)):
        return False
    cache_dtype = spec.kv_cache_dtype or config.cache_config.cache_dtype
    scheme = getattr(quant_config, "kv_cache_scheme", None)
    if scheme is not None and scheme.get("strategy") == "attn_head":
        return False
    if cache_dtype == "auto" and scheme is not None:
        cache_dtype = "fp8"
    if cache_dtype in _FA3_CACHE_DTYPES:
        return True

    # Attention can override a quantized cache back to auto for individual
    # layers. DFlash numbers its layers after the last target layer.
    skip_layers = config.cache_config.kv_cache_dtype_skip_layers
    if not skip_layers:
        return False
    from vllm.model_executor.models.qwen3_dflash import _resolve_layer_attention

    first = config.model_config.get_total_num_hidden_layers()
    return all(
        str(first + i) in skip_layers
        or (
            "sliding_window" in skip_layers
            and (config.cache_config.sliding_window is not None or _resolve_layer_attention(hf, i)[0] is not None)
        )
        for i in range(hf.num_hidden_layers)
    )


def supports_fa3_model(config: VllmConfig) -> bool:
    spec = config.speculative_config
    v2 = getattr(config, "use_v2_model_runner", False)
    # V2 retains the target config while loading a tagged draft. The legacy
    # MiniMax route instead receives the currently loaded model's own config.
    if v2:
        from vllm.compilation import backends

        tag = backends.model_tag
    else:
        tag = "dspark_head"
    if spec is None or spec.draft_model_config is None:
        return False
    if tag == "backbone" and (
        getattr(config.model_config, "architecture", None) not in _FA3_TARGET_ARCHITECTURES
        or getattr(config.quant_config, "enable_c8_quant", False)
    ):
        return False
    target_hf = getattr(spec.target_model_config, "hf_config", None)
    target_architectures = getattr(target_hf, "architectures", None) or ()
    draft = spec.draft_model_config if v2 else config.model_config.hf_config
    draft_architectures = draft.architectures or ()
    for targets, drafts in _FA3_DSPARK_DRAFT_ARCHITECTURES.items():
        if not targets.isdisjoint(target_architectures):
            return (
                (tag, spec.method) == ("dspark_head", "dspark")
                and not drafts.isdisjoint(draft_architectures)
                and config.cache_config.cache_dtype in _FA3_CACHE_DTYPES
                and not getattr(config.quant_config, "enable_c8_quant", False)
            )
    if (
        tag not in ("backbone", "dflash_head")
        or spec.method != "dflash"
        or _FA3_DFLASH_DRAFT_ARCHITECTURES.isdisjoint(draft_architectures)
    ):
        return False
    return _supports_dflash_fa3(config, for_target=tag == "backbone")
