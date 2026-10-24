# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""EAGLE-4 chain configuration; the optional harness is imported only on use."""

from copy import copy
from typing import Any

EAGLE4_METHOD = "eagle4"
EAGLE4_ARCH = "AscendEagle4ForCausalLM"
CHECKPOINT_ARCH = "LlamaForCausalLMEagle4"
_CHAIN_OPTIONS = frozenset({"mode", "intensity", "node_top_k", "max_verify_width", "tree_depth"})


def normalize_eagle4_config(config: Any) -> Any:
    """Select the NPU model without racing the harness's CUDA model registration."""
    if CHECKPOINT_ARCH in (getattr(config, "architectures", None) or ()):
        from eagle_harness.models.eagle4_config import resolve_draft_sliding_window

        window = resolve_draft_sliding_window(config)
        config.layer_types = ["sliding_attention" if window is not None else "full_attention"]
        config.architectures = [EAGLE4_ARCH]
        config.fc_norm = getattr(config, "fc_norm", False) or getattr(config, "enable_fc_norm", False)
    return config


def validate_draft_config(config: Any) -> int | None:
    # Pure model helpers intentionally avoid eagle_harness.vllm.models, which
    # imports CUDA candidate kernels even when only its model class is needed.
    from eagle_harness.models.eagle4_config import check_ported, resolve_draft_sliding_window

    checked = copy(config)
    checked.__dict__ = dict(config.__dict__, architectures=[CHECKPOINT_ARCH])
    for key in ("eagle_aux_hidden_state_layer_ids", "target_layer_count"):
        checked.__dict__.pop(key, None)
    check_ported(checked)
    window = resolve_draft_sliding_window(checked)
    if not (getattr(config, "eagle_config", None) or {}).get("use_aux_hidden_state", True):
        raise ValueError("Ascend EAGLE-4 requires auxiliary target hidden states.")
    if not getattr(config, "share_draft_embedding_from_target", True):
        raise NotImplementedError("Ascend EAGLE-4 currently requires sharing the target embedding.")
    head = config.efficient_lm_head_config
    for name in ("rank", "num_candidates"):
        value = head.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"efficient_lm_head_config.{name} must be a positive integer.")
    if head["rank"] > min(config.hidden_size, config.vocab_size):
        raise ValueError("EAGLE-4 candidate head rank exceeds its hidden/vocab size.")
    return window


def configure_eagle4(spec: Any) -> None:
    try:
        from eagle_harness.vllm.config import aux_hidden_state_layers, configure_speculative_config, parse_method_config
    except ModuleNotFoundError as exc:
        if exc.name == "eagle_harness":
            raise ImportError(
                "Ascend EAGLE-4 requires eagle-harness==0.1.1 (install without the CUDA extras)."
            ) from exc
        raise

    extra = set(spec.method_config) - _CHAIN_OPTIONS
    if extra:
        raise ValueError(f"Unsupported Ascend EAGLE-4 method_config options: {sorted(extra)}")
    config = {"mode": "static", "intensity": "chain_intensity", "node_top_k": 1, **spec.method_config}
    if "max_verify_width" not in config:
        if spec.num_speculative_tokens is None:
            raise ValueError("Set num_speculative_tokens or method_config.max_verify_width for Ascend EAGLE-4.")
        config["max_verify_width"] = spec.num_speculative_tokens + 1
    parsed = parse_method_config(config)
    if parsed.node_top_k != 1:
        raise NotImplementedError(
            "Ascend EAGLE-4 supports chain mode only (node_top_k=1); tree attention is not ported."
        )
    if parsed.harness.mode.value != "static" or parsed.harness.intensity.value != "chain_intensity":
        raise NotImplementedError(
            "Ascend EAGLE-4 requires mode='static', intensity='chain_intensity'; NPU costs are not calibrated."
        )
    if spec.rejection_sample_method != "standard":
        raise NotImplementedError("Ascend EAGLE-4 requires standard lossless rejection sampling.")
    if spec.num_speculative_tokens_per_batch_size is not None:
        raise NotImplementedError("Ascend EAGLE-4 currently requires a fixed chain length.")
    if spec.use_heterogeneous_vocab:
        raise NotImplementedError("Ascend EAGLE-4 requires the target vocabulary and tokenizer.")
    if spec.parallel_drafting or spec.use_local_argmax_reduction or spec.enable_adaptive_verification:
        raise NotImplementedError(
            "Ascend EAGLE-4 does not support parallel drafting, local argmax or adaptive verification."
        )
    if spec.quantization is not None or spec.draft_lm_head_quantization is not None:
        raise NotImplementedError("Ascend EAGLE-4 requires unquantized draft weights and candidate heads.")
    if spec.draft_model_config is None:
        raise ValueError("Ascend EAGLE-4 requires a draft model checkpoint.")
    draft = spec.draft_model_config.hf_config
    if EAGLE4_ARCH not in (draft.architectures or ()):
        raise ValueError("method='eagle4' requires a LlamaForCausalLMEagle4 checkpoint.")
    if getattr(draft, "quantization_config", None):
        raise NotImplementedError("Ascend EAGLE-4 does not support quantized draft checkpoints.")
    validate_draft_config(draft)
    if draft.vocab_size != spec.target_model_config.get_vocab_size():
        raise ValueError("The EAGLE-4 draft and target must have the same vocabulary size.")
    expected_hidden_size = getattr(draft, "target_hidden_size", draft.hidden_size)
    if expected_hidden_size != spec.target_model_config.get_hidden_size():
        raise ValueError("The EAGLE-4 checkpoint's target_hidden_size does not match the target model.")
    named_layers = (getattr(draft, "eagle_config", None) or {}).get("eagle_aux_hidden_state_layer_ids") or ()
    if any(isinstance(layer, bool) or not isinstance(layer, int) for layer in named_layers):
        raise ValueError("EAGLE-4 auxiliary output layer IDs must be integers.")
    layers = aux_hidden_state_layers(draft)
    target_layers = spec.target_model_config.get_total_num_hidden_layers()
    if (
        len(layers) != 3
        or len(set(layers)) != 3
        or layers != tuple(sorted(layers))
        or not all(0 < layer < target_layers for layer in layers)
    ):
        # EagleModelMixin emits states in layer order, so accepting an unordered
        # checkpoint would silently permute the trained FC/norm inputs.
        raise ValueError(
            "EAGLE-4 requires three distinct auxiliary output layers in ascending order within the target."
        )
    draft.eagle_aux_hidden_state_layer_ids = layers
    spec.method_config = config
    configure_speculative_config(spec)


def validate_eagle4_runtime(vllm_config: Any) -> None:
    """Reject execution modes whose scheduling/collectives are not validated."""
    spec = vllm_config.speculative_config
    if spec is None or spec.method != EAGLE4_METHOD:
        return
    if not vllm_config.use_v2_model_runner:
        raise NotImplementedError("Ascend EAGLE-4 requires Model Runner V2.")
    parallel = vllm_config.parallel_config
    if (
        parallel.pipeline_parallel_size != 1
        or parallel.data_parallel_size != 1
        or parallel.decode_context_parallel_size != 1
        or parallel.prefill_context_parallel_size != 1
        or parallel.enable_dbo
    ):
        raise NotImplementedError("Ascend EAGLE-4 currently supports TP only, without PP, DP, CP or DBO.")
    if vllm_config.lora_config is not None or vllm_config.model_config.is_multimodal_model:
        raise NotImplementedError("Ascend EAGLE-4 currently supports text targets without LoRA.")
    if vllm_config.kv_transfer_config is not None:
        raise NotImplementedError("Ascend EAGLE-4 does not yet support disaggregated KV transfer.")
    if vllm_config.cache_config.cache_dtype not in ("auto", "float16", "bfloat16"):
        raise NotImplementedError("Ascend EAGLE-4 currently requires an unquantized KV cache.")


def register_eagle4_method() -> None:
    # The release lane has no method registry. Leave it usable for other methods;
    # EAGLE-4 requires the vLLM branch containing the generic speculative hooks.
    try:
        from vllm.v1.spec_decode.registry import SpeculativeMethodSpec, register_speculative_method
    except ModuleNotFoundError as exc:
        if exc.name == "vllm.v1.spec_decode.registry":
            return
        raise

    register_speculative_method(
        EAGLE4_METHOD,
        SpeculativeMethodSpec(
            speculator_cls="vllm_ascend.worker.v2.spec_decode.eagle4.speculator:AscendEagle4Speculator",
            configure=configure_eagle4,
            uses_target_hidden_states=True,
            uses_aux_hidden_states=True,
        ),
    )
