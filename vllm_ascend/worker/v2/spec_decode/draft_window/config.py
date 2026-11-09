# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
from __future__ import annotations

from typing import Any, Literal

from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.logger import logger

from vllm_ascend.ascend_config import get_ascend_config


def draft_uses_eager(config: VllmConfig) -> bool:
    """Use the execution request that the original Draft managers receive."""
    speculative = config.speculative_config
    return bool(
        speculative.enforce_eager
        or speculative.draft_model_config.enforce_eager
        or config.compilation_config.cudagraph_mode is CUDAGraphMode.NONE
    )


def resolve_draft_window_policy(
    *,
    requested_window_size: int | None,
    method: str,
    target_model_type: str = "",
    builtin_v4_dspark: bool | None = None,
    dynamic_k: bool = False,
    is_dflash2: bool = False,
) -> int | None:
    """Return the resolved window without constructing a runtime or policy object.

    ``None`` is CONFIG_OFF. MTP and an explicitly identified co-trained V4
    DSpark route are forced off before number or capability validation. The
    caller identifies the dedicated built-in category from normalized target
    and draft config. Paths and revisions do not define that category, and
    target model type alone does not establish it. ``False`` explicitly
    identifies another draft category; ``None`` leaves that boundary open.

    A resolved window establishes method policy only. Runtime consumers must
    independently validate their actual backend, geometry and ownership.
    """
    method = method.strip().lower()
    model_type = target_model_type.strip().lower()

    if requested_window_size is None:
        return None
    if method == "mtp":
        return None
    if method == "dspark" and builtin_v4_dspark is True:
        return None

    if method not in {"eagle3", "dflash", "dspark"}:
        raise NotImplementedError(f"draft_window_size is not supported for speculative method {method!r}")
    if is_dflash2 and method != "dflash":
        raise NotImplementedError("DFlash2 draft_window_size requires speculative method 'dflash'")
    if dynamic_k:
        raise NotImplementedError("draft_window_size requires static speculative decoding")
    if method == "dspark" and model_type.startswith("deepseek_v4") and builtin_v4_dspark is not False:
        raise ValueError(
            "DSpark with a V4 target requires a known model-route identity before enabling draft_window_size"
        )
    if isinstance(requested_window_size, bool) or not isinstance(requested_window_size, int):
        raise TypeError("draft_window_size must be an integer or None")
    if requested_window_size <= 0:
        raise ValueError("draft_window_size must be greater than zero")
    return requested_window_size


_COMMON_CONSUMER_CONFIG = {
    "attention_bias": False,
    "hidden_size": 2048,
}


_CONSUMER_CONFIGS = {
    "dflash_full8": {
        **_COMMON_CONSUMER_CONFIG,
        **{
            "architectures": ["DFlashDraftModel"],
            "dflash_config": {"mask_token_id": 248070, "target_layer_ids": [1, 10, 19, 28, 37]},
            "head_dim": 128,
            "layer_types": ["full_attention"] * 8,
            "model_type": "qwen3",
            "num_attention_heads": 32,
            "num_hidden_layers": 8,
            "num_key_value_heads": 4,
            "rope_parameters": {
                "beta_fast": 32.0,
                "beta_slow": 1.0,
                "factor": 64.0,
                "original_max_position_embeddings": 4096,
                "rope_type": "yarn",
                "rope_theta": 10000000,
            },
            "sliding_window": None,
            "use_sliding_window": False,
        },
    },
    "eagle3": {
        **_COMMON_CONSUMER_CONFIG,
        **{
            "architectures": ["LlamaForCausalLMEagle3"],
            "eagle_config": {"eagle_aux_hidden_state_layer_ids": [3, 19, 35]},
            "head_dim": 256,
            "model_type": "llama",
            "num_attention_heads": 16,
            "num_hidden_layers": 1,
            "num_key_value_heads": 2,
            "rope_parameters": {"rope_theta": 10000000, "rope_type": "default"},
            "use_sliding_window": False,
            "eagle_aux_hidden_state_layer_ids": [3, 19, 35],
        },
    },
}


_FULL8_QUERY_TOKENS = 8


def _consumer_profile(speculative):
    """Use the same normalized configuration that constructs the actual model."""
    hf = speculative.draft_model_config.hf_config
    model_type = getattr(hf, "model_type", None)
    if model_type == "eagle":
        from vllm.transformers_utils.configs.eagle import EAGLEConfig

        # Fixed vLLM wraps DFlash and Eagle configs before worker creation.
        # Keep checking the outer, effective projection/attention fields;
        # only the architecture's model type belongs to the wrapped model.
        if isinstance(hf, EAGLEConfig) and hf.model is not None:
            model_type = hf.model.model_type
    architectures = getattr(hf, "architectures", ()) or ()
    if architectures == ["DFlashDraftModel"] and model_type == "qwen3":
        if (
            getattr(hf, "layer_types", None) == ["sliding_attention"] * 5 + ["full_attention"]
            and getattr(hf, "sliding_window", None) == 4096
        ):
            return "dflash_swa6"
        # DFlashDraftModel is also the fixed GLM draft's upstream model class.
        # Its target-feature projection is not the Qwen Full8 consumer.
        target = getattr(speculative, "target_model_config", None)
        target_hf = getattr(target, "hf_text_config", None)
        if target is not None and target_hf is not None and getattr(target_hf, "model_type", None) == "glm_moe_dsa":
            dflash = getattr(hf, "dflash_config", None)
            aux = dflash.get("target_layer_ids", ()) if isinstance(dflash, dict) else ()
            layers = getattr(hf, "num_hidden_layers", None)
            if not (
                type(layers) is int
                and layers > 0
                and getattr(hf, "layer_types", None) == ["full_attention"] * layers
                and getattr(hf, "sliding_window", None) is None
                and getattr(hf, "use_sliding_window", False) is False
                and getattr(hf, "target_hidden_size", None) == target.get_hidden_size()
                and isinstance(aux, list)
                and aux
                and all(type(layer) is int and 0 <= layer < target_hf.num_hidden_layers for layer in aux)
                and len(set(aux)) == len(aux)
            ):
                raise NotImplementedError("GLM DFlash Graph requires its Full draft and matching Target feature layout")
            return "glm_dflash"
        profile = "dflash_full8"
    elif architectures == ["LlamaForCausalLMEagle3"]:
        profile = "eagle3"
    else:
        return None
    expected = _CONSUMER_CONFIGS[profile]
    for field, value in expected.items():
        actual = model_type if field == "model_type" else getattr(hf, field, None)
        if field == "rope_parameters" and isinstance(actual, dict):
            actual = dict(actual)
            actual.pop("type", None)  # Transformers can retain this legacy alias.
        if actual != value:
            raise NotImplementedError(f"Draft Graph consumer {field} differs from its qualified configuration")
    # These actual model overrides change projection/auxiliary/attention work;
    # none belongs to the qualified consumer. Packaging keys do not enter here.
    if any(
        getattr(hf, field, None) is not None
        for field in (
            "is_causal",
            "target_hidden_size",
            "num_aux_hidden_states",
            "norm_before_fc",
            "use_aux_hidden_state",
            "add_swa_attention_sink_bias",
            "quantization_config",
        )
    ) or (profile == "dflash_full8" and getattr(hf, "eagle_config", None)):
        raise NotImplementedError("Draft Graph consumer overrides have not been qualified")
    return profile


def resolve_graph_support(vllm_config) -> Literal[True] | None:
    """Parse startup support without changing the upstream configuration.

    Return True only for the protected qualified routes, and None otherwise.
    Method/capacity facts remain on the actual speculator; VllmConfig receives
    no undeclared fields because upstream DFlash loading copies it with replace.
    """
    speculative = vllm_config.speculative_config
    if speculative is None or speculative.method not in ("dflash", "eagle3") or draft_uses_eager(vllm_config):
        return None
    mode = vllm_config.compilation_config.cudagraph_mode
    decode = getattr(mode, "decode_mode", None)
    effective = decode() if callable(decode) else mode
    if getattr(effective, "name", effective) == "NONE":
        return None
    profile = _consumer_profile(speculative)
    if profile == "dflash_swa6":
        raise NotImplementedError(
            "This six-layer native-SWA DFlash Graph combination has an unresolved valid-output counterexample; "
            "use the independently supported Draft eager path. No fallback is selected."
        )
    if profile in ("dflash_full8", "eagle3"):
        return _validate_graph_consumer_config(vllm_config, profile)
    if profile != "glm_dflash":
        hf = speculative.draft_model_config.hf_config
        # Familiar names do not open an unknown normalized consumer geometry.
        architectures = getattr(hf, "architectures", ()) or ()
        if (speculative.method == "eagle3" and "LlamaForCausalLMEagle3" in architectures) or (
            speculative.method == "dflash"
            and "DFlashDraftModel" in architectures
            and getattr(hf, "model_type", None) == "qwen3"
        ):
            raise NotImplementedError("This dense Draft Graph model configuration has not been qualified")
    return None


def _validate_graph_consumer_config(vllm_config, profile):
    speculative = vllm_config.speculative_config
    parallel = vllm_config.parallel_config
    draft_parallel = speculative.draft_parallel_config
    compilation = vllm_config.compilation_config
    scheduler = vllm_config.scheduler_config
    additional = vllm_config.additional_config or {}
    eagle = profile == "eagle3"
    method, k, window, query = ("eagle3", 3, 1024, 4) if eagle else ("dflash", 7, 32768, 8)
    dp, rows = parallel.data_parallel_size, scheduler.max_num_seqs
    target_phase = (getattr(parallel, "all2all_backend", None), getattr(parallel, "use_sequence_parallel_moe", None))
    target_phase_valid = not eagle or target_phase in (
        ("allgather_reducescatter", True),
        ("flashinfer_all2allv", False),
    )
    captures = compilation.cudagraph_capture_sizes
    maximum = vllm_config.model_config.max_model_len
    if (
        speculative.method != method
        or str(getattr(speculative.draft_model_config, "dtype", "")) != "torch.bfloat16"
        or speculative.uses_dynamic_speculative_decoding()
        or speculative.num_speculative_tokens != k
        or speculative.enable_adaptive_verification is not False
        or dp not in (1, 2)
        or rows not in (1, 2)
        or (dp == 2 and rows != 1)
        or parallel.tensor_parallel_size != 4
        or any(
            getattr(parallel, field) != 1
            for field in ("pipeline_parallel_size", "prefill_context_parallel_size", "decode_context_parallel_size")
        )
        or parallel.enable_expert_parallel is not True
        or getattr(draft_parallel, "tensor_parallel_size", None) != 4
        or getattr(draft_parallel, "data_parallel_size", None) != 1
        or (eagle and getattr(draft_parallel, "enable_expert_parallel", None) is not False)
        or not target_phase_valid
        or getattr(compilation.cudagraph_mode, "name", None) != "FULL_DECODE_ONLY"
        or additional.get("draft_window_size") not in (None, window)
        or (additional.get("draft_window_size") is not None and type(additional["draft_window_size"]) is not int)
        or additional.get("enable_dsa_cp", False) is not False
        or type(additional.get("enable_flashcomm1", False)) is not bool
        or not captures
        or any(size <= 0 for size in captures)
        or not {query * count for count in range(1, rows + 1)}.issubset(captures)
        or (not eagle and captures != [query * count for count in range(1, rows + 1)])
        or compilation.max_cudagraph_capture_size is None
        or compilation.max_cudagraph_capture_size < query * rows
        or maximum <= k
    ):
        raise NotImplementedError(
            f"Qualified {method} Draft Graph requires TP4, DP1 local capacity 1/2 or DP2 local capacity 1, "
            f"static K{k}, Window OFF/W{window}, exact admitted padded rows and corresponding Q{query} buckets; "
            "larger/unknown shapes remain protected. No eager fallback is selected."
        )
    if not eagle:
        _validate_full8_capacity(vllm_config)
    return True


def note_graph_request_lifecycle(owner, scheduler_output):
    """Record a qualified runner's preemption IDs until real completion.

    Fixed MRV2 resumption is re-emitted as scheduled_new_reqs, including the case
    where no output token exists yet. prefill_len==prompt_len cannot detect
    that case. No scheduler, request state, or device object is modified.
    """
    blocked = getattr(owner, "_draft_window_preempted_request_ids", None)
    if blocked is None:
        blocked = set()
        owner._draft_window_preempted_request_ids = blocked
    blocked.update(scheduler_output.preempted_req_ids or ())
    blocked.difference_update(scheduler_output.finished_req_ids)


def validate_graph_request_bounds(owner, input_batch):
    """Check a qualified runner's CPU limits before Target attention writes.

    The fixed worker serializes request updates and proposals. Prompt/prefill
    limits change only when a request is added, not during Target sampling or
    Draft consumers; use that actual owner rather than a second bounds snapshot.
    """
    req_states = owner._draft_window_request_state
    blocked = getattr(owner, "_draft_window_preempted_request_ids", ())
    lookahead = owner.num_speculative_steps if owner.method == "eagle3" else owner.num_query_per_req
    for req_id, raw_index in zip(input_batch.req_ids, input_batch.idx_mapping_np, strict=True):
        if req_id in blocked:
            raise NotImplementedError("Draft Graph does not qualify resumption after preemption")
        row = int(raw_index)
        prompt = int(req_states.prompt_len.np[row])
        if int(req_states.prefill_len.np[row]) != prompt:
            raise NotImplementedError("Draft Graph does not qualify resumed request history")
        maximum = int(req_states.max_seq_len[row])
        if not 0 < prompt < maximum or maximum + lookahead > owner.max_model_len:
            raise NotImplementedError("Draft Graph prompt/output budget and query lookahead exceed allocated capacity")


def validate_eagle_graph_exact(owner, input_batch, exact, *, num_reqs, padded, step):
    """Inspect the existing CPU exact owner, never an upper bound or new D2H."""
    scope = getattr(owner, "_draft_window_graph_support", None)
    if scope is None or owner.method != "eagle3":
        return
    # The real Eagle producer owns CPU int32 storage and validates the step.
    if num_reqs != len(input_batch.idx_mapping_np) or padded != num_reqs:
        raise RuntimeError("Eagle Graph lost its exact admitted-row CPU consumer input")
    state = owner._draft_window_request_state
    provider = getattr(owner, "_draft_window_graph_binding_provider", None)
    continuing_rows = getattr(provider, "proposal", None) or ()
    for position, raw_index in enumerate(input_batch.idx_mapping_np):
        row = int(raw_index)
        prompt = int(state.prompt_len.np[row])
        value = exact[position].item()
        target_end = int(input_batch.seq_lens_np[position])
        query = int(input_batch.query_start_loc_np[position + 1] - input_batch.query_start_loc_np[position])
        if query < 1 or not 0 <= target_end <= owner.max_model_len:
            raise RuntimeError("Eagle Graph lost its current Target query/exact extent")
        # A final uniform Target query may extend past max_tokens, which is a
        # stopping budget, not its initialized KV extent. The first consumer
        # keeps that exact end; AR removes at most the row's draft queries and
        # advances by its ordinal, as prepare_decode_inputs does upstream.
        rejected = min(owner.num_speculative_steps, query - 1) if step else 0
        lower = max(prompt, min(target_end - rejected + step, owner.max_model_len))
        upper = min(target_end + step, owner.max_model_len)
        if continuing_rows and continuing_rows[position]:
            # The proposal epoch already classified and validated this row from
            # canonical InputBatch counters. In a mixed prefill, its first
            # consumer ends before the prompt; later Eagle AR consumers advance
            # that end by their step, while the row remains a Full-source read.
            end = int(input_batch.num_computed_prefill_tokens_np[position]) + int(
                input_batch.num_scheduled_tokens[position]
            )
            lower = upper = min(end + step, owner.max_model_len)
        if not lower <= value <= upper:
            raise NotImplementedError(
                "Eagle consumer exact length is outside its request and model capacity: "
                f"row={position}, step={step}, exact={value!r}, expected=[{lower}, {upper}]"
            )


def validate_eagle_graph_descriptor(owner, *, first, n, tokens):
    """Check the executed Graph shape, including Window OFF consumers."""
    scope = getattr(owner, "_draft_window_graph_support", None)
    if scope is None or owner.method != "eagle3":
        return
    batch = owner.input_batch
    # Propose already admitted active/padded rows and their request bounds.
    # Only the selected runtime descriptor can change before this replay.
    if n != batch.num_reqs or tokens != n * (owner.num_speculative_steps + 1 if first else 1):
        raise NotImplementedError(
            "Eagle Graph requires padded1 or explicitly admitted padded2, first Q4 or AR Q1 before replay"
        )


def _validate_full8_capacity(vllm_config):
    """Use the allocated sequence capacity and both actual model ceilings."""
    target = vllm_config.model_config
    draft = vllm_config.speculative_config.draft_model_config
    capacity = target.max_model_len
    for config in (target, draft):
        ceiling = getattr(config.hf_text_config, "max_position_embeddings", None)
        if type(ceiling) is not int or not _FULL8_QUERY_TOKENS < capacity <= ceiling:
            raise NotImplementedError(
                "Full8 Graph requires an exact positive configured capacity within both model position ceilings"
            )


def validate_graph_requests(owner, input_batch, *, dummy_run=False, is_profile=False, context_kv_only=False):
    """Check the actual execution shape before changing Graph ownership.

    Request identity, preemption and immutable prompt/output bounds were checked
    against RequestState when the runner constructed this InputBatch. Fixed
    MRV2 does not change those CPU fields before propose; only device lengths
    advance during Target postprocessing and remain checked by each consumer.
    """
    scope = getattr(owner, "_draft_window_graph_support", None)
    if scope is None or dummy_run or is_profile or getattr(input_batch, "is_dummy", False):
        return
    if owner.method == "dflash" and type(input_batch.num_reqs) is int and input_batch.num_reqs == 0:
        return
    active, padded = input_batch.num_reqs, input_batch.num_reqs_after_padding
    if (
        type(active) is not int
        or not 1 <= active <= owner.max_num_reqs
        or (not context_kv_only and (type(padded) is not int or padded != active))
        or len(input_batch.req_ids) != active
        or len(input_batch.idx_mapping_np) != active
    ):
        raise NotImplementedError("Draft Graph requires admitted actual rows and their corresponding padded bank")


def _builtin_v4_dspark_identity(speculative_config, target_model_config) -> bool | None:
    """Identify the dedicated V4 DSpark category from normalized config.

    Upstream registers these architectures for V4 DSpark stages under mtp.*.
    Paths, revisions and packaging do not change that model category. This is
    route identification, not a claim to reconstruct checkpoint training history.
    """
    from vllm_ascend.utils import is_deepseek_v41

    draft = speculative_config.draft_model_config
    target_hf = target_model_config.hf_config
    model_type = "deepseek_v41" if is_deepseek_v41(target_hf) else target_hf.model_type
    expected_architectures = {
        "deepseek_v4": ("DSparkDraftModel",),
        # Ascend normalizes the upstream name after multimodal conversion.
        "deepseek_v41": ("DSparkV41DraftModel", "DeepseekV41DSparkModel"),
    }.get(model_type, ())
    if (
        bool(expected_architectures)
        and draft.hf_config.model_type == model_type
        and any(architecture in draft.architectures for architecture in expected_architectures)
    ):
        return True
    # Recognized external categories follow the switch even with a V4 target.
    # Compatibility with that target remains the model loader's contract.
    external_architectures = ("Qwen3DSparkModel", "Qwen3OmniDSparkModel", "Glm5DSparkForCausalLM")
    if any(architecture in draft.architectures for architecture in external_architectures):
        return False
    return None


def _resolve_window_policy(vllm_config: VllmConfig) -> int | None:
    requested = get_ascend_config().draft_window_size
    if requested is None:
        return None
    speculative_config = vllm_config.speculative_config
    method = speculative_config.method
    # Forced OFF must precede numerical checks and accesses to optional model
    # fields, dynamic-K state, or backend capabilities.
    if method == "mtp":
        forced_off: str | None = "MTP"
    else:
        target = vllm_config.model_config
        target_model_type = target.hf_config.model_type
        builtin_v4 = _builtin_v4_dspark_identity(speculative_config, target) if method == "dspark" else None
        forced_off = "the built-in DeepSeek V4 DSpark route" if builtin_v4 is True else None
    if forced_off is not None:
        logger.warning(
            "MRV2 draft_window_size is ignored for %s; Draft Sliding Window is disabled. "
            "The original speculative method is retained.",
            forced_off,
        )
        return None
    architectures = speculative_config.draft_model_config.architectures
    if method == "dflash" and (
        not architectures
        or len(set(architectures)) != 1
        or architectures[0] not in {"DFlashDraftModel", "DFlash2DraftModel"}
    ):
        raise NotImplementedError(
            "draft_window_size with method 'dflash' requires a supported "
            "DFlashDraftModel or DFlash2DraftModel architecture"
        )
    return resolve_draft_window_policy(
        requested_window_size=requested,
        method=method,
        target_model_type=target_model_type,
        builtin_v4_dspark=builtin_v4,
        dynamic_k=speculative_config.uses_dynamic_speculative_decoding(),
        is_dflash2="DFlash2DraftModel" in architectures,
    )


def _use_graph_window_owner(vllm_config: VllmConfig, policy: int | None, support: Any) -> bool:
    """Keep the ordinary eager runtime and persistent Graph owner exclusive."""
    if policy is None or draft_uses_eager(vllm_config):
        return False
    validate_draft_window_graph_config(vllm_config, support=support)
    return True


def validate_draft_window_graph_config(vllm_config: Any, *, support: Any) -> None:
    """Validate topology after the factory resolves policy and Graph support."""
    from vllm_ascend.device.hardware_profile import AttentionBackendFamily, get_current_hardware_profile

    if get_current_hardware_profile().attention_backend_family is not AttentionBackendFamily.STANDARD:
        raise NotImplementedError("Draft Window Graph requires the STANDARD hardware family")
    # The qualified consumer contract already validates its exact topology,
    # buckets, static K and OFF/ON window. Do not parse/check it a second time.
    if support is not None:
        return
    parallel = vllm_config.parallel_config
    speculative = vllm_config.speculative_config
    if parallel.pipeline_parallel_size != 1:
        raise RuntimeError("Draft Window Graph requires PP=1")
    if parallel.prefill_context_parallel_size != 1:
        draft_parallel = speculative.draft_parallel_config
        # Official PCP restores global requests before the replicated DSpark
        # draft prepares its own positions, Full slots and exact lengths.
        # The loaded provider still validates its PCP1/DCP1 reader and tables.
        if not (
            speculative.method == "dspark"
            and vllm_config.model_config.hf_text_config.model_type == "glm_moe_dsa"
            and not speculative.draft_model_config.use_mla
            and parallel.tensor_parallel_size == parallel.prefill_context_parallel_size == 4
            and parallel.decode_context_parallel_size == parallel.data_parallel_size == 1
            and parallel.enable_expert_parallel
            and speculative.num_speculative_tokens == 8
            and not speculative.enable_adaptive_verification
            and draft_parallel is not None
            and draft_parallel.prefill_context_parallel_size == draft_parallel.decode_context_parallel_size == 1
        ):
            raise NotImplementedError("Draft Window Graph PCP requires replicated GLM DSpark TP4/PCP4, DP1/DCP1 and K8")
    if parallel.decode_context_parallel_size != 1:
        # Upstream gives a dense parallel drafter replicated KV even when the
        # GLM Target shards its KV across the TP group. The loaded provider
        # checks that actual Draft group before allocating its read bank.
        if not (
            speculative.method in ("dspark", "dflash")
            and vllm_config.model_config.hf_text_config.model_type == "glm_moe_dsa"
            and not speculative.draft_model_config.use_mla
            and parallel.decode_context_parallel_size == parallel.tensor_parallel_size
        ):
            raise NotImplementedError("Draft Window Graph requires replicated Draft KV under Target DCP")
    additional = vllm_config.additional_config or {}
    if additional.get("draft_window_size") != 1024:
        raise ValueError("Draft Window Graph requires W1024 for this consumer")
    draft_parallel = speculative.draft_parallel_config
    tp = vllm_config.parallel_config.tensor_parallel_size
    if tp not in (2, 4, 16) or draft_parallel is None or draft_parallel.tensor_parallel_size != tp:
        raise ValueError("replicated TP Graph support requires matching actual Target/Draft TP in (2, 4, 16)")
    if speculative.method == "eagle3" and (tp != 2 or speculative.num_speculative_tokens != 3):
        raise ValueError("Eagle3 replicated Graph support requires static K3 and TP2 or the named dense TP4 geometry")
    dp = getattr(vllm_config.parallel_config, "data_parallel_size", 1)
    if dp not in (1, 2):
        raise ValueError("replicated Graph support requires integer DP1 or the explicit DFlash TP4/DP2 route")
    if dp == 2 and (
        speculative.method != "dflash"
        or tp != 4
        # Pinned SpeculativeConfig.create_draft_parallel_config builds a
        # TP-only model descriptor with DP1. The actual two replicas live
        # in the parent engine/groups, retained by the runtime fingerprint.
        # Requiring descriptor DP2 rejects the official DP2 constructor.
        or draft_parallel.data_parallel_size != 1
        or getattr(vllm_config.parallel_config, "enable_expert_parallel", False) is not True
    ):
        raise ValueError(
            "DP2 Graph support requires actual DFlash TP4/DP2, the pinned TP-only Draft descriptor and expert parallel"
        )
    if getattr(vllm_config.compilation_config.cudagraph_mode, "name", None) != "FULL_DECODE_ONLY":
        raise ValueError("replicated TP Graph support requires FULL_DECODE_ONLY")
