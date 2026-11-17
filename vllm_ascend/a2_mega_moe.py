# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU policy for the opt-in DeepSeek V4.1 A2 prefill backend."""

from collections.abc import Callable
from typing import Any, Literal

from pydantic import ConfigDict, Field
from pydantic.dataclasses import dataclass

DEEPSEEK_V41_ROUTED_EXPERTS = 384


@dataclass(config=ConfigDict(extra="forbid", frozen=True))
class A2MegaMoEPrefillConfig:
    enabled: bool = Field(default=False, strict=True)
    scope: Literal["prefill"] = "prefill"


class _ReplicatedPCPDraftSettings(dict[str, Any]):
    """Internal draft identity carried by an otherwise ordinary settings dict.

    Nested dict values survive VllmConfig validation, copy and replace. Keeping
    the original keys also preserves JSON hashing and AscendConfig validation.
    """


def a2_prefill_enabled(ascend_config: Any) -> bool:
    return getattr(getattr(ascend_config, "a2_mega_moe_prefill", None), "enabled", False) is True


def is_draft_model(config: Any) -> bool:
    additional = getattr(config, "additional_config", None)
    if isinstance(additional, dict) and isinstance(additional.get("a2_mega_moe_prefill"), _ReplicatedPCPDraftSettings):
        return True
    model = config.model_config
    architectures = getattr(model, "architectures", None) or [getattr(model, "architecture", None)]
    return getattr(model, "runner_type", None) == "draft" or "DeepseekV41DSparkModel" in architectures


def rebuild_replicated_pcp_draft(source: Any, rebuild: Callable[..., Any], **changes: Any) -> Any:
    """Identify the internal PCP=1 draft before its model identity is replaced.

    Preserve the target's enabled additional_config: AscendConfig is shared by
    target and draft in a worker. Disabling that flag could disable the target.
    """
    additional = getattr(source, "additional_config", None)
    settings = additional.get("a2_mega_moe_prefill", {}) if isinstance(additional, dict) else {}
    enabled = settings.get("enabled", False) if isinstance(settings, dict) else getattr(settings, "enabled", False)
    if enabled is not True:
        return rebuild(source, **changes)
    draft_settings = _ReplicatedPCPDraftSettings(
        settings if isinstance(settings, dict) else {"enabled": settings.enabled, "scope": settings.scope}
    )
    # Copy only draft input; the target and its public opt-in remain untouched.
    draft_additional = {**additional, "a2_mega_moe_prefill": draft_settings}
    return rebuild(source, additional_config=draft_additional, **changes)


def validate_a2_mega_moe_prefill(config: Any, *, dynamic_eplb: bool, redundant_experts: int, mix_placement: bool):
    # Draft configs inherit additional_config while loading in the P worker.
    # They retain ordinary MoE weights and never select the new backend.
    if is_draft_model(config):
        return
    parallel = config.parallel_config
    model = config.model_config
    architectures = getattr(model, "architectures", None) or [getattr(model, "architecture", None)]
    if "DeepseekV41ForCausalLM" not in architectures:
        raise ValueError("a2_mega_moe_prefill requires DeepSeek-V4.1-Flash")
    if (
        not parallel.enable_expert_parallel
        or (parallel.data_parallel_size, parallel.tensor_parallel_size, parallel.prefill_context_parallel_size)
        != (2, 1, 8)
        or parallel.pipeline_parallel_size != 1
        or parallel.decode_context_parallel_size != 1
        or parallel.use_ubatching
        or getattr(parallel, "expert_placement_strategy", "linear") != "linear"
    ):
        raise ValueError("a2_mega_moe_prefill requires DP2/TP1/PCP8, EP16, PP1/DCP1, linear experts, no microbatching")
    transfer = getattr(config, "kv_transfer_config", None)
    if transfer is None or transfer.kv_role != "kv_producer":
        raise ValueError("a2_mega_moe_prefill is restricted to the PD kv_producer (P) worker")
    if not model.enforce_eager or not config.use_v2_model_runner:
        raise ValueError("a2_mega_moe_prefill requires --enforce-eager and model runner V2")
    if str(model.dtype) != "torch.bfloat16":
        raise ValueError("a2_mega_moe_prefill requires BF16 activations")
    text = model.hf_text_config
    if (text.hidden_size, text.moe_intermediate_size, model.get_num_experts(), text.num_experts_per_tok) != (
        5120,
        2304,
        DEEPSEEK_V41_ROUTED_EXPERTS,
        6,
    ):
        raise ValueError("a2_mega_moe_prefill requires H=5120, I=2304, E=384 and top_k=6")
    if (
        getattr(config, "lora_config", None) is not None
        or parallel.enable_eplb
        or dynamic_eplb
        or redundant_experts
        or mix_placement
    ):
        raise ValueError("a2_mega_moe_prefill requires static contiguous experts without LoRA/EPLB/mix placement")


def prefill_vote(metadata: Any) -> tuple[int, int]:
    """Return (prefill evidence, veto) using explicit CPU DSA request counts.

    Empty replicas may participate with no active rows. Unknown, decode or
    mixed phases veto fusion across the entire EP group. Query length is never
    used: one-token prefix-cache tails and decoder replay are still prefill.
    """
    entries = metadata.values() if isinstance(metadata, dict) else (metadata,)
    evidence = False
    known = False
    for meta in entries:
        # PCP keeps a placeholder request on an empty shard. Its request phase
        # can look like decode even though every routed row is padding.
        actual_tokens = getattr(meta, "num_actual_tokens", None)
        if isinstance(actual_tokens, int) and actual_tokens == 0:
            known = True
            continue
        counts = tuple(getattr(meta, name, None) for name in ("num_reqs", "num_prefills", "num_decodes"))
        if not all(isinstance(value, int) and value >= 0 for value in counts):
            return 0, 1
        requests, prefills, decodes = counts
        if decodes or prefills != requests:
            return 0, 1
        known = True
        evidence |= prefills > 0
    return int(evidence), int(not known)


def select_prefill_from_votes(prefill: int, veto: int) -> bool:
    return prefill > 0 and veto == 0


def prefill_metadata_summary(metadata: Any) -> list[dict[str, Any]]:
    """Describe vote inputs without reading device tensors or logging token IDs."""
    entries = metadata.items() if isinstance(metadata, dict) else (("metadata", metadata),)
    groups: dict[int, dict[str, Any]] = {}
    fields = ("num_actual_tokens", "num_reqs", "num_prefills", "num_decodes")
    for name, meta in entries:
        key = id(meta)
        if key in groups:
            groups[key]["aliases"] += 1
            continue
        values = {field: getattr(meta, field, None) for field in fields}
        vote = prefill_vote(meta)
        actual_tokens = values["num_actual_tokens"]
        if isinstance(actual_tokens, int) and actual_tokens == 0:
            reason = "empty_tokens"
        elif vote[1]:
            counts = tuple(values[field] for field in fields[1:])
            if not all(isinstance(value, int) and value >= 0 for value in counts):
                reason = "invalid_or_missing_counts"
            elif values["num_decodes"]:
                reason = "decode_or_mixed"
            else:
                reason = "request_count_mismatch"
        else:
            reason = "prefill" if vote[0] else "empty_requests"
        # Never stringify tensor values: repr(Tensor) can synchronize the NPU.
        values = {
            field: value if value is None or isinstance(value, int) else type(value).__name__
            for field, value in values.items()
        }
        groups[key] = dict(
            layer=str(name), aliases=1, metadata_type=type(meta).__name__, **values, vote=vote, reason=reason
        )
    return list(groups.values())
