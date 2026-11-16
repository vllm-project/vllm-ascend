# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU policy for the opt-in DeepSeek V4.1 A2 prefill backend."""

from collections.abc import Callable
from contextvars import ContextVar
from typing import Any, Literal

from pydantic import ConfigDict, Field
from pydantic.dataclasses import dataclass

DEEPSEEK_V41_ROUTED_EXPERTS = 384
# Only active during synchronous VllmConfig reconstruction. Context-local state
# keeps concurrent/nested draft construction isolated and is reset on failure.
_REPLICATED_PCP_SOURCE: ContextVar[Any] = ContextVar("a2_replicated_pcp_source", default=None)


@dataclass(config=ConfigDict(extra="forbid", frozen=True))
class A2MegaMoEPrefillConfig:
    enabled: bool = Field(default=False, strict=True)
    scope: Literal["prefill"] = "prefill"


def a2_prefill_enabled(ascend_config: Any) -> bool:
    return getattr(getattr(ascend_config, "a2_mega_moe_prefill", None), "enabled", False) is True


def is_draft_model(config: Any) -> bool:
    if getattr(config, "_ascend_a2_replicated_pcp_draft", False) is True:
        return True
    source = _REPLICATED_PCP_SOURCE.get()
    if (
        source is not None
        and config is not source
        and config.model_config is source.model_config
        and source.parallel_config.prefill_context_parallel_size > 1
        and config.parallel_config.prefill_context_parallel_size == 1
    ):
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
    token = _REPLICATED_PCP_SOURCE.set(source)
    try:
        draft = rebuild(source, **changes)
    finally:
        _REPLICATED_PCP_SOURCE.reset(token)
    draft._ascend_a2_replicated_pcp_draft = True
    return draft


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
