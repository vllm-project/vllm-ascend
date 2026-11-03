# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A2 prefill policy and BF16 execution weights; no vLLM/NPU imports."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import ConfigDict, Field
from pydantic.dataclasses import dataclass

if TYPE_CHECKING:
    import torch


@dataclass(config=ConfigDict(extra="forbid", frozen=True))
class A2MegaMoEPrefillConfig:
    """Opt-in EP8 BF16 prefill experiment, configured in additional_config."""

    enabled: bool = Field(default=False, strict=True)

    def selects(self, metadata: Any, *, in_profile_run: bool = False) -> bool:
        # Profiling exercises fused allocation even without live phase metadata.
        return self.enabled and (in_profile_run or is_pure_prefill(metadata))


def is_pure_prefill(metadata: Any) -> bool:
    """Use explicit CPU phase evidence, including Qwen's complete QSA plan.

    GDN metadata may not have phase counters. QSA's request plan covers all
    active rows, so unknown GDN entries do not invalidate that evidence.
    A decode/mixed QSA plan or any explicit decode counter always vetoes it.
    Never infer the phase from query length or copy a device tensor to CPU.
    """
    entries = metadata.values() if isinstance(metadata, dict) else (metadata,)
    evidence = []
    for meta in entries:
        if hasattr(meta, "qsa_request_plan"):
            plan = meta.qsa_request_plan
            requests = getattr(plan, "requests", ())
            if not requests or not all(req.is_prefill is True for req in requests):
                return False
            evidence.append(True)
        prefills = getattr(meta, "num_prefills", None)
        decodes = getattr(meta, "num_decode_tokens", None)
        if isinstance(decodes, int) and decodes > 0:
            return False
        if isinstance(prefills, int) and isinstance(decodes, int):
            if prefills <= 0 or decodes != 0:
                return False
            evidence.append(True)
    return bool(evidence)


def validate_a2_prefill_model(config: Any, *, dynamic_eplb: bool, redundant_experts: int, mix_placement: bool) -> None:
    """Reject unsupported topologies before allocating collective resources."""
    parallel = config.parallel_config
    model = config.model_config
    architectures = getattr(model, "architectures", None) or [getattr(model, "architecture", None)]
    allowed = {"Qwen4ExpForCausalLM", "Qwen4ExpForConditionalGeneration"}
    if not any(arch in allowed for arch in architectures):
        raise ValueError("a2_mega_moe_prefill requires Qwen3.8-Flash-Next (Qwen4Exp)")
    if (
        not parallel.enable_expert_parallel
        or parallel.tensor_parallel_size != 8
        or parallel.data_parallel_size != 1
        or parallel.pipeline_parallel_size != 1
        or parallel.prefill_context_parallel_size != 1
    ):
        raise ValueError("a2_mega_moe_prefill requires TP8/EP8, DP1, PP1 and PCP1")
    if getattr(config, "speculative_config", None) is not None or getattr(config, "lora_config", None) is not None:
        raise ValueError("a2_mega_moe_prefill does not support speculative decoding or LoRA")
    if dynamic_eplb or redundant_experts or mix_placement:
        raise ValueError("a2_mega_moe_prefill requires static contiguous experts without EPLB or mix placement")
    if getattr(config, "quant_config", None) is not None or str(model.dtype) != "torch.bfloat16":
        raise ValueError("a2_mega_moe_prefill currently requires unquantized BF16 weights and activations")
    text = model.hf_text_config
    if (text.hidden_size, text.moe_intermediate_size, model.get_num_experts(), text.num_experts_per_tok) != (
        2560,
        640,
        512,
        10,
    ):
        raise ValueError("a2_mega_moe_prefill requires H=2560, I=640, E=512 and top_k=10")


def pad_bf16_expert_weights(w1: torch.Tensor, w2: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Build one shared ND TensorList for fused prefill and ordinary fallback.

    GMM direction is [E,H,2I] and [E,I,H]. Pad gate and up separately;
    appending to the concatenated W1 would move its gate/up split.
    """
    # Defer torch during config/platform discovery; weights are processed in the worker.
    import torch

    if w1.dtype != torch.bfloat16 or w2.dtype != torch.bfloat16:
        raise ValueError("A2 prefill weights must be BF16")
    if w1.ndim != 3 or w2.ndim != 3 or w1.shape[:2] != (w2.shape[0], w2.shape[2]):
        raise ValueError("Expected expert weights [E,H,2I] and [E,I,H]")
    intermediate = w2.shape[1]
    if w1.shape[2] != 2 * intermediate:
        raise ValueError("W1 must contain equal gate and up halves")
    physical = (intermediate + 255) // 256 * 256
    first, second = [], []
    for gate_up, down in zip(w1.unbind(0), w2.unbind(0)):
        padded = gate_up.new_zeros((gate_up.shape[0], 2 * physical))
        padded[:, :intermediate].copy_(gate_up[:, :intermediate])
        padded[:, physical : physical + intermediate].copy_(gate_up[:, intermediate:])
        padded_down = down.new_zeros((physical, down.shape[1]))
        padded_down[:intermediate].copy_(down)
        first.append(padded)
        second.append(padded_down)
    return first, second
