# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A2 prefill policy and BF16/INT8 execution weights; no vLLM/NPU imports."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import ConfigDict, Field
from pydantic.dataclasses import dataclass

if TYPE_CHECKING:
    import torch

_A2_INTERMEDIATE_SIZE_MIN = 512
_A2_INTERMEDIATE_SIZE_ALIGNMENT = 128


@dataclass(config=ConfigDict(extra="forbid", frozen=True))
class A2MegaMoEPrefillConfig:
    """Opt-in EP8 BF16/W8A8 prefill experiment, configured in additional_config."""

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
    if dynamic_eplb or redundant_experts or mix_placement or getattr(parallel, "enable_eplb", False) is True:
        raise ValueError("a2_mega_moe_prefill requires static contiguous experts without EPLB or mix placement")
    if str(model.dtype) != "torch.bfloat16":
        raise ValueError("a2_mega_moe_prefill requires BF16 activations; expert quantization is checked per layer")
    text = model.hf_text_config
    if (text.hidden_size, text.moe_intermediate_size, model.get_num_experts(), text.num_experts_per_tok) != (
        2560,
        640,
        512,
        10,
    ):
        raise ValueError("a2_mega_moe_prefill requires H=2560, I=640, E=512 and top_k=10")


def prepare_bf16_expert_weights(w1: torch.Tensor, w2: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Share native-width ND expert views between MegaMoE and ordinary GMM."""
    # Defer torch during config/platform discovery; weights are processed in the worker.
    import torch

    if w1.dtype != torch.bfloat16 or w2.dtype != torch.bfloat16:
        raise ValueError("A2 prefill weights must be BF16")
    if w1.ndim != 3 or w2.ndim != 3 or w1.shape[:2] != (w2.shape[0], w2.shape[2]):
        raise ValueError("Expected expert weights [E,H,2I] and [E,I,H]")
    intermediate = w2.shape[1]
    if w1.shape[2] != 2 * intermediate:
        raise ValueError("W1 must contain equal gate and up halves")
    if intermediate < _A2_INTERMEDIATE_SIZE_MIN or intermediate % _A2_INTERMEDIATE_SIZE_ALIGNMENT != 0:
        raise ValueError("A2 MegaMoE requires intermediate size >= 512 and aligned to 128")
    return list(w1.unbind(0)), list(w2.unbind(0))


def validate_a2_expert_quantization(quant_type: Any) -> None:
    """Validate the resolved routed scheme, including partially quantized checkpoints."""
    if getattr(quant_type, "name", None) not in ("NONE", "W8A8"):
        raise ValueError("A2 MegaMoE prefill supports only BF16 or dynamic INT8 W8A8 routed experts")


def validate_w8a8_expert_weights(w1, w2, scale1, offset1, scale2, offset2) -> None:
    """Validate native-width symmetric per-channel INT8 before NZ conversion.

    Weights are in GMM direction. Scale1/offset1 have 2I output channels;
    scale2/offset2 have H output channels. No weight or scale padding is needed
    with an A2 operator supporting 128-aligned intermediate sizes.
    Called only during weight loading.
    """
    import torch

    if w1.dtype != torch.int8 or w2.dtype != torch.int8:
        raise ValueError("W8A8 MegaMoE requires INT8 weights")
    if w1.ndim != 3 or w2.ndim != 3 or w1.shape[:2] != (w2.shape[0], w2.shape[2]):
        raise ValueError("Expected expert weights [E,H,2I] and [E,I,H]")
    experts, hidden, two_i = w1.shape
    intermediate = w2.shape[1]
    if two_i != 2 * intermediate:
        raise ValueError("W1 must contain equal gate and up halves")
    if intermediate < _A2_INTERMEDIATE_SIZE_MIN or intermediate % _A2_INTERMEDIATE_SIZE_ALIGNMENT != 0:
        raise ValueError("A2 MegaMoE requires intermediate size >= 512 and aligned to 128")
    parameters = (scale1, offset1, scale2, offset2)
    for parameter, channels in zip(parameters, (two_i, two_i, hidden, hidden)):
        if parameter.shape[0] != experts or parameter.numel() != experts * channels:
            raise ValueError("W8A8 requires per-expert, per-output-channel scale and offset")
    # This API has no INT8 weight zero-point input. Never silently drop it.
    if torch.count_nonzero(offset1).item() or torch.count_nonzero(offset2).item():
        raise ValueError("A2 MegaMoE W8A8 requires symmetric weights (zero weight offsets)")
    if any(not torch.isfinite(scale).all().item() or not (scale > 0).all().item() for scale in (scale1, scale2)):
        raise ValueError("W8A8 scales must be finite and positive")
