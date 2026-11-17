# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prefix expert reduction for Qwen3.5 functional debugging."""

import re
from collections.abc import Iterable, Iterator
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    from vllm.config import VllmConfig

_EXPERT = re.compile(r"(?:^|\.)mlp\.experts\.(\d+)\.")


def configure_expert_reduction(vllm_config: "VllmConfig", count: int) -> tuple[int, int] | None:
    if count == 0:
        return None
    config = vllm_config.model_config.hf_text_config
    if config.model_type != "qwen3_5_moe_text":
        raise ValueError("Expert reduction currently supports only Qwen3.5 MoE")
    original = getattr(config, "_ascend_original_num_experts", config.num_experts)
    if not config.num_experts_per_tok <= count <= original:
        raise ValueError(f"Expected top_k <= reduced experts <= {original}, got {count}")
    parallel = vllm_config.parallel_config
    if parallel.enable_eplb:
        raise ValueError("Expert reduction does not support EPLB")
    ep_size = parallel.tensor_parallel_size * parallel.data_parallel_size if parallel.enable_expert_parallel else 1
    if count % ep_size:
        raise ValueError("Reduced experts must be divisible by EP size")
    config._ascend_original_num_experts = original
    config.num_experts = count
    return original, count


def reduce_expert_weights(
    weights: Iterable[tuple[str, "torch.Tensor"]], reduction: tuple[int, int] | None
) -> Iterator[tuple[str, "torch.Tensor"]]:
    if reduction is None or reduction[0] == reduction[1]:
        yield from weights
        return
    original, count = reduction
    for name, weight in weights:
        match = _EXPERT.search(name)
        if match:
            if int(match.group(1)) >= count:
                continue
        elif ".mlp.experts." in "." + name:
            raise ValueError(f"Expert reduction requires per-expert checkpoint tensors, got {name}")
        if name.endswith(("mlp.gate.weight", "mlp.gate.bias")):
            if weight.shape[0] != original:
                raise ValueError(f"{name}: expected router dimension {original}, got {weight.shape[0]}")
            weight = weight[:count].contiguous()
        yield name, weight


class Qwen3_5ExpertReductionMixin:
    """Reduce before model allocation and filter before upstream weight mapping."""

    def __init__(self, *, vllm_config: "VllmConfig", **kwargs):
        # Resolve the worker's config lazily to keep model inspection dependency-light.
        from vllm_ascend.ascend_config import get_ascend_config

        count = get_ascend_config().expert_reduction.num_experts
        reduction = configure_expert_reduction(vllm_config, count)
        super().__init__(vllm_config=vllm_config, **kwargs)
        self._ascend_expert_reduction = reduction

    def load_weights(self, weights: Iterable[tuple[str, "torch.Tensor"]]) -> set[str]:
        return super().load_weights(reduce_expert_weights(weights, self._ascend_expert_reduction))
