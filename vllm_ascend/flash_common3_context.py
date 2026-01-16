from dataclasses import dataclass

import torch
from vllm.model_executor.layers.linear import LinearBase


@dataclass
class FlashCommon3Context:
    gate: LinearBase | None = None
    topk_weights: torch.Tensor | None = None
    topk_ids: torch.Tensor | None = None
    row_idx: torch.Tensor | None = None
    shared_experts: torch.nn.Module | None = None
    shared_out: torch.Tensor | None = None


_flash_common3_context: FlashCommon3Context | None = None


def get_flash_common3_context() -> FlashCommon3Context | None:
    return _flash_common3_context


def set_flash_common3_context(
    topk_weights: torch.Tensor | None = None,
    topk_ids: torch.Tensor | None = None,
    shared_experts: torch.nn.Module | None = None,
    shared_out: torch.Tensor | None = None,
):
    global _flash_common3_context
    if _flash_common3_context is None:
        _flash_common3_context = FlashCommon3Context()

    if topk_weights is not None:
        _flash_common3_context.topk_weights = topk_weights
    if topk_ids is not None:
        _flash_common3_context.topk_ids = topk_ids
    if shared_experts is not None:
        _flash_common3_context.shared_experts = shared_experts
    if shared_out is not None:
        _flash_common3_context.shared_out = shared_out
