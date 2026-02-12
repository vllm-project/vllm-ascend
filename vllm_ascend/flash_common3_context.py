from dataclasses import dataclass
from typing import Optional

import torch
from vllm.model_executor.layers.linear import LinearBase


@dataclass
class FlashCommon3Context:
    gate: Optional[LinearBase] = None
    topk_weights: Optional[torch.Tensor] = None
    topk_ids: Optional[torch.Tensor] = None
    row_idx: Optional[torch.Tensor] = None
    shared_experts: Optional[torch.nn.Module] = None
    shared_out: Optional[torch.Tensor] = None


_flash_common3_context: Optional[FlashCommon3Context] = None


def get_flash_common3_context() -> Optional[FlashCommon3Context]:
    return _flash_common3_context


def set_flash_common3_context(
    topk_weights: Optional[torch.Tensor] = None,
    topk_ids: Optional[torch.Tensor] = None,
    shared_experts: Optional[torch.nn.Module] = None,
    shared_out: Optional[torch.Tensor] = None,
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
