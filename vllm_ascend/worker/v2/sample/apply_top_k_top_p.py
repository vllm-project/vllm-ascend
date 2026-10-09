import torch
from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p_pytorch

from vllm_ascend.sample.topk_topp import apply_top_k_top_p_with_fallback


def apply_top_k_top_p_npu(logits: torch.Tensor, k: torch.Tensor | None, p: torch.Tensor | None) -> torch.Tensor:
    """Use the same hardware and compiler guards as the V1 sampler."""
    return apply_top_k_top_p_with_fallback(logits, k, p, apply_top_k_top_p_pytorch)
