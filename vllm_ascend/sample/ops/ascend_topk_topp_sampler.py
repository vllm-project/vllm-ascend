from typing import Dict, Optional

import torch
from vllm.v1.sample.ops.topk_topp_sampler import TopKTopPSampler, random_sample


class AscendTopKTopPSampler(TopKTopPSampler):

    def forward_native(
        self,
        logits: torch.Tensor,
        generators: Dict[int, torch.Generator],
        k: Optional[torch.Tensor],
        p: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Optimized implementation of top-k and top-p sampling on NPU."""
        logits = apply_top_k_top_p_npu(logits, k, p)
        probs = logits.softmax(dim=-1, dtype=torch.float32)
        return random_sample(probs, generators)


def apply_top_k_top_p_npu(
    logits: torch.Tensor,
    k: Optional[torch.Tensor],
    p: Optional[torch.Tensor],
) -> torch.Tensor:
    """Apply top-k and/or top-p optimized for NPU."""
    if k is None and p is None:
        return logits

    batch_size, vocab_size = logits.shape
    device = logits.device
    logits_sort, logits_idx = logits.sort(dim=-1, descending=False)
    if k is not None:
        safe_k = torch.clamp(k, min=1, max=vocab_size)
        boundary_idx = (vocab_size - safe_k).unsqueeze(1)
        boundary = logits_sort.gather(1, boundary_idx)
        top_k_mask = logits_sort < boundary
        logits_sort = logits_sort.masked_fill(top_k_mask, -float("inf"))
    else:
        top_k_mask = torch.zeros_like(logits_sort, dtype=torch.bool)

    cutoffs = top_k_mask.sum(dim=-1)
    strides = torch.arange(0,
                           batch_size * vocab_size,
                           vocab_size,
                           device=device).unsqueeze(1)
    if p is not None:
        global_cutoff = cutoffs.min()
        active_part = logits_idx[:, global_cutoff:]
        probs_sort = logits_sort[:, global_cutoff:].softmax(dim=-1)
        cumprob = probs_sort.cumsum(dim=-1)
        top_p_mask = (cumprob <= (1 - p.unsqueeze(1))) | (torch.arange(
            probs_sort.size(1), device=device) == probs_sort.size(1) - 1)
    else:
        active_part = logits_idx
        top_p_mask = torch.arange(vocab_size, device=device).expand(
            batch_size, -1) >= cutoffs.unsqueeze(1)

    valid_idx = (active_part + strides).masked_select(top_p_mask)
    logits_flatten = logits.flatten()
    output = torch.full_like(logits_flatten, -float('inf'))
    output[valid_idx] = logits_flatten[valid_idx]
    return output.reshape(batch_size, vocab_size)
