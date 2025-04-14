from typing import Dict, Optional

import torch
import torch.nn as nn

from vllm.v1.sample.ops.topk_topp_sampler import TopKTopPSampler, random_sample
from vllm.logger import init_logger


logger = init_logger(__name__)


class AscendTopKTopPSampler(TopKTopPSampler):

    def __init__(self):
        super().__init__()
        # TODO(linfeng): eliminate warning for FlashInfer here
        self.forward = self.forward_npu

    def forward_npu(
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
    """Apply top-k and top-p optimized for NPU.

    This algorithm avoids using torch.scatter which is time-consuming on NPU.
    """
    # TODO(linfeng): consider the case taht either p or k is applied
    if k is None and p is None:
        return logits
    batch_size, vocab_size = logits.shape
    logits_sort, logits_idx = logits.sort(dim=-1, descending=False)

    boundary = logits_sort.gather(1, (vocab_size - k).unsqueeze(dim=1))
    top_k_mask = logits_sort < boundary
    logits_sort.masked_fill_(top_k_mask, -float("inf"))
    cutoff = top_k_mask.sum(dim=-1).min()
    probs_sort = logits_sort.softmax(dim=-1)[:, cutoff:]
    probs_sum = probs_sort.cumsum(dim=-1)
    top_p_mask = probs_sum > 1 - p.unsqueeze(dim=1)
    top_p_mask[:, -1] = True
    strides = torch.arange(0, batch_size*vocab_size, vocab_size, device=logits.device)
    flatten_idx = logits_idx[:, cutoff:] + strides.unsqueeze(dim=1)
    valid_idx = torch.masked_select(flatten_idx, top_p_mask)

    logits_flatten = logits.flatten()
    valid_logits = torch.index_select(logits_flatten, 0, valid_idx)
    logits = torch.empty_like(logits_flatten).fill_(-float("inf"))
    logits[valid_idx] = valid_logits
    return logits.reshape(batch_size, vocab_size)