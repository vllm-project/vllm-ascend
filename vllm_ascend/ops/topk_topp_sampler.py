# SPDX-License-Identifier: Apache-2.0
from typing import Optional
import torch
from vllm.v1.sample.ops.topk_topp_sampler import TopKTopPSampler, apply_top_k_top_p_tpu, random_sample


def forward_npu(
    self,
    logits: torch.Tensor,
    generators: dict[int, torch.Generator],
    k: Optional[torch.Tensor],
    p: Optional[torch.Tensor],
) -> torch.Tensor:
    logits = apply_top_k_top_p_tpu(logits, k, p)
    probs = logits.softmax(dim=-1, dtype=torch.float32)
    return random_sample(probs, generators)


TopKTopPSampler.forward_native = forward_npu