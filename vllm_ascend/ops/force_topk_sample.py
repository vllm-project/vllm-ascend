#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""force_topk sampler: small-operator reference implementation.

See FORCE_TOPK_DESIGN.md §4.3.1 for the full design. This module provides
the pre-fusion implementation assembled from standard torch operators. It is
numerically equivalent to the future fused NPU kernel and serves as:

  1. The step-2 implementation (get it correct first).
  2. The regression golden for the step-4 fused kernel.
  3. The fallback path when the fused kernel is unavailable.

Key invariants (design §4.6):
  I1: greedy == full-vocab argmax (guaranteed by caller, not this function).
  I2: top_k/top_p/min_p are exact when the effective candidate set ⊆ top-k.
  I3: logprobs use the full-vocab LSE (logsumexp), matching raw_logprobs.
"""

import torch

from vllm.logger import logger
from vllm_ascend.sample.topk_map import CompactDist

__all__ = ["build_compact_for_logprobs", "force_topk_sample"]


def build_compact_for_logprobs(logits: torch.Tensor, k: int) -> CompactDist:
    """Build a CompactDist for logprobs reporting (no sampling).

    Used by the greedy branch of AscendSampler.sample() when the caller only
    needs the compact logprob representation, not a random sample.

    Steps (design §4.3.1):
      1. lse_full = logsumexp(logits)          # full-vocab normalizer L
      2. topv, token_index = topk(logits, k)   # descending
      3. logprobs = topv - lse_full            # I3: full-vocab normalization

    Args:
        logits: [B, V] float32, post-logits-processors, pre-temperature.
        k:      global candidate ceiling.

    Returns:
        CompactDist with token_index [B, k] i32 and logprobs [B, k] f32.
    """
    k = min(k, logits.shape[-1])
    lse_full = torch.logsumexp(logits, dim=-1, keepdim=True)   # [B, 1]
    topv, token_index = torch.topk(logits, k, dim=-1)          # [B, k] descending
    logprobs = topv - lse_full                                 # [B, k] I3
    return CompactDist(token_index.to(torch.int32), logprobs)


def force_topk_sample(
    logits: torch.Tensor,
    temperature: torch.Tensor,
    top_p: torch.Tensor,
    top_k: torch.Tensor,
    min_p: torch.Tensor | None,
    generators: dict[int, torch.Generator],
    k: int,
) -> tuple[torch.Tensor, CompactDist]:
    """Sample from logits using the force_topk compact-space path.

    All sampling logic (temperature, top_k, top_p, min_p, Gumbel-max) is
    performed in the [B, k] local-rank space after a single full-vocab topk.
    The returned CompactDist carries the vocab-id restoration mapping and
    full-vocab-normalized logprobs.

    Args:
        logits:      [B, V] float32, post-logits-processors, **pre-temperature**.
        temperature: [B] float32, per-request temperature.
        top_p:       [B] float32, per-request (1.0 = disabled).
        top_k:       [B] int32, per-request (<=0 = disabled → use k).
        min_p:       [B] float32 or None, per-request (None = disabled).
        generators:  per-request torch.Generator dict (may be empty).
        k:           global candidate ceiling (env VLLM_ASCEND_SAMPLER_FORCE_TOPK).

    Returns:
        (sampled, cdist):
          sampled: [B] int64, sampled vocab ids.
          cdist:   CompactDist with token_index [B, k] i32 and logprobs [B, k] f32.
    """
    B, V = logits.shape
    k = min(k, V)
    logger.warning("[force_topk_sample] enter: B=%d, V=%d, k=%d", B, V, k)  # TODO: remove after debugging

    # Step 1: full-vocab normalizer L (single pass, O(V))
    lse_full = torch.logsumexp(logits, dim=-1, keepdim=True)   # [B, 1]

    # Step 2: top-k candidates (the only O(V log k) operation)
    topv, token_index = torch.topk(logits, k, dim=-1)          # [B, k] descending
    logger.warning("[force_topk_sample] topk done: topv.shape=%s", topv.shape)  # TODO: remove after debugging

    # Step 3: full-vocab-normalized logprobs for reporting (I3)
    logprobs = topv - lse_full                                 # [B, k]

    # ---- Below: all computation in [B, k] compact space ----

    # Temperature scaling
    s = topv / temperature.unsqueeze(1)                        # [B, k]

    # True probabilities (full-vocab normalized, NOT k-renormalized)
    true_p = torch.exp(topv - lse_full)                        # [B, k]

    neg = torch.finfo(s.dtype).min

    # top_k: mask ranks >= min(top_k, k). Already descending, so just cut.
    # top_k <= 0 (disabled) → use k as cap (no-op).
    k_cap = torch.where(
        top_k > 0,
        torch.minimum(top_k, torch.full_like(top_k, k)),
        torch.full_like(top_k, k),
    )                                                          # [B] int32
    rank = torch.arange(k, device=logits.device)               # [k]
    s = s.masked_fill(rank[None, :] >= k_cap[:, None], neg)    # [B, k]
    logger.warning("[force_topk_sample] top_k mask applied")  # TODO: remove after debugging

    # top_p: nucleus based on true probabilities (full-vocab consistent).
    # keep[r] = (cumulative prob BEFORE r) < top_p  → includes the
    # threshold-crossing item (smallest prefix s.t. cumsum >= p).
    cdf = true_p.cumsum(dim=-1)                                # [B, k]
    keep = (cdf - true_p) < top_p[:, None]                     # [B, k]
    s = s.masked_fill(~keep, neg)                              # [B, k]
    logger.warning("[force_topk_sample] top_p nucleus applied")  # TODO: remove after debugging

    # min_p: threshold = min_p * max_prob. max is top1 (already in top-k).
    if min_p is not None:
        thr = min_p[:, None] * true_p[:, :1]                   # [B, 1]
        s = s.masked_fill(true_p < thr, neg)                   # [B, k]
        logger.warning("[force_topk_sample] min_p mask applied")  # TODO: remove after debugging

    # Softmax over k candidates (renormalization after masking)
    probs_k = torch.softmax(s, dim=-1)                         # [B, k]

    # Gumbel-max sampling via exponential noise (no CPU-NPU sync, design N1).
    # Aligned with vllm_ascend/sample/sampler.py::random_sample.
    q = torch.empty_like(probs_k)
    if len(generators) != B:
        q.exponential_()
    if generators:
        for i, generator in generators.items():
            q[i].exponential_(generator=generator)

    local = (probs_k / q).argmax(dim=-1)                       # [B] local rank
    logger.warning("[force_topk_sample] local rank sampled: shape=%s", local.shape)  # TODO: remove after debugging

    # Map local rank back to vocab id via token_index (π mapping)
    sampled = token_index.gather(
        1, local[:, None]
    ).squeeze(1).to(torch.int64)                               # [B]

    logger.warning("[force_topk_sample] done: sampled.shape=%s, cdist.k=%d", sampled.shape, k)  # TODO: remove after debugging
    return sampled, CompactDist(token_index.to(torch.int32), logprobs)
