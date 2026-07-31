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

Structure mirrors the private fork's _fused_sample kernel:
  Phase 1: topk (full-vocab → [B, k])
  Phase 2: raw logprobs (only when return_raw_logprobs=True)
  Phase 3: _apply_sampling_constraints (temperature + top_k + top_p + min_p + softmax)
  Phase 4: _sample (Gumbel-max via exponential noise)
  Phase 5: logprobs selection (raw vs processed)
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


def _apply_sampling_constraints(
    topv: torch.Tensor,
    temperature: torch.Tensor,
    top_p: torch.Tensor,
    top_k: torch.Tensor,
    min_p: torch.Tensor | None,
    true_p: torch.Tensor,
    k: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply temperature + top_k + top_p + min_p masks + softmax.

    Corresponds to _fused_sample steps ①-④:
      ① temperature scaling
      ② top_k mask (rank-based cut)
      ③ top_p nucleus (cumsum-based threshold)
      ④ min_p threshold
      ⑤ softmax (renormalization after masking)

    Args:
        topv:        [B, k] float32, top-k logits (descending).
        temperature: [B] float32, per-request.
        top_p:       [B] float32, per-request (1.0 = disabled).
        top_k:       [B] int32, per-request (<=0 = disabled → use k).
        min_p:       [B] float32 or None, per-request.
        true_p:      [B, k] float32, probabilities for nucleus/min_p judgment.
                     raw mode: exp(topv - lse_full) (full-vocab normalized).
                     processed mode: softmax(topv) (k-dim normalized).
        k:           candidate ceiling.
        device:      NPU device.

    Returns:
        s_masked: [B, k] float32, masked logits after temperature + top_k/top_p/min_p.
        probs_k:  [B, k] float32, softmax probabilities (renormalized).
    """
    # ① Temperature scaling
    s = topv / temperature.unsqueeze(1)                        # [B, k]

    neg = torch.finfo(s.dtype).min

    # ② top_k: mask ranks >= min(top_k, k). Already descending, so just cut.
    k_cap = torch.where(
        top_k > 0,
        torch.minimum(top_k, torch.full_like(top_k, k)),
        torch.full_like(top_k, k),
    )                                                          # [B] int32
    rank = torch.arange(k, device=device)                      # [k]
    s = s.masked_fill(rank[None, :] >= k_cap[:, None], neg)    # [B, k]

    # ③ top_p: nucleus based on true probabilities.
    # keep[r] = (cumulative prob BEFORE r) < top_p → includes threshold-crossing item.
    cdf = true_p.cumsum(dim=-1)                                # [B, k]
    keep = (cdf - true_p) < top_p[:, None]                     # [B, k]
    s = s.masked_fill(~keep, neg)                              # [B, k]

    # ④ min_p: threshold = min_p * max_prob. max is top1 (already in top-k).
    if min_p is not None:
        thr = min_p[:, None] * true_p[:, :1]                   # [B, 1]
        s = s.masked_fill(true_p < thr, neg)                   # [B, k]

    # ⑤ Softmax over k candidates (renormalization after masking)
    probs_k = torch.softmax(s, dim=-1)                         # [B, k]
    return s, probs_k


def _sample(
    probs_k: torch.Tensor,
    generators: dict[int, torch.Generator],
    B: int,
    token_index: torch.Tensor,
) -> torch.Tensor:
    """Gumbel-max sampling via exponential noise (no CPU-NPU sync, design N1).

    Corresponds to _fused_sample step ⑤: random sampling.

    Args:
        probs_k:     [B, k] float32, renormalized probabilities.
        generators:  per-request torch.Generator dict (may be empty).
        B:           batch size.
        token_index: [B, k] int64, vocab id mapping (for π mapping).

    Returns:
        [B] int64, sampled vocab ids.
    """
    q = torch.empty_like(probs_k)
    if len(generators) != B:
        q.exponential_()
    if generators:
        for i, generator in generators.items():
            q[i].exponential_(generator=generator)

    local = (probs_k / q).argmax(dim=-1)                       # [B] local rank
    sampled = token_index.gather(
        1, local[:, None]
    ).squeeze(1).to(torch.int64)                               # [B]
    return sampled


def force_topk_sample(
    logits: torch.Tensor,
    temperature: torch.Tensor,
    top_p: torch.Tensor,
    top_k: torch.Tensor,
    min_p: torch.Tensor | None,
    generators: dict[int, torch.Generator],
    k: int,
    return_raw_logprobs: bool = True,
) -> tuple[torch.Tensor, CompactDist]:
    """Sample from logits using the force_topk compact-space path.

    All sampling logic (temperature, top_k, top_p, min_p, Gumbel-max) is
    performed in the [B, k] local-rank space after a single full-vocab topk.
    The returned CompactDist carries the vocab-id restoration mapping and
    logprobs (raw or processed depending on return_raw_logprobs).

    Structure mirrors the private fork's _fused_sample kernel:
      Phase 1: topk (full-vocab → [B, k])
      Phase 2: raw logprobs (only when return_raw_logprobs=True)
      Phase 3: _apply_sampling_constraints (temperature + top_k + top_p + min_p + softmax)
      Phase 4: _sample (Gumbel-max via exponential noise)
      Phase 5: logprobs selection (raw vs processed)

    Args:
        logits:              [B, V] float32, post-logits-processors, **pre-temperature**.
        temperature:         [B] float32, per-request temperature.
        top_p:               [B] float32, per-request (1.0 = disabled).
        top_k:               [B] int32, per-request (<=0 = disabled → use k).
        min_p:               [B] float32 or None, per-request (None = disabled).
        generators:          per-request torch.Generator dict (may be empty).
        k:                   global candidate ceiling (env VLLM_ASCEND_SAMPLER_FORCE_TOPK).
        return_raw_logprobs: if True, logprobs = topv - LSE(z_raw) (full-vocab normalized,
                             requires full-vocab logsumexp). If False, logprobs =
                             log_softmax(s_masked) (processed, k-dim only, no full-vocab scan).

    Returns:
        (sampled, cdist):
          sampled: [B] int64, sampled vocab ids.
          cdist:   CompactDist with token_index [B, k] i32 and logprobs [B, k] f32.
    """
    B, V = logits.shape
    k = min(k, V)
    logger.warning("[force_topk_sample] enter: B=%d, V=%d, k=%d, return_raw=%s", B, V, k, return_raw_logprobs)  # TODO: remove after debugging

    # Phase 1: top-k candidates (the only O(V log k) operation)
    topv, token_index = torch.topk(logits, k, dim=-1)          # [B, k] descending
    logger.warning("[force_topk_sample] topk done: topv.shape=%s", topv.shape)  # TODO: remove after debugging

    # Phase 2: raw logprobs + true_p (only when return_raw_logprobs=True)
    if return_raw_logprobs:
        lse_full = torch.logsumexp(logits, dim=-1, keepdim=True)   # [B, 1] full-vocab LSE
        raw_logprobs = topv - lse_full                              # [B, k] raw logprob
        true_p = torch.exp(topv - lse_full)                         # [B, k] full-vocab normalized probs
    else:
        raw_logprobs = None
        true_p = torch.softmax(topv, dim=-1)                        # [B, k] k-dim normalized probs

    # Phase 3: apply sampling constraints (temperature + top_k + top_p + min_p + softmax)
    # Corresponds to _fused_sample steps ①-⑤
    s_masked, probs_k = _apply_sampling_constraints(
        topv, temperature, top_p, top_k, min_p, true_p, k, logits.device
    )
    logger.warning("[force_topk_sample] sampling constraints applied")  # TODO: remove after debugging

    # Phase 4: random sampling
    # Corresponds to _fused_sample step ⑥
    sampled = _sample(probs_k, generators, B, token_index)
    logger.warning("[force_topk_sample] sampled: shape=%s", sampled.shape)  # TODO: remove after debugging

    # Phase 5: select logprobs based on mode
    if return_raw_logprobs:
        logprobs = raw_logprobs                                  # topv - LSE(z_raw): raw distribution
    else:
        logprobs = torch.log_softmax(s_masked, dim=-1)          # log_softmax(s_masked): processed distribution

    logger.warning("[force_topk_sample] done: sampled.shape=%s, cdist.k=%d, mode=%s", sampled.shape, k, "raw" if return_raw_logprobs else "processed")  # TODO: remove after debugging
    return sampled, CompactDist(token_index.to(torch.int32), logprobs)
