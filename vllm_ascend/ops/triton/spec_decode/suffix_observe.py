# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Acceptance observation and bookkeeping for MRV2 suffix decoding.

Adapted from vLLM #60339, merge 190df0b90007eead441d97d3abf6270f84f2c63e.
"""

import torch
from vllm.triton_utils import tl, triton

# Preserve the existing 256-lane clear tile and four-warp reduction geometry.
_OBSERVE_RESET_BLOCK = 256
_OBSERVE_NUM_WARPS = 4


@triton.jit
def _len_bucket(match_len):
    # Match the upstream statistics partition: six exact and three coarse buckets.
    EXACT_MATCH_BUCKETS = 6
    MEDIUM_MATCH_MAX = 10
    LONG_MATCH_MAX = 20
    COARSE_MATCH_BUCKETS = 3
    NUM_LEN_BUCKETS = EXACT_MATCH_BUCKETS + COARSE_MATCH_BUCKETS
    return tl.where(
        match_len <= EXACT_MATCH_BUCKETS,
        match_len - 1,
        tl.where(
            match_len <= MEDIUM_MATCH_MAX,
            EXACT_MATCH_BUCKETS,
            tl.where(match_len <= LONG_MATCH_MAX, EXACT_MATCH_BUCKETS + 1, NUM_LEN_BUCKETS - 1),
        ),
    )


@triton.jit
def _suffix_observe_kernel(
    idx_mapping_ptr,
    num_sampled_ptr,
    num_rejected_ptr,
    stats_ptr,
    last_bucket_ptr,
    last_valid_ptr,
    num_reqs,
    K: tl.constexpr,
    K_PO2: tl.constexpr,
    REQS_PO2: tl.constexpr,
):
    """Reduce batch counts in int32 and update each bucket in int64."""
    STAT_FIELDS = 2  # Statistics entries contain (accepted, observed).
    OBSERVED_OFFSET = 1
    bucket = tl.program_id(0)
    b = tl.arange(0, REQS_PO2)
    req = tl.load(idx_mapping_ptr + b, mask=b < num_reqs, other=0).to(tl.int64)
    previous_bucket = tl.load(last_bucket_ptr + req)
    valid = tl.load(last_valid_ptr + req)
    sampled = tl.load(num_sampled_ptr + b, mask=b < num_reqs, other=0)
    rejected = tl.load(num_rejected_ptr + b, mask=b < num_reqs, other=0)
    accepted = sampled - 1  # Exclude the target model's bonus token.
    verified = accepted + rejected
    if K != K_PO2:
        # The current backend fails on the padded 2D reduction / strided
        # int64 update path for K=3. Use scalar updates for unaligned K,
        # keeping the vectorized path for power-of-two draft lengths.
        for k in tl.static_range(K):
            observed = (
                (b < num_reqs)
                & (previous_bucket == bucket)
                & (sampled > 0)
                & (k <= accepted)
                & (k < verified)
                & (k < valid)
            )
            num_observed = tl.sum(observed.to(tl.int32), axis=0).to(tl.int64)
            num_accepted = tl.sum((observed & (k < accepted)).to(tl.int32), axis=0).to(tl.int64)
            stat = stats_ptr + (bucket * K + k) * STAT_FIELDS
            # Load both fields before either store; interleaving them also
            # triggers a BiShengHIR pipeline failure on the current backend.
            old_accepted = tl.load(stat)
            old_observed = tl.load(stat + OBSERVED_OFFSET)
            tl.store(stat, old_accepted + num_accepted)
            tl.store(stat + OBSERVED_OFFSET, old_observed + num_observed)
    else:
        k = tl.arange(0, K_PO2)
        observed = (
            (b[:, None] < num_reqs)
            & (previous_bucket[:, None] == bucket)
            & (sampled[:, None] > 0)
            & (k[None, :] < K)
            & (k[None, :] <= accepted[:, None])
            & (k[None, :] < verified[:, None])
            & (k[None, :] < valid[:, None])
        )
        num_observed = tl.sum(observed.to(tl.int32), axis=0).to(tl.int64)
        num_accepted = tl.sum((observed & (k[None, :] < accepted[:, None])).to(tl.int32), axis=0).to(tl.int64)
        stat = stats_ptr + (bucket * K + k) * STAT_FIELDS
        old_accepted = tl.load(stat, mask=k < K, other=0)
        old_observed = tl.load(stat + OBSERVED_OFFSET, mask=k < K, other=0)
        tl.store(stat, old_accepted + num_accepted, mask=k < K)
        tl.store(stat + OBSERVED_OFFSET, old_observed + num_observed, mask=k < K)


@triton.jit
def _suffix_clear_observed_kernel(
    idx_mapping_ptr,
    num_sampled_ptr,
    last_bucket_ptr,
    num_reqs,
    BLOCK: tl.constexpr,
):
    """Consume previous-draft bookkeeping after all bucket reductions finish."""
    INVALID_BUCKET = -1
    b = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    req = tl.load(idx_mapping_ptr + b, mask=b < num_reqs, other=0).to(tl.int64)
    sampled = tl.load(num_sampled_ptr + b, mask=b < num_reqs, other=0)
    bucket = tl.load(last_bucket_ptr + req)
    tl.store(last_bucket_ptr + req, INVALID_BUCKET, mask=(b < num_reqs) & (sampled > 0) & (bucket >= 0))


def launch_suffix_observe(
    idx_mapping_ptr: torch.Tensor,
    num_sampled_ptr: torch.Tensor,
    num_rejected_ptr: torch.Tensor,
    stats_ptr: torch.Tensor,
    last_bucket_ptr: torch.Tensor,
    last_valid_ptr: torch.Tensor,
    K: int,
    K_PO2: int,
    **launch_options,
) -> None:
    """Update bucket counts, then clear consumed markers on the same stream."""
    launch_options.update(
        K=K,
        K_PO2=K_PO2,
    )
    num_reqs = idx_mapping_ptr.shape[0]
    launch_options["K_PO2"] = triton.next_power_of_2(launch_options["K"])
    launch_options["REQS_PO2"] = triton.next_power_of_2(num_reqs)
    launch_options["num_warps"] = _OBSERVE_NUM_WARPS
    _suffix_observe_kernel[(stats_ptr.shape[0],)](
        idx_mapping_ptr,
        num_sampled_ptr,
        num_rejected_ptr,
        stats_ptr,
        last_bucket_ptr,
        last_valid_ptr,
        num_reqs,
        **launch_options,
    )
    _suffix_clear_observed_kernel[(triton.cdiv(num_reqs, _OBSERVE_RESET_BLOCK),)](
        idx_mapping_ptr,
        num_sampled_ptr,
        last_bucket_ptr,
        num_reqs,
        BLOCK=_OBSERVE_RESET_BLOCK,
        num_warps=_OBSERVE_NUM_WARPS,
    )
