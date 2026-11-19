# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Acceptance observation and bookkeeping for MRV2 suffix decoding.

Adapted from vLLM #60339, merge 190df0b90007eead441d97d3abf6270f84f2c63e.
Host launch wrappers retain the upstream positional ABI and derive NPU launch
metadata from tensors and scalar arguments. Request lifecycle and execution
ordering remain with the upstream speculator.
"""

import torch
from vllm.triton_utils import tl, triton

# Adaptive verification: acceptance is tracked per (draft source, match-length
# bucket, position). Match lengths 1-6 get their own bucket, then 7-10, 11-20
# and 21+. Counts are halved every _STATS_HALF_LIFE steps so the estimate
# follows the workload.
_NUM_LEN_BUCKETS = tl.constexpr(9)
_STATS_HALF_LIFE = 1024
_OBSERVE_RESET_BLOCK = 256


@triton.jit
def _len_bucket(L):
    return tl.where(L <= 6, L - 1, tl.where(L <= 10, 6, tl.where(L <= 20, 7, 8)))


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
    """One program owns each statistics bucket; no contended int64 atomics.

    Reduce batch-local counts in int32, then accumulate in int64. This avoids
    the divergent early return / vector-atomic path in the GPU observer and
    gives every TP rank the same integer result.
    """
    bucket = tl.program_id(0)
    b = tl.arange(0, REQS_PO2)
    req = tl.load(idx_mapping_ptr + b, mask=b < num_reqs, other=0).to(tl.int64)
    previous_bucket = tl.load(last_bucket_ptr + req)
    valid = tl.load(last_valid_ptr + req)
    sampled = tl.load(num_sampled_ptr + b, mask=b < num_reqs, other=0)
    rejected = tl.load(num_rejected_ptr + b, mask=b < num_reqs, other=0)
    accepted = sampled - 1
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
            stat = stats_ptr + (bucket * K + k) * 2
            # Load both fields before either store; interleaving them also
            # triggers a BiShengHIR pipeline failure on the current backend.
            old_accepted = tl.load(stat)
            old_observed = tl.load(stat + 1)
            tl.store(stat, old_accepted + num_accepted)
            tl.store(stat + 1, old_observed + num_observed)
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
        stat = stats_ptr + (bucket * K + k) * 2
        old_accepted = tl.load(stat, mask=k < K, other=0)
        old_observed = tl.load(stat + 1, mask=k < K, other=0)
        tl.store(stat, old_accepted + num_accepted, mask=k < K)
        tl.store(stat + 1, old_observed + num_observed, mask=k < K)


@triton.jit
def _suffix_clear_observed_kernel(
    idx_mapping_ptr,
    num_sampled_ptr,
    last_bucket_ptr,
    num_reqs,
    BLOCK: tl.constexpr,
):
    """Consume previous-draft bookkeeping after all bucket reductions finish."""
    b = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    req = tl.load(idx_mapping_ptr + b, mask=b < num_reqs, other=0).to(tl.int64)
    sampled = tl.load(num_sampled_ptr + b, mask=b < num_reqs, other=0)
    bucket = tl.load(last_bucket_ptr + req)
    tl.store(last_bucket_ptr + req, -1, mask=(b < num_reqs) & (sampled > 0) & (bucket >= 0))


def launch_suffix_observe(
    idx_mapping: torch.Tensor,
    num_sampled: torch.Tensor,
    num_rejected: torch.Tensor,
    accept_stats: torch.Tensor,
    last_bucket: torch.Tensor,
    last_valid: torch.Tensor,
    **launch_options,
) -> None:
    """Reduce active requests into NPU statistics buckets.

    Preserve the parent observer's consumption semantics. Clearing previous
    buckets inside the reduction kernel would race the other bucket programs,
    so it runs as a second launch on the same stream after all reductions.
    """
    num_reqs = idx_mapping.shape[0]
    launch_options["K_PO2"] = triton.next_power_of_2(launch_options["K"])
    launch_options["REQS_PO2"] = triton.next_power_of_2(num_reqs)
    launch_options["num_warps"] = 4
    _suffix_observe_kernel[(accept_stats.shape[0],)](
        idx_mapping,
        num_sampled,
        num_rejected,
        accept_stats,
        last_bucket,
        last_valid,
        num_reqs,
        **launch_options,
    )
    _suffix_clear_observed_kernel[(triton.cdiv(num_reqs, _OBSERVE_RESET_BLOCK),)](
        idx_mapping,
        num_sampled,
        last_bucket,
        num_reqs,
        BLOCK=_OBSERVE_RESET_BLOCK,
        num_warps=4,
    )
