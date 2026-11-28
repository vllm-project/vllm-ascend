# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Draft selection and adaptive confidence for MRV2 suffix decoding.

Adapted from vLLM #60339, merge 190df0b90007eead441d97d3abf6270f84f2c63e.
"""

import torch
from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.spec_decode.suffix_corpus import _corpus_draft
from vllm_ascend.ops.triton.spec_decode.suffix_observe import (
    _INVALID_BUCKET,
    _NUM_LEN_BUCKETS,
    _OBSERVED_OFFSET,
    _STAT_FIELDS,
    _len_bucket,
)

# Beta(1, 1) smoothing starts valid slots at confidence 1/2.
_ACCEPTED_PRIOR = tl.constexpr(1.0)
_OBSERVED_PRIOR = tl.constexpr(2.0)
# A5 cannot lower the corpus boolean gather with a single internal draft lane.
_MIN_VOTING_WIDTH = 2


@triton.jit
def _suffix_finalize_kernel(
    token_ids_ptr,  # *int32  [max_num_reqs, token_ids_stride]
    token_ids_stride,
    idx_mapping_ptr,  # *int32  [B]
    total_len_ptr,  # *int32  [max_num_reqs]
    num_sampled_ptr,  # *int32  [B]
    last_sampled_ptr,  # *int64  [max_num_reqs]
    scratch_ptr,  # *int64  [B, scratch_stride]
    scratch_stride,
    num_blocks,
    Lp1,
    corpus_ptr,  # *int32  [corpus_len]
    corpus_len,
    table_ptr,  # *int64  [2, NUM_BUCKETS, SLOTS]
    drafts_ptr,  # *int64  [B, K]  (output)
    num_valid_ptr,  # *int32  [B]  (output)
    stats_ptr,  # *int64  [2 * _NUM_LEN_BUCKETS, K, 2]  (accepted, observed)
    last_bucket_ptr,  # *int32  [max_num_reqs]  (output)
    last_valid_ptr,  # *int32  [max_num_reqs]  (output)
    conf_ptr,  # *fp32  [B, K]  (output)
    HAS_CORPUS: tl.constexpr,
    ADAPTIVE: tl.constexpr,
    HASH_MUL: tl.constexpr,
    NUM_BUCKETS: tl.constexpr,
    SLOTS: tl.constexpr,
    MAX_DEPTH: tl.constexpr,
    DEPTH_PO2: tl.constexpr,
    K: tl.constexpr,
    K_PO2: tl.constexpr,
    NB_PO2: tl.constexpr,
):
    b = tl.program_id(0).to(tl.int64)
    req = tl.load(idx_mapping_ptr + b).to(tl.int64)
    seq_len = tl.load(total_len_ptr + req).to(tl.int64)
    num_sampled = tl.load(num_sampled_ptr + b)
    last_tok = tl.load(last_sampled_ptr + req)
    row = token_ids_ptr + req * token_ids_stride

    # Best match in the request's own history.
    nb = tl.arange(0, NB_PO2).to(tl.int64)
    l_score = tl.max(
        tl.load(scratch_ptr + b * scratch_stride + nb, mask=nb < num_blocks, other=0),
        axis=0,
    )
    l_len = l_score // Lp1
    l_pos = l_score - l_len * Lp1

    k_iota = tl.arange(0, K_PO2).to(tl.int64)
    k_mask = k_iota < K
    idx = l_pos + k_iota
    draft_ok = k_mask & (idx < seq_len) & (l_len > 0)
    draft = tl.load(row + tl.minimum(idx, token_ids_stride - 1), mask=draft_ok, other=0).to(tl.int64)

    if HAS_CORPUS:
        g_len, g_draft, g_n = _corpus_draft(
            row,
            seq_len,
            table_ptr,
            corpus_ptr,
            corpus_len,
            HASH_MUL,
            NUM_BUCKETS,
            SLOTS,
            MAX_DEPTH,
            DEPTH_PO2,
            K,
            K_PO2,
        )
        # The longer match wins; ties go to the request's own history.
        use_corpus = g_len > l_len
        draft = tl.where(use_corpus, g_draft, draft)
        draft_ok = tl.where(use_corpus, k_iota < g_n, draft_ok)
        match_len = tl.where(use_corpus, g_len, l_len)
        source = use_corpus.to(tl.int64)
    else:
        match_len = l_len
        source = tl.zeros((), tl.int64)

    draft_ok = draft_ok & (num_sampled > 0)
    num_valid = tl.sum(draft_ok.to(tl.int32), axis=0)
    # Fill invalid slots with the last sampled token.
    out = tl.where(draft_ok, draft, last_tok)
    tl.store(drafts_ptr + b * K + k_iota, out, mask=k_mask)
    tl.store(num_valid_ptr + b, num_valid)

    if ADAPTIVE:
        # Filler slots receive zero confidence.
        bucket = source * _NUM_LEN_BUCKETS + _len_bucket(tl.maximum(match_len, 1))
        if K != K_PO2:
            # As in the observer, avoid padded strided statistics accesses
            # for non-power-of-two draft lengths on the current backend.
            for d in tl.static_range(K):
                stat = stats_ptr + (bucket * K + d) * _STAT_FIELDS
                accepted = tl.load(stat).to(tl.float32)
                observed = tl.load(stat + _OBSERVED_OFFSET).to(tl.float32)
                is_valid = tl.sum(tl.where(k_iota == d, draft_ok.to(tl.int32), 0), axis=0) > 0
                conf = tl.where(is_valid, (accepted + _ACCEPTED_PRIOR) / (observed + _OBSERVED_PRIOR), 0.0)
                tl.store(conf_ptr + b * K + d, conf)
        else:
            stat = stats_ptr + (bucket * K + k_iota) * _STAT_FIELDS
            accepted = tl.load(stat, mask=k_mask, other=0).to(tl.float32)
            observed = tl.load(stat + _OBSERVED_OFFSET, mask=k_mask, other=0).to(tl.float32)
            conf = tl.where(draft_ok, (accepted + _ACCEPTED_PRIOR) / (observed + _OBSERVED_PRIOR), 0.0)
            tl.store(conf_ptr + b * K + k_iota, conf, mask=k_mask)
        tl.store(last_bucket_ptr + req, tl.where(num_valid > 0, bucket, _INVALID_BUCKET).to(tl.int32))
        tl.store(last_valid_ptr + req, num_valid)


def launch_suffix_finalize(
    token_ids_ptr: torch.Tensor,
    token_ids_stride: int,
    idx_mapping_ptr: torch.Tensor,
    total_len_ptr: torch.Tensor,
    num_sampled_ptr: torch.Tensor,
    last_sampled_ptr: torch.Tensor,
    scratch_ptr: torch.Tensor,
    scratch_stride: int,
    num_blocks: int,
    Lp1: int,
    corpus_ptr: torch.Tensor,
    corpus_len: int,
    table_ptr: torch.Tensor,
    drafts_ptr: torch.Tensor,
    num_valid_ptr: torch.Tensor,
    stats_ptr: torch.Tensor,
    last_bucket_ptr: torch.Tensor,
    last_valid_ptr: torch.Tensor,
    conf_ptr: torch.Tensor,
    HAS_CORPUS: bool,
    ADAPTIVE: bool,
    HASH_MUL: int,
    NUM_BUCKETS: int,
    SLOTS: int,
    MAX_DEPTH: int,
    DEPTH_PO2: int,
    K: int,
    K_PO2: int,
    NB_PO2: int,
    **launch_options,
) -> None:
    """Select history/corpus drafts and emit confidence with NPU kernels."""
    launch_options.update(
        HAS_CORPUS=HAS_CORPUS,
        ADAPTIVE=ADAPTIVE,
        HASH_MUL=HASH_MUL,
        NUM_BUCKETS=NUM_BUCKETS,
        SLOTS=SLOTS,
        MAX_DEPTH=MAX_DEPTH,
        DEPTH_PO2=DEPTH_PO2,
        K=K,
        K_PO2=K_PO2,
        NB_PO2=NB_PO2,
    )
    num_reqs = idx_mapping_ptr.shape[0]
    launch_options["K_PO2"] = max(_MIN_VOTING_WIDTH, triton.next_power_of_2(launch_options["K"]))
    launch_options["DEPTH_PO2"] = triton.next_power_of_2(launch_options["MAX_DEPTH"])
    launch_options["NB_PO2"] = triton.next_power_of_2(num_blocks)
    _suffix_finalize_kernel[(num_reqs,)](
        token_ids_ptr,
        token_ids_ptr.stride(0),
        idx_mapping_ptr,
        total_len_ptr,
        num_sampled_ptr,
        last_sampled_ptr,
        scratch_ptr,
        scratch_ptr.stride(0),
        num_blocks,
        Lp1,
        corpus_ptr,
        corpus_len,
        table_ptr,
        drafts_ptr,
        num_valid_ptr,
        stats_ptr,
        last_bucket_ptr,
        last_valid_ptr,
        conf_ptr,
        **launch_options,
    )
