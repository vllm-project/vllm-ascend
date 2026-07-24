import torch
import torch.nn as nn
import triton
import triton.language as tl
import triton.runtime.driver as driver
import torch_npu


@triton.jit
def ngram_spec_decode_kernel(
    token_ids_ptr,
    num_tokens_ptr,
    sampled_ptr,
    discard_ptr,
    next_token_ids_ptr,
    draft_token_ids_ptr,
    num_valid_draft_ptr,
    raw_valid_count_ptr,
    max_seq_len: tl.constexpr,
    max_new_tokens: tl.constexpr,
    vocab_size: tl.constexpr,
    min_n: tl.constexpr,
    max_n: tl.constexpr,
    k: tl.constexpr,
    batch_size,
):
    pid = tl.program_id(0)
    num_cores = tl.num_programs(0)

    BLOCK: tl.constexpr = (
        1024 if max_n <= 5 else (512 if max_n <= 10 else (256 if max_n <= 16 else 128))
    )
    NUM_BLOCKS: tl.constexpr = (max_seq_len + BLOCK - 1) // BLOCK
    NO_MATCH_F: tl.constexpr = 1.0e9

    for batch_idx in range(pid, batch_size, num_cores):
        seq_len = tl.load(num_tokens_ptr + batch_idx)
        discard = tl.load(discard_ptr + batch_idx)
        row_off = batch_idx * max_seq_len

        # ── Filter phase ────────────────────────────────────────────────
        s_off = tl.arange(0, max_new_tokens)
        sampled_vals = tl.load(
            sampled_ptr + batch_idx * max_new_tokens + s_off,
            care_padding=False,
        )

        if discard != 0:
            filtered = tl.full([max_new_tokens], -1, tl.int32)
            valid_count = 0
        else:
            is_valid = (sampled_vals != -1) & (sampled_vals < vocab_size)
            filtered = tl.where(
                is_valid, sampled_vals, tl.full([max_new_tokens], -1, tl.int32)
            )
            valid_count = tl.sum(tl.cast(is_valid, tl.int32))

        tl.store(raw_valid_count_ptr + batch_idx, valid_count)

        avail_space = max_seq_len - seq_len
        if avail_space < 0:
            avail_space = 0
        if valid_count > avail_space:
            valid_count = avail_space

        nt = seq_len + valid_count

        # next_token = filtered[valid_count - 1] (masked-sum extraction).
        # Stored to HBM immediately to keep its live register short.
        if valid_count > 0:
            sel = s_off == (valid_count - 1)
            next_token = tl.sum(tl.where(sel, filtered, 0), axis=0)
        else:
            backup_pos = seq_len - 1
            if backup_pos < 0:
                backup_pos = 0
            next_token = tl.load(token_ids_ptr + row_off + backup_pos)
        tl.store(next_token_ids_ptr + batch_idx, next_token)

        # Append the valid prefix to token_ids; the matching phase below searches
        # this row including the appended suffix.
        if valid_count > 0:
            c_mask = s_off < valid_count
            tl.store(
                token_ids_ptr + row_off + seq_len + s_off,
                filtered,
                mask=c_mask,
            )

        # ── Longest n-gram match + draft extraction phase ──────────────
        # Each block overlap-reads at most max_n tokens past its owned range to
        # evaluate every n-gram length purely in registers.
        #
        # For owned position i, L[i] = longest n in [min_n, max_n] such that
        #   token_ids[i .. i+n-1] == token_ids[nt-n .. nt-1]   (and i < nt-n).
        # best = globally longest L, tie-broken by earliest pos.
        #
        # For length n, D_n[i] = OR_{j<n} (token_ids[i+j] - suffix_n[j]) with
        # suffix_n[j] = token_ids[nt-n+j]. D_n[i] == 0 iff the window matches. The
        # inner j loop is unrolled (tl.static_range) into a single int32
        # accumulator `d` per n — fully vectorized, never scalar.
        best_pos = -1
        best_len = 0

        if valid_count > 0 and nt >= min_n:
            g_best_len = 0.0
            g_best_pos = NO_MATCH_F

            if NUM_BLOCKS > 1:
                s_tail = tl.arange(0, max_n)
                s_pos = nt - max_n + s_tail
                s_pos = tl.where(s_pos < 0, 0, s_pos)
                S_tail = tl.load(token_ids_ptr + row_off + s_pos)

            for bidx in tl.range(NUM_BLOCKS):
                base = bidx * BLOCK
                off = tl.arange(0, BLOCK)
                pos = base + off
                pos_f = tl.cast(pos, tl.float32)

                L = tl.full([BLOCK], 0.0, tl.float32)
                for n_val in tl.static_range(min_n, max_n + 1):
                    d = tl.full([BLOCK], 0, tl.int32)
                    for j in tl.static_range(0, n_val):
                        gp = pos + j
                        tok_j = tl.load(
                            token_ids_ptr + row_off + gp,
                            mask=gp < nt,
                            care_padding=False,
                        )
                        if NUM_BLOCKS > 1:
                            # Extract S_tail[max_n - n_val + j] via masked selection.
                            sel_mask = s_tail == (max_n - n_val + j)
                            suf = tl.sum(tl.where(sel_mask, S_tail, 0), axis=0)
                        else:
                            # Single-block path: a direct scalar load is cheaper than
                            # the masked-selection overhead.
                            sidx = nt - n_val + j
                            sidx = tl.where(sidx < 0, 0, sidx)
                            suf = tl.load(token_ids_ptr + row_off + sidx)
                        d = d | (tok_j - suf)
                    match_n = d == 0
                    valid_n = pos_f < (nt - n_val)
                    L = tl.where(match_n & valid_n, tl.cast(n_val, tl.float32), L)

                # Per-block best: longest L, tie-break earliest pos.
                block_best_len = tl.max(L, axis=0)
                eq_best = L == block_best_len
                has_match = block_best_len > 0.0
                cand_pos = tl.where(eq_best & has_match, pos_f, NO_MATCH_F)
                block_best_pos = tl.min(cand_pos, axis=0)

                # Reduce into the global best (longer wins; equal -> earlier pos).
                new_better = block_best_len > g_best_len
                same_earlier = (
                    block_best_len == g_best_len
                ) & (block_best_pos < g_best_pos)
                update = new_better | same_earlier
                g_best_len = tl.where(update, block_best_len, g_best_len)
                g_best_pos = tl.where(update, block_best_pos, g_best_pos)

            if g_best_len > 0.0:
                best_pos = tl.cast(g_best_pos, tl.int32)
                best_len = tl.cast(g_best_len, tl.int32)

        # Draft extraction (vectorized).
        draft_start = best_pos + best_len
        tokens_avail = nt - draft_start
        if tokens_avail < 0:
            tokens_avail = 0

        d_off = tl.arange(0, k)
        if best_pos >= 0:
            can_copy = d_off < tokens_avail
            draft_vals = tl.load(
                token_ids_ptr + row_off + draft_start + d_off,
                mask=can_copy,
                other=-1,
                care_padding=False,
            )
            tl.store(draft_token_ids_ptr + batch_idx * k + d_off, draft_vals)
            # Draft tokens copied from the sequence are real (>= 0) token ids, so
            # the valid count is simply min(k, tokens_avail) — no reload/reduce.
            valid_draft = tokens_avail
            if valid_draft > k:
                valid_draft = k
        else:
            tl.store(
                draft_token_ids_ptr + batch_idx * k + d_off,
                tl.full([k], -1, tl.int32),
            )
            valid_draft = 0

        tl.store(num_valid_draft_ptr + batch_idx, valid_draft)


def triton_ngram_spec_decode(
    token_ids,
    num_tokens_no_spec,
    sampled_token_ids,
    discard_request_mask,
    vocab_size,
    min_n,
    max_n,
    k,
):
    device = torch_npu.npu.current_device()
    properties = driver.active.utils.get_device_properties(device)
    vectorcore_num = properties["num_vectorcore"]

    batch_size = token_ids.shape[0]
    device = token_ids.device
    max_seq_len = token_ids.shape[1]

    # Normalize sampled_token_ids: accept a list[list[int]] (padded to a
    # rectangular int32 tensor) or a pre-batched tensor, then slice to the
    # active batch_size derived from token_ids.
    if isinstance(sampled_token_ids, list):
        max_len = max((len(sublist) for sublist in sampled_token_ids), default=0)
        max_len = max(max_len, 1)
        padded_list = [
            sublist + [-1] * (max_len - len(sublist))
            for sublist in sampled_token_ids
        ]
        sampled_token_ids = torch.tensor(
            padded_list, dtype=torch.int32, device=device
        )
    sampled_token_ids = sampled_token_ids[:batch_size]
    max_new_tokens = sampled_token_ids.shape[1]

    base = torch.empty(
        batch_size * 3 + batch_size * k, dtype=torch.int32, device=device
    )
    next_token_ids = base[0 * batch_size : 1 * batch_size]
    num_valid_draft_tokens = base[1 * batch_size : 2 * batch_size]
    valid_sampled_tokens_count = base[2 * batch_size : 3 * batch_size]
    draft_token_ids = base[
        3 * batch_size : 3 * batch_size + batch_size * k
    ].reshape(batch_size, k)

    grid = (
        batch_size
        if batch_size < vectorcore_num
        else vectorcore_num,
    )

    ngram_spec_decode_kernel[grid](
        token_ids,
        num_tokens_no_spec,
        sampled_token_ids,
        discard_request_mask,
        next_token_ids,
        draft_token_ids,
        num_valid_draft_tokens,
        valid_sampled_tokens_count,
        max_seq_len=max_seq_len,
        max_new_tokens=max_new_tokens,
        vocab_size=vocab_size,
        min_n=min_n,
        max_n=max_n,
        k=k,
        batch_size=batch_size,
    )

    return (
        next_token_ids,
        draft_token_ids,
        num_valid_draft_tokens,
        valid_sampled_tokens_count,
    )
