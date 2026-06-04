# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import numpy as np
import torch


def compute_sliding_window_block_table(
    full_block_table: torch.Tensor,
    full_seq_lens: torch.Tensor,
    num_speculative_tokens: int,
    window_size: int,
    block_size: int,
    max_window_blocks: int,
    out: torch.Tensor,
) -> None:
    """Vectorized, sync-free computation of the cropped sliding-window block table.

    For each request, the ``ceil((seq_len + K - start) / B)`` most recent blocks
    of ``full_block_table`` are gathered into ``out[:num_reqs]``, zero-padded to
    ``max_window_blocks`` columns.

        window[i, :needed] = full_block_table[i, start_idx : start_idx + needed]

    (indices beyond the full table, or beyond ``needed``, are left as 0 -- those
    trailing block ids are never read because draft attention is bounded by
    ``seq_lens``).

    Args:
        full_block_table: ``[num_reqs, full_cols]`` block table.
        full_seq_lens: ``[num_reqs]`` sequence lengths (before the K draft steps).
        num_speculative_tokens: K, the number of speculative tokens.
        window_size: W, the sliding window size in tokens.
        block_size: B, the block size.
        max_window_blocks: fixed column count of the output window table
            (constant across graph capture and runtime).
        out: pre-allocated ``[>= num_reqs, max_window_blocks]`` buffer; the
            gathered window table is written into ``out[:num_reqs]`` in place.

    Returns:
        None; the cropped window table is written into ``out[:num_reqs]`` in
        place. (``start_block_indices`` / ``needed_blocks_per_req`` are computed
        internally but no longer returned -- no caller consumed them.)
    """
    num_reqs = full_seq_lens.shape[0]
    k = num_speculative_tokens
    w = window_size
    b = block_size

    final_seq_lens = full_seq_lens + k
    start_tokens = (final_seq_lens - w).clamp(min=0)
    start_blocks = (start_tokens // b) * b
    start_block_indices = (start_blocks // b).to(torch.int64)

    tokens_to_cover = final_seq_lens - start_blocks
    needed_blocks_per_req = ((tokens_to_cover + b - 1) // b).to(torch.int64)

    full_cols = full_block_table.shape[1]
    # column offset grid [1, max_window_blocks]
    cols = torch.arange(max_window_blocks, device=full_block_table.device).unsqueeze(0)
    # source column per (row, col): start_block_indices[:, None] + cols
    src_cols = start_block_indices.unsqueeze(1) + cols
    # clamp to the valid full-block-table column range so gather never goes OOB
    src_cols_clamped = src_cols.clamp(max=full_cols - 1)

    gathered = torch.gather(full_block_table, 1, src_cols_clamped)
    needed = torch.clamp(needed_blocks_per_req, max=max_window_blocks).unsqueeze(1)
    # keep only columns within `needed` and within the full table; zero the rest
    valid_mask = (cols < needed) & (src_cols < full_cols)
    out[:num_reqs].copy_(gathered * valid_mask.to(gathered.dtype))


def update_num_computed_tokens_for_batch_change(
    num_computed_tokens: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    prev_positions: torch.Tensor,
    valid_sampled_token_count: torch.Tensor,
    prev_num_draft_tokens: torch.Tensor,
    cpu_num_computed_tokens: torch.Tensor,
) -> None:
    """Correct num_computed_tokens for async spec decode drift.

    Requests that had drafts: corrected = prev_gpu + valid_count.
    New requests or non-draft (e.g. prefills): use CPU value directly.
    """
    # Clamp because prev_positions can be -1 for new requests
    gather_indices = prev_positions.clamp(min=0)

    valid_counts = valid_sampled_token_count[gather_indices]
    prev_computed = num_computed_tokens[gather_indices]
    prev_drafts = prev_num_draft_tokens[gather_indices]

    participating = (prev_positions >= 0) & (prev_drafts > 0)
    corrected = prev_computed + valid_counts.int()

    n = prev_positions.shape[0]
    num_computed_tokens[:n].copy_(torch.where(participating, corrected, cpu_num_computed_tokens))
    num_accepted_tokens.copy_(torch.where(participating, valid_counts, num_accepted_tokens))


def correct_optimistic_seq_lens_cpu(
    optimistic_seq_lens_cpu_np: np.ndarray,
    prev_positions_np: np.ndarray,
    prev_num_draft_tokens_np: np.ndarray,
    valid_sampled_token_count_np: np.ndarray,
    num_reqs: int,
) -> None:
    """Correct ``optimistic_seq_lens_cpu`` for async spec decode drift.

    The scheduler optimistically advances ``num_computed_tokens_cpu`` by the
    full number of tokens scheduled in the previous step (``prev_drafts + 1``
    per spec-decode request), assuming all drafts were accepted. The actual
    number of valid sampled tokens is ``valid_count = 1 + accepted_drafts``.
    The drift, equal to the number of rejected tokens, is therefore::

        rejected = prev_drafts + 1 - valid_count

    Subtracting this from the optimistic seq_lens recovers the true seq_lens
    that ``self.seq_lens`` (GPU) carries for participating requests, without
    touching the device. New requests (``prev_positions < 0``) and prefills
    (``prev_drafts == 0``) need no correction.

    Mirrors ``update_num_computed_tokens_for_batch_change`` on the CPU side.

    All arrays are sliced to ``num_reqs``; ``optimistic_seq_lens_cpu_np`` is
    modified in place.
    """
    prev_positions = prev_positions_np[:num_reqs]
    # Clamp negative entries (new requests) to 0; the participating mask zeroes
    # out their correction so the gathered values are don't-care.
    gather_indices = np.maximum(prev_positions, 0)
    prev_drafts = prev_num_draft_tokens_np[gather_indices]
    valid_counts = valid_sampled_token_count_np[gather_indices]

    participating = (prev_positions >= 0) & (prev_drafts > 0)
    # rejected_for_participating == correction; non-participating reqs end up
    # at zero via the mask multiply.
    correction = (prev_drafts + 1 - valid_counts) * participating
    optimistic_seq_lens_cpu_np[:num_reqs] -= correction.astype(optimistic_seq_lens_cpu_np.dtype, copy=False)
