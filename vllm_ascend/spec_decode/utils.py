# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import numpy as np
import torch


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


class SlidingWindowAdapter:
    """
    Sliding-window draft attention for the EAGLE3 draft model.
    Caps the draft model's self-attention to the most recent ``window_size`` (W)
    tokens by (a) cropping its block table to the window's blocks and (b) keeping
    every KV-length tensor the FIA kernel can read (notably ``_ seq_lens_cpu``)
    capped at W. Slot-mapping is untouched and still addresses the full, absolute
    KV cache via :attr:`full_block_table`.
    """

    def __init__(
        self,
        window_size: int,
        block_size: int,
        max_num_reqs: int,
        num_speculative_tokens: int,
        device: torch.device,
    ) -> None:
        self.window_size: int = window_size
        self.block_size: int = block_size
        self.window_blocks = (window_size + block_size - 1) // block_size
        self.max_window_blocks = self.window_blocks + 1
        self.num_speculative_tokens = num_speculative_tokens
        self._block_table_clone = torch.zeros(
            (max_num_reqs, self.max_window_blocks),
            dtype=torch.int32,
            device=device,
        )

    def compute_sliding_window_block_table(
        self,
        common_attn_metadata,
        out: torch.Tensor,
    ) -> None:
        k = self.num_speculative_tokens
        w = self.window_size
        b = self.block_size
        num_reqs = common_attn_metadata.seq_lens.shape[0]

        # Window math on the (NPU) seq_lens. Pure arithmetic -> stays on NPU.
        self.start_tokens_in_window_rounding = ((common_attn_metadata.seq_lens + k - w).clamp(min=0) // b) * b
        self._windowed_seq_lens = common_attn_metadata.seq_lens - self.start_tokens_in_window_rounding
        start_block_indices = self.start_tokens_in_window_rounding // b
        needed_blocks_per_req = (self._windowed_seq_lens + b - 1) // b

        full_cols = self.full_block_table.shape[1]
        # column offset grid [1, max_window_blocks]
        cols = torch.arange(self.max_window_blocks, device=self.full_block_table.device).unsqueeze(0)
        # source column per (row, col): start_block_indices[:, None] + cols
        src_cols = start_block_indices.unsqueeze(1) + cols
        # clamp to the valid full-block-table column range so gather never goes OOB
        src_cols_clamped = src_cols.clamp(max=full_cols - 1)

        gathered = torch.gather(self.full_block_table, 1, src_cols_clamped)
        needed = torch.clamp(needed_blocks_per_req, max=self.max_window_blocks).unsqueeze(1)
        # keep only columns within `needed` and within the full table; zero the rest
        valid_mask = (cols < needed) & (src_cols < full_cols)
        out[:num_reqs].copy_(gathered * valid_mask.to(gathered.dtype))

    def apply(
        self,
        common_attn_metadata,
    ) -> None:
        self.full_block_table = common_attn_metadata.block_table_tensor
        num_reqs = common_attn_metadata.seq_lens.shape[0]
        k = self.num_speculative_tokens
        w = self.window_size
        b = self.block_size

        self.compute_sliding_window_block_table(common_attn_metadata, self._block_table_clone)
        common_attn_metadata.block_table_tensor = self._block_table_clone[:num_reqs]

        # update NPU seq_lens: reuse the value computed in compute().
        common_attn_metadata.seq_lens = self._windowed_seq_lens

        # update CPU mirrors: recompute from each one's own CPU tensor -> stays on CPU,
        # no D2H sync. numerically identical to the NPU
        for name in ("seq_lens_cpu", "_seq_lens_cpu", "seq_lens_cpu_upper_bound"):
            src = getattr(common_attn_metadata, name, None)
            if src is not None:
                _windowed_cpu = src - ((src + k - w).clamp(min=0) // b) * b
                setattr(common_attn_metadata, name, _windowed_cpu)
