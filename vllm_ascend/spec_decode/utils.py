# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import math
from contextlib import contextmanager
from dataclasses import replace
from typing import Any

import numpy as np
import torch
import vllm.distributed.parallel_state as _ps  # type: ignore[import-not-found]
from vllm.config import CompilationMode
from vllm.triton_utils import HAS_TRITON

if HAS_TRITON:
    from vllm_ascend.ops.triton.update_num_computed_tokens import (
        update_num_computed_tokens_triton,
    )
else:
    update_num_computed_tokens_triton = None


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
    if update_num_computed_tokens_triton is not None and num_computed_tokens.device.type != "cpu":
        update_num_computed_tokens_triton(
            num_computed_tokens,
            num_accepted_tokens,
            prev_positions,
            valid_sampled_token_count,
            prev_num_draft_tokens,
            cpu_num_computed_tokens,
        )
        return
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
    Sliding-window draft attention for the draft model (EAGLE3 / DFlash / DSpark).
    Caps the draft model's attention to the most recent ``window_size`` (W) tokens
    by (a) cropping its block table to the window's blocks and (b) keeping every
    KV-length tensor the FIA kernel can read (notably ``_seq_lens_cpu`` for EAGLE3,
    GPU ``seq_lens`` for DFlash/DSpark ``parallel_drafting``) capped at W.
    Slot-mapping is untouched and still addresses the full, absolute KV cache via
    :attr:`full_block_table`.

    ``future_offset`` is the number of tokens beyond ``seq_lens`` (at :meth:`apply`
    time) that the window end must cover:
      * EAGLE3 passes ``num_speculative_tokens`` — its ``seq_lens`` is context-only
        and the K draft positions lie beyond it, so ``final = seq_lens + K``.
      * DFlash / DSpark pass ``0`` — ``set_inputs_first_pass`` already bakes the
        query stretch (bonus + mask) into ``seq_lens``, so ``final = seq_lens``.
    """

    def __init__(
        self,
        window_size: int,
        block_size: int,
        max_num_reqs: int,
        future_offset: int,
        device: torch.device,
    ) -> None:
        self.window_size: int = window_size
        self.block_size: int = block_size
        self.window_blocks = (window_size + block_size - 1) // block_size
        self.max_window_blocks = self.window_blocks + 1
        self._future_offset: int = future_offset
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
        k_future = self._future_offset
        w = self.window_size
        b = self.block_size
        num_reqs = common_attn_metadata.seq_lens.shape[0]
        full_cols = self.full_block_table.shape[1]

        # Window math on the (NPU) seq_lens. Pure arithmetic -> stays on NPU.
        self.start_tokens_in_window_rounding = ((common_attn_metadata.seq_lens + k_future - w).clamp(min=0) // b) * b
        self._windowed_seq_lens = common_attn_metadata.seq_lens - self.start_tokens_in_window_rounding
        start_block_indices = self.start_tokens_in_window_rounding // b

        # column offset grid [1, max_window_blocks]
        cols = torch.arange(self.max_window_blocks, device=self.full_block_table.device).unsqueeze(0)
        # source column per (row, col): start_block_indices[:, None] + cols
        src_cols = start_block_indices.unsqueeze(1) + cols
        # clamp to the valid full-block-table column range so gather never goes OOB
        src_cols_clamped = src_cols.clamp(max=full_cols - 1)

        gathered = torch.gather(self.full_block_table, 1, src_cols_clamped)

        needed = torch.clamp((self._windowed_seq_lens + b - 1) // b, max=self.max_window_blocks).unsqueeze(1)
        valid_mask = (cols < needed) & (src_cols < full_cols)
        out[:num_reqs].copy_(gathered * valid_mask.to(gathered.dtype))

    def apply(
        self,
        common_attn_metadata,
    ) -> None:
        self.full_block_table = common_attn_metadata.block_table_tensor
        num_reqs = common_attn_metadata.seq_lens.shape[0]
        k_future = self._future_offset
        w = self.window_size
        b = self.block_size

        self.compute_sliding_window_block_table(common_attn_metadata, self._block_table_clone)
        common_attn_metadata.block_table_tensor = self._block_table_clone[:num_reqs]

        # update NPU seq_lens: reuse the value computed in compute().
        common_attn_metadata.seq_lens = self._windowed_seq_lens

        for name in ("seq_lens_cpu", "_seq_lens_cpu", "seq_lens_cpu_upper_bound"):
            src = getattr(common_attn_metadata, name, None)
            if src is not None:
                _windowed = src - ((src + k_future - w).clamp(min=0) // b) * b
                setattr(common_attn_metadata, name, _windowed)


@contextmanager
def patch_tensor_parallel_group(tp_group):
    """Temporarily swap the global TP group for draft-model spec decode.

    vllm-ascend local implementation for swapping the global TP group so the
    draft model can run with a TP degree that differs from the target model.
    """
    old_tp_group = _ps.get_tp_group()
    _ps._TP_STATE_PATCHED = True
    _ps._TP = tp_group
    try:
        yield
    finally:
        _ps._TP_STATE_PATCHED = False
        _ps._TP = old_tp_group


# TODO: Remove it when the bug of fx-graph is solved
# patch vllm_config to be in CompilationMode.NONE temporarily
@contextmanager
def _maybe_eager_context(vllm_config):
    target_compilation_config = vllm_config.compilation_config
    draft_compilation_config = replace(
        target_compilation_config,
        mode=CompilationMode.NONE,
    )
    # Model layers use these registries even when compilation is disabled.
    draft_compilation_config.static_forward_context = target_compilation_config.static_forward_context
    draft_compilation_config.static_all_moe_layers = target_compilation_config.static_all_moe_layers
    vllm_config.compilation_config = draft_compilation_config
    try:
        yield
    finally:
        vllm_config.compilation_config = target_compilation_config


class DynamicSpecScheduler:
    """Dynamic verification scheduler shared by DFlash and DSpark.

    Both DFlash and DSpark use the same scheduling algorithm:

       method-specific confidence
           -> token acceptance probabilities [B, D]
           -> cumulative survival probabilities [B, D]
           -> shared verify budget
           -> per-request verify lengths [B]

    The only method-specific part is how token acceptance probabilities are
    estimated:

    * DFlash:
       probability of the argmax draft token.

    * DSpark:
       sigmoid output of the confidence head.

    Everything after token-probability estimation is identical.
    """

    def __init__(
        self,
        *,
        method: str,
        method_params: dict[str, Any],
        max_batch_size: int,
        num_speculative_tokens: int,
        device: torch.device,
    ) -> None:
        if method not in ("dflash", "dspark"):
            raise ValueError(f"Unsupported dynamic speculative method: {method}")

        self.method = method

        self.max_batch_size = max_batch_size
        self.num_speculative_tokens = num_speculative_tokens
        self.device = device

        # Shared configuration

        self.initial_verify_budget_per_req = int(
            method_params.get(
                "initial_verify_budget_per_req",
                5,
            )
        )

        self.budget_update_interval = int(
            method_params.get(
                "budget_update_interval",
                16,
            )
        )

        self.budget_threshold = float(
            method_params.get(
                "budget_threshold",
                0.3,
            )
        )

        self.min_k = int(
            method_params.get(
                "min_verify_tokens",
                1,
            )
        )

        self.budget_k = max(
            self.min_k,
            min(
                self.initial_verify_budget_per_req,
                self.num_speculative_tokens,
            ),
        )

        self._steps_since_budget_update = 0

        # Shared buffers

        # Conditional acceptance probability for every proposed token.
        # token_probs[b, i] ~= P(token_i accepted | prefix accepted)
        # Shape: [B, D]
        self._token_probs_buffer = torch.empty(
            (
                self.max_batch_size,
                self.num_speculative_tokens,
            ),
            dtype=torch.float32,
            device=device,
        )

        # Cumulative survival probability.
        # survival[b, i] = prod(token_probs[b, :i + 1])
        # Shape: [B, D]
        self._survival_buffer = torch.empty(
            (
                self.max_batch_size,
                self.num_speculative_tokens,
            ),
            dtype=torch.float32,
            device=device,
        )

        # Final verification length selected for each request.
        # Shape: [B]
        self._num_verify_tokens_buffer = torch.empty(
            self.max_batch_size,
            dtype=torch.int32,
            device=device,
        )

        # Reused scatter_add source.
        self._scatter_ones_buffer = torch.ones(
            self.max_batch_size * self.num_speculative_tokens,
            dtype=torch.int32,
            device=device,
        )

        # Latest result consumed by the model runner.
        self.num_verify_tokens: torch.Tensor | None = None

    def update(
        self,
        *,
        logits: torch.Tensor | None = None,
        model=None,
        last_hidden_states: torch.Tensor | None = None,
        draft_token_ids: torch.Tensor | None = None,
        num_reqs: int | None = None,
    ) -> torch.Tensor:
        if self.method == "dflash":
            if logits is None:
                raise ValueError("DFlash requires logits.")

            token_probs = self._compute_dflash_token_probs(
                logits,
            )
        elif self.method == "dspark":
            if num_reqs is None:
                raise ValueError("DSpark requires num_reqs.")

            token_probs = self._compute_dspark_token_probs(
                model,
                last_hidden_states,
                draft_token_ids,
                num_reqs,
            )
        else:
            raise RuntimeError(f"Unsupported dynamic speculative method: {self.method}")

        return self._update_from_token_probs(token_probs)

    def _compute_dflash_token_probs(
        self,
        logits: torch.Tensor,
    ) -> torch.Tensor:
        """Estimate DFlash token acceptance probabilities.

        DFlash has no confidence head, so the softmax probability of the
        argmax draft token is used as the acceptance-confidence proxy.

        Input:
            logits: [B * D, V]

        Output:
            token_probs: [B, D]
        """
        num_rows = logits.shape[0]
        num_draft_tokens = self.num_speculative_tokens
        num_reqs = num_rows // num_draft_tokens

        token_probs = self._token_probs_buffer[:num_reqs]
        # max(softmax(logits)) per row; PyTorch keeps this ACLGraph-safe.
        token_probs.copy_(torch.softmax(logits.float(), dim=-1).max(dim=-1).values.view(num_reqs, num_draft_tokens))
        token_probs.clamp_(
            min=1e-6,
            max=1.0,
        )

        return token_probs

    def _compute_dspark_token_probs(
        self,
        model,
        last_hidden_states: torch.Tensor,
        draft_token_ids: torch.Tensor,
        num_reqs: int,
    ) -> torch.Tensor:
        """Estimate DSpark token acceptance probabilities.

        ``compute_confidence`` already returns per-position acceptance
        probabilities (sigmoid of the confidence-head logits).

        Output:
            token_probs: [B, D]
        """
        num_draft_tokens = self.num_speculative_tokens
        num_tokens = num_reqs * num_draft_tokens

        flat_hidden = last_hidden_states.reshape(
            num_tokens,
            last_hidden_states.shape[-1],
        )

        # draft_token_ids normally has shape [B, D + 1] for DSpark:
        # [seed, draft_1, ..., draft_D]
        # The confidence prediction for D positions uses the first D
        # Markov inputs.
        markov_embs = model.markov_embed(
            draft_token_ids[
                :num_reqs,
                :num_draft_tokens,
            ]
        )

        flat_markov = markov_embs.reshape(
            num_tokens,
            markov_embs.shape[-1],
        ).to(flat_hidden.dtype)

        confidence = model.compute_confidence(
            flat_hidden,
            flat_markov,
        )

        token_probs = self._token_probs_buffer[:num_reqs]

        token_probs.copy_(
            confidence.reshape(
                num_reqs,
                num_draft_tokens,
            )
        )

        token_probs.clamp_(
            min=1e-6,
            max=1.0,
        )

        return token_probs

    def _update_from_token_probs(
        self,
        token_probs: torch.Tensor,
    ) -> torch.Tensor:
        """Run the shared dynamic speculative scheduling pipeline."""
        num_reqs, num_draft_tokens = token_probs.shape

        survival = self._survival_buffer[:num_reqs]

        # survival[b, i] estimates the probability that request b reaches
        # and accepts the draft prefix through position i.
        torch.cumprod(
            token_probs,
            dim=1,
            out=survival,
        )

        self.compute_verify_budget(survival)

        self.num_verify_tokens = self.allocate_verify_budget(survival)

        return self.num_verify_tokens

    def compute_verify_budget(
        self,
        survival: torch.Tensor,
    ) -> None:
        """Periodically recompute the shared per-request verify budget."""
        self._steps_since_budget_update += 1

        if self._steps_since_budget_update < self.budget_update_interval:
            return

        self._steps_since_budget_update = 0

        num_reqs = survival.shape[0]

        if num_reqs == 0:
            return

        # Count cumulative-prefix positions whose estimated probability of
        # being reached and accepted exceeds the configured threshold.
        # `.item()` introduces an NPU -> CPU synchronization, but only on
        # budget-update steps.
        mean_k = float((survival >= self.budget_threshold).sum().item()) / float(num_reqs)

        new_budget_k = math.ceil(mean_k)

        # Previously measured on Qwen3-8B on A3:
        # verification costs of adjacent budgets differ only slightly,
        # and the next odd speculative budget may be approximately equal
        # to or cheaper than the previous even one.
        # Example: batch=64 K=6 -> 52.9 K=7 -> 54.3
        # Verification also includes the bonus token, so an odd K gives an
        # even verification width. Current kernels can process these widths
        # more efficiently, potentially due to padding / next_power_of_2().
        if new_budget_k % 2 == 0 and new_budget_k < self.num_speculative_tokens:
            new_budget_k += 1

        self.budget_k = max(
            self.min_k,
            min(
                new_budget_k,
                self.num_speculative_tokens,
            ),
        )

    def allocate_verify_budget(
        self,
        survival: torch.Tensor,
    ) -> torch.Tensor:
        """Distribute the global verification budget across requests.

        Every request receives at least `min_k` tokens.

        The remaining global token budget is assigned to the largest
        cumulative survival probabilities across the whole batch.

        Because cumulative survival is monotonically non-increasing inside
        each request, selecting the globally highest positions naturally
        produces prefix lengths.
        """
        num_reqs, num_draft_tokens = survival.shape

        keep_lens = self._num_verify_tokens_buffer[:num_reqs]

        keep_lens.fill_(self.min_k)

        extra_budget_per_req = max(
            self.budget_k - self.min_k,
            0,
        )

        # Positions [0:min_k] have already been guaranteed.
        candidate_window = survival[
            :,
            self.min_k :,
        ]

        num_candidates = candidate_window.numel()

        num_budget_tokens = min(
            num_reqs * extra_budget_per_req,
            num_candidates,
        )

        if num_budget_tokens > 0:
            candidate_cols = num_draft_tokens - self.min_k

            flat_survival = candidate_window.reshape(-1)

            _, top_indices = torch.topk(
                flat_survival,
                k=num_budget_tokens,
                largest=True,
                sorted=False,
            )

            chosen_requests = torch.div(
                top_indices,
                candidate_cols,
                rounding_mode="floor",
            )

            keep_lens.scatter_add_(
                0,
                chosen_requests,
                self._scatter_ones_buffer[:num_budget_tokens],
            )

        keep_lens.clamp_(
            min=self.min_k,
            max=num_draft_tokens,
        )

        return keep_lens
