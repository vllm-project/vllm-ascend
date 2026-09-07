# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import math
from typing import Any

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

        self.initial_verify_budget_per_req = int(method_params.get("initial_verify_budget_per_req", 5))

        self.budget_update_interval = int(method_params.get("budget_update_interval", 16))

        self.budget_threshold = float(method_params.get("budget_threshold", 0.3))

        self.min_k = int(method_params.get("min_verify_tokens", 1))

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
