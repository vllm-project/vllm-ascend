# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Dynamic speculative verification scheduling.

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

from __future__ import annotations

import math
from typing import Any

import torch

from vllm_ascend.spec_decode.dynamic_utils.kernel_utils.parallel_kernel import (
   _max_softmax_kernel,
   )


class DynamicSpecScheduler:
    """Dynamic verification scheduler shared by DFlash and DSpark."""

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
            raise ValueError(
               f"Unsupported dynamic speculative method: {method}"
           )

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

            token_probs = self._compute_dspark_token_probs(
               model,
               last_hidden_states,
               draft_token_ids,
               num_reqs,
            )

        else:
            raise RuntimeError(
               f"Unsupported dynamic speculative method: {self.method}"
            )

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
        num_rows, vocab_size = logits.shape
        num_draft_tokens = self.num_speculative_tokens
        num_reqs = num_rows // num_draft_tokens

        token_probs = self._token_probs_buffer[:num_reqs]
        flat_token_probs = token_probs.reshape(-1)

        _max_softmax_kernel[(num_rows,)](
           logits,
           flat_token_probs,
           vocab_size,
           logits.stride(0),
           logits.stride(1),
           BLOCK_SIZE=2048,
        )

        flat_token_probs.clamp_(
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
       The DSpark confidence head produces logits for each speculative
       position. Sigmoid converts them to conditional token acceptance
       probabilities.

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

        confidence_logits = model.confidence_logits(
           flat_hidden,
           flat_markov,
        )

        token_probs = self._token_probs_buffer[:num_reqs]

        token_probs.copy_(
           confidence_logits.reshape(
               num_reqs,
               num_draft_tokens,
           )
        )

        token_probs.sigmoid_()
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

        self._compute_verify_budget(
           survival
        )

        self.num_verify_tokens = self._allocate_verify_budget(
           survival
        )

        return self.num_verify_tokens

    def _compute_verify_budget(
       self,
       survival: torch.Tensor,
    ) -> None:
        """Periodically recompute the shared per-request verify budget."""
        self._steps_since_budget_update += 1

        if (
           self._steps_since_budget_update
           < self.budget_update_interval
        ):
            return

        self._steps_since_budget_update = 0

        num_reqs = survival.shape[0]

        if num_reqs == 0:
            return

        # Count cumulative-prefix positions whose estimated probability of
        # being reached and accepted exceeds the configured threshold.
        # `.item()` introduces an NPU -> CPU synchronization, but only on
        # budget-update steps.
        mean_k = float(
            (
               survival >= self.budget_threshold
            ).sum().item()
        ) / float(num_reqs)

        new_budget_k = math.ceil(
           mean_k
        )

        # Previously measured on Qwen3-8B on A3:
        # verification costs of adjacent budgets differ only slightly,
        # and the next odd speculative budget may be approximately equal
        # to or cheaper than the previous even one.
        # Example: batch=64 K=6 -> 52.9 K=7 -> 54.3
        # Verification also includes the bonus token, so an odd K gives an
        # even verification width. Current kernels can process these widths
        # more efficiently, potentially due to padding / next_power_of_2().
        if (
           new_budget_k % 2 == 0
           and new_budget_k < self.num_speculative_tokens
        ):
            new_budget_k += 1

        self.budget_k = max(
            self.min_k,
            min(
                new_budget_k,
                self.num_speculative_tokens,
            ),
        )

    def _allocate_verify_budget(
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

        keep_lens = self._num_verify_tokens_buffer[
            :num_reqs
        ]

        keep_lens.fill_(
            self.min_k
        )

        extra_budget_per_req = max(
            self.budget_k - self.min_k,
            0,
        )

        # Positions [0:min_k] have already been guaranteed.
        candidate_window = survival[
            :,
            self.min_k:,
        ]

        num_candidates = candidate_window.numel()

        num_budget_tokens = min(
            num_reqs * extra_budget_per_req,
            num_candidates,
        )

        if num_budget_tokens > 0:
            candidate_cols = (
                num_draft_tokens - self.min_k
            )

            flat_survival = candidate_window.reshape(
                -1
            )

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
                self._scatter_ones_buffer[
                    :num_budget_tokens
                ],
            )

        keep_lens.clamp_(
            min=self.min_k,
            max=num_draft_tokens,
        )

        return keep_lens