# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Hardware-aware allocation of speculative verification prefixes."""

from __future__ import annotations

import torch

from vllm_ascend.spec_decode.dynamic.cost_model import HardwareCostModel


class HardwareAwarePrefixPolicy:
    """Choose per-request prefixes by maximizing expected token throughput.

    The policy consumes cumulative prefix survival probabilities shaped
    ``[num_requests, max_draft_tokens]``.  Because each row is monotonically
    non-increasing, selecting the globally highest marginal scores preserves a
    valid prefix for every request.  A deterministic tiny tie-break favors an
    earlier position when a model emits equal scores.
    """

    def __init__(
        self,
        *,
        cost_model: HardwareCostModel,
        min_k: int,
        max_batch_size: int,
        max_draft_tokens: int,
        device: torch.device,
        decision_interval: int = 16,
    ) -> None:
        if min_k < 0 or min_k > max_draft_tokens:
            raise ValueError("min_k must be in [0, max_draft_tokens]")
        if decision_interval <= 0:
            raise ValueError("decision_interval must be > 0")
        self.cost_model = cost_model
        self.min_k = min_k
        self.max_batch_size = max_batch_size
        self.max_draft_tokens = max_draft_tokens
        self.device = device
        self.decision_interval = decision_interval
        self._steps = 0
        self._best_total_tokens: int | None = None
        self._last_num_reqs: int | None = None
        self._last_num_draft_tokens: int | None = None
        self.last_goodput: float | None = None

        # The profile is tiny, so keeping the lookup on device avoids a CPU
        # round-trip for each candidate m.  Zero means that a shape was not
        # profiled and is handled with the nearest profiled shape on CPU before
        # constructing the tensor.
        max_tokens = max_batch_size + max_draft_tokens * max_batch_size
        costs = [0.0] * (max_tokens + 1)
        for token_count in range(1, max_tokens + 1):
            costs[token_count] = cost_model.latency(token_count)
        self._latency_ms = torch.tensor(costs, dtype=torch.float32, device=device)

    def _lookup_latency(self, token_counts: torch.Tensor) -> torch.Tensor:
        return self._latency_ms[token_counts.clamp(min=1, max=self._latency_ms.numel() - 1)]

    def allocate(self, survival: torch.Tensor) -> torch.Tensor:
        num_reqs, num_draft_tokens = survival.shape
        if num_reqs == 0:
            return torch.empty((0,), dtype=torch.int32, device=survival.device)
        if num_draft_tokens > self.max_draft_tokens:
            raise ValueError(
                f"at most {self.max_draft_tokens} draft positions are supported, "
                f"got {num_draft_tokens}"
            )

        # Batch-level dynamic-K may temporarily use a smaller physical width.
        # Recompute the hardware optimum for that width instead of reusing a
        # budget computed for a different candidate set.
        if self._last_num_draft_tokens != num_draft_tokens:
            self._best_total_tokens = None
            self._last_num_draft_tokens = num_draft_tokens

        self._steps += 1
        mandatory = min(self.min_k, num_draft_tokens)
        max_total = num_reqs * num_draft_tokens
        base_total = num_reqs * mandatory

        # Recompute the global optimum periodically.  Confidence still drives
        # the per-request allocation on every step, but the hardware budget is
        # stable across a short interval and does not need a top-k search each
        # time.
        should_recompute = (
            self._best_total_tokens is None
            or self._last_num_reqs != num_reqs
            or self._steps % self.decision_interval == 0
        )
        if should_recompute:
            mandatory_accepts = survival[:, :mandatory].sum() if mandatory else survival.new_zeros(())
            candidates = survival[:, mandatory:]
            candidate_count = candidates.numel()
            if candidate_count:
                # Favor earlier positions in exact ties.  The perturbation is
                # far below fp32 confidence resolution and does not change the
                # ordering of materially different scores.
                cols = torch.arange(
                    mandatory,
                    num_draft_tokens,
                    device=survival.device,
                    dtype=survival.dtype,
                ).repeat(num_reqs, 1)
                tie_break = (num_draft_tokens - cols) * torch.finfo(survival.dtype).eps
                ranked = (candidates + tie_break).reshape(-1)
                ranked_values, _ = torch.sort(ranked, descending=True)
                prefix = torch.cumsum(ranked_values, dim=0)
                candidate_totals = torch.arange(
                    base_total + 1,
                    max_total + 1,
                    device=survival.device,
                    dtype=torch.int64,
                )
                expected_accepts = num_reqs + mandatory_accepts + prefix
                token_counts = candidate_totals
                goodput = expected_accepts / self._lookup_latency(token_counts)
                baseline_expected = num_reqs + mandatory_accepts
                baseline_total = torch.tensor(base_total, device=survival.device, dtype=torch.int64)
                baseline_goodput = baseline_expected / self._lookup_latency(baseline_total)
                all_goodput = torch.cat((baseline_goodput.reshape(1), goodput))
                best_offset = int(torch.argmax(all_goodput).item())
                best_total = base_total + best_offset
                self.last_goodput = float(all_goodput[best_offset].item())
            else:
                best_total = base_total
                self.last_goodput = float(
                    ((num_reqs + (survival[:, :mandatory].sum() if mandatory else 0.0))
                     / self.cost_model.latency(max(1, base_total)))
                )
            self._best_total_tokens = max(base_total, min(best_total, max_total))
            self._last_num_reqs = num_reqs

        selected_total = self._best_total_tokens
        assert selected_total is not None
        extra = selected_total - base_total
        lengths = torch.full(
            (num_reqs,), mandatory, dtype=torch.int32, device=survival.device
        )
        if extra <= 0:
            return lengths

        candidates = survival[:, mandatory:]
        cols = torch.arange(
            mandatory,
            num_draft_tokens,
            device=survival.device,
            dtype=survival.dtype,
        ).repeat(num_reqs, 1)
        tie_break = (num_draft_tokens - cols) * torch.finfo(survival.dtype).eps
        ranked = (candidates + tie_break).reshape(-1)
        _, selected = torch.topk(ranked, k=min(extra, ranked.numel()), largest=True, sorted=False)
        request_indices = torch.div(
            selected,
            num_draft_tokens - mandatory,
            rounding_mode="floor",
        )
        lengths.scatter_add_(0, request_indices, torch.ones_like(request_indices, dtype=torch.int32))
        return lengths.clamp_(min=mandatory, max=num_draft_tokens)
