# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Freeze Eagle's proposal anchor and prepare each real Graph consumer's read bank."""

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import numpy as np
import torch

from vllm_ascend.worker.v2.spec_decode.draft_window_graph import DraftWindowGraphBindingProvider, DraftWindowGraphBucket
from vllm_ascend.worker.v2.spec_decode.draft_window_plan import compute_draft_window_start
from vllm_ascend.worker.v2.spec_decode.draft_window_runtime import continuing_prefill_rows


def _eagle3_graph_scheduled_consumers(static_k: int, scheduled: int | None) -> int:
    """Separate a zero-draft first reader from the fixed capture geometry."""
    if type(static_k) is not int or static_k < 1:
        raise ValueError("Eagle3 Graph requires a positive configured static K")
    if scheduled is None:
        return static_k
    if type(scheduled) is not int or scheduled not in (0, static_k):
        raise NotImplementedError(
            "Eagle3 Window Graph permits only zero-draft first-only or its configured static K; "
            "partial merged AR Graph proposals are not qualified"
        )
    return 1 if scheduled == 0 else static_k


class Eagle3DraftWindowMixin:
    """Proposal scope and frozen window geometry for static-K Eagle Graph."""

    _draft_window_graph_binding_provider: DraftWindowGraphBindingProvider
    _eagle3_window_intervals: tuple[tuple[int, int], ...] | None
    _eagle3_window_next_step: int
    input_batch: Any
    input_buffers: Any
    block_tables: Any
    max_model_len: int
    method: str
    num_speculative_steps: int

    def _initialize_eagle3_draft_window_state(self) -> None:
        """Initialize proposal state after source/configuration validation."""
        self._eagle3_window_intervals = None
        self._eagle3_window_next_step = 0
        self._eagle3_window_expected_steps = self.num_speculative_steps

    @contextmanager
    def _eagle3_draft_window_proposal_scope(
        self,
        input_batch: Any,
        *,
        dummy_run: bool,
        is_profile: bool,
        num_speculative_tokens: int | None = None,
    ) -> Iterator[None]:
        """Own one real proposal, or explicitly run the baseline path."""
        if not hasattr(self, "_draft_window_graph_binding_provider"):
            yield
            return
        expected_steps = _eagle3_graph_scheduled_consumers(self.num_speculative_steps, num_speculative_tokens)
        provider = self._draft_window_graph_binding_provider
        rows: tuple[bool, ...] = ()
        if not dummy_run and not is_profile and not bool(getattr(input_batch, "is_dummy", False)):
            rows = tuple(map(bool, continuing_prefill_rows(input_batch, input_batch.num_reqs)))
        if not rows or all(rows):
            with provider.proposal_scope(rows):
                # An idle DP replica executes the original Full/PAD first/AR
                # graphs. Keep that work in the same owning proposal scope,
                # so any failed preparation or UPDATE quarantines its banks.
                idle = (
                    bool(provider._carriers)
                    and getattr(self, "dp_size", None) == 2
                    and not is_profile
                    and getattr(self, "_eagle3_capture_call", None) is None
                    and (dummy_run or getattr(input_batch, "is_dummy", False))
                )
                if idle:
                    if (
                        dummy_run is not True
                        or input_batch.is_dummy is not True
                        or self.input_batch.is_dummy is not True
                    ):
                        raise RuntimeError(
                            "Eagle DP idle requires the fixed core's real dummy input, not a K0 business row"
                        )
                    provider.require_reusable()
                self._eagle3_graph_dp_dummy_active = idle
                try:
                    yield
                finally:
                    self._eagle3_graph_dp_dummy_active = False
            return

        if not provider._carriers:
            raise RuntimeError("Eagle3 Draft Window business proposals require their installed Graph manager")
        self._eagle3_window_intervals = None
        self._eagle3_window_next_step = 0
        self._eagle3_window_expected_steps = expected_steps
        try:
            with provider.proposal_scope(rows):
                yield
                if self._eagle3_window_next_step != expected_steps:
                    raise RuntimeError(
                        "eligible proposal completed without consuming all Eagle3 steps: "
                        f"consumed={self._eagle3_window_next_step} expected={expected_steps} "
                        f"static_K={self.num_speculative_steps}"
                    )
        finally:
            self._eagle3_window_intervals = None
            self._eagle3_window_next_step = 0
            self._eagle3_window_expected_steps = self.num_speculative_steps

    def _eagle3_window_is_active(self) -> bool:
        provider = getattr(self, "_draft_window_graph_binding_provider", None)
        rows = provider.proposal if provider is not None else None
        return bool(rows) and not all(rows)

    def _materialize_eagle3_step(
        self,
        capture: DraftWindowGraphBucket,
        step: int,
        cpu_full_exact: torch.Tensor | None = None,
    ) -> DraftWindowGraphBucket:
        """Freeze the first anchor, then project each AR end into its own bank."""
        if (
            type(step) is not int
            or not 0 <= step < self._eagle3_window_expected_steps
            or step != self._eagle3_window_next_step
        ):
            raise RuntimeError("Eagle3 Window step is stale or its previous reader remains active")
        provider = self._draft_window_graph_binding_provider
        rows = provider.proposal
        if rows is None:
            raise RuntimeError("Eagle3 Draft Window operation has no active proposal")
        batch = self.input_batch
        num_reqs, num_reqs_padded = batch.num_reqs, capture.descriptor.num_reqs
        if num_reqs != len(rows):
            raise RuntimeError("Eagle3 request geometry changed within one proposal")
        proposal = self._eagle3_window_intervals
        if step == 0:
            if proposal is not None:
                raise RuntimeError("Eagle3 first-pass Window state is already materialized")
            values = np.asarray(batch.seq_lens_np)
            if values.ndim != 1 or values.shape[0] < num_reqs or not np.issubdtype(values.dtype, np.integer):
                raise ValueError("proposal Full exact lengths must cover the active integer request rows")
            if bool(np.any(values[:num_reqs] < 0)):
                raise ValueError("proposal Full exact lengths must be non-negative")
            exact = torch.tensor(values[:num_reqs], dtype=torch.int32)
            proposal = tuple(
                (
                    compute_draft_window_start(
                        full_exact_end=int(value),
                        future_offset=self.num_speculative_steps,
                        window_size=provider.window_size,
                        block_span_tokens=provider.group.table_column_span,
                    ),
                    int(value) + self.num_speculative_steps,
                )
                for value in exact[:num_reqs]
            )
            if batch.has_prefill:
                widths = np.diff(batch.query_start_loc_np[: num_reqs + 1])
                if any(
                    not rows[row] and query > int(exact[row]) - interval[0]
                    for row, (query, interval) in enumerate(zip(widths, proposal))
                ):
                    raise NotImplementedError(
                        "Eagle3 Window terminal prefill query begins before its frozen read window; "
                        "use a smaller chunked-prefill budget"
                    )
        else:
            if proposal is None or cpu_full_exact is None:
                raise RuntimeError("Eagle3 decode step requires its frozen anchor and corrected exact length")
            exact = cpu_full_exact
        from vllm_ascend.worker.v2.spec_decode.draft_graph_review_limits import (
            validate_eagle_single_request_review_exact,
        )

        validate_eagle_single_request_review_exact(
            self, batch, exact, num_reqs=num_reqs, padded=num_reqs_padded, step=step
        )
        bank = provider.prepare(
            capture,
            frozen_intervals=proposal,
            cpu_full_exact=exact,
            idx_mapping_np=batch.idx_mapping_np,
        )
        self._eagle3_window_intervals = proposal
        return bank
