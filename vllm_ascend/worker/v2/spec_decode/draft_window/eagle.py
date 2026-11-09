# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Eagle's frozen proposal anchor and capture-consumer state, composed by its owner."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import numpy as np
import torch
from vllm.config.compilation import CUDAGraphMode

from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.worker.v2.spec_decode.draft_window.graph import (
    DraftWindowGraphBindingProvider,
    DraftWindowGraphBucket,
    bind_draft_window_read,
    copy_draft_attention_metadata,
)
from vllm_ascend.worker.v2.spec_decode.draft_window.read import compute_draft_window_start
from vllm_ascend.worker.v2.spec_decode.draft_window.runtime import continuing_prefill_rows


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


class EagleDraftWindow:
    def __init__(self, owner: Any, provider: DraftWindowGraphBindingProvider):
        self.owner, self.provider = owner, provider
        self.intervals: tuple[tuple[int, int], ...] | None = None
        self.next_step = 0
        self.expected_steps = owner.num_speculative_steps
        self.capture_call: Iterator[DraftWindowGraphBucket] | None = None
        self.dp_dummy = False
        # The merged AR graph stages all consumers together, with separate banks.
        owner.supports_decode_metadata_update = False

    @contextmanager
    def proposal_scope(
        self,
        input_batch: Any,
        *,
        dummy_run: bool,
        is_profile: bool,
        num_speculative_tokens: int | None = None,
    ) -> Iterator[None]:
        """Own one real proposal, or explicitly run the baseline path."""
        expected_steps = _eagle3_graph_scheduled_consumers(self.owner.num_speculative_steps, num_speculative_tokens)
        provider = self.provider
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
                    and getattr(self.owner, "dp_size", None) == 2
                    and not is_profile
                    and getattr(self, "capture_call", None) is None
                    and (dummy_run or getattr(input_batch, "is_dummy", False))
                )
                if idle:
                    if (
                        dummy_run is not True
                        or input_batch.is_dummy is not True
                        or self.owner.input_batch.is_dummy is not True
                    ):
                        raise RuntimeError(
                            "Eagle DP idle requires the fixed core's real dummy input, not a K0 business row"
                        )
                    provider.require_reusable()
                self.dp_dummy = idle
                try:
                    yield
                finally:
                    self.dp_dummy = False
            return

        if not provider._carriers:
            raise RuntimeError("Eagle3 Draft Window business proposals require their installed Graph manager")
        self.intervals = None
        self.next_step = 0
        self.expected_steps = expected_steps
        try:
            with provider.proposal_scope(rows):
                yield
                if self.next_step != expected_steps:
                    raise RuntimeError(
                        "eligible proposal completed without consuming all Eagle3 steps: "
                        f"consumed={self.next_step} expected={expected_steps} "
                        f"static_K={self.owner.num_speculative_steps}"
                    )
        finally:
            self.intervals = None
            self.next_step = 0
            self.expected_steps = self.owner.num_speculative_steps

    def active(self) -> bool:
        rows = self.provider.proposal
        return rows is not None and len(rows) > 0 and not all(rows)

    def materialize(
        self,
        capture: DraftWindowGraphBucket,
        step: int,
        cpu_full_exact: torch.Tensor | None = None,
    ) -> DraftWindowGraphBucket:
        """Freeze the first anchor, then project each AR end into its own bank."""
        if type(step) is not int or not 0 <= step < self.expected_steps or step != self.next_step:
            raise RuntimeError("Eagle3 Window step is stale or its previous reader remains active")
        provider = self.provider
        rows = provider.proposal
        if rows is None:
            raise RuntimeError("Eagle3 Draft Window operation has no active proposal")
        batch = self.owner.input_batch
        num_reqs, num_reqs_padded = batch.num_reqs, capture.descriptor.num_reqs
        if num_reqs != len(rows):
            raise RuntimeError("Eagle3 request geometry changed within one proposal")
        proposal = self.intervals
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
                        future_offset=self.owner.num_speculative_steps,
                        window_size=provider.window_size,
                        block_span_tokens=provider.group.table_column_span,
                    ),
                    int(value) + self.owner.num_speculative_steps,
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
        from vllm_ascend.worker.v2.spec_decode.draft_window.config import (
            validate_eagle_graph_exact,
        )

        validate_eagle_graph_exact(self.owner, batch, exact, num_reqs=num_reqs, padded=num_reqs_padded, step=step)
        bank = provider.prepare(
            capture,
            frozen_intervals=proposal,
            cpu_full_exact=exact,
            idx_mapping_np=batch.idx_mapping_np,
        )
        self.intervals = proposal
        return bank

    def install(self):
        owner, provider = self.owner, self.provider
        capacity = min(owner.max_model_len, provider.window_size + provider.group.table_column_span - 1)
        for first, manager in ((True, owner.prefill_cudagraph_manager), (False, owner.decode_cudagraph_manager)):
            if owner.update_stream is None or manager.update_stream is not owner.update_stream:
                raise RuntimeError("Eagle3 capture managers must share their owner UPDATE stream")
            q = owner.num_speculative_steps + 1 if first else 1
            for descriptors in manager._capture_descs.values():
                for desc in descriptors:
                    if desc.cg_mode != CUDAGraphMode.FULL:
                        raise NotImplementedError("Eagle3 Window capture requires an actual FULL descriptor")
                    n = desc.num_reqs
                    if n is None or desc.num_tokens != n * q:
                        raise RuntimeError("Eagle3 startup descriptor does not match its request/query width")
                    for ordinal in range(1 if first else owner.num_speculative_steps - 1):
                        bank = provider.register_capture_identity(descriptor=desc, consumer_ordinal=ordinal)
                        # Seed before GLOBAL capture; no H2D belongs in its callback.
                        bank.cpu_window_exact.fill_(capacity)
        if not provider._carriers:
            raise RuntimeError("Eagle3 Window requires a nonempty capture bucket inventory")

    @contextmanager
    def capture_scope(self, *, n: int, tokens: int):
        if self.capture_call is not None:
            raise RuntimeError("nested Eagle3 startup capture invocation")
        self.capture_call = iter(self.provider.capture_banks_for(num_reqs=n, num_tokens=tokens))
        try:
            yield
            if next(self.capture_call, None) is not None:
                raise RuntimeError("Eagle3 capture did not visit every actual first-pass/AR model consumer")
        finally:
            self.capture_call = None

    def native_metadata(self, input_batch, attn_metadata, slot_mappings, cg_mode):
        if not self.active() or cg_mode == CUDAGraphMode.FULL:
            return attn_metadata, slot_mappings
        if self.next_step != 0 or cg_mode not in (CUDAGraphMode.NONE, CUDAGraphMode.PIECEWISE):
            raise RuntimeError("Eagle3 Window decode uses its FULL Graph managers; no eager fallback")
        # Native first retains the target's query/write operands. Only its
        # private Draft header and the first reader's stable bank change.
        provider = self.provider
        n = input_batch.num_reqs
        queries = input_batch.query_start_loc_np[1 : n + 1].tolist()
        layer = attn_metadata[provider.group.layers[0]]
        if (
            layer.actual_seq_lengths_q != queries
            or layer.num_actual_tokens != queries[-1]
            or queries[-1] > input_batch.num_tokens_after_padding
            or layer.block_tables.shape[0] < n
            or len(layer.seq_lens_list) < n
        ):
            raise NotImplementedError(
                "Eagle3 Window native prefill requires canonical effective query rows; "
                "additional padding requests are not qualified for this read bank"
            )
        bank = provider.capture_banks_for(num_reqs=n, num_tokens=n * (self.owner.num_speculative_steps + 1))[0]
        self.owner.block_tables.gather_block_tables(input_batch.idx_mapping, num_reqs_padded=n)
        bank = self.materialize(bank, 0)
        attn_metadata = copy_draft_attention_metadata(attn_metadata, provider.group.layers)
        layer = attn_metadata[provider.group.layers[0]]
        bind_draft_window_read(layer, bank)
        layer.attn_state = AscendAttentionState.ChunkedPrefill
        return attn_metadata, slot_mappings

    def native_done(self):
        if self.active() and self.next_step == 0:
            # This releases Host preparation, not the device reader. Any later
            # proposal failure still quarantines the strongly owned banks.
            self.provider.complete_native()
            self.next_step = 1

    def capture_metadata(self, attn_metadata, cudagraph_runtime_mode):
        capture_call = getattr(self, "capture_call", None)
        if capture_call is not None:
            provider = self.provider
            bank = next(capture_call, None)
            if bank is None:
                raise RuntimeError("Eagle3 capture visited more consumers than its real descriptor")
            n = bank.descriptor.num_reqs
            attn_metadata = copy_draft_attention_metadata(attn_metadata, provider.group.layers)
            query = self.owner.input_buffers.query_start_loc
            if query.dtype != torch.int32 or query.ndim != 1 or query.numel() < n + 1:
                raise RuntimeError("Eagle3 capture requires its owned int32 Full input query")
            layer = attn_metadata[provider.group.layers[0]]
            bind_draft_window_read(layer, bank)
            layer.causal = True
            # Use the owned Full query, not a builder's transient GPU copy.
            layer.query_start_loc = query[: n + 1]
        elif self.active() and (
            self.provider._active_bank is None
            or cudagraph_runtime_mode not in (CUDAGraphMode.NONE, CUDAGraphMode.PIECEWISE)
        ):
            raise RuntimeError("Eagle3 Window native first requires its prepared Draft bank; no eager fallback")
        return attn_metadata

    def prepare_decode(self, num_reqs, skip_attn, batch_desc, num_speculative_steps):
        if self.active():
            provider = self.provider
            padded = batch_desc.num_reqs or num_reqs
            if (
                self.expected_steps == 1
                or self.next_step != 1
                or num_reqs != self.owner.input_batch.num_reqs
                or padded < num_reqs
                or batch_desc.cg_mode != CUDAGraphMode.FULL
                or skip_attn
                or num_speculative_steps not in (None, self.owner.num_speculative_steps)
            ):
                raise RuntimeError("Eagle3 Window requires its actual static-K FULL AR descriptor; no eager fallback")
            # Rejection correction is already applied. Keep its sole D2H
            # snapshot local while staging all merged AR consumers.
            exact = self.owner.input_buffers.seq_lens[:num_reqs].to(device="cpu", dtype=torch.int32, copy=True)
            for step in range(1, self.owner.num_speculative_steps):
                capture = provider.capture_banks_for(num_reqs=padded, num_tokens=batch_desc.num_tokens)[step - 1]
                self.materialize(capture, step, exact.add(step - 1).clamp_(max=self.owner.max_model_len))
                provider.bind_current(capture.descriptor)
                self.next_step = step + 1

    def prepare_first_graph(self, desc):
        bank = self.provider.capture_banks_for(num_reqs=desc.num_reqs, num_tokens=desc.num_tokens)[0]
        self.materialize(bank, 0)
        self.provider.bind_current(bank.descriptor)
        self.next_step = 1
