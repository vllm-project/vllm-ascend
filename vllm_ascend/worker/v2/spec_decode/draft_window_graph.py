# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Persistent Draft Graph read banks and runtime binding validation.

Startup capture fixes addresses. Each proposal updates only its private read
banks and metadata; canonical Full write tables, slots, and absolute positions
remain unchanged. One proposal scope owns preparation and UPDATE, retaining
capture banks across failure and until the original stream can reuse them.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor

from vllm_ascend.worker.v2.spec_decode.draft_window_backends import DraftWindowGroupRoute
from vllm_ascend.worker.v2.spec_decode.draft_window_plan import (
    compute_draft_window_start,
    window_capacity_tokens,
)


def draft_window_all_continuing(owner: Any) -> bool:
    """Select the original Full prefill reader before metadata/padding."""
    provider = getattr(owner, "_draft_window_graph_binding_provider", None)
    rows = provider.proposal if provider is not None else None
    return bool(rows) and all(rows)


@dataclass(slots=True, eq=False)
class DraftWindowGraphBucket:
    """Actual capture consumer and its strongly owned read tensors.

    Only its provider writes these private fields. Capture and replay borrow
    the same tensors; a proposal updates their contents, never their address.
    """

    descriptor: BatchExecutionDescriptor
    consumer_ordinal: int
    window_block_table: torch.Tensor
    cpu_window_exact: torch.Tensor
    owner: DraftWindowGraphBindingProvider


class DraftWindowGraphBindingProvider:
    """Own canonical source geometry, capture banks and one proposal binding.

    Borrow Full tables from the actual BlockTables owner at prepare; its CPU
    counts remain canonical even while the UVA count transport rotates.
    """

    def __init__(self, *, policy: int | None, group: DraftWindowGroupRoute, block_tables: Any) -> None:
        assert policy is not None
        self.window_size = policy
        self.group, self.block_tables = group, block_tables
        if self.table.shape[1] * group.table_column_span < window_capacity_tokens(
            self.window_size, group.table_column_span
        ):
            raise ValueError("Draft Graph canonical table cannot cover the Window capacity")
        self.proposal: tuple[bool, ...] | None = None
        self._active_bank: DraftWindowGraphBucket | None = None
        self.quarantined = False
        self._carriers: dict[tuple[int, int], tuple[DraftWindowGraphBucket, ...]] = {}
        self._pending: list[tuple[DraftWindowGraphBucket, dict[str, Any]]] = []

    @property
    def table(self) -> torch.Tensor:
        return self.block_tables.input_block_tables[self.group.gid]

    def quarantine(self) -> None:
        """Retain all banks after uncertain work, even without active inputs."""
        self.quarantined = True

    @contextmanager
    def proposal_scope(self, full_read_rows: tuple[bool, ...]) -> Iterator[None]:
        # The preparation boundary checks the actual stream before writing a
        # bank. Merely entering a proposal need not select a bank or its stream.
        if self.quarantined:
            raise RuntimeError("Draft Window bank is quarantined after uncertain submission")
        if self.proposal is not None or self._active_bank is not None or self._pending:
            raise RuntimeError("a Draft Window proposal is already active")
        self.proposal = full_read_rows
        try:
            yield
            if self._pending or self._active_bank is not None:
                raise RuntimeError("Draft Window proposal has unfinished Graph consumers")
        except BaseException:
            self.quarantine()
            raise
        finally:
            self.proposal = None

    def prepare(
        self,
        capture: DraftWindowGraphBucket,
        *,
        cpu_full_exact: torch.Tensor,
        idx_mapping_np: np.ndarray,
        frozen_intervals: tuple[tuple[int, int], ...] | None = None,
    ) -> DraftWindowGraphBucket:
        """Select and update the actual consumer before any preparation write."""
        self.require_reusable()
        proposal = self.proposal
        if proposal is None or capture.owner is not self:
            raise RuntimeError("Draft Window bank has no current owning proposal")
        if self._active_bank is not None:
            raise RuntimeError("previous Draft Window consumer is still active")
        if len(proposal) > capture.descriptor.num_reqs:
            raise ValueError("Draft Window active rows exceed the selected capture bucket")
        if capture.consumer_ordinal != len(self._pending):
            raise RuntimeError("Graph Window consumer is pending or outside its actual serial ordinal order")
        if self._pending and self._pending[0][0].descriptor != capture.descriptor:
            raise RuntimeError("Graph consumers belong to different execution buckets")
        self._active_bank = capture
        # Both method preparation entries construct this completed int32 CPU
        # snapshot; only its current row extent changes between proposals.
        if len(cpu_full_exact) < len(proposal) or len(idx_mapping_np) < len(proposal):
            raise ValueError("current CPU exact lengths and request slots are required")
        if frozen_intervals is not None and len(frozen_intervals) != len(proposal):
            raise ValueError("frozen Window anchors must match active request rows")
        # BlockTables owns initialized counts in persistent request-slot order.
        # Gather only real rows here; padding belongs to this bank, not to a
        # second count tensor passed through every method and provider layer.
        counts = self.block_tables.num_blocks.cpu[self.group.gid].index_select(
            0, torch.from_numpy(idx_mapping_np[: len(proposal)])
        )
        source_table, span = self.table, self.group.table_column_span
        intervals = []
        # Validate the whole current preparation before the first tensor write.
        for row, full_read in enumerate(proposal):
            exact, valid = int(cpu_full_exact[row]), int(counts[row])
            if exact < 0 or not 0 <= valid <= source_table.shape[1]:
                raise RuntimeError("current exact length or initialized block count is invalid")
            if full_read:
                start, required, visible = 0, valid, exact
            else:
                if frozen_intervals is None:
                    start_token = compute_draft_window_start(
                        full_exact_end=exact, window_size=self.window_size, block_span_tokens=span
                    )
                else:
                    start_token, horizon = frozen_intervals[row]
                    if not start_token <= exact <= horizon:
                        raise ValueError("exact end is outside the frozen proposal window/horizon")
                if start_token % span:
                    raise NotImplementedError("Draft Graph cannot read an intra-block Window start")
                start, visible = start_token // span, exact - start_token
                required = (visible + span - 1) // span
            if required > capture.window_block_table.shape[1]:
                raise RuntimeError("draft-window table capacity exceeded")
            if start + required > valid or visible > required * span:
                raise RuntimeError("draft-window source does not cover the exact readable interval")
            intervals.append((start, required, visible))
        capture.window_block_table.zero_()
        capture.cpu_window_exact.zero_()
        for row, (start, required, visible) in enumerate(intervals):
            if required:
                capture.window_block_table[row, :required].copy_(source_table[row, start : start + required])
            capture.cpu_window_exact[row] = visible

        return capture

    def complete_native(self) -> None:
        """Release the Host preparation after the real native first forward.

        Tensor ownership stays with capture and reuse keeps the existing
        current-stream dependency; this is not a device-completion claim.
        """
        if self.quarantined or self._active_bank is None:
            raise RuntimeError("Draft Window has no current native first consumer")
        self._active_bank = None

    def capture_banks_for(self, *, num_reqs: int, num_tokens: int) -> tuple[DraftWindowGraphBucket, ...]:
        """The fixed capture callbacks expose geometry, so index it once here."""
        try:
            return self._carriers[num_reqs, num_tokens]
        except KeyError as error:
            raise RuntimeError("Draft Window has no captured consumer for this geometry") from error

    def capture_identity_for(
        self,
        *,
        descriptor: BatchExecutionDescriptor,
    ) -> DraftWindowGraphBucket:
        bank = self.capture_banks_for(num_reqs=descriptor.num_reqs, num_tokens=descriptor.num_tokens)[0]
        if bank.descriptor != descriptor:
            raise RuntimeError(f"runtime descriptor does not resolve a captured consumer: {descriptor!r}")
        return bank

    def register_capture_identity(
        self,
        *,
        descriptor: BatchExecutionDescriptor,
        consumer_ordinal: int = 0,
    ) -> DraftWindowGraphBucket:
        """Bind the manager's exact frozen descriptor to one persistent bank."""
        if type(descriptor) is not BatchExecutionDescriptor:
            raise TypeError("capture requires the actual BatchExecutionDescriptor")
        if descriptor.cg_mode.name != "FULL":
            raise ValueError("Draft Window read capture requires a FULL attention descriptor")
        if descriptor.num_active_loras or descriptor.num_ubatches != 1:
            raise NotImplementedError("Draft Window capture does not support LoRA or microbatch variants")
        num_reqs = descriptor.num_reqs
        if type(num_reqs) is not int or num_reqs < 1 or type(consumer_ordinal) is not int or consumer_ordinal < 0:
            raise ValueError("capture request count and consumer ordinal must be valid integers")
        if type(descriptor.num_tokens) is not int or descriptor.num_tokens < 1 or descriptor.num_tokens % num_reqs:
            raise ValueError("capture token count must be positive and divisible by request count")
        query_width = descriptor.num_tokens // num_reqs
        if descriptor.uniform_token_count is not None and descriptor.uniform_token_count != query_width:
            raise ValueError("capture query width differs from the actual descriptor")
        key = num_reqs, descriptor.num_tokens
        banks = self._carriers.get(key, ())
        if banks and banks[0].descriptor != descriptor:
            raise RuntimeError("Draft Window capture descriptor geometry is ambiguous")
        if consumer_ordinal != len(banks):
            raise RuntimeError("capture consumer identity is duplicate or outside serial ordinal order")
        if num_reqs > self.table.shape[0]:
            raise RuntimeError("capture request bucket exceeds source capacity")
        table = self.table
        if table.ndim != 2 or table.dtype != torch.int32:
            raise ValueError("Draft capture requires a two-dimensional int32 table")
        bucket = DraftWindowGraphBucket(
            descriptor,
            consumer_ordinal,
            table.new_zeros((num_reqs, table.shape[1])),
            torch.zeros(num_reqs, dtype=torch.int32),
            self,
        )
        self._carriers[key] = (*banks, bucket)
        return bucket

    def require_reusable(self) -> None:
        """Keep every proposal on its established completion stream."""
        if self.quarantined:
            raise RuntimeError("Draft Window bank is quarantined after uncertain submission")
        source = self.table
        if source.device.type != "cpu":
            stream = int(torch.npu.current_stream().npu_stream)
            previous = getattr(self, "_owner_stream", stream)
            if stream != previous:
                raise RuntimeError("Draft Window bank reuse changed stream without a completion dependency")
            self._owner_stream = stream

    def bind_current(self, descriptor: BatchExecutionDescriptor) -> None:
        """Publish the prepared bank directly to the original Graph resolver."""
        bank = self._active_bank
        if self.quarantined or bank is None or bank.descriptor != descriptor:
            raise RuntimeError("Graph replay does not own the current unconsumed preparation")
        params = {"actual_seq_lengths_kv": bank.cpu_window_exact.tolist()}
        self._pending.append((bank, params))
        self._active_bank = None

    def get(self, _provider: Any) -> tuple[dict[str, Any], ...]:
        """Resolve one payload per actual layer consumer, in capture order."""
        return tuple(params for _, params in self._pending)

    def validate_captured_graphs(self, graphs: Mapping[BatchExecutionDescriptor, Any]) -> None:
        """Check actual FIA tasks once, after the manager has stored its graphs.

        The original manager owns these graphs by descriptor. Replays use its
        task resolver and the already registered bank/consumer, without a
        second graph-to-route inventory.
        """
        import torch_npu

        from vllm_ascend.attention.attention_v1 import FIAParamProvider

        for descriptor, graph in graphs.items():
            if descriptor.cg_mode.name != "FULL":
                continue
            if not graph.tasks:
                raise RuntimeError("Draft Window Graph captured no attention consumer")
            banks = self.capture_banks_for(num_reqs=descriptor.num_reqs, num_tokens=descriptor.num_tokens)
            q = descriptor.num_tokens // descriptor.num_reqs
            queries = list(range(q, descriptor.num_tokens + 1, q))
            for task in graph.tasks:
                if (
                    not isinstance(task.provider, FIAParamProvider)
                    or not task.provider.is_draft_model
                    or task.provider.is_non_pa
                    or task.provider.layer_name not in self.group.layers
                    or task.operation != torch_npu.npu_fused_infer_attention_score.out
                ):
                    raise RuntimeError("Draft Window Graph requires its supported Draft FIA consumer")
                # Task kwargs are capture-owned copies; stable banks supply their addresses.
                if (
                    task.kwargs["actual_seq_lengths"] != queries
                    or task.kwargs["block_table"].data_ptr() != banks[task.provider_index].window_block_table.data_ptr()
                ):
                    raise RuntimeError("Draft Window capture changed its fixed query or bank address")

    def finish_update(self) -> None:
        """Drop staged Host bindings after UPDATE; actual streams still own reuse."""
        self._pending.clear()
