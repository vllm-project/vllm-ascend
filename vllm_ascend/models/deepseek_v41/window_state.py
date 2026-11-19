# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Window metadata and one runner-owned chronological staging workspace."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from vllm.v1.kv_cache_interface import KVCacheConfig

from vllm_ascend.core.kv_cache_interface import AscendKVCacheConfig
from vllm_ascend.models.deepseek_v41.cache_config import MLA_HEAD_DIM, WINDOW, RequestWindowSpec, unwrap


@dataclass(frozen=True)
class RequestWindowMetadata:
    """Per-request ring IDs and the corresponding chronological workspace views."""

    state_ids: torch.Tensor
    row_starts: torch.Tensor
    lengths: torch.Tensor
    block_table: torch.Tensor
    rows_used: int
    max_seq_len: int
    capacity: int


class RequestWindowMetadataBuilder:
    """Own persistent decode buffers; update contents without replacing storage."""

    def __init__(self, seq_lens: torch.Tensor, capacity: int):
        self.capacity = capacity
        self.lengths = torch.empty_like(seq_lens)
        rows = torch.arange(seq_lens.numel(), dtype=torch.int32, device=seq_lens.device)
        self.row_starts = rows * WINDOW
        self.block_table = rows.view(-1, 1)

    def build(self, common, result) -> RequestWindowMetadata:
        n = result.num_reqs
        if result.max_query_len <= 1:
            # Graph replay requires the same buffers across Decode steps.
            assert common.block_table_tensor.shape[1] == 1
            assert n <= self.lengths.numel() and n * WINDOW <= self.capacity
            lengths = self.lengths[:n]
            from vllm_ascend.models.deepseek_v41.request_state import state_swa_lengths

            lengths.copy_(state_swa_lengths(result.seq_lens, common.query_start_loc))
            return RequestWindowMetadata(
                common.block_table_tensor[:n, 0],
                self.row_starts[:n],
                lengths,
                self.block_table[:n],
                n * WINDOW,
                WINDOW,
                self.capacity,
            )
        qcpu = common.query_start_loc_cpu
        seqcpu = getattr(common, "seq_lens_cpu", None)
        if seqcpu is None:
            seqcpu = common._seq_lens_cpu
        assert qcpu.device.type == seqcpu.device.type == "cpu", "Use CPU scheduling metadata without device sync"
        queries = qcpu[: n + 1].tolist()
        sequences = seqcpu[:n].tolist()
        bases, lengths, cursor = [], [], 0
        for i in range(n):
            query = queries[i + 1] - queries[i]
            old = min(max(0, sequences[i] - query), WINDOW - 1)
            length = old + query if 0 < query <= sequences[i] else 0
            bases.append(cursor)
            lengths.append(length)
            cursor += ((length + WINDOW - 1) // WINDOW) * WINDOW
        assert cursor <= self.capacity
        max_pages = max(1, max(((length + WINDOW - 1) // WINDOW for length in lengths), default=0))
        table = [
            [base // WINDOW + j if j < (length + WINDOW - 1) // WINDOW else 0 for j in range(max_pages)]
            for base, length in zip(bases, lengths)
        ]
        device = result.seq_lens.device
        return RequestWindowMetadata(
            common.block_table_tensor[:n, 0].contiguous(),
            torch.tensor(bases, dtype=torch.int32, device=device),
            torch.tensor(lengths, dtype=torch.int32, device=device),
            torch.tensor(table, dtype=torch.int32, device=device),
            cursor,
            max(lengths, default=0),
            self.capacity,
        )


def prepare_request_window_metadata(builder, common, result):
    window_builder = builder.request_window_builder
    if window_builder is None:
        return result
    result.request_state = window_builder.build(common, result)
    return result


@dataclass(frozen=True)
class WindowStagingWorkspace:
    """Shared across sequential attention layers, never across runner instances.

    Allocate before warmup/capture. Do not resize or release per request: graphs
    retain the address, and the runner/model teardown owns the final lifetime.
    Concurrent forwards on independent streams require separate workspaces and
    are outside this contract.
    """

    tensor: torch.Tensor

    def validate(self, ring: torch.Tensor, capacity: int) -> None:
        if (
            self.tensor.shape != (capacity // WINDOW, *ring.shape[1:])
            or self.tensor.device != ring.device
            or self.tensor.dtype != ring.dtype
        ):
            raise ValueError("Window staging layout changed; recreate the runner and its graphs")


def bind_window_staging_workspace(
    config: KVCacheConfig,
    layers: Mapping[str, Any],
    existing: WindowStagingWorkspace | None = None,
) -> WindowStagingWorkspace | None:
    """Bind the single planned staging allocation to this runner's SWA layers."""
    if not isinstance(config, AscendKVCacheConfig) or config.request_state_plan is None:
        return None
    windows = [
        layers[name]
        for group in config.kv_cache_groups
        if all(isinstance(spec, RequestWindowSpec) for spec in unwrap(group.kv_cache_spec).values())
        for name in group.layer_names
    ]
    # Bind from declared cache-group membership, never model-name suffixes.
    if not windows:
        raise ValueError("Request-state worker layout has no window layers")
    capacity = config.staging_rows
    if capacity <= 0 or capacity % WINDOW:
        raise ValueError("Window staging capacity must be a positive whole number of pages")
    rings = [layer.kv_cache for layer in windows]
    first = rings[0]
    if any(
        ring.shape[1:] != (WINDOW, 1, MLA_HEAD_DIM) or ring.dtype != first.dtype or ring.device != first.device
        for ring in rings
    ):
        raise ValueError("Window staging requires matching unpacked 128x1x512 cache layouts")
    if existing is None:
        existing = WindowStagingWorkspace(
            torch.empty(
                (capacity // WINDOW, WINDOW, 1, MLA_HEAD_DIM),
                dtype=first.dtype,
                device=first.device,
            )
        )
    else:
        existing.validate(first, capacity)
    for layer in windows:
        layer.request_state_workspace = existing
    return existing
