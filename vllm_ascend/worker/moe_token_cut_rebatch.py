# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")

"""Decode-only row projection helpers for cross-layer MoE token refill."""

from __future__ import annotations

from copy import copy
from dataclasses import dataclass, field
from typing import Any

import torch

_REQUEST_TENSOR_FIELDS = (
    "seq_lens",
    "seq_lens_cpu",
    "block_tables",
)


@dataclass
class NativeAttentionResult:
    """Row-aligned rebatch outputs kept alive until the next model step.

    The source metadata identifies the KV layout. Different KV groups cannot
    reuse each other's block/slot addresses merely because row counts match.
    """

    request_ids: tuple[str, ...]
    source_metadata: Any
    slots: torch.Tensor
    seq_lens: torch.Tensor
    block_tables: torch.Tensor
    _cpu_seq_lens: torch.Tensor | None = field(default=None, init=False, repr=False)

    @property
    def cpu_seq_lens(self):
        if self._cpu_seq_lens is None:
            self._cpu_seq_lens = self.seq_lens.to(device="cpu")
        return self._cpu_seq_lens

    def project(self, metadata: Any, request_ids: tuple[str, ...]) -> Any:
        if request_ids != self.request_ids:
            raise RuntimeError("native Attention output request order no longer matches the model rows")
        source_blocks = self.source_metadata.block_tables
        target_blocks = metadata.block_tables
        if (source_blocks.data_ptr(), source_blocks.shape, source_blocks.device) != (
            target_blocks.data_ptr(),
            target_blocks.shape,
            target_blocks.device,
        ):
            raise RuntimeError("native rebatch Attention propagation requires a shared KV block-table layout")
        n = len(request_ids)
        if (
            self.slots.shape != (n,)
            or self.seq_lens.shape != (n,)
            or self.block_tables.shape != (n, target_blocks.shape[1])
        ):
            raise RuntimeError("native rebatch Attention outputs do not match the compact row layout")
        projected = copy(metadata)
        for field_name, value in (
            ("slot_mapping", self.slots),
            ("seq_lens", self.seq_lens),
            ("block_tables", self.block_tables),
        ):
            original = getattr(metadata, field_name)
            if field_name == "seq_lens" and original.device.type == "cpu":
                value = self.cpu_seq_lens
            setattr(projected, field_name, value.to(device=original.device, dtype=original.dtype))
        # v0.27 eager FIA still consumes host lengths. Materialize the native
        # result once here; never reconstruct these lengths from old requests.
        # Full graph must instead bind device lengths or task-update them.
        lengths_cpu = self.cpu_seq_lens
        if isinstance(getattr(metadata, "seq_lens_cpu", None), torch.Tensor):
            projected.seq_lens_cpu = lengths_cpu.to(dtype=metadata.seq_lens_cpu.dtype)
        if getattr(metadata, "seq_lens_list", None) is not None:
            projected.seq_lens_list = lengths_cpu.tolist()
        _set_decode_query_layout(projected, n)
        return projected


def _set_decode_query_layout(projected: Any, row_count: int) -> None:
    for field_name in ("query_start_loc", "query_start_loc_cpu"):
        value = getattr(projected, field_name, None)
        if isinstance(value, torch.Tensor):
            setattr(projected, field_name, torch.arange(row_count + 1, dtype=value.dtype, device=value.device))
    projected.actual_seq_lengths_q = list(range(1, row_count + 1))
    for field_name in ("num_actual_tokens", "num_decode_tokens", "num_decodes"):
        if hasattr(projected, field_name):
            setattr(projected, field_name, row_count)
    if hasattr(projected, "num_prefills"):
        projected.num_prefills = 0
    if hasattr(projected, "max_query_len"):
        projected.max_query_len = 1 if row_count else 0


def project_decode_attention_metadata(metadata: Any, row_indices: torch.Tensor) -> Any:
    """Return a shallow metadata copy containing only selected decode rows.

    Token-cut is restricted to ordinary one-token decode. Consequently request
    rows and token rows have the same index, and the cumulative query lengths
    are always ``[1, ..., n]``. Tensor payloads not related to batch rows (for
    example the attention mask) continue to be shared with the original
    metadata object.
    """
    projected = copy(metadata)
    device_indices = row_indices
    cpu_indices = row_indices.to(device="cpu")

    for field_name in _REQUEST_TENSOR_FIELDS:
        value = getattr(metadata, field_name, None)
        if isinstance(value, torch.Tensor):
            indices = device_indices.to(device=value.device)
            setattr(projected, field_name, value.index_select(0, indices))

    slot_mapping = getattr(metadata, "slot_mapping", None)
    if isinstance(slot_mapping, torch.Tensor):
        projected.slot_mapping = slot_mapping.index_select(0, device_indices.to(device=slot_mapping.device))

    row_count = int(row_indices.numel())
    seq_lens_list = getattr(metadata, "seq_lens_list", None)
    if seq_lens_list is not None:
        projected.seq_lens_list = [seq_lens_list[index] for index in cpu_indices.tolist()]
    _set_decode_query_layout(projected, row_count)
    return projected


def project_layer_attention_metadata(
    per_layer_metadata: Any,
    layer_name: str,
    row_indices: torch.Tensor,
) -> Any:
    """Project one layer entry while preserving the outer metadata mapping."""
    if not isinstance(per_layer_metadata, dict):
        raise RuntimeError("MoE token-cut cross-layer refill requires per-layer metadata")
    if layer_name not in per_layer_metadata:
        raise RuntimeError(f"missing attention metadata for layer {layer_name!r}")
    projected = dict(per_layer_metadata)
    projected[layer_name] = project_decode_attention_metadata(per_layer_metadata[layer_name], row_indices)
    return projected
