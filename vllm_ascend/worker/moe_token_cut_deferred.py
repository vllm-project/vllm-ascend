# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from __future__ import annotations

from dataclasses import dataclass

import torch

from vllm_ascend.worker.moe_token_cut_restore import MoETokenDeferredBatch


@dataclass(slots=True)
class _DeferredLayerStorage:
    request_ids: list[str]
    hidden_states: torch.Tensor
    residuals: torch.Tensor
    topk_ids: torch.Tensor
    topk_weights: torch.Tensor
    count: int = 0


class MoETokenDeferredStore:
    """Fixed-capacity device storage for rows deferred across forwards.

    Request identity stays on the host while tensor payloads remain on their
    original device. The store is deliberately independent from Scheduler
    mutation so its enqueue/pop semantics can be validated before enabling the
    production rebatch path.
    """

    def __init__(self, *, num_layers: int, capacity_per_layer: int):
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if capacity_per_layer <= 0:
            raise ValueError("capacity_per_layer must be positive")
        self.num_layers = num_layers
        self.capacity_per_layer = capacity_per_layer
        self._layers: dict[int, _DeferredLayerStorage] = {}

    def count(self, layer_idx: int) -> int:
        self._validate_layer(layer_idx)
        storage = self._layers.get(layer_idx)
        return 0 if storage is None else storage.count

    def request_ids(self, layer_idx: int) -> tuple[str, ...]:
        self._validate_layer(layer_idx)
        storage = self._layers.get(layer_idx)
        if storage is None:
            return ()
        return tuple(storage.request_ids[: storage.count])

    def enqueue(self, batch: MoETokenDeferredBatch) -> None:
        self._validate_layer(batch.layer_idx)
        row_count = len(batch.request_ids)
        self._validate_batch(batch, row_count)
        if row_count == 0:
            return

        storage = self._layers.get(batch.layer_idx)
        if storage is None:
            storage = self._allocate(batch)
            self._layers[batch.layer_idx] = storage
        else:
            self._validate_storage_compatibility(storage, batch)

        duplicate_ids = set(storage.request_ids[: storage.count]).intersection(
            batch.request_ids
        )
        if duplicate_ids:
            duplicate = min(duplicate_ids)
            raise ValueError(
                f"request {duplicate!r} is already deferred at layer {batch.layer_idx}"
            )
        end = storage.count + row_count
        if end > self.capacity_per_layer:
            raise OverflowError(
                f"layer {batch.layer_idx} deferred capacity exceeded: "
                f"{end} > {self.capacity_per_layer}"
            )

        destination = slice(storage.count, end)
        storage.hidden_states[destination].copy_(batch.hidden_states)
        storage.residuals[destination].copy_(batch.residuals)
        storage.topk_ids[destination].copy_(batch.topk_ids)
        storage.topk_weights[destination].copy_(batch.topk_weights)
        storage.request_ids[storage.count:end] = batch.request_ids
        storage.count = end

    def peek(self, layer_idx: int, max_rows: int | None = None) -> MoETokenDeferredBatch:
        self._validate_layer(layer_idx)
        storage = self._require_storage(layer_idx)
        row_count = self._resolve_row_count(storage.count, max_rows)
        return self._slice_batch(layer_idx, storage, row_count, clone=False)

    def pop(self, layer_idx: int, max_rows: int | None = None) -> MoETokenDeferredBatch:
        self._validate_layer(layer_idx)
        storage = self._require_storage(layer_idx)
        row_count = self._resolve_row_count(storage.count, max_rows)
        batch = self._slice_batch(layer_idx, storage, row_count, clone=True)
        remaining = storage.count - row_count
        if remaining:
            source = torch.arange(
                row_count,
                storage.count,
                dtype=torch.long,
                device=storage.hidden_states.device,
            )
            storage.hidden_states[:remaining].copy_(storage.hidden_states.index_select(0, source))
            storage.residuals[:remaining].copy_(storage.residuals.index_select(0, source))
            storage.topk_ids[:remaining].copy_(storage.topk_ids.index_select(0, source))
            storage.topk_weights[:remaining].copy_(storage.topk_weights.index_select(0, source))
        storage.request_ids[:remaining] = storage.request_ids[row_count : storage.count]
        storage.request_ids[remaining : storage.count] = [""] * row_count
        storage.count = remaining
        return batch

    def discard_requests(self, request_ids: set[str]) -> int:
        """Remove cancelled or finished requests from every layer."""
        removed = 0
        if not request_ids:
            return removed
        for storage in self._layers.values():
            keep_rows = [
                index
                for index, request_id in enumerate(storage.request_ids[: storage.count])
                if request_id not in request_ids
            ]
            if len(keep_rows) == storage.count:
                continue
            old_count = storage.count
            new_count = len(keep_rows)
            if new_count:
                indices = torch.tensor(
                    keep_rows,
                    dtype=torch.long,
                    device=storage.hidden_states.device,
                )
                storage.hidden_states[:new_count].copy_(
                    storage.hidden_states.index_select(0, indices)
                )
                storage.residuals[:new_count].copy_(storage.residuals.index_select(0, indices))
                storage.topk_ids[:new_count].copy_(storage.topk_ids.index_select(0, indices))
                storage.topk_weights[:new_count].copy_(
                    storage.topk_weights.index_select(0, indices)
                )
            kept_ids = [storage.request_ids[index] for index in keep_rows]
            storage.request_ids[:new_count] = kept_ids
            storage.request_ids[new_count:old_count] = [""] * (old_count - new_count)
            storage.count = new_count
            removed += old_count - new_count
        return removed

    def _allocate(self, batch: MoETokenDeferredBatch) -> _DeferredLayerStorage:
        hidden_shape = (self.capacity_per_layer, *batch.hidden_states.shape[1:])
        topk_shape = (self.capacity_per_layer, *batch.topk_ids.shape[1:])
        return _DeferredLayerStorage(
            request_ids=[""] * self.capacity_per_layer,
            hidden_states=torch.empty(
                hidden_shape,
                dtype=batch.hidden_states.dtype,
                device=batch.hidden_states.device,
            ),
            residuals=torch.empty(
                hidden_shape,
                dtype=batch.residuals.dtype,
                device=batch.residuals.device,
            ),
            topk_ids=torch.empty(
                topk_shape,
                dtype=batch.topk_ids.dtype,
                device=batch.topk_ids.device,
            ),
            topk_weights=torch.empty(
                topk_shape,
                dtype=batch.topk_weights.dtype,
                device=batch.topk_weights.device,
            ),
        )

    @staticmethod
    def _validate_batch(batch: MoETokenDeferredBatch, row_count: int) -> None:
        if not all(batch.request_ids) or len(set(batch.request_ids)) != row_count:
            raise ValueError("deferred request IDs must be non-empty and unique")
        tensors = (
            batch.hidden_states,
            batch.residuals,
            batch.topk_ids,
            batch.topk_weights,
        )
        if any(tensor.ndim < 1 or tensor.shape[0] != row_count for tensor in tensors):
            raise ValueError("deferred tensors must have one row per request ID")
        if batch.hidden_states.shape != batch.residuals.shape:
            raise ValueError("deferred hidden_states and residuals must have matching shapes")
        if batch.topk_ids.shape != batch.topk_weights.shape:
            raise ValueError("deferred top-k IDs and weights must have matching shapes")

    @staticmethod
    def _validate_storage_compatibility(
        storage: _DeferredLayerStorage,
        batch: MoETokenDeferredBatch,
    ) -> None:
        pairs = (
            (storage.hidden_states, batch.hidden_states),
            (storage.residuals, batch.residuals),
            (storage.topk_ids, batch.topk_ids),
            (storage.topk_weights, batch.topk_weights),
        )
        if any(
            stored.shape[1:] != incoming.shape[1:]
            or stored.dtype != incoming.dtype
            or stored.device != incoming.device
            for stored, incoming in pairs
        ):
            raise ValueError("deferred tensor shape, dtype, or device changed")

    def _validate_layer(self, layer_idx: int) -> None:
        if not 0 <= layer_idx < self.num_layers:
            raise ValueError(f"layer index {layer_idx} is outside [0, {self.num_layers})")

    def _require_storage(self, layer_idx: int) -> _DeferredLayerStorage:
        storage = self._layers.get(layer_idx)
        if storage is None or storage.count == 0:
            raise IndexError(f"layer {layer_idx} has no deferred rows")
        return storage

    @staticmethod
    def _resolve_row_count(count: int, max_rows: int | None) -> int:
        if max_rows is None:
            return count
        if max_rows <= 0:
            raise ValueError("max_rows must be positive")
        return min(count, max_rows)

    @staticmethod
    def _slice_batch(
        layer_idx: int,
        storage: _DeferredLayerStorage,
        row_count: int,
        *,
        clone: bool,
    ) -> MoETokenDeferredBatch:
        def select(tensor: torch.Tensor) -> torch.Tensor:
            view = tensor[:row_count]
            return view.clone() if clone else view

        return MoETokenDeferredBatch(
            layer_idx=layer_idx,
            request_ids=tuple(storage.request_ids[:row_count]),
            hidden_states=select(storage.hidden_states),
            residuals=select(storage.residuals),
            topk_ids=select(storage.topk_ids),
            topk_weights=select(storage.topk_weights),
        )
