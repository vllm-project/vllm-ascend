# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

from __future__ import annotations

from dataclasses import dataclass

import torch

from vllm_ascend.worker.moe_token_cut_state import MoETokenCutLayerCapture


@dataclass(frozen=True, slots=True)
class MoETokenRestorePlan:
    """Validated reference plan for one rank's compact/defer operation.

    Indices are kept on CPU intentionally. Building the plan is an explicit
    diagnostic synchronization point in capture mode, not part of the serving
    hot path. A production AscendC restore backend must match these semantics
    without copying decision tensors to the host.
    """

    layer_idx: int
    request_ids: tuple[str, ...]
    kept_indices: torch.Tensor
    deferred_indices: torch.Tensor
    padding_indices: torch.Tensor
    actual_bs: int

    @property
    def kept_request_ids(self) -> tuple[str, ...]:
        return tuple(
            self.request_ids[index]
            for index in self.kept_indices.tolist()
            if self.request_ids[index]
        )

    @property
    def deferred_request_ids(self) -> tuple[str, ...]:
        return tuple(
            self.request_ids[index]
            for index in self.deferred_indices.tolist()
            if self.request_ids[index]
        )


@dataclass(frozen=True, slots=True)
class MoETokenDeferredBatch:
    """Original pre-dispatch rows parked by one sparse MoE layer."""

    layer_idx: int
    request_ids: tuple[str, ...]
    hidden_states: torch.Tensor
    residuals: torch.Tensor
    topk_ids: torch.Tensor
    topk_weights: torch.Tensor


@dataclass(frozen=True, slots=True)
class MoETokenGlobalRestorePlan:
    """Host-side merge of the contiguous token shards owned by EP ranks."""

    layer_idx: int
    request_ids: tuple[str, ...]
    kept_request_ids: tuple[str, ...]
    deferred_request_ids: tuple[str, ...]
    row_locations: dict[str, tuple[int, int]]


def build_restore_plan(capture: MoETokenCutLayerCapture) -> MoETokenRestorePlan:
    """Validate a captured device decision and build its reference row map."""
    batch_size = len(capture.request_ids)
    cut_idx = capture.cut_idx.detach().to(device="cpu", dtype=torch.int64)
    prefix_index = capture.prefix_index[:batch_size].detach().to(
        device="cpu", dtype=torch.int64
    )
    active_mask = capture.active_mask.detach().to(device="cpu", dtype=torch.bool)
    actual_bs = int(capture.actual_bs.detach().to(device="cpu").reshape(-1)[0])

    if cut_idx.shape != (batch_size,):
        raise ValueError(
            f"layer {capture.layer_idx}: cut_idx must have {batch_size} entries"
        )
    if torch.any((cut_idx != 0) & (cut_idx != 1)):
        raise ValueError(f"layer {capture.layer_idx}: cut_idx must be binary")

    expected_prefix = torch.cumsum(cut_idx, dim=0)
    if not torch.equal(prefix_index, expected_prefix):
        raise ValueError(
            f"layer {capture.layer_idx}: prefix_index does not match cut_idx"
        )
    expected_actual_bs = int(expected_prefix[-1]) if batch_size else 0
    if actual_bs != expected_actual_bs:
        raise ValueError(
            f"layer {capture.layer_idx}: actual_bs={actual_bs} does not match "
            f"the {expected_actual_bs} kept rows"
        )
    expected_active_mask = torch.arange(batch_size) < actual_bs
    if active_mask.shape != (batch_size,) or not torch.equal(
        active_mask, expected_active_mask
    ):
        raise ValueError(
            f"layer {capture.layer_idx}: active_mask must mark the compacted prefix"
        )

    request_mask = torch.tensor(
        [bool(request_id) for request_id in capture.request_ids],
        dtype=torch.bool,
    )
    if torch.any((cut_idx != 0) & ~request_mask):
        raise ValueError(
            f"layer {capture.layer_idx}: padding rows cannot be marked as kept"
        )
    kept_indices = torch.nonzero((cut_idx != 0) & request_mask, as_tuple=False).flatten()
    deferred_indices = torch.nonzero((cut_idx == 0) & request_mask, as_tuple=False).flatten()
    padding_indices = torch.nonzero(~request_mask, as_tuple=False).flatten()
    return MoETokenRestorePlan(
        layer_idx=capture.layer_idx,
        request_ids=capture.request_ids,
        kept_indices=kept_indices,
        deferred_indices=deferred_indices,
        padding_indices=padding_indices,
        actual_bs=actual_bs,
    )


def materialize_deferred_batch(
    capture: MoETokenCutLayerCapture,
    plan: MoETokenRestorePlan,
) -> MoETokenDeferredBatch:
    """Gather the rows that a production backend must persist for restore."""
    _check_matching_layer(capture, plan)
    indices = plan.deferred_indices.to(device=capture.hidden_states.device)
    return MoETokenDeferredBatch(
        layer_idx=plan.layer_idx,
        request_ids=tuple(plan.request_ids[index] for index in plan.deferred_indices.tolist()),
        hidden_states=capture.hidden_states.index_select(0, indices),
        residuals=capture.residuals.index_select(0, indices),
        topk_ids=capture.topk_ids.index_select(0, indices),
        topk_weights=capture.topk_weights.index_select(0, indices),
    )


def merge_rank_restore_plans(
    plans: tuple[MoETokenRestorePlan, ...],
) -> MoETokenGlobalRestorePlan:
    """Merge rank-local plans in MC2 shard order without integer request IDs."""
    if not plans:
        raise ValueError("at least one rank-local restore plan is required")
    layer_idx = plans[0].layer_idx
    if any(plan.layer_idx != layer_idx for plan in plans):
        raise ValueError("rank-local restore plans must belong to the same layer")

    request_ids: list[str] = []
    kept_request_ids: list[str] = []
    deferred_request_ids: list[str] = []
    row_locations: dict[str, tuple[int, int]] = {}
    for rank_index, plan in enumerate(plans):
        kept_rows = set(plan.kept_indices.tolist())
        deferred_rows = set(plan.deferred_indices.tolist())
        for row_index, request_id in enumerate(plan.request_ids):
            if not request_id:
                continue
            if request_id in row_locations:
                raise ValueError(f"duplicate request ID in restore plans: {request_id!r}")
            if row_index not in kept_rows and row_index not in deferred_rows:
                raise ValueError(f"request {request_id!r} is not classified by its restore plan")
            request_ids.append(request_id)
            row_locations[request_id] = (rank_index, row_index)
            if row_index in kept_rows:
                kept_request_ids.append(request_id)
            else:
                deferred_request_ids.append(request_id)

    return MoETokenGlobalRestorePlan(
        layer_idx=layer_idx,
        request_ids=tuple(request_ids),
        kept_request_ids=tuple(kept_request_ids),
        deferred_request_ids=tuple(deferred_request_ids),
        row_locations=row_locations,
    )


def restore_original_order(
    compacted_active_rows: torch.Tensor,
    deferred_rows: torch.Tensor,
    plan: MoETokenRestorePlan,
) -> torch.Tensor:
    """Reference scatter used to verify a future device restore kernel."""
    if compacted_active_rows.ndim < 1 or deferred_rows.ndim != compacted_active_rows.ndim:
        raise ValueError("active and deferred tensors must have the same positive rank")
    if compacted_active_rows.shape[1:] != deferred_rows.shape[1:]:
        raise ValueError("active and deferred tensors must have matching trailing dimensions")
    if compacted_active_rows.dtype != deferred_rows.dtype:
        raise ValueError("active and deferred tensors must have matching dtypes")
    if compacted_active_rows.shape[0] != plan.actual_bs:
        raise ValueError("active row count does not match restore plan")
    if deferred_rows.shape[0] != plan.deferred_indices.numel():
        raise ValueError("deferred row count does not match restore plan")

    restored = torch.zeros(
        (len(plan.request_ids), *compacted_active_rows.shape[1:]),
        dtype=compacted_active_rows.dtype,
        device=compacted_active_rows.device,
    )
    kept_indices = plan.kept_indices.to(device=restored.device)
    deferred_indices = plan.deferred_indices.to(device=restored.device)
    restored.index_copy_(0, kept_indices, compacted_active_rows)
    restored.index_copy_(0, deferred_indices, deferred_rows.to(restored.device))
    return restored


def _check_matching_layer(
    capture: MoETokenCutLayerCapture,
    plan: MoETokenRestorePlan,
) -> None:
    if capture.layer_idx != plan.layer_idx or capture.request_ids != plan.request_ids:
        raise ValueError("capture and restore plan identify different layer inputs")
