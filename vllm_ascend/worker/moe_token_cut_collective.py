# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist


def build_local_deferred_mask(
    request_ids: tuple[str, ...],
    local_request_ids: tuple[str, ...],
    cut_idx: torch.Tensor,
) -> torch.Tensor:
    """Map one MC2 shard's cut decisions into global Worker row order.

    Empty local request IDs are MC2 padding rows. ``cut_idx`` uses the kernel
    convention 1=kept and 0=deferred.
    """
    if not request_ids or not all(isinstance(request_id, str) and request_id for request_id in request_ids):
        raise ValueError("worker request IDs must be non-empty strings")
    if len(set(request_ids)) != len(request_ids):
        raise ValueError("worker request IDs must be unique")
    if cut_idx.ndim != 1 or cut_idx.shape[0] != len(local_request_ids):
        raise ValueError("cut_idx must have one row per local request ID")

    non_padding_ids = tuple(request_id for request_id in local_request_ids if request_id)
    if len(set(non_padding_ids)) != len(non_padding_ids):
        raise ValueError("local non-padding request IDs must be unique")
    request_id_to_index = {request_id: index for index, request_id in enumerate(request_ids)}
    unknown = set(non_padding_ids).difference(request_id_to_index)
    if unknown:
        raise ValueError(f"local token-cut shard contains unknown requests: {sorted(unknown)!r}")

    local_mask = torch.zeros(
        len(request_ids),
        dtype=torch.bool,
        device=cut_idx.device,
    )
    if not non_padding_ids:
        return local_mask

    source_rows = torch.tensor(
        [index for index, request_id in enumerate(local_request_ids) if request_id],
        dtype=torch.long,
        device=cut_idx.device,
    )
    destination_rows = torch.tensor(
        [request_id_to_index[request_id] for request_id in non_padding_ids],
        dtype=torch.long,
        device=cut_idx.device,
    )
    local_mask.index_copy_(
        0,
        destination_rows,
        cut_idx.index_select(0, source_rows).eq(0),
    )
    return local_mask


def all_reduce_deferred_mask(
    local_mask: torch.Tensor,
    process_group: Any,
) -> torch.Tensor:
    """Return the union of deferred rows across one MC2 process group."""
    if local_mask.ndim != 1 or local_mask.dtype != torch.bool:
        raise ValueError("local deferred mask must be a one-dimensional bool tensor")
    global_mask = local_mask.to(dtype=torch.int32)
    dist.all_reduce(
        global_mask,
        op=dist.ReduceOp.MAX,
        group=process_group,
    )
    return global_mask.to(dtype=torch.bool)


def agree_token_cut_execution(
    *,
    requested: bool,
    compatible: bool,
    device: torch.device,
    process_group: Any,
) -> bool:
    """Agree on entering the collective kernel, never on DP request identity.

    Every EP participant, including idle DP ranks, must call this once per MC2
    layer. Unsupported inputs on any rank disable this invocation everywhere.
    An eligible rank can cut independently: protected ranks enter communication
    with real routing counts and a no-delete threshold, then keep their original
    dispatch input. Never give those ranks enable_cut=0. The host read is
    required to keep the collective call order consistent in eager mode.
    """
    flags = torch.tensor([not compatible, requested], dtype=torch.int32, device=device)
    dist.all_reduce(flags, op=dist.ReduceOp.MAX, group=process_group)
    any_incompatible, any_requested = flags.to(device="cpu").tolist()
    return not any_incompatible and bool(any_requested)


def deferred_request_ids_from_mask(
    request_ids: tuple[str, ...],
    deferred_mask: torch.Tensor,
) -> tuple[str, ...]:
    """Synchronize the final mask at the sampling-bookkeeping boundary."""
    if deferred_mask.ndim != 1 or deferred_mask.shape[0] != len(request_ids):
        raise ValueError("deferred mask must have one entry per worker request")
    if deferred_mask.dtype != torch.bool:
        raise ValueError("deferred mask must have bool dtype")
    mask = deferred_mask.detach().to(device="cpu").tolist()
    return tuple(request_id for request_id, is_deferred in zip(request_ids, mask) if is_deferred)
