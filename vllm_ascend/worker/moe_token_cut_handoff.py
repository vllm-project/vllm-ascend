# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

DEFERRED_REQUEST_IDS_ATTR = "moe_token_cut_deferred_request_ids"


@dataclass(frozen=True, slots=True)
class WorkerDeferredPlan:
    """Stable Worker-side projection of EP-wide deferred requests."""

    request_ids: tuple[str, ...]
    deferred_request_ids: tuple[str, ...]
    deferred_indices: tuple[int, ...]
    discard_indices: tuple[int, ...]


def build_worker_deferred_plan(
    request_ids: tuple[str, ...],
    deferred_request_ids_by_rank: tuple[tuple[str, ...], ...],
    existing_discard_indices: tuple[int, ...] = (),
) -> WorkerDeferredPlan:
    """Merge EP-rank decisions and existing sampling discards.

    The returned request IDs follow the original model-runner row order, not
    collective rank order. This makes the result deterministic even if a
    collective implementation changes how rank payloads are represented.
    """
    if not request_ids or not all(
        isinstance(request_id, str) and request_id for request_id in request_ids
    ):
        raise ValueError("worker request IDs must be non-empty strings")
    if len(set(request_ids)) != len(request_ids):
        raise ValueError("worker request IDs must be unique")
    if not deferred_request_ids_by_rank:
        raise ValueError("at least one EP-rank deferred result is required")

    scheduled = set(request_ids)
    deferred: set[str] = set()
    for rank_index, rank_request_ids in enumerate(deferred_request_ids_by_rank):
        if not all(
            isinstance(request_id, str) and request_id
            for request_id in rank_request_ids
        ):
            raise ValueError(
                f"EP rank {rank_index} deferred request IDs must be non-empty strings"
            )
        if len(set(rank_request_ids)) != len(rank_request_ids):
            raise ValueError(
                f"EP rank {rank_index} deferred request IDs must be unique"
            )
        unknown = set(rank_request_ids).difference(scheduled)
        if unknown:
            raise ValueError(
                f"EP rank {rank_index} deferred unknown requests: {sorted(unknown)!r}"
            )
        deferred.update(rank_request_ids)

    row_count = len(request_ids)
    if len(set(existing_discard_indices)) != len(existing_discard_indices):
        raise ValueError("existing discard indices must be unique")
    invalid_indices = tuple(
        index
        for index in existing_discard_indices
        if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < row_count
    )
    if invalid_indices:
        raise ValueError(
            f"existing discard indices are outside the worker batch: {invalid_indices!r}"
        )

    deferred_indices = tuple(
        index for index, request_id in enumerate(request_ids) if request_id in deferred
    )
    discard_indices = tuple(
        sorted(set(existing_discard_indices).union(deferred_indices))
    )
    return WorkerDeferredPlan(
        request_ids=request_ids,
        deferred_request_ids=tuple(request_ids[index] for index in deferred_indices),
        deferred_indices=deferred_indices,
        discard_indices=discard_indices,
    )


def get_deferred_request_ids(model_runner_output: Any) -> tuple[str, ...]:
    """Read the token-cut handoff marker from a model-runner output."""
    request_ids = getattr(model_runner_output, DEFERRED_REQUEST_IDS_ATTR, ())
    if request_ids is None:
        return ()
    if not isinstance(request_ids, (tuple, list)):
        raise TypeError(
            f"{DEFERRED_REQUEST_IDS_ATTR} must be a tuple or list of request IDs"
        )
    request_ids = tuple(request_ids)
    if not all(isinstance(request_id, str) and request_id for request_id in request_ids):
        raise ValueError("deferred request IDs must be non-empty strings")
    if len(set(request_ids)) != len(request_ids):
        raise ValueError("deferred request IDs must be unique")
    return request_ids


def attach_deferred_request_ids(
    model_runner_output: Any,
    deferred_request_ids: tuple[str, ...],
) -> None:
    """Attach a validated Worker-to-Scheduler deferred-request marker.

    Deferred rows remain present in ``ModelRunnerOutput`` so the upstream
    scheduler can retire their in-flight accounting. Their sampled-token row
    must already be empty; this helper never discards a sampled token itself.
    """
    request_ids = tuple(deferred_request_ids)
    if not request_ids:
        setattr(model_runner_output, DEFERRED_REQUEST_IDS_ATTR, ())
        return
    if not all(isinstance(request_id, str) and request_id for request_id in request_ids):
        raise ValueError("deferred request IDs must be non-empty strings")
    if len(set(request_ids)) != len(request_ids):
        raise ValueError("deferred request IDs must be unique")

    output_request_ids = tuple(model_runner_output.req_ids)
    unknown = set(request_ids).difference(output_request_ids)
    if unknown:
        raise ValueError(
            "deferred requests are absent from ModelRunnerOutput: "
            f"{sorted(unknown)!r}"
        )

    sampled_token_ids = model_runner_output.sampled_token_ids
    for request_id in request_ids:
        try:
            request_index = model_runner_output.req_id_to_index[request_id]
        except KeyError as error:
            raise ValueError(
                f"deferred request {request_id!r} has no output row"
            ) from error
        if request_index < 0 or request_index >= len(sampled_token_ids):
            raise ValueError(
                f"deferred request {request_id!r} has invalid output row "
                f"{request_index}"
            )
        if sampled_token_ids[request_index]:
            raise ValueError(
                f"deferred request {request_id!r} still has sampled tokens"
            )

    setattr(model_runner_output, DEFERRED_REQUEST_IDS_ATTR, request_ids)
