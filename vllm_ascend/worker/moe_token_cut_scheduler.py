# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from vllm_ascend.worker.moe_token_cut_handoff import get_deferred_request_ids

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import CachedRequestData, SchedulerOutput

    from vllm_ascend.worker.moe_token_cut_deferred import MoETokenDeferredStore


@dataclass(frozen=True, slots=True)
class MoETokenSchedulerProjection:
    """Immutable request-row projection for a deferred decode batch."""

    original_request_ids: tuple[str, ...]
    kept_request_ids: tuple[str, ...]
    deferred_request_ids: tuple[str, ...]
    kept_indices: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class MoETokenSchedulerRollback:
    """One validated scheduler-state rollback for a deferred decode row."""

    request_id: str
    request: Any
    num_tokens: int


def build_scheduler_projection(
    request_ids: tuple[str, ...],
    deferred_request_ids: tuple[str, ...],
) -> MoETokenSchedulerProjection:
    """Build a stable-order projection without mutating Scheduler state."""
    if not request_ids or not all(request_ids):
        raise ValueError("scheduled request IDs must be non-empty")
    if len(set(request_ids)) != len(request_ids):
        raise ValueError("scheduled request IDs must be unique")
    if not all(deferred_request_ids):
        raise ValueError("deferred request IDs must be non-empty")
    if len(set(deferred_request_ids)) != len(deferred_request_ids):
        raise ValueError("deferred request IDs must be unique")

    scheduled = set(request_ids)
    unknown = set(deferred_request_ids).difference(scheduled)
    if unknown:
        raise ValueError(
            f"deferred requests are not present in the scheduled batch: {sorted(unknown)!r}"
        )
    deferred = set(deferred_request_ids)
    kept_indices = tuple(
        index for index, request_id in enumerate(request_ids) if request_id not in deferred
    )
    return MoETokenSchedulerProjection(
        original_request_ids=request_ids,
        kept_request_ids=tuple(request_ids[index] for index in kept_indices),
        deferred_request_ids=tuple(
            request_id for request_id in request_ids if request_id in deferred
        ),
        kept_indices=kept_indices,
    )


def build_projection_from_store(
    request_ids: tuple[str, ...],
    store: MoETokenDeferredStore,
    layer_idx: int,
) -> MoETokenSchedulerProjection:
    return build_scheduler_projection(request_ids, store.request_ids(layer_idx))


def prepare_scheduler_rollbacks(
    scheduler: Any,
    scheduler_output: Any,
    model_runner_output: Any,
) -> tuple[MoETokenSchedulerRollback, ...]:
    """Validate a deferred handoff without mutating Scheduler state."""
    deferred_request_ids = get_deferred_request_ids(model_runner_output)
    if not deferred_request_ids:
        return ()

    scheduler_config = getattr(scheduler, "scheduler_config", None)
    if scheduler_config is None:
        scheduler_config = getattr(
            getattr(scheduler, "vllm_config", None),
            "scheduler_config",
            None,
        )
    if getattr(scheduler_config, "async_scheduling", False):
        raise ValueError("token-cut deferred handoff does not support async scheduling")
    if scheduler_output.scheduled_new_reqs:
        raise ValueError("token-cut deferred handoff does not support new requests")

    scheduled_spec = scheduler_output.scheduled_spec_decode_tokens
    unexpected_spec = set(deferred_request_ids).intersection(scheduled_spec)
    if unexpected_spec:
        raise ValueError(
            "token-cut deferred handoff does not support speculative tokens: "
            f"{sorted(unexpected_spec)!r}"
        )
    scheduled_encoder_inputs = getattr(
        scheduler_output,
        "scheduled_encoder_inputs",
        {},
    )
    unexpected_encoder_inputs = set(deferred_request_ids).intersection(
        scheduled_encoder_inputs
    )
    if unexpected_encoder_inputs:
        raise ValueError(
            "token-cut deferred handoff does not support encoder inputs: "
            f"{sorted(unexpected_encoder_inputs)!r}"
        )
    if getattr(model_runner_output, "kv_connector_output", None) is not None:
        raise ValueError("token-cut deferred handoff does not support KV connectors")
    if getattr(model_runner_output, "ec_connector_output", None) is not None:
        raise ValueError("token-cut deferred handoff does not support EC connectors")
    if getattr(model_runner_output, "routed_experts", None) is not None:
        raise ValueError("token-cut deferred handoff does not support routed-expert output")

    prompt_logprobs = getattr(model_runner_output, "prompt_logprobs_dict", {})
    num_nans = getattr(model_runner_output, "num_nans_in_logits", None) or {}
    sampled_token_ids = model_runner_output.sampled_token_ids
    pooler_output = getattr(model_runner_output, "pooler_output", None)
    rollbacks: list[MoETokenSchedulerRollback] = []
    for request_id in deferred_request_ids:
        num_tokens = scheduler_output.num_scheduled_tokens.get(request_id)
        if num_tokens != 1:
            raise ValueError(
                "token-cut deferred handoff requires one scheduled token for "
                f"{request_id!r}; got {num_tokens!r}"
            )
        request = scheduler.requests.get(request_id)
        if request is None or request.is_finished():
            raise ValueError(
                f"deferred request {request_id!r} is absent or already finished"
            )
        if getattr(request, "num_output_placeholders", 0):
            raise ValueError(
                f"deferred request {request_id!r} has async output placeholders"
            )
        if getattr(request, "num_in_flight_tokens", num_tokens) < num_tokens:
            raise ValueError(
                f"deferred request {request_id!r} has insufficient in-flight tokens"
            )
        if request.num_computed_tokens < num_tokens:
            raise ValueError(
                f"deferred request {request_id!r} has insufficient computed tokens"
            )
        num_prompt_tokens = getattr(request, "num_prompt_tokens", None)
        if (
            num_prompt_tokens is not None
            and request.num_computed_tokens <= num_prompt_tokens
        ):
            raise ValueError(
                f"deferred request {request_id!r} is not in decode state"
            )

        request_index = model_runner_output.req_id_to_index.get(request_id)
        if request_index is None:
            raise ValueError(f"deferred request {request_id!r} has no output row")
        if request_index < 0 or request_index >= len(sampled_token_ids):
            raise ValueError(
                f"deferred request {request_id!r} has invalid output row "
                f"{request_index}"
            )
        if (
            request_index >= len(model_runner_output.req_ids)
            or model_runner_output.req_ids[request_index] != request_id
        ):
            raise ValueError(
                f"deferred request {request_id!r} has an inconsistent output mapping"
            )
        if sampled_token_ids[request_index]:
            raise ValueError(
                f"deferred request {request_id!r} still has sampled tokens"
            )
        if pooler_output:
            if request_index >= len(pooler_output):
                raise ValueError(
                    f"deferred request {request_id!r} has no pooling-output row"
                )
            if pooler_output[request_index] is not None:
                raise ValueError(
                    f"deferred request {request_id!r} has a pooling output"
                )
        if prompt_logprobs.get(request_id) is not None:
            raise ValueError(
                f"deferred request {request_id!r} has prompt logprobs"
            )
        if request_id in num_nans:
            raise ValueError(
                f"deferred request {request_id!r} has logits diagnostics"
            )
        rollbacks.append(
            MoETokenSchedulerRollback(request_id, request, num_tokens)
        )
    return tuple(rollbacks)


def apply_scheduler_rollbacks(
    rollbacks: tuple[MoETokenSchedulerRollback, ...],
) -> None:
    for rollback in rollbacks:
        rollback.request.num_computed_tokens -= rollback.num_tokens


def restore_scheduler_rollbacks(
    rollbacks: tuple[MoETokenSchedulerRollback, ...],
) -> None:
    for rollback in rollbacks:
        rollback.request.num_computed_tokens += rollback.num_tokens


def project_scheduler_output(
    scheduler_output: SchedulerOutput,
    projection: MoETokenSchedulerProjection,
) -> SchedulerOutput:
    """Return a filtered SchedulerOutput while leaving the source untouched.

    This helper intentionally rejects new requests and speculative tokens. The
    runtime configuration already excludes those modes, and silently projecting
    either would corrupt request/KV ownership.
    """
    if scheduler_output.scheduled_new_reqs:
        raise ValueError("token-cut scheduler projection does not support new requests")
    if scheduler_output.scheduled_spec_decode_tokens:
        raise ValueError("token-cut scheduler projection does not support speculative tokens")

    cached = scheduler_output.scheduled_cached_reqs
    if tuple(cached.req_ids) != projection.original_request_ids:
        raise ValueError("cached request order does not match the token-cut projection")
    missing_counts = set(projection.original_request_ids).difference(
        scheduler_output.num_scheduled_tokens
    )
    if missing_counts:
        raise ValueError(
            f"scheduled token counts are missing requests: {sorted(missing_counts)!r}"
        )
    non_decode_counts = {
        request_id: scheduler_output.num_scheduled_tokens[request_id]
        for request_id in projection.original_request_ids
        if scheduler_output.num_scheduled_tokens[request_id] != 1
    }
    if non_decode_counts:
        raise ValueError(
            "token-cut scheduler projection requires one scheduled token per request; "
            f"got {non_decode_counts!r}"
        )

    projected_cached = _project_cached_requests(cached, projection)
    kept = set(projection.kept_request_ids)
    num_scheduled_tokens = {
        request_id: scheduler_output.num_scheduled_tokens[request_id]
        for request_id in projection.kept_request_ids
    }
    return replace(
        scheduler_output,
        scheduled_cached_reqs=projected_cached,
        num_scheduled_tokens=num_scheduled_tokens,
        total_num_scheduled_tokens=sum(num_scheduled_tokens.values()),
        scheduled_encoder_inputs={
            request_id: value
            for request_id, value in scheduler_output.scheduled_encoder_inputs.items()
            if request_id in kept
        },
        num_invalid_spec_tokens=(
            None
            if scheduler_output.num_invalid_spec_tokens is None
            else {
                request_id: value
                for request_id, value in scheduler_output.num_invalid_spec_tokens.items()
                if request_id in kept
            }
        ),
    )


def _project_cached_requests(
    cached: CachedRequestData,
    projection: MoETokenSchedulerProjection,
) -> CachedRequestData:
    row_count = len(projection.original_request_ids)

    def project_rows(values: list, field_name: str, *, allow_empty: bool = False) -> list:
        if allow_empty and not values:
            return []
        if len(values) != row_count:
            raise ValueError(
                f"cached request field {field_name!r} has {len(values)} rows; "
                f"expected {row_count}"
            )
        return [values[index] for index in projection.kept_indices]

    kept = set(projection.kept_request_ids)
    return replace(
        cached,
        req_ids=list(projection.kept_request_ids),
        resumed_req_ids=cached.resumed_req_ids.intersection(kept),
        new_token_ids=project_rows(
            cached.new_token_ids,
            "new_token_ids",
            allow_empty=True,
        ),
        all_token_ids={
            request_id: value
            for request_id, value in cached.all_token_ids.items()
            if request_id in kept
        },
        new_block_ids=project_rows(cached.new_block_ids, "new_block_ids"),
        num_computed_tokens=project_rows(
            cached.num_computed_tokens,
            "num_computed_tokens",
        ),
        num_output_tokens=project_rows(
            cached.num_output_tokens,
            "num_output_tokens",
        ),
    )
