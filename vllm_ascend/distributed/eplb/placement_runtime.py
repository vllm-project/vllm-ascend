# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""Collective deployment loading and one-time installation, outside the hot path."""

import json
from collections.abc import Callable, Mapping, Sequence
from typing import Any, TypeVar

import numpy as np
import torch
import torch.distributed as dist
from vllm.distributed.eplb import eplb_state as upstream_state
from vllm.utils.gpu_sync_debug import gpu_sync_allowed

from vllm_ascend.distributed.eplb.explicit_transfer import stage_explicit_layer_transfer
from vllm_ascend.distributed.eplb.placement import (
    EplbPlacement,
    plan_placement_sources,
)

_T = TypeVar("_T")


def collective_call(operation: Callable[[], _T], group: Any, description: str) -> _T:
    """Report local validation/I/O failures before any peer starts migration."""
    result = None
    error = None
    try:
        result = operation()
    except Exception as exception:
        error = f"{type(exception).__name__}: {exception}"
    errors: list[Any] = [None] * group.size()
    dist.all_gather_object(errors, error, group=group)
    failures = [f"rank {rank}: {message}" for rank, message in enumerate(errors) if message is not None]
    if failures:
        raise ValueError(f"EPLB {description} failed: {'; '.join(failures)}")
    return result  # type: ignore[return-value]


def verify_placement_consensus(placement: EplbPlacement, group: Any) -> str:
    digest = collective_call(placement.digest, group, "placement validation")
    digests: list[Any] = [None] * group.size()
    dist.all_gather_object(digests, digest, group=group)
    if len(set(digests)) != 1:
        raise ValueError("EPLB placement differs across EP ranks")
    return digest


def read_placement_collectively(
    path: str,
    metadata: dict[str, Any],
    group: Any,
    *,
    draft_metadata: dict[str, Any] | None = None,
) -> EplbPlacement:
    """Read the deployment on the EPLB leader and broadcast it to every rank.

    Only rank 0 touches the filesystem. Other nodes do not need a local copy,
    and a stale copy on some other node cannot make the ranks disagree. Every rank still validates the broadcast document
    against its own runtime metadata.
    """
    payload: list[str | None] = [None]
    if group.rank() == 0:
        try:
            with open(path, encoding="utf-8") as source:
                payload[0] = source.read()
        except OSError as error:
            payload[0] = f"ERROR: {type(error).__name__}: {error}"
    dist.broadcast_object_list(payload, src=dist.get_global_rank(group, 0), group=group)
    agreed = payload[0]
    if agreed is None or agreed.startswith("ERROR:"):
        raise FileNotFoundError(f"EPLB placement {path!r} could not be read on EPLB rank 0: {agreed}")

    def decode() -> EplbPlacement:
        placement = EplbPlacement.from_dict(json.loads(agreed))
        placement.validate(metadata, draft_metadata)
        return placement

    placement = collective_call(decode, group, "placement loading")
    verify_placement_consensus(placement, group)
    return placement


def _synchronize(stream: Any) -> None:
    with gpu_sync_allowed():
        stream.synchronize()


def _commit_layer_mapping(model_state: Any, target: torch.Tensor, layer_idx: int) -> None:
    logical_map, replicas = upstream_state.compute_logical_maps(
        target.unsqueeze(0),
        model_state.model.num_logical_experts,
    )
    destination = model_state.logical_to_physical_map[layer_idx]
    padding = destination.shape[-1] - logical_map.shape[-1]
    if padding < 0:
        raise ValueError("Static EPLB placement exceeds the allocated replica capacity")
    logical_map = torch.nn.functional.pad(logical_map[0], (0, padding), value=-1)
    model_state.physical_to_logical_map[layer_idx].copy_(target)
    destination.copy_(logical_map)
    model_state.logical_replica_count[layer_idx].copy_(replicas[0])


def _move_from_buffer_for_expert_views(
    expert_weights: Sequence[Any],
    expert_weight_buffers: Sequence[Any],
    transfer_metadata: Any,
    new_indices: np.ndarray,
    ep_rank: int,
) -> None:
    """Install buffered weights for either tensor or Tensor List views.

    Upstream ``move_from_buffer`` installs rows by indexing and copying them
    (``w[dst].copy_(b[dst])``), so a per-expert list of tensors can be passed
    through unchanged. Stacking the rows first would materialize a batched
    tensor and fails for Ascend internal formats such as FRACTAL_NZ, whose
    aclnn Stack operator is not implemented.
    """
    upstream_state.move_from_buffer(
        expert_weights=expert_weights,
        expert_weights_buffers=expert_weight_buffers,
        transfer_metadata=transfer_metadata,
        new_indices=new_indices,
        ep_rank=ep_rank,
    )


def install_placement(
    state: Any,
    placement: EplbPlacement,
    group: Any,
    stream: Any,
    *,
    model_states: Mapping[str, Any] | None = None,
) -> None:
    """Install a specified deployment once, before profiling or graph capture.

    Every rank stages from the old live weights. No rank overwrites a source
    until all reads have completed. Metadata and lookup are published only
    after the live compute weights have been installed.
    """

    def prepare() -> dict[str, tuple[Any, torch.Tensor, torch.Tensor, list[Any]]]:
        if state.async_worker is not None:
            raise RuntimeError("Static EPLB installation must precede the background worker")
        if model_states is None:
            if len(state.model_states) != 1 or placement.draft is not None:
                raise ValueError("Static EPLB with one model requires exactly one model state")
            resolved_states = {"target": next(iter(state.model_states.values()))}
        else:
            if "target" not in model_states:
                raise ValueError("Static EPLB placement is missing its target model state")
            if placement.draft is not None and "draft" not in model_states:
                raise ValueError("Static EPLB placement is missing its draft model state")
            resolved_states = dict(model_states)

        requested_placements = {"target": placement}
        if placement.draft is not None:
            requested_placements["draft"] = placement.draft

        prepared: dict[str, tuple[Any, torch.Tensor, torch.Tensor, list[Any]]] = {}
        for role, requested in requested_placements.items():
            model_state = resolved_states[role]
            with gpu_sync_allowed():
                old = model_state.physical_to_logical_map.cpu().clone()
            target = torch.tensor(
                requested.physical_to_logical,
                dtype=old.dtype,
                device="cpu",
            )
            if target.shape != old.shape:
                raise ValueError(f"Static EPLB cannot change the number of allocated expert slots for {role}")
            plans = [
                plan_placement_sources(
                    tuple(before.tolist()),
                    tuple(after.tolist()),
                    group.size(),
                )
                for before, after in zip(old, target)
            ]
            prepared[role] = (model_state, old, target, plans)
        return prepared

    prepared = collective_call(prepare, group, "installation preflight")
    # Validate the complete plan on every rank before backend registration.
    verify_placement_consensus(placement, group)
    for role, (model_state, old, _, _) in prepared.items():
        record = placement if role == "target" else placement.draft
        assert record is not None
        verify_placement_consensus(
            EplbPlacement(
                record.metadata,
                tuple(tuple(layer) for layer in old.tolist()),
                {},
            ),
            group,
        )

    changed_layers = {
        role: torch.nonzero((old != target).any(dim=1)).flatten().tolist()
        for role, (_, old, target, _) in prepared.items()
    }
    if not any(changed_layers.values()):
        return
    state.prepare_movement_resources()
    _synchronize(stream)
    for role, (model_state, old, target, plans) in prepared.items():
        if not changed_layers[role]:
            continue
        communicator = model_state.communicator
        communicator.set_stream(stream)
        for layer_idx in changed_layers[role]:
            source_ranks, source_slots = plans[layer_idx]
            metadata = stage_explicit_layer_transfer(
                old[layer_idx],
                target[layer_idx],
                np.asarray(source_ranks),
                np.asarray(source_slots),
                model_state.model.expert_weights[layer_idx],
                model_state.expert_buffer,
                state.placement_device_group(),
                communicator,
                stream,
                layer_idx,
            )
            _synchronize(stream)
            dist.barrier(group=group)
            _move_from_buffer_for_expert_views(
                expert_weights=model_state.model.expert_weights[layer_idx],
                expert_weight_buffers=model_state.expert_buffer,
                transfer_metadata=metadata,
                new_indices=target[layer_idx].numpy(),
                ep_rank=group.rank(),
            )
            _commit_layer_mapping(model_state, target[layer_idx], layer_idx)
            state.refresh_placement_layer(model_state, layer_idx)
            _synchronize(stream)
            dist.barrier(group=group)
    state.release_movement_resources()
