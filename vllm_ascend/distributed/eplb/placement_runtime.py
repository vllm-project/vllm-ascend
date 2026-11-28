# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""Collective deployment export I/O, outside the hot path."""

from collections.abc import Callable
from typing import Any, TypeVar

import torch.distributed as dist
from vllm.utils.gpu_sync_debug import gpu_sync_allowed

from vllm_ascend.distributed.eplb.placement import EplbPlacement, save_placement

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


def export_committed_placement(
    state: Any,
    metadata: dict[str, Any],
    path: str,
    workload: str,
    group: Any,
    model_state: Any | None = None,
    *,
    draft_metadata: dict[str, Any] | None = None,
    draft_model_state: Any | None = None,
) -> dict[str, Any]:
    """Called collectively between forwards, with no new work being published."""

    def validate_request() -> Any:
        selected_model_state = model_state
        if not isinstance(path, str) or not path:
            raise ValueError("Export path must be a non-empty string")
        if not isinstance(workload, str) or not workload.strip():
            raise ValueError("Describe the representative workload used to generate the placement")
        if (draft_metadata is None) != (draft_model_state is None):
            raise ValueError("EPLB draft placement export requires both draft metadata and draft model state")
        if selected_model_state is None:
            if len(state.model_states) != 1:
                raise ValueError("EPLB deployment export requires a target MoE model state")
            selected_model_state = next(iter(state.model_states.values()))
        return selected_model_state

    model_state = collective_call(validate_request, group, "export request")
    state.finish_pending_commits()

    # CPU transfer is intentional: export is an explicit offline operation.
    def snapshot() -> EplbPlacement:
        provenance = {
            "workload": workload,
            "policy": state.parallel_config.eplb_config.policy,
            "window_size": state.parallel_config.eplb_config.window_size,
            "step_interval": state.parallel_config.eplb_config.step_interval,
            "load_collection_phase": getattr(state, "load_collection_phase", "all"),
        }
        with gpu_sync_allowed():
            mapping = model_state.physical_to_logical_map.cpu().tolist()
        draft_placement = None
        if draft_metadata is not None:
            assert draft_model_state is not None
            with gpu_sync_allowed():
                draft_mapping = draft_model_state.physical_to_logical_map.cpu().tolist()
            draft_placement = EplbPlacement(
                draft_metadata,
                tuple(tuple(layer) for layer in draft_mapping),
                {**provenance, "role": "draft"},
            )
            draft_placement.validate()
        placement = EplbPlacement(
            metadata,
            tuple(tuple(layer) for layer in mapping),
            provenance,
            draft=draft_placement,
        )
        placement.validate(metadata, draft_metadata)
        return placement

    placement = collective_call(snapshot, group, "committed snapshot")
    digest = verify_placement_consensus(placement, group)
    collective_call(
        lambda: save_placement(path, placement) if group.rank() == 0 else None,
        group,
        "placement writing",
    )
    result = {
        "path": path,
        "sha256": digest,
        "num_moe_layers": len(placement.physical_to_logical),
    }
    if placement.draft is not None:
        result["num_draft_moe_layers"] = len(placement.draft.physical_to_logical)
    return result
