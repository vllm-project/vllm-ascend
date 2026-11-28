# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""Version-1 EPLB placement snapshot loading and load-time expert slot binding."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, TypeVar

import torch.distributed as dist

from vllm_ascend.distributed.eplb.placement import (
    PLACEMENT_SNAPSHOT_VERSION,
    validate_placement_snapshot,
)

_T = TypeVar("_T")


@dataclass(frozen=True)
class RolePlacement:
    num_logical_experts: int
    num_redundant_experts: int
    physical_to_logical_map: tuple[tuple[int, ...], ...]

    @property
    def num_physical_experts(self) -> int:
        return self.num_logical_experts + self.num_redundant_experts

    @property
    def num_layers(self) -> int:
        return len(self.physical_to_logical_map)


@dataclass(frozen=True)
class PlacementSnapshot:
    version: int
    ep_size: int
    target: RolePlacement
    draft: RolePlacement | None = None

    def digest(self) -> str:
        payload: dict[str, Any] = {
            "version": self.version,
            "ep_size": self.ep_size,
            "target": {
                "num_logical_experts": self.target.num_logical_experts,
                "num_redundant_experts": self.target.num_redundant_experts,
                "physical_to_logical_map": [list(row) for row in self.target.physical_to_logical_map],
            },
        }
        if self.draft is not None:
            payload["draft"] = {
                "num_logical_experts": self.draft.num_logical_experts,
                "num_redundant_experts": self.draft.num_redundant_experts,
                "physical_to_logical_map": [list(row) for row in self.draft.physical_to_logical_map],
            }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class RuntimeRoleSpec:
    num_logical_experts: int
    num_redundant_experts: int
    num_layers: int


def build_model_role_spec(moe_model: Any | None) -> RuntimeRoleSpec | None:
    if moe_model is None:
        return None
    moe_layers = list(getattr(moe_model, "moe_layers", ()))
    if not moe_layers:
        return None
    num_logical = int(moe_model.num_logical_experts)
    num_physical = int(moe_model.num_physical_experts)
    num_redundant = num_physical - num_logical
    if num_logical < 1 or num_redundant < 0:
        raise ValueError("Invalid runtime MoE expert dimensions")
    return RuntimeRoleSpec(
        num_logical_experts=num_logical,
        num_redundant_experts=num_redundant,
        num_layers=len(moe_layers),
    )


def _to_role_placement(entry: dict[str, Any]) -> RolePlacement:
    return RolePlacement(
        num_logical_experts=entry["num_logical_experts"],
        num_redundant_experts=entry["num_redundant_experts"],
        physical_to_logical_map=tuple(tuple(row) for row in entry["physical_to_logical_map"]),
    )


def parse_placement_snapshot(snapshot: Any) -> PlacementSnapshot:
    validate_placement_snapshot(snapshot)
    draft = _to_role_placement(snapshot["draft"]) if "draft" in snapshot else None
    return PlacementSnapshot(
        version=PLACEMENT_SNAPSHOT_VERSION,
        ep_size=snapshot["ep_size"],
        target=_to_role_placement(snapshot["target"]),
        draft=draft,
    )


def load_placement_snapshot(path: str | os.PathLike[str]) -> PlacementSnapshot:
    with open(path, encoding="utf-8") as source:
        return parse_placement_snapshot(json.load(source))


def validate_role_against_spec(placement: RolePlacement, role: str, spec: RuntimeRoleSpec) -> None:
    if placement.num_logical_experts != spec.num_logical_experts:
        raise ValueError(
            f"EPLB {role} num_logical_experts mismatch: "
            f"file={placement.num_logical_experts}, runtime={spec.num_logical_experts}"
        )
    if placement.num_redundant_experts != spec.num_redundant_experts:
        raise ValueError(
            f"EPLB {role} num_redundant_experts mismatch: "
            f"file={placement.num_redundant_experts}, runtime={spec.num_redundant_experts}"
        )
    if placement.num_layers != spec.num_layers:
        raise ValueError(f"EPLB {role} layer count mismatch: file={placement.num_layers}, runtime={spec.num_layers}")


def slice_pp_target_role(
    snapshot: PlacementSnapshot,
    ep_size: int,
    pp_rank: int,
    stage_specs: Sequence[RuntimeRoleSpec | None],
) -> RolePlacement | None:
    if snapshot.ep_size != ep_size:
        raise ValueError(f"EPLB placement ep_size mismatch: file={snapshot.ep_size}, runtime={ep_size}")
    present = [spec for spec in stage_specs if spec is not None]
    if not present:
        raise ValueError("EPLB deployments require at least one PP stage with target MoE layers")
    for spec in present:
        if snapshot.target.num_logical_experts != spec.num_logical_experts:
            raise ValueError(
                f"EPLB target num_logical_experts mismatch: "
                f"file={snapshot.target.num_logical_experts}, runtime={spec.num_logical_experts}"
            )
        if snapshot.target.num_redundant_experts != spec.num_redundant_experts:
            raise ValueError(
                f"EPLB target num_redundant_experts mismatch: "
                f"file={snapshot.target.num_redundant_experts}, runtime={spec.num_redundant_experts}"
            )
    total_layers = sum(spec.num_layers for spec in present)
    if snapshot.target.num_layers != total_layers:
        raise ValueError(f"EPLB target layer count mismatch: file={snapshot.target.num_layers}, runtime={total_layers}")
    local_spec = stage_specs[pp_rank]
    if local_spec is None:
        return None
    offset = sum(spec.num_layers for spec in stage_specs[:pp_rank] if spec is not None)
    return RolePlacement(
        num_logical_experts=snapshot.target.num_logical_experts,
        num_redundant_experts=snapshot.target.num_redundant_experts,
        physical_to_logical_map=snapshot.target.physical_to_logical_map[offset : offset + local_spec.num_layers],
    )


def collective_call(operation: Callable[[], _T], group: Any, description: str) -> _T:
    result = None
    error = None
    try:
        result = operation()
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    if group.size() <= 1:
        if error is not None:
            raise ValueError(f"EPLB {description} failed: rank 0: {error}")
        return result  # type: ignore[return-value]
    errors: list[Any] = [None] * group.size()
    dist.all_gather_object(errors, error, group=group)
    failures = [f"rank {r}: {msg}" for r, msg in enumerate(errors) if msg is not None]
    if failures:
        raise ValueError(f"EPLB {description} failed: {'; '.join(failures)}")
    return result  # type: ignore[return-value]


def read_and_slice_target_placement(
    path: str,
    local_spec_fn: Callable[[], RuntimeRoleSpec | None],
    ep_group: Any,
    pp_group: Any | None = None,
    pp_rank: int = 0,
) -> tuple[PlacementSnapshot, RolePlacement | None]:
    """Read snapshot on global rank 0, broadcast across PP+EP, and slice target layers."""
    local_spec = collective_call(local_spec_fn, ep_group, "static target spec")
    if pp_group is None or pp_group.size() <= 1:
        payload: list[str | None] = [None]
        if ep_group.rank() == 0:
            try:
                with open(path, encoding="utf-8") as source:
                    payload[0] = source.read()
            except OSError as error:
                payload[0] = f"ERROR: {type(error).__name__}: {error}"
        if ep_group.size() > 1:
            dist.broadcast_object_list(payload, src=dist.get_global_rank(ep_group, 0), group=ep_group)
        agreed = payload[0]
        if agreed is None or agreed.startswith("ERROR:"):
            raise FileNotFoundError(f"EPLB placement {path!r} could not be read on EPLB rank 0: {agreed}")

        def decode_single() -> tuple[PlacementSnapshot, RolePlacement | None]:
            if local_spec is None:
                raise ValueError("EPLB deployments require target MoE layers")
            snapshot = parse_placement_snapshot(json.loads(agreed))
            if snapshot.ep_size != ep_group.size():
                raise ValueError(f"EPLB placement ep_size mismatch: file={snapshot.ep_size}, runtime={ep_group.size()}")
            validate_role_against_spec(snapshot.target, "target", local_spec)
            return snapshot, snapshot.target

        return collective_call(decode_single, ep_group, "placement loading")

    packet: list[Any] = [None, None]
    if ep_group.rank() == 0:
        pp_text: list[str | None] = [None]
        if pp_rank == 0:
            try:
                with open(path, encoding="utf-8") as source:
                    pp_text[0] = source.read()
            except OSError as error:
                pp_text[0] = f"ERROR: {type(error).__name__}: {error}"
        dist.broadcast_object_list(pp_text, src=dist.get_global_rank(pp_group, 0), group=pp_group)
        stage_specs: list[Any] = [None] * pp_group.size()
        dist.all_gather_object(stage_specs, local_spec, group=pp_group)
        packet = [pp_text[0], stage_specs]
    if ep_group.size() > 1:
        dist.broadcast_object_list(packet, src=dist.get_global_rank(ep_group, 0), group=ep_group)
    agreed_text, stage_specs = packet
    if agreed_text is None or agreed_text.startswith("ERROR:"):
        raise FileNotFoundError(f"EPLB placement {path!r} could not be read on global rank 0: {agreed_text}")

    def decode_pp() -> tuple[PlacementSnapshot, RolePlacement | None]:
        snapshot = parse_placement_snapshot(json.loads(agreed_text))
        role = slice_pp_target_role(snapshot, ep_group.size(), pp_rank, stage_specs)
        return snapshot, role

    return collective_call(decode_pp, ep_group, "PP placement loading")


def bind_role_placement_to_model(moe_model: Any, role_placement: RolePlacement, ep_size: int, ep_rank: int) -> None:
    moe_layers = list(moe_model.moe_layers)
    if len(moe_layers) != role_placement.num_layers:
        raise ValueError(f"MoE layer count ({len(moe_layers)}) does not match placement ({role_placement.num_layers})")
    slots_per_rank = role_placement.num_physical_experts // ep_size
    start = ep_rank * slots_per_rank
    moe_model._static_role_placement = role_placement
    for layer, row in zip(moe_layers, role_placement.physical_to_logical_map):
        slot_slice = row[start : start + slots_per_rank]
        layer._static_logical_to_local_slot = {logical_id: slot for slot, logical_id in enumerate(slot_slice)}


def local_expert_ids_for_role(role_placement: RolePlacement, ep_size: int, ep_rank: int) -> set[int] | None:
    if ep_size <= 1:
        return None
    slots_per_rank = role_placement.num_physical_experts // ep_size
    start = ep_rank * slots_per_rank
    return {eid for row in role_placement.physical_to_logical_map for eid in row[start : start + slots_per_rank]}


ACTIVE_STATIC_CONTROLLER: Any = None
ACTIVE_STATIC_ROLE_PLACEMENT: Any = None


def _init_static_context_vars() -> tuple[Any, Any]:
    from contextvars import ContextVar

    return (
        ContextVar("vllm_ascend_static_eplb_controller", default=None),
        ContextVar("vllm_ascend_static_role_placement", default=None),
    )


ACTIVE_STATIC_CONTROLLER, ACTIVE_STATIC_ROLE_PLACEMENT = _init_static_context_vars()
