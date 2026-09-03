# SPDX-License-Identifier: Apache-2.0
"""Atomic update-plan construction for policy-4 shared expert slots."""

from __future__ import annotations

import numpy as np
import torch
from vllm.logger import logger

_GLOBAL_SLOT_BATCH_PER_RANK = 4


def _global_maps_to_table(global_maps: np.ndarray, local_slots: int) -> np.ndarray:
    num_layers, num_ranks, num_experts = global_maps.shape
    table = np.full((num_layers, num_ranks, local_slots), -1, dtype=np.int64)
    for layer_id in range(num_layers):
        for rank_id in range(num_ranks):
            for expert_id in range(num_experts):
                local_slot = int(global_maps[layer_id, rank_id, expert_id])
                if local_slot < 0:
                    continue
                if local_slot >= local_slots or table[layer_id, rank_id, local_slot] >= 0:
                    raise ValueError("invalid global expert map for policy 4")
                table[layer_id, rank_id, local_slot] = expert_id
    return table


def _table_to_global_maps(table: np.ndarray, num_experts: int) -> np.ndarray:
    maps = np.full((table.shape[0], table.shape[1], num_experts), -1, dtype=np.int32)
    for layer_id in range(table.shape[0]):
        for rank_id in range(table.shape[1]):
            for local_slot, expert_id in enumerate(table[layer_id, rank_id]):
                if expert_id < 0:
                    continue
                if maps[layer_id, rank_id, expert_id] >= 0:
                    raise ValueError("a rank cannot hold two copies of one logical expert")
                maps[layer_id, rank_id, expert_id] = local_slot
    return maps


def _log2phy_map(global_map: np.ndarray, ep_rank: int, local_slots: int) -> list[int]:
    result = np.empty(global_map.shape[1], dtype=np.int32)
    for expert_id in range(global_map.shape[1]):
        holders = [
            int(global_map[rank_id, expert_id]) + rank_id * local_slots
            for rank_id in range(global_map.shape[0])
            if global_map[rank_id, expert_id] >= 0
        ]
        if not holders:
            raise ValueError(f"logical expert {expert_id} has no physical copy")
        result[expert_id] = holders[ep_rank % len(holders)]
    return result.tolist()


def _layer_updates(
    global_maps: np.ndarray,
    layer_ids: set[int],
    rank_id: int,
    local_slots: int,
) -> list[dict]:
    return [
        {
            "layer_id": layer_id,
            "rank_map": global_maps[layer_id, rank_id].tolist(),
            "log2phy_map": _log2phy_map(global_maps[layer_id], rank_id, local_slots),
        }
        for layer_id in sorted(layer_ids)
    ]


def do_global_slot_update(worker):
    global_maps_value = worker.shared_dict.get("expert_maps", None)
    workload_value = worker.shared_dict.get("moe_load", None)
    if global_maps_value is None or workload_value is None:
        return {"kind": "global_slots", "changed": False}
    global_maps = np.asarray(
        global_maps_value.numpy() if isinstance(global_maps_value, torch.Tensor) else global_maps_value,
        dtype=np.int32,
    )
    workload = np.asarray(
        workload_value.numpy() if isinstance(workload_value, torch.Tensor) else workload_value,
        dtype=np.float64,
    )
    local_slots = workload.shape[-1]
    old_table = _global_maps_to_table(global_maps, local_slots)
    changed, _, placement = worker.policy.rebalance_experts(old_table, workload)
    decision = getattr(worker.policy, "last_decision", None)
    if not changed:
        if worker.rank_id == 0 and decision is not None:
            logger.debug(
                "[eplb/global] Skip plan reason=%s gain=%.6f changed_slots=%s",
                decision.reason,
                decision.gain,
                decision.changed_slots,
            )
        return {"kind": "global_slots", "changed": False}

    new_table = np.asarray(placement, dtype=np.int64)
    num_layers, num_ranks, _ = new_table.shape
    num_experts = global_maps.shape[-1]
    base_slots = num_experts // num_ranks
    new_maps = _table_to_global_maps(new_table, num_experts)
    transitions_by_slot: list[list[dict]] = [[] for _ in range(local_slots - base_slots)]
    for dst_rank in range(num_ranks):
        for shared_slot in range(local_slots - base_slots):
            local_slot = base_slots + shared_slot
            old_owners = [
                (layer_id, int(old_table[layer_id, dst_rank, local_slot]))
                for layer_id in range(num_layers)
                if old_table[layer_id, dst_rank, local_slot] >= 0
            ]
            new_owners = [
                (layer_id, int(new_table[layer_id, dst_rank, local_slot]))
                for layer_id in range(num_layers)
                if new_table[layer_id, dst_rank, local_slot] >= 0
            ]
            if len(old_owners) > 1 or len(new_owners) > 1:
                raise ValueError("a global physical slot cannot be owned by multiple layers")
            old_owner = old_owners[0] if old_owners else None
            new_owner = new_owners[0] if new_owners else None
            if new_owner == old_owner:
                continue
            transitions_by_slot[shared_slot].append(
                {
                    "dst_rank": dst_rank,
                    "local_slot": local_slot,
                    "old_owner": old_owner,
                    "new_owner": new_owner,
                }
            )

    working_maps = global_maps.copy()
    steps: list[dict] = []
    for first_slot in range(0, len(transitions_by_slot), _GLOBAL_SLOT_BATCH_PER_RANK):
        transitions = [
            transition
            for slot_transitions in transitions_by_slot[first_slot : first_slot + _GLOBAL_SLOT_BATCH_PER_RANK]
            for transition in slot_transitions
        ]
        if not transitions:
            continue

        deactivate_layers: set[int] = set()
        for transition in transitions:
            old_owner = transition["old_owner"]
            if old_owner is None:
                continue
            layer_id, expert_id = old_owner
            dst_rank = transition["dst_rank"]
            local_slot = transition["local_slot"]
            if working_maps[layer_id, dst_rank, expert_id] != local_slot:
                raise ValueError("policy 4 old slot owner does not match the active map")
            working_maps[layer_id, dst_rank, expert_id] = -1
            deactivate_layers.add(layer_id)
        deactivated_maps = working_maps.copy()

        send: list[list[int]] = []
        recv: list[list[int]] = []
        activate_layers: set[int] = set()
        for transition in transitions:
            new_owner = transition["new_owner"]
            if new_owner is None:
                continue
            layer_id, expert_id = new_owner
            dst_rank = transition["dst_rank"]
            local_slot = transition["local_slot"]
            source_locations = np.argwhere(new_table[layer_id, :, :base_slots] == expert_id)
            if source_locations.shape[0] != 1:
                raise ValueError("policy 4 requires one immutable base source for every expert")
            src_rank, source_slot = (int(value) for value in source_locations[0])
            if worker.rank_id == src_rank:
                send.append([dst_rank, layer_id, source_slot])
            if worker.rank_id == dst_rank:
                recv.append([src_rank, layer_id, local_slot])
            working_maps[layer_id, dst_rank, expert_id] = local_slot
            activate_layers.add(layer_id)

        steps.append(
            {
                "deactivate": _layer_updates(
                    deactivated_maps,
                    deactivate_layers,
                    worker.rank_id,
                    local_slots,
                ),
                "send": send,
                "recv": recv,
                "activate": _layer_updates(working_maps, activate_layers, worker.rank_id, local_slots),
                "changed_slots": len(transitions),
            }
        )

    if not np.array_equal(working_maps, new_maps):
        raise ValueError("policy 4 staged update does not reach the planned map")

    changed_layers = int(np.count_nonzero(np.any(new_table != old_table, axis=(1, 2))))
    if worker.rank_id == 0:
        logger.info(
            "[eplb/global] Apply plan gain=%.6f changed_slots=%s changed_layers=%s steps=%s active_slots=%s",
            decision.gain if decision is not None else 0.0,
            decision.changed_slots if decision is not None else sum(step["changed_slots"] for step in steps),
            changed_layers,
            len(steps),
            int(np.count_nonzero(new_table[:, :, base_slots:] >= 0)),
        )
    return {
        "kind": "global_slots",
        "changed": True,
        "steps": steps,
        "global_maps": new_maps.tolist(),
    }


__all__ = ["do_global_slot_update"]
