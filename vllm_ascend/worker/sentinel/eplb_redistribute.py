# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Expert redistribution and weight reload for fault-tolerance scale-down.

After dead EP ranks are removed, renumber the routing tables so they only
refer to the live ranks, and copy the moved experts' weights from the
checkpoint into each rank's per-expert weight tensors.
"""

import os
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import torch
from safetensors import safe_open
from vllm.config import LoadConfig, ModelConfig
from vllm.distributed import get_ep_group
from vllm.logger import logger
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader

from vllm_ascend.worker.sentinel.eplb_quant_schemes import (
    _MOE_WEIGHT_NAMES,
    _WANTED_SUFFIXES,
    _find_reload_plans,
    _QuantReloadPlans,
    _WeightBuilder,
)

__all__ = [
    "densify_routing_tables",
    "reload_experts_from_disk",
]

# How many experts are reloaded together as one stacked tensor.
_EXPERTS_PER_STACK = 32

# Host threads that prepare one stack of expert tensors in parallel.
_GATHER_MAX_WORKERS = 8


def densify_routing_tables(
    moe_layers: list,
    orig_to_dense_rank: torch.Tensor,
    num_local_experts: int,
) -> None:
    """Renumber every layer's routing table so it refers to live EP ranks only.

    Each entry is a physical expert id; it becomes
    ``dense_rank * num_local_experts + slot``, where ``dense_rank`` is the
    0-based position of the hosting rank among the live ranks (dead ranks are
    skipped). Entries are updated in place, so captured graphs keep pointing
    at valid storage.
    """
    for layer in moe_layers:
        routing_table = getattr(getattr(layer, "eplb_state", None), "expert_replica_routing_table", None)
        if routing_table is None:
            continue
        ids = routing_table.to(torch.int64)
        if bool((ids < 0).any()):
            raise RuntimeError(
                "[FT] expert replica routing table references empty slots after "
                "redistribution; every logical expert must have a live replica."
            )
        orig_rank = torch.div(ids, num_local_experts, rounding_mode="floor")
        dense_rank = orig_to_dense_rank.to(device=ids.device, dtype=torch.int64)[orig_rank]
        if bool((dense_rank < 0).any()):
            raise RuntimeError(
                "[FT] expert replica routing table references dead EP ranks after "
                "redistribution; the placement did not vacate the dead ranks."
            )
        dense_ids = dense_rank * num_local_experts + ids % num_local_experts
        routing_table.copy_(dense_ids.to(routing_table.dtype))


def _reload_local_expert_slots(
    routed_layers: list,
    local_slots: dict[tuple[int, int], int],
    expert_weights: dict[tuple[int, int], dict[str, torch.Tensor]],
) -> int:
    """Store the dead ranks' experts, now redistributed onto this rank, in its local per-expert tensors.

    Returns the number of experts copied.
    """
    pairs_by_slot_shape: dict[tuple, list[tuple[int, int]]] = {}
    for (layer_idx, logical_id), slot in sorted(local_slots.items()):
        routed_layer = routed_layers[layer_idx]
        if getattr(routed_layer, "w13_bias", None) is not None or getattr(routed_layer, "w2_bias", None) is not None:
            raise NotImplementedError("[FT] scale_down weight reload does not support MoE expert bias yet.")
        reload_plans = _find_reload_plans(routed_layer)
        slot_shapes = []
        for weight_name in reload_plans.pipelines:
            slot_list = getattr(routed_layer, f"{weight_name}_list", None)
            tensor = slot_list[slot] if slot_list is not None else getattr(routed_layer, weight_name).data[slot]
            slot_shapes.append((weight_name, tensor.shape, tensor.dtype))
        pairs_by_slot_shape.setdefault((reload_plans, tuple(slot_shapes)), []).append((layer_idx, logical_id))

    reloaded = 0
    for (reload_plans, _slot_shapes), layer_experts in pairs_by_slot_shape.items():
        for start in range(0, len(layer_experts), _EXPERTS_PER_STACK):
            expert_stack = layer_experts[start : start + _EXPERTS_PER_STACK]
            reloaded += _reload_expert_stack(reload_plans, routed_layers, expert_stack, local_slots, expert_weights)
    return reloaded


def _convert_expert_weights(
    layer_expert: tuple[int, int],
    routed_layers: list,
    expert_weights: dict[tuple[int, int], dict[str, torch.Tensor]],
    pipelines: dict[str, _WeightBuilder],
) -> dict[str, torch.Tensor]:
    """Read one expert's checkpoint tensors and apply the same conversions as a normal weight load."""
    # EP-only deployments report tp_size=1, so the checkpoint is read unsharded.
    parallel_config = routed_layers[layer_expert[0]].moe_config.moe_parallel_config
    tp_rank, tp_size = parallel_config.tp_rank, parallel_config.tp_size
    tensors = expert_weights[layer_expert]
    return {
        weight_name: pipeline.processor(pipeline.source(tensors, tp_rank, tp_size))
        for weight_name, pipeline in pipelines.items()
    }


def _reload_expert_stack(
    reload_plans: _QuantReloadPlans,
    routed_layers: list,
    layer_experts: list[tuple[int, int]],
    local_slots: dict[tuple[int, int], int],
    expert_weights: dict[tuple[int, int], dict[str, torch.Tensor]],
) -> int:
    """Reload one stack of experts that use the same reload rules.

    Their tensors are prepared on the CPU pool, then stacked, NZ-cast and
    written into the per-expert tensors.
    """
    # Convert the pairs in parallel on the CPU (read-only shared state).
    convert_expert = partial(
        _convert_expert_weights,
        routed_layers=routed_layers,
        expert_weights=expert_weights,
        pipelines=reload_plans.pipelines,
    )
    with ThreadPoolExecutor(max_workers=min(len(layer_experts), _GATHER_MAX_WORKERS)) as pool:
        converted_experts = list(pool.map(convert_expert, layer_experts))

    # Device phase: one H2D copy and one NZ cast per weight tensor, converted
    # to the destination dtype (int8 weights, int64 bit scales, fp32).
    first_layer = routed_layers[layer_experts[0][0]]
    first_slot = local_slots[layer_experts[0]]
    device_tensors: dict[str, torch.Tensor] = {}
    for weight_name in reload_plans.pipelines:
        slot_list = getattr(first_layer, f"{weight_name}_list", None)
        if slot_list is not None:
            destination = slot_list[first_slot]
        else:
            destination = getattr(first_layer, weight_name).data[first_slot]
        stacked = torch.stack([expert[weight_name] for expert in converted_experts])
        device_tensors[weight_name] = stacked.to(device=destination.device, dtype=destination.dtype)
    for weight_name in _MOE_WEIGHT_NAMES:
        device_tensors[weight_name] = reload_plans.cast_weights_to_nz(device_tensors[weight_name])

    # Scatter phase: non-blocking copies drained by one sync.
    for i, layer_expert in enumerate(layer_experts):
        layer, slot = routed_layers[layer_expert[0]], local_slots[layer_expert]
        for weight_name in device_tensors:
            slot_list = getattr(layer, f"{weight_name}_list", None)
            target = slot_list[slot] if slot_list is not None else getattr(layer, weight_name).data[slot]
            target.copy_(device_tensors[weight_name][i], non_blocking=True)
    torch.npu.synchronize()
    return len(layer_experts)


def _get_ckpt_name_mapper(model: torch.nn.Module) -> Callable[[str], str]:
    """Map checkpoint weight names to the names the model's own parameters use."""
    mapper = getattr(model, "hf_to_vllm_mapper", None)
    if mapper is None:
        # No mapping declared: the checkpoint already uses the model's parameter
        # names (e.g. Qwen3-MoE).
        return lambda name: name
    return lambda name: mapper._map_name(name) or name


def _match_expert_tensor(
    name: str,
    wanted_experts: dict[str, dict[int, tuple[int, int]]],
) -> tuple[tuple[int, int], str] | None:
    """Split a mapped checkpoint name into (layer_expert, suffix) if it names a wanted expert tensor.

    E.g. ``model.layers.3.ffn.experts.12.gate_proj.weight`` -> ``((3, 12), "gate_proj.weight")``.
    """
    head, _, suffix = name.rpartition(".")
    layer_name, _, expert_str = head.rpartition(".")
    if not expert_str.isdigit():
        # 2-segment suffix ("down_proj.weight"): peel one more segment.
        layer_name, _, last = layer_name.rpartition(".")
        suffix = f"{expert_str}.{suffix}"
        expert_str = last
    by_expert = wanted_experts.get(layer_name)
    if by_expert is None or not expert_str.isdigit():
        return None
    layer_expert = by_expert.get(int(expert_str))
    return (layer_expert, suffix) if layer_expert is not None else None


def _collect_safetensors_files(
    model_config: ModelConfig,
    load_config: LoadConfig,
    model: torch.nn.Module,
) -> list[str]:
    """Checkpoint safetensors file paths, resolved the same way the normal loader does."""
    loader = DefaultModelLoader(load_config)
    _, files, use_safetensors, _ = loader._prepare_weights(
        model_name_or_path=model_config.model,
        subfolder=None,
        revision=model_config.revision,
        fall_back_to_pt=getattr(model, "fall_back_to_pt_during_load", True),
        allow_patterns_overrides=getattr(model, "allow_patterns_overrides", None),
    )
    if not use_safetensors:
        sample_files = ", ".join(os.path.basename(f) for f in files[:3]) or "no weight files"
        raise RuntimeError(
            f"[FT] scale_down expert reload requires a safetensors checkpoint, but "
            f"'{model_config.model}' resolved to non-safetensors files ({sample_files})."
        )
    return sorted(f for f in files if f.endswith(".safetensors"))


def _collect_matching_weights(
    files: list[str],
    name_mapper: Callable[[str], str],
    wanted_experts: dict[str, dict[int, tuple[int, int]]],
) -> dict[tuple[int, int], dict[str, torch.Tensor]]:
    """Return ``{(layer_idx, logical_expert_id): {suffix: tensor}}`` for the wanted experts.

    Only file headers are read, so tensors of other experts are never loaded.
    """
    expert_weights: dict[tuple[int, int], dict[str, torch.Tensor]] = {}
    for file_path in files:
        with safe_open(file_path, framework="pt") as f:
            for raw_name in f.keys():  # noqa: SIM118
                match = _match_expert_tensor(name_mapper(raw_name), wanted_experts)
                if match is None:
                    continue
                layer_expert, suffix = match
                if suffix in _WANTED_SUFFIXES:
                    expert_weights.setdefault(layer_expert, {})[suffix] = f.get_tensor(raw_name)
    return expert_weights


def _find_local_expert_slots(
    routed_layers: list,
    moe_layers: list,
    moved_experts: set[tuple[int, int]],
    ep_rank: int,
) -> dict[tuple[int, int], int]:
    """Map each moved expert to the index of its per-expert tensor on this rank.

    An expert may have replicas on several ranks; only the replica hosted by
    this rank is returned.
    """
    local_slots: dict[tuple[int, int], int] = {}
    for layer_idx, logical_id in moved_experts:
        layer_state = getattr(moe_layers[layer_idx], "eplb_state", None)
        logical_to_physical = getattr(layer_state, "logical_to_physical_map", None)
        if logical_to_physical is None:
            raise RuntimeError(
                f"[FT] MoE layer {layer_idx} has no EPLB placement map; the moved expert has nowhere to land."
            )
        num_local_experts = routed_layers[layer_idx].moe_config.num_local_experts
        start = ep_rank * num_local_experts
        for physical_id in logical_to_physical[logical_id].tolist():
            if start <= physical_id < start + num_local_experts:
                local_slots[(layer_idx, logical_id)] = physical_id - start
                break
    return local_slots


def reload_experts_from_disk(
    model: torch.nn.Module,
    model_config: ModelConfig,
    load_config: LoadConfig,
    moved_experts: set[tuple[int, int]],
) -> int:
    """Reload the moved experts' weights into this rank's per-expert tensors.

    The tensors must end up exactly as ``process_weights_after_loading`` leaves
    them (weights transposed and cast to NZ, quant scales converted, one tensor
    per expert slot), which ``model.load_weights`` cannot produce. Each model is
    read with its own checkpoint config and ``hf_to_vllm_mapper``; returns the
    number of (layer, expert) pairs reloaded.
    """
    if not moved_experts:
        return 0

    moe_layers = list(model.moe_layers)
    routed_layers = [getattr(layer, "routed_experts", layer) for layer in moe_layers]
    local_slots = _find_local_expert_slots(routed_layers, moe_layers, moved_experts, get_ep_group().rank_in_group)
    if not local_slots:
        return 0

    name_mapper = _get_ckpt_name_mapper(model)

    # "<layer_name>.<expert_id>" -> (layer_idx, logical_id) for O(1) matching.
    wanted_experts: dict[str, dict[int, tuple[int, int]]] = {}
    for layer_idx, logical_id in local_slots:
        wanted_experts.setdefault(routed_layers[layer_idx].layer_name, {})[logical_id] = (layer_idx, logical_id)

    logger.info("[FT] Reloading %d moved (layer, expert) pair(s) on this rank from disk.", len(local_slots))
    files = _collect_safetensors_files(model_config, load_config, model)
    expert_weights = _collect_matching_weights(files, name_mapper, wanted_experts)
    # A missing weight would leave the previous expert's weights in that tensor, so fail.
    unmatched = [layer_expert for layer_expert in local_slots if layer_expert not in expert_weights]
    if unmatched:
        raise RuntimeError(
            f"[FT] {len(unmatched)} (layer, expert) pair(s) had no matching "
            f"checkpoint weight, e.g. {unmatched[:5]}. The model's expert "
            "weights likely use checkpoint names that do not follow "
            "'<layer_name>.<expert_id>.' (e.g. fused experts), or the "
            "checkpoint's naming differs from the model's parameter names and "
            "no hf_to_vllm_mapper is declared on the model class."
        )

    reloaded = _reload_local_expert_slots(routed_layers, local_slots, expert_weights)

    logger.info("[FT] Expert weight reload complete: %d (layer, expert) pair(s).", reloaded)
    return reloaded
