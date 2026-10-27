# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""NPU expert redistribution and weight reload for fault-tolerance scale-down.

After dead EP ranks are removed and their experts reassigned to surviving
ranks, this module renumbers the expert-replica routing tables into the
densified rank space and reloads the reassigned experts' weights from the
checkpoint into each rank's local slots.
"""

import glob
import os
from collections.abc import Callable, Set
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import torch
from safetensors import safe_open
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME
from vllm.config import VllmConfig
from vllm.distributed import get_ep_group
from vllm.logger import logger
from vllm.model_executor.model_loader.weight_utils import (
    filter_duplicate_safetensors_files,
)

from vllm_ascend.worker.sentinel.eplb_quant_schemes import (
    _WANTED_SUFFIXES,
    _WEIGHT_ATTRS,
    _ReloadScheme,
    _resolve_scheme,
    _WeightPipeline,
)

__all__ = [
    "build_orig_to_dense_rank_table",
    "densify_routing_table_physical_ids",
    "reload_experts_from_disk",
]

# (layer, expert) pairs per chunk: bigger chunks amortise device rounds but grow the stacked buffer.
_RELOAD_CHUNK_SIZE = 32

# Threads assembling one chunk; bounded because worker processes share host memory bandwidth.
_GATHER_MAX_WORKERS = 8


def build_orig_to_dense_rank_table(ep_world_size: int, dead_ranks: set[int]) -> torch.Tensor:
    """Map each original EP rank to its densified rank (-1 for dead ranks).

    Returns the ``[ep_world_size]`` int32 table the dispatch kernel reads as
    ``table1``; survivors are numbered in ascending original-rank order.
    """
    table = torch.full((ep_world_size,), -1, dtype=torch.int32)
    alive = sorted(set(range(ep_world_size)) - dead_ranks)
    for dense_rank, orig_rank in enumerate(alive):
        table[orig_rank] = dense_rank
    return table


def densify_routing_table_physical_ids(
    routing_table: torch.Tensor,
    orig_to_dense_rank: torch.Tensor,
    num_local_experts: int,
) -> None:
    """Renumber one layer's ``expert_replica_routing_table`` into the densified id space.

    The MC2 dispatch kernel indexes ``table2[expert_id // num_local]`` and the
    combine kernel drops ids beyond the shrunk expert count, so ids must be
    dense-rank-major (``dense_rank * num_local + slot``); a dead rank in the
    middle would otherwise misroute tokens. The update is an in-place
    ``copy_``, so captured graphs keep pointing at valid storage.
    """
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


def _slot_tensor(layer: torch.nn.Module, attr: str, slot: int) -> torch.Tensor:
    """The runtime storage of one per-slot tensor."""

    per_slot = getattr(layer, f"{attr}_list", None)
    return per_slot[slot] if per_slot is not None else getattr(layer, attr).data[slot]


def _reload_local_slots(
    routed_layers: list,
    local_slots: dict[tuple[int, int], int],
    expert_weights: dict[tuple[int, int], dict[str, torch.Tensor]],
) -> int:
    """Reload all local (layer, expert) pairs in scheme-uniform chunks.

    Pairs are grouped by (scheme, per-slot layout) so every stacked chunk is
    shape- and dtype-uniform; the layout comes from the runtime slots, so a new
    scheme needs no change here.
    """
    groups: dict[tuple, list[tuple[int, int]]] = {}
    for (layer_idx, logical_id), slot in sorted(local_slots.items()):
        routed = routed_layers[layer_idx]
        if getattr(routed, "w13_bias", None) is not None or getattr(routed, "w2_bias", None) is not None:
            raise NotImplementedError("[FT] scale_down weight reload does not support MoE expert bias yet.")
        reload_scheme = _resolve_scheme(routed)
        layout = []
        for attr in reload_scheme.pipelines:
            tensor = _slot_tensor(routed, attr, slot)
            layout.append((attr, tensor.shape, tensor.dtype))
        groups.setdefault((reload_scheme, tuple(layout)), []).append((layer_idx, logical_id))

    reloaded = 0
    for (reload_scheme, _layout), layer_experts in groups.items():
        for start in range(0, len(layer_experts), _RELOAD_CHUNK_SIZE):
            chunk = layer_experts[start : start + _RELOAD_CHUNK_SIZE]
            reloaded += _reload_chunk(reload_scheme, routed_layers, chunk, local_slots, expert_weights)
    return reloaded


def _process_expert(
    layer_expert: tuple[int, int],
    routed_layers: list,
    expert_weights: dict[tuple[int, int], dict[str, torch.Tensor]],
    pipelines: dict[str, _WeightPipeline],
) -> dict[str, torch.Tensor]:
    """CPU-side gather + transform of one expert, keyed by runtime attribute."""

    # EP reports (0, 1); the shard helpers still split for TP>1 deployments.
    parallel_config = routed_layers[layer_expert[0]].moe_config.moe_parallel_config
    tp_rank, tp_size = parallel_config.tp_rank, parallel_config.tp_size
    tensors = expert_weights[layer_expert]
    return {
        attr: pipeline.processor(pipeline.source(tensors, tp_rank, tp_size)) for attr, pipeline in pipelines.items()
    }


def _reload_chunk(
    scheme: _ReloadScheme,
    routed_layers: list,
    layer_experts: list[tuple[int, int]],
    local_slots: dict[tuple[int, int], int],
    expert_weights: dict[tuple[int, int], dict[str, torch.Tensor]],
) -> int:
    """Apply one chunk of pairs sharing a scheme and per-slot layout.

    Assemble on the CPU pool, then do one H2D and one layout cast per attribute
    for the whole stack, and one non-blocking copy per slot drained by a single
    sync -- instead of doing all of that per pair.
    """
    # CPU phase: assemble pairs in parallel (read-only shared state).
    assemble = partial(
        _process_expert, routed_layers=routed_layers, expert_weights=expert_weights, pipelines=scheme.pipelines
    )
    with ThreadPoolExecutor(max_workers=min(len(layer_experts), _GATHER_MAX_WORKERS)) as pool:
        per_expert = list(pool.map(assemble, layer_experts))

    # Device phase: one H2D and one layout cast per attribute. Landing at the
    # slot's dtype covers every scheme (int8 weights, int64 bit scales, fp32).
    first_layer = routed_layers[layer_experts[0][0]]
    first_slot = local_slots[layer_experts[0]]
    device_tensors: dict[str, torch.Tensor] = {}
    for attr in scheme.pipelines:
        ref = _slot_tensor(first_layer, attr, first_slot)
        stacked = torch.stack([expert[attr] for expert in per_expert])
        device_tensors[attr] = stacked.to(device=ref.device, dtype=ref.dtype)
    for attr in _WEIGHT_ATTRS:
        device_tensors[attr] = scheme.cast_layout(device_tensors[attr])

    # Scatter phase: non-blocking copies drained by one sync.
    for i, layer_expert in enumerate(layer_experts):
        layer, slot = routed_layers[layer_expert[0]], local_slots[layer_expert]
        for attr in device_tensors:
            _slot_tensor(layer, attr, slot).copy_(device_tensors[attr][i], non_blocking=True)
    torch.npu.synchronize()
    return len(layer_experts)


def _get_ckpt_name_mapper(model: torch.nn.Module) -> Callable[[str], str]:
    """Map raw checkpoint weight names into the model's runtime namespace."""
    mapper = getattr(model, "hf_to_vllm_mapper", None)
    if mapper is None:
        # No mapping declared: the checkpoint already uses runtime names (e.g. Qwen3-MoE).
        return lambda name: name
    return lambda name: mapper._map_name(name) or name


def _resolve_expert_tensor(
    name: str,
    wanted: dict[str, dict[int, tuple[int, int]]],
) -> tuple[tuple[int, int], str] | None:
    """Split a mapped checkpoint name into (key, suffix) if it names a wanted expert tensor.

    E.g. ``model.layers.3.ffn.experts.12.gate_proj.weight`` -> ``((3, 12), "gate_proj.weight")``.
    """
    head, _, suffix = name.rpartition(".")
    layer_name, _, expert_str = head.rpartition(".")
    if not expert_str.isdigit():
        # 2-segment suffix ("down_proj.weight"): peel one more segment.
        layer_name, _, last = layer_name.rpartition(".")
        suffix = f"{expert_str}.{suffix}"
        expert_str = last
    by_expert = wanted.get(layer_name)
    if by_expert is None or not expert_str.isdigit():
        return None
    key = by_expert.get(int(expert_str))
    return (key, suffix) if key is not None else None


def _collect_safetensors_shards(vllm_config: VllmConfig, model: torch.nn.Module) -> list[str]:
    """Collect the safetensors shards the model loads from the local checkpoint directory."""
    model_path = vllm_config.model_config.model
    if not os.path.isdir(model_path):
        raise RuntimeError(
            f"[FT] scale_down expert reload reads a local checkpoint only, but '{model_path}' is not a directory."
        )
    # A model may narrow the patterns it loads; honour that over our default.
    allow_patterns = getattr(model, "allow_patterns_overrides", None) or ["*.safetensors"]
    shards: list[str] = []
    for pattern in allow_patterns:
        shards += glob.glob(os.path.join(model_path, pattern))
        if shards:
            break
    # An override may select non-safetensors files; the header walk has
    # nothing to read there.
    shards = [shard for shard in shards if shard.endswith(".safetensors")]
    if len(shards) > 1:
        # Sharded and consolidated safetensors can coexist; the index file
        # selects the set the model loads (the filter no-ops without one).
        shards = filter_duplicate_safetensors_files(shards, model_path, SAFE_WEIGHTS_INDEX_NAME)
    return sorted(shards)


def _collect_matching_weights(
    shards: list[str],
    name_mapper: Callable[[str], str],
    wanted: dict[str, dict[int, tuple[int, int]]],
    wanted_suffixes: Set[str],
    expert_weights: dict[tuple[int, int], dict[str, torch.Tensor]],
    matched: set[tuple[int, int]],
) -> None:
    """Read only the wanted experts' tensors by walking shard headers.

    ``safe_open(...).keys()`` reads just the JSON header, so rejected names
    never reach ``get_tensor``: reads stay limited to the wanted experts.
    """
    for st_file in shards:
        with safe_open(st_file, framework="pt") as f:
            for raw_name in f.keys():  # noqa: SIM118
                found = _resolve_expert_tensor(name_mapper(raw_name), wanted)
                if found is None:
                    continue
                key, suffix = found
                matched.add(key)
                if suffix in wanted_suffixes:
                    expert_weights.setdefault(key, {})[suffix] = f.get_tensor(raw_name)


def reload_experts_from_disk(
    model: torch.nn.Module,
    vllm_config: VllmConfig,
    reassignments: set[tuple[int, int]],
) -> int:
    """Reload reassigned (moe_layer_idx, logical_expert_id) weights from disk.

    Ascend keeps expert weights in runtime layout (transposed, NZ-cast, per-slot
    lists, derived quant scales), so ``model.load_weights`` cannot write them
    back. Only experts landing in this rank's physical block are reloaded, into
    the slot from ``eplb_state.logical_to_physical_map``; the call must follow
    ``rebuild_model_expert_maps``.
    """
    if not reassignments:
        return 0

    moe_layers = list(model.moe_layers)
    routed_layers = [getattr(layer, "routed_experts", layer) for layer in moe_layers]
    ep_rank = get_ep_group().rank_in_group

    local_slots: dict[tuple[int, int], int] = {}
    for layer_idx, logical_id in reassignments:
        layer_state = getattr(moe_layers[layer_idx], "eplb_state", None)
        l2p = getattr(layer_state, "logical_to_physical_map", None)
        if l2p is None:
            raise RuntimeError(
                f"[FT] MoE layer {layer_idx} has no EPLB placement map; the reassigned expert has nowhere to land."
            )
        num_local = routed_layers[layer_idx].moe_config.num_local_experts
        start = ep_rank * num_local
        for physical_id in l2p[logical_id].tolist():
            if start <= physical_id < start + num_local:
                local_slots[(layer_idx, logical_id)] = physical_id - start
                break

    if not local_slots:
        return 0

    name_mapper = _get_ckpt_name_mapper(model)

    # "<layer_name>.<expert_id>" -> key for O(1) matching.
    wanted: dict[str, dict[int, tuple[int, int]]] = {}
    for layer_idx, logical_id in local_slots:
        wanted.setdefault(routed_layers[layer_idx].layer_name, {})[logical_id] = (layer_idx, logical_id)

    expert_weights: dict[tuple[int, int], dict[str, torch.Tensor]] = {}
    matched: set[tuple[int, int]] = set()

    logger.info("[FT] Reloading %d reassigned (layer, expert) pair(s) on this rank from disk.", len(local_slots))
    shards = _collect_safetensors_shards(vllm_config, model)
    if not shards:
        raise RuntimeError(
            "[FT] scale_down expert reload requires a safetensors checkpoint; "
            f"{vllm_config.model_config.model} has none."
        )
    _collect_matching_weights(shards, name_mapper, wanted, _WANTED_SUFFIXES, expert_weights, matched)
    # A pair with no checkpoint weight would silently keep its slot's previous occupant, so fail.
    unmatched = [layer_expert for layer_expert in local_slots if layer_expert not in matched]
    if unmatched:
        raise RuntimeError(
            f"[FT] {len(unmatched)} (layer, expert) pair(s) had no matching "
            f"checkpoint weight, e.g. {unmatched[:5]}. The model's expert "
            "weights likely use a layout that does not follow "
            "'<layer_name>.<expert_id>.' (e.g. fused experts), or the "
            "checkpoint's naming differs from the runtime namespace without "
            "an hf_to_vllm_mapper declared on the model class."
        )

    reloaded = _reload_local_slots(routed_layers, local_slots, expert_weights)

    logger.info("[FT] Expert weight reload complete: %d (layer, expert) pair(s).", reloaded)
    return reloaded
