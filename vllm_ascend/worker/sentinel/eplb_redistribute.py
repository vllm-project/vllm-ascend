# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""NPU expert redistribution and weight reload for fault-tolerance scale-down.

Device-agnostic placement math is imported from the upstream
``vllm.v1.worker.sentinel.eplb_redistribute``; this module only carries the
Ascend-specific pieces:

- ``reload_experts_from_disk``: reload reassigned experts from checkpoint and
  write them back in place, mirroring the runtime layout produced by
  ``process_weights_after_loading`` (transpose / NZ cast / per-slot lists /
  quant scales), so captured graphs keep working. It mirrors the upstream
  signature (a set of ``(layer, logical)`` reassignments) so the shared
  sentinel flow can drive it directly; the destination local slot on this rank
  is recovered from the rebuilt ``logical_to_physical_map``.

The EPLB structures (physical_to_logical / logical_to_physical /
logical_replica_count) keep the upstream slot model end to end: full width,
original physical ids. The one exception is the kernel-facing
``expert_replica_routing_table`` values: in scale-down mode the MC2 kernels
override their world view from elastic_info (dense ep size / dense physical
expert count / own rank = table1[orig]) and route each token via
``table2[expert_id // num_local]``, while the combine kernel silently drops
any id >= the shrunk physical expert count — so the ids consumed by the
kernels must live in the densified space ``dense_rank * num_local + slot``
(see ``densify_routing_table_physical_ids``). Only the table *values* are
renumbered; shapes never change and updates are in-place, so captured graphs
stay valid. Densified rank numbering otherwise lives only in the elastic_info
tensor (table1/table2, consumed by the kernels) and in the rebuilt gloo cpu
groups, exactly like upstream.

All functions are deterministic with stable iteration order, so every
surviving rank running them with the same inputs produces bit-identical
results — no cross-rank communication during recovery.
"""

from collections.abc import Callable, Generator

import torch
import torch_npu
from vllm.config import VllmConfig
from vllm.distributed import get_ep_group
from vllm.logger import logger
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader

# Re-exported upstream helpers, kept in one place so the sentinel has a
# single import site for the redistribution building blocks.
from vllm.v1.worker.sentinel.eplb_redistribute import (
    check_redundancy_sufficient,
    compute_dead_ep_ranks,
    mark_dead_expert_slots_inplace,
    rebuild_logical_expert_maps,
    redistribute_expert_placement,
)

from vllm_ascend.ops.fused_moe.routed_experts import AscendUnquantizedFusedMoEMethod
from vllm_ascend.quantization.methods.w8a8_dynamic import (
    AscendW8A8DynamicFusedMoEMethod,
    scale_from_float_to_int64,
)
from vllm_ascend.utils import ACL_FORMAT_FRACTAL_NZ, maybe_trans_nz

__all__ = [
    "build_orig_to_dense_rank_table",
    "check_redundancy_sufficient",
    "compute_dead_ep_ranks",
    "densify_routing_table_physical_ids",
    "mark_dead_expert_slots_inplace",
    "rebuild_logical_expert_maps",
    "redistribute_expert_placement",
    "reload_experts_from_disk",
]

# Checkpoint name suffixes (relative to "<layer_name>.<expert_id>.") of the
# tensors a reload may consume. weight_offset is loaded at startup but never
# consumed by the MoE apply path, so it is not reloaded here.
_W13_WEIGHT_SUFFIXES = ("gate_up_proj.weight", "gate_proj.weight", "up_proj.weight")
_W2_WEIGHT_SUFFIX = "down_proj.weight"
_W13_SCALE_SUFFIXES = ("gate_up_proj.weight_scale", "gate_proj.weight_scale", "up_proj.weight_scale")
_W2_SCALE_SUFFIX = "down_proj.weight_scale"


def build_orig_to_dense_rank_table(ep_world_size: int, dead_ranks: set[int]) -> torch.Tensor:
    """Build the orig-rank -> densified-rank mapping table.

    Returns a ``[ep_world_size]`` int32 tensor where ``table[orig_rank]`` is
    the densified rank (-1 for dead ranks), i.e. the kernel's table1 view of
    the elastic_info layout. Densified ranks are assigned in ascending
    original-rank order over the survivors.
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
    """Renumber a routing table's physical ids into the densified id space.

    In scale-down mode the MC2 dispatch kernel computes a token's destination
    as ``table2[expert_id // num_local]`` and the combine kernel drops any id
    >= the shrunk physical expert count, so the ids produced by the EPLB
    mapping must be dense-rank-major: ``dense_rank * num_local + slot``.
    Keeping original ids only works when the dead ranks happen to be a suffix
    (then table2 is the identity on the alive prefix); a dead rank in the
    middle misroutes tokens or crashes the kernel on a -1 rank lookup.

    The update is an in-place ``copy_`` with an unchanged shape, so captured
    graphs keep pointing at valid storage.

    Args:
        routing_table: ``expert_replica_routing_table`` of one MoE layer
            (device, int32), holding original global physical ids.
        orig_to_dense_rank: ``[ep_world_size]`` original EP rank -> densified
            rank (-1 for dead ranks), i.e. elastic_info's table1.
        num_local_experts: physical slots per EP rank (unchanged by
            scale-down).
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


def _tp_shard_info(layer) -> tuple[int, int]:
    """(tp_rank, tp_size) of the MoE weights of a routed-experts module."""
    parallel_config = layer.moe_config.moe_parallel_config
    return parallel_config.tp_rank, parallel_config.tp_size


def _shard_row(t: torch.Tensor, tp_rank: int, tp_size: int) -> torch.Tensor:
    """Take this TP rank's row shard (dim 0) of a full checkpoint tensor."""
    if tp_size == 1:
        return t
    shard = t.shape[0] // tp_size
    return t.narrow(0, tp_rank * shard, shard)


def _shard_col(t: torch.Tensor, tp_rank: int, tp_size: int) -> torch.Tensor:
    """Take this TP rank's column shard (dim 1) of a full checkpoint tensor."""
    if tp_size == 1:
        return t
    shard = t.shape[1] // tp_size
    return t.narrow(1, tp_rank * shard, shard)


def _gather_w13(
    tensors: dict[str, torch.Tensor],
    tp_rank: int,
    tp_size: int,
) -> torch.Tensor:
    """Assemble one expert's w13 ([2I_local, H]) from checkpoint tensors."""
    fused = tensors.get("gate_up_proj.weight")
    if fused is not None:
        half = fused.shape[0] // 2
        gate, up = fused[:half], fused[half:]
    else:
        gate, up = tensors["gate_proj.weight"], tensors["up_proj.weight"]
    return torch.cat([_shard_row(gate, tp_rank, tp_size), _shard_row(up, tp_rank, tp_size)], dim=0)


def _gather_w13_scale(
    tensors: dict[str, torch.Tensor],
    tp_rank: int,
    tp_size: int,
) -> torch.Tensor:
    """1-D w13 scale ([2I_local]) from checkpoint tensors."""
    fused = tensors.get("gate_up_proj.weight_scale")
    if fused is not None:
        half = fused.shape[0] // 2
        gate, up = fused[:half], fused[half:]
    else:
        gate = tensors["gate_proj.weight_scale"]
        up = tensors["up_proj.weight_scale"]
    return torch.cat([_shard_row(gate, tp_rank, tp_size), _shard_row(up, tp_rank, tp_size)], dim=0).view(-1)


def _reload_unquantized(layer, slot: int, tensors: dict[str, torch.Tensor]) -> None:
    """Mirror AscendUnquantizedFusedMoEMethod.process_weights_after_loading
    for a single expert slot (ROCm-only padding intentionally skipped)."""
    if getattr(layer, "w13_bias", None) is not None or getattr(layer, "w2_bias", None) is not None:
        raise NotImplementedError("[FT] scale_down weight reload does not support MoE expert bias yet.")
    tp_rank, tp_size = _tp_shard_info(layer)
    w13_weight_list = getattr(layer, "w13_weight_list", None)
    target = w13_weight_list[slot] if w13_weight_list is not None else layer.w13_weight
    device, dtype = target.device, target.dtype

    # [2I, H] -> transpose -> [H, 2I], then NZ cast (whole-tensor policy
    # function, same as process_weights_after_loading's non-fused path).
    w13 = _gather_w13(tensors, tp_rank, tp_size).transpose(0, 1).contiguous()
    w13 = maybe_trans_nz(w13.to(device=device, dtype=dtype))
    w2 = _shard_col(tensors[_W2_WEIGHT_SUFFIX], tp_rank, tp_size).transpose(0, 1).contiguous()
    w2 = maybe_trans_nz(w2.to(device=device, dtype=dtype))

    if w13_weight_list is not None:
        w13_weight_list[slot].copy_(w13)
        layer.w2_weight_list[slot].copy_(w2)
    else:
        # Whole-tensor NZ layout: the slot slice is one expert matrix.
        layer.w13_weight.data[slot].copy_(w13)
        layer.w2_weight.data[slot].copy_(w2)


def _reload_w8a8_dynamic(layer, slot: int, tensors: dict[str, torch.Tensor]) -> None:
    """Mirror AscendW8A8DynamicFusedMoEMethod.process_weights_after_loading
    for a single expert slot (v2 + EPLB always stores per-slot lists)."""
    tp_rank, tp_size = _tp_shard_info(layer)
    device = layer.w13_weight_list[slot].device

    w13 = _gather_w13(tensors, tp_rank, tp_size).transpose(0, 1).contiguous()
    w13 = torch_npu.npu_format_cast(w13.to(device=device), ACL_FORMAT_FRACTAL_NZ)
    w2 = _shard_col(tensors[_W2_WEIGHT_SUFFIX], tp_rank, tp_size).transpose(0, 1).contiguous()
    w2 = torch_npu.npu_format_cast(w2.to(device=device), ACL_FORMAT_FRACTAL_NZ)
    layer.w13_weight_list[slot].copy_(w13)
    layer.w2_weight_list[slot].copy_(w2)

    w13_scale = _gather_w13_scale(tensors, tp_rank, tp_size)
    w2_scale = tensors[_W2_SCALE_SUFFIX].view(-1)
    layer.w13_weight_scale_fp32_list[slot].copy_(w13_scale.to(torch.float32))
    w2_scale_target = layer.w2_weight_scale_list[slot]
    layer.w2_weight_scale_list[slot].copy_(w2_scale.to(w2_scale_target.dtype))

    # fused_w*_scale_list only exist when enable_fused_mc2 == 1 (currently
    # rejected for scale_down); keep the mirror for future support.
    fused_w1_scale_list = getattr(layer, "fused_w1_scale_list", None)
    fused_w2_scale_list = getattr(layer, "fused_w2_scale_list", None)
    if fused_w1_scale_list is not None and fused_w2_scale_list is not None:
        fused_w1_scale_list[slot].copy_(scale_from_float_to_int64(w13_scale))
        fused_w2_scale_list[slot].copy_(scale_from_float_to_int64(w2_scale))


# Dispatch on the quant method type; register new schemes here following the
# same pattern.
_RELOADERS: dict[type, Callable[[torch.nn.Module, int, dict[str, torch.Tensor]], None]] = {
    AscendUnquantizedFusedMoEMethod: _reload_unquantized,
    AscendW8A8DynamicFusedMoEMethod: _reload_w8a8_dynamic,
}


def reload_experts_from_disk(
    model: torch.nn.Module,
    vllm_config: VllmConfig,
    reassignments: set[tuple[int, int]],
) -> int:
    """Reload reassigned (layer, logical) expert weights from disk.

    Mirrors the upstream ``reload_experts_from_disk`` signature (a set of
    ``(moe_layer_idx, logical_expert_id)`` reassignments produced by
    ``redistribute_expert_placement``), so the shared sentinel flow can drive
    it unchanged. Ascend keeps expert weights in runtime layout (transposed,
    NZ-cast, split into per-slot lists, with derived quant scales), so the
    standard ``model.load_weights`` path cannot write them back; checkpoint
    tensors are read via DefaultModelLoader and converted per quant method,
    then copied into the existing slot storage in place.

    The destination local slot of each reassigned logical expert is recovered
    from the freshly rebuilt per-layer ``logical_to_physical_map`` (rebuilt by
    the upstream flow before this is called). Only reassigned experts whose
    replica falls in this rank's physical block are reloaded.

    Returns:
        Number of (layer, slot) pairs reloaded.
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
            continue
        num_local = routed_layers[layer_idx].moe_config.num_local_experts
        start = ep_rank * num_local
        for physical_id in l2p[logical_id].tolist():
            if start <= physical_id < start + num_local:
                local_slots[(layer_idx, logical_id)] = physical_id - start
                break

    if not local_slots:
        return 0

    prefixes: dict[str, tuple[int, int]] = {
        f"{routed_layers[layer_idx].layer_name}.{logical_id}.": (layer_idx, logical_id)
        for layer_idx, logical_id in local_slots
    }

    loader = DefaultModelLoader(vllm_config.load_config)
    # Produce every expert, not just the ones local at startup.
    loader.local_expert_ids = None
    all_weights = loader.get_all_weights(vllm_config.model_config, model)

    wanted_suffixes = set(_W13_WEIGHT_SUFFIXES + _W13_SCALE_SUFFIXES)
    wanted_suffixes.add(_W2_WEIGHT_SUFFIX)
    wanted_suffixes.add(_W2_SCALE_SUFFIX)

    buckets: dict[tuple[int, int], dict[str, torch.Tensor]] = {}
    matched: set[str] = set()

    def filtered_iter() -> Generator[tuple[tuple[int, int], str, torch.Tensor], None, None]:
        for name, tensor in all_weights:
            for prefix, key in prefixes.items():
                if name.startswith(prefix):
                    matched.add(prefix)
                    suffix = name[len(prefix) :]
                    if suffix in wanted_suffixes:
                        yield key, suffix, tensor
                    break

    logger.info("[FT] Reloading %d reassigned (layer, expert) pair(s) on this rank from disk.", len(local_slots))
    for key, suffix, tensor in filtered_iter():
        buckets.setdefault(key, {})[suffix] = tensor

    unmatched = [pair for prefix, pair in prefixes.items() if prefix not in matched]
    if unmatched:
        raise RuntimeError(
            f"[FT] {len(unmatched)} (layer, expert) pair(s) had no matching "
            f"checkpoint weight, e.g. {unmatched[:5]}. The model's expert "
            "weights likely use a layout that does not follow "
            "'<layer_name>.<expert_id>.' (e.g. fused experts)."
        )

    reloaded = 0
    for (layer_idx, logical_id), slot in sorted(local_slots.items()):
        routed = routed_layers[layer_idx]
        # Quantized layers carry the AscendFusedMoEMethod wrapper; the actual
        # scheme (the _RELOADERS key) lives in its .quant_method attribute.
        # Unquantized layers hold the bare scheme, so fall back to the object
        # itself (same idiom as AscendRoutedExperts.quant_type).
        quant_method = getattr(routed.quant_method, "quant_method", routed.quant_method)
        reloader = _RELOADERS.get(type(quant_method))
        if reloader is None:
            raise NotImplementedError(
                f"[FT] scale_down weight reload is not implemented for quant method {type(quant_method).__name__}."
            )
        tensors = buckets[(layer_idx, logical_id)]
        reloader(routed, slot, tensors)
        reloaded += 1

    logger.info("[FT] Expert weight reload complete: %d (layer, slot) pair(s).", reloaded)
    return reloaded
