# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Transport-independent sparse runtime weight patch operations."""

import logging
from math import prod

import regex as re
import torch
from vllm.distributed.weight_transfer.sparse_nccl_engine import SparseWeightPatch as SparseWeightPatch
from vllm.model_executor.model_loader.checkpoint_weight_patch import (
    CheckpointWeightPatch,
    load_checkpoint_weight_patches,
)


def _validate_sparse_hf_patches(patches):
    """Validate before any write, batching equal bounds to avoid tiny kernels."""
    invalid = []
    groups = {}

    def check(parts, bound):
        indices = torch.cat(parts) if len(parts) > 1 else parts[0]
        invalid.append(((indices < 0) | (indices >= bound)).any())

    for patch, shape in patches:
        if not patch.name or any(dimension < 0 for dimension in shape):
            raise ValueError("Sparse HF name/shape metadata is invalid")
        if tuple(shape) != tuple(patch.full_shape):
            raise ValueError(f"Sparse HF full_shape mismatch: {patch.name}")
        if not patch.values.is_floating_point():
            raise TypeError(f"Sparse HF values must be floating point: {patch.name}")
        invalid.append(torch.isnan(patch.values).any())
        if patch.indices.numel():
            sorted_indices = torch.sort(patch.indices).values
            invalid.append((sorted_indices[1:] == sorted_indices[:-1]).any())
        if patch.indices.dtype != torch.int32 or patch.indices.ndim != 1 or patch.values.ndim != 1:
            raise ValueError(f"Invalid sparse HF tensor metadata: {patch.name}")
        if patch.indices.numel() != patch.values.numel():
            raise ValueError(f"Sparse HF indices/value length mismatch: {patch.name}")
        if prod(shape) >= 2**31:
            raise OverflowError(f"Sparse HF int32 index limit exceeded: {patch.name}")
        if patch.indices.numel():
            key = (patch.indices.device, prod(shape))
            parts, count = groups.get(key, ([], 0))
            if parts and count + patch.indices.numel() > 4 * 1024 * 1024:
                check(parts, key[1])
                parts, count = [], 0
            parts.append(patch.indices)
            groups[key] = (parts, count + patch.indices.numel())
    for (_, bound), (parts, _) in groups.items():
        check(parts, bound)
    if invalid and bool(torch.stack(invalid).any()):
        raise ValueError("Sparse HF flush contains NaN values, duplicate indices or out of bounds indices")


@torch.no_grad()
def apply_sparse_hf_patches_with_loader(
    model,
    patches,
    *,
    chunk_bytes=512 * 1024 * 1024,
    direct_moe=False,
    verify=False,
):
    """Validate before optimized writes; delegate all fallback loading to vLLM."""
    if chunk_bytes <= 0:
        raise ValueError("chunk_bytes must be positive")
    _validate_sparse_hf_patches(patches)
    if direct_moe:
        patches = _apply_direct_moe_patches(model, patches, verify=verify)
    checkpoint_patches = [
        CheckpointWeightPatch(patch.name, tuple(shape), patch.values.dtype, patch.values, patch.indices)
        for patch, shape in patches
    ]
    load_checkpoint_weight_patches(
        _VerifiedLoader(model) if verify else model,
        checkpoint_patches,
        max_chunk_bytes=chunk_bytes,
        validate_unique_indices=False,  # Already checked before any MoE writes.
    )


class _VerifiedLoader:
    """Observe vLLM's masked final writes without reimplementing masked loading."""

    def __init__(self, model):
        self.model = model

    def load_weights(self, weights):
        masked_copy = torch.Tensor.copy_

        def verify_copy(destination, source, *args, **kwargs):
            result = masked_copy(destination, source, *args, **kwargs)
            cast = source.to(dtype=destination.dtype, device=destination.device)
            mask = ~torch.isnan(cast)
            if not torch.equal(destination[mask], cast[mask]):
                raise RuntimeError("Sparse loader rank-local verification failed")
            return result

        torch.Tensor.copy_ = verify_copy
        try:
            return self.model.load_weights(weights)
        finally:
            torch.Tensor.copy_ = masked_copy


def _apply_moe_group(param, entries, shape, tp_rank, tp_size, down, *, verify=False):
    """Compact once per runtime weight, preserving HF absolute-copy semantics.

    SGLang avoids per-expert dynamic boolean indexing in its masked loader.
    Here the same TP selection is performed for a whole runtime w13/w2 group.
    Explicit output_size prevents repeat_interleave from reading a device count.
    """
    lengths = [patch.indices.numel() for patch, _, _ in entries]
    total = sum(lengths)
    if not total:
        return
    device = entries[0][0].indices.device
    indices = torch.cat([patch.indices for patch, _, _ in entries]).long()
    values = torch.cat([patch.values for patch, _, _ in entries])
    width = shape[1 if down else 0] // tp_size
    descriptors = torch.tensor(
        [[local_expert, width if projection == "up_proj" else 0] for _, local_expert, projection in entries],
        dtype=torch.int64,
        device=device,
    )
    descriptors = torch.repeat_interleave(
        descriptors,
        torch.tensor(lengths, dtype=torch.int64, device=device),
        dim=0,
        output_size=total,
    )
    row, col = indices // shape[1], indices % shape[1]
    coordinate = col if down else row
    selected = (
        ((coordinate >= tp_rank * width) & (coordinate < (tp_rank + 1) * width) & ~torch.isnan(values))
        .nonzero()
        .view(-1)
    )
    if down:
        target = descriptors[:, 0] * width * shape[0] + (col - tp_rank * width) * shape[0] + row
    else:
        target = (
            descriptors[:, 0] * shape[1] * (2 * width) + col * (2 * width) + descriptors[:, 1] + row - tp_rank * width
        )
    local_indices = target.index_select(0, selected)
    local_values = values.index_select(0, selected)
    flat = param.data.view(-1)
    flat.index_copy_(0, local_indices, local_values)
    if verify:
        if not torch.equal(flat.index_select(0, local_indices), local_values):
            raise RuntimeError("Sparse MoE rank-local verification failed")


def _apply_direct_moe_patches(
    model,
    patches,
    *,
    verify=False,
):
    # Exact implementation and layout guards: unfamiliar models, quantization,
    # EP/EPLB, padded or NZ storage retain the SGLang-style native loader.
    if getattr(getattr(model, "config", None), "model_type", None) != "qwen3_moe":
        return patches
    # Preserve ordered overwrite semantics for split/repeated parameter names.
    names = [patch.name for patch, _ in patches]
    if len(names) != len(set(names)):
        return patches
    from vllm_ascend.ops.fused_moe.routed_experts import (
        AscendRoutedExperts,
        AscendUnquantizedFusedMoEMethod,
    )

    fallback, grouped = [], {}
    for patch, shape in patches:
        match = re.fullmatch(r"(.+\.experts)\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight", patch.name)
        if match is None or len(shape) != 2:
            fallback.append((patch, shape))
            continue
        prefix, expert_text, projection = match.groups()
        try:
            runner = model.get_submodule(prefix)
        except AttributeError:
            fallback.append((patch, shape))
            continue
        layer = getattr(runner, "routed_experts", runner)
        if type(layer) is not AscendRoutedExperts or type(layer.quant_method) is not AscendUnquantizedFusedMoEMethod:
            fallback.append((patch, shape))
            continue
        config = layer.moe_config
        parallel = config.moe_parallel_config
        expert = int(expert_text)
        tp_size, tp_rank = parallel.tp_size, config.tp_rank
        param = getattr(layer, "w2_weight" if projection == "down_proj" else "w13_weight", None)
        shard_dim = 1 if projection == "down_proj" else 0
        expected = (
            (shape[1] // tp_size, shape[0]) if projection == "down_proj" else (shape[1], 2 * (shape[0] // tp_size))
        )
        expert_map = getattr(layer, "ascend_expert_map", None)
        eplb_enabled = getattr(config, "enable_eplb", False) or getattr(layer.quant_method, "dynamic_eplb", False)
        if parallel.ep_size == 1:
            local_expert = expert
        elif expert_map is not None and not eplb_enabled:
            cached_map = getattr(layer, "_sparse_expert_map_cpu", None)
            if cached_map is None:
                cached_map = expert_map.detach().cpu()
                layer._sparse_expert_map_cpu = cached_map
            if not 0 <= expert < cached_map.numel():
                raise IndexError(f"Global expert {expert} outside expert map")
            local_expert = int(cached_map[expert])
            if local_expert < 0:
                # The expert belongs to another EP rank.  Drop it before any
                # full-shape temporary is allocated by the native loader.
                continue
        else:
            local_expert = -1
        if (
            local_expert < 0
            or eplb_enabled
            or not config.is_act_and_mul
            or param is None
            or getattr(param, "is_transposed", False)
            or shape[shard_dim] % tp_size
            or param.ndim != 3
            or tuple(param.shape[1:]) != expected
            or param.dtype != patch.values.dtype
            or not param.is_contiguous()
            or not _supports_logical_flat_index_copy(param)
            or not 0 <= local_expert < param.shape[0]
        ):
            fallback.append((patch, shape))
            continue
        group = grouped.setdefault(
            id(param),
            (param, [], tuple(shape), tp_rank, tp_size, projection == "down_proj"),
        )
        if group[2:] != (tuple(shape), tp_rank, tp_size, projection == "down_proj"):
            fallback.append((patch, shape))
            continue
        group[1].append((patch, local_expert, projection))
    for param, entries, shape, tp_rank, tp_size, down in grouped.values():
        _apply_moe_group(param, entries, shape, tp_rank, tp_size, down, verify=verify)
    logging.getLogger(__name__).info(
        "Sparse MoE apply: direct_patches=%d runtime_parameters=%d fallback_patches=%d",
        len(patches) - len(fallback),
        len(grouped),
        len(fallback),
    )
    return fallback


def _supports_logical_flat_index_copy(tensor: torch.Tensor) -> bool:
    """Reject transformed Ascend layouts until physical mapping is defined."""
    if tensor.device.type != "npu":
        return True
    try:
        import torch_npu
    except ImportError:
        return False
    npu_format = str(torch_npu.get_npu_format(tensor)).upper()
    return npu_format in {"ND", "2"}
