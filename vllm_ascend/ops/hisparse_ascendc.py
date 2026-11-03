"""Tensor-only native resolver dispatch; no address-keyed launch ownership."""

from pathlib import Path

import torch


def scratch_layout(hot, max_union_rows):
    """Size hash/LRU scratch from the request capacity, not a model shape."""
    if not 0 < max_union_rows <= hot <= 32768:
        raise ValueError("HiSparse union must fit the hot cache and int16 LRU slot representation")
    hash_size = 1 << max_union_rows.bit_length()
    tiles = (hot + 31) // 32
    scratch_ints = ((2 * hash_size + hot + 2 * tiles + 64 + 15) // 16) * 16
    return hash_size, scratch_ints * 4 + 512


def _load_resolver():
    library = Path(__file__).with_name("torch_binding.so")
    if not library.is_file():
        return None
    if str(library) not in torch.ops.loaded_libraries:
        torch.ops.load_library(str(library))
    namespace = torch.ops.hisparse_fused60103
    if not hasattr(namespace, "scratch_capacity") or not hasattr(namespace, "resolve_tensors"):
        raise RuntimeError("Rebuild the HiSparse native extension for the Tensor resolver ABI")
    return namespace


def resolve_tensors(native, arguments, resolved_scratch, errors):
    """Pass live tensors; C++ derives geometry and protects queued storage.

    During replay, external tensors remain owned by the runtime/workspace.
    Callers must destroy graphs before releasing or rebinding those owners.
    """
    a = arguments
    native.resolve_tensors(
        a["global_indices"],
        a["request_ids"],
        a["request_state_indices"],
        a["source_block_table"],
        a["resident_block_table"],
        a["hot_block_table"],
        a["device_global_indices"],
        a["lru_slots"],
        a["hot_indices"],
        a["attention_indices"],
        resolved_scratch,
        a["valid_counts"],
        a["swap_host_physical_rows"],
        a["swap_device_physical_rows"],
        a["swap_counts"],
        a["miss_mask"],
        a["stats"],
        errors,
        a["num_valid_rows"],
        a["host_cache"].shape[0],
        a["hot_cache"].shape[1],
        a["source_block_size"],
        a["resident_block_size"],
        a["resident_null_block"],
        a["attention_block_stride"],
        a["max_union_rows"],
    )


def try_resolve(owner, scope):
    # The native resolver reads indices, not KV payload bytes. The optional
    # mask ABI is int32; other masks keep the existing Triton fallback.
    if scope["miss_mask"] is not None and scope["miss_mask"].dtype != torch.int32:
        return False
    hot = scope["device_global_indices"].shape[1]
    _, scratch_bytes = scratch_layout(hot, scope["max_union_rows"])
    native = _load_resolver()
    if native is None or scratch_bytes > native.scratch_capacity():
        return False
    rows, topk = scope["global_indices"].shape
    # Native and fallback resolvers execute serially on the same group copy
    # stream. Reuse group-owned scratch without retaining invocation inputs.
    workspace = owner.workspace
    resolved = scope["resolved_global_indices"]
    if resolved is None:
        resolved = workspace.first_refs[: rows * topk].view(rows, topk)
    resolve_tensors(native, scope, resolved, workspace.errors[:rows])
    return True
