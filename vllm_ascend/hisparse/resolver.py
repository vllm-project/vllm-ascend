# SPDX-License-Identifier: Apache-2.0
"""Tensor-only native resolver dispatch; no address-keyed launch ownership."""

import torch


def scratch_layout(hot, max_union_rows, capacity=None):
    """Size hash/LRU scratch from the request capacity, not a model shape."""
    if not 0 < max_union_rows <= hot <= 32768:
        raise ValueError("HiSparse union must fit the hot cache and int16 LRU slot representation")
    hash_size = 1 << max_union_rows.bit_length()
    tiles = (hot + 31) // 32
    fixed_words = hot + 2 * tiles + 64
    if capacity is not None:
        available_words = max(0, (capacity - 512) // 64) * 16
        max_hash_size = (available_words - fixed_words) // 2
        if max_hash_size <= max_union_rows:
            minimum_bytes = ((2 * (max_union_rows + 1) + fixed_words + 15) // 16) * 64 + 512
            raise RuntimeError(
                f"HiSparse hot_size={hot}, max_union_rows={max_union_rows} needs at least {minimum_bytes} UB bytes; "
                f"the compiled target allows {capacity}. Lower device_buffer_size or speculative tokens."
            )
        hash_size = min(hash_size, max_hash_size)
    scratch_ints = ((2 * hash_size + hot + 2 * tiles + 64 + 15) // 16) * 16
    return hash_size, scratch_ints * 4 + 512


def _load_resolver():
    namespace = torch.ops._C_ascend
    if not all(
        hasattr(namespace, name) for name in ("hisparse_resolver_scratch_capacity", "hisparse_resolve_residency")
    ):
        raise RuntimeError("HiSparse requires its compiled AscendC resolver. Rebuild vllm-ascend for this target.")
    return namespace


def resolve_tensors(native, arguments, resolved_scratch, errors):
    """Pass live tensors; C++ derives geometry and protects queued storage.

    During replay, external tensors remain owned by the runtime/workspace.
    Callers must destroy graphs before releasing or rebinding those owners.
    """
    a = arguments
    native.hisparse_resolve_residency(
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


def resolve(owner, scope):
    """Resolve with the native request-union kernel; no Triton fallback."""
    rows, topk = scope["global_indices"].shape
    workspace = owner.workspace
    resolved = scope["resolved_global_indices"]
    if resolved is None:
        resolved = workspace.first_refs[: rows * topk].view(rows, topk)
    resolve_tensors(workspace.native, scope, resolved, workspace.errors[:rows])
