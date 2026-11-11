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


def resolve(owner, arguments):
    """Dispatch live tensors to the native request-union resolver.

    During replay, external tensors remain owned by the runtime/workspace.
    Callers must destroy graphs before releasing or rebinding those owners.
    """
    rows, topk = arguments["global_indices"].shape
    workspace = owner.workspace
    resolved = arguments["resolved_global_indices"]
    if resolved is None:
        resolved = workspace.first_refs[: rows * topk].view(rows, topk)
    workspace.native.hisparse_resolve_residency(
        arguments["global_indices"],
        arguments["request_ids"],
        arguments["request_state_indices"],
        arguments["source_block_table"],
        arguments["resident_block_table"],
        arguments["hot_block_table"],
        arguments["device_global_indices"],
        arguments["lru_slots"],
        arguments["hot_indices"],
        arguments["attention_indices"],
        resolved,
        arguments["valid_counts"],
        arguments["swap_host_physical_rows"],
        arguments["swap_device_physical_rows"],
        arguments["swap_counts"],
        arguments["miss_mask"],
        arguments["stats"],
        workspace.errors[:rows],
        arguments["num_valid_rows"],
        arguments["host_cache"].shape[0],
        arguments["hot_cache"].shape[1],
        arguments["source_block_size"],
        arguments["resident_block_size"],
        arguments["resident_null_block"],
        arguments["attention_block_stride"] if owner.attention_block_stride is None else owner.attention_block_stride,
        arguments["max_union_rows"],
        owner.attention_capacity,
    )
