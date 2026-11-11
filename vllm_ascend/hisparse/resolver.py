# SPDX-License-Identifier: Apache-2.0
"""Tensor-only native resolver dispatch; no address-keyed launch ownership."""

import torch


def _load_resolver():
    namespace = torch.ops._C_ascend
    if not all(
        hasattr(namespace, name) for name in ("hisparse_resolver_validate_scratch", "hisparse_resolve_residency")
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
