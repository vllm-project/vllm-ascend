# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import re
from functools import cache, lru_cache
from importlib import import_module
from importlib.util import find_spec
from typing import NamedTuple

import torch
from vllm.logger import logger

_ASCEND_950_MAX_TOPK = 8192


class LightningIndexerV2Plan(NamedTuple):
    cu_query: torch.Tensor
    visible_lengths: torch.Tensor
    block_table: torch.Tensor
    valid_pages: torch.Tensor
    metadata: torch.Tensor
    windows: int
    capacity: int


@cache
def get_lightning_indexer_v2_metadata_size(device: torch.device) -> int:
    """Resolve the installed BF16 ACLNN output size once per worker device before capture."""
    cu_query = torch.tensor([0, 1], device=device, dtype=torch.int32)
    lengths = torch.ones(1, device=device, dtype=torch.int32)
    op = torch.ops._C_ascend.npu_lightning_indexer_v2_metadata
    try:
        metadata = op(cu_query, lengths, 32, 2048, metadata_size=1024)
    except RuntimeError as exc:
        # The synchronous workspace check reports its required output length
        # before any kernel is queued. Only retry this specific ABI mismatch.
        message = str(exc).partition("TraceBack (most recent call last):")[0]
        match = re.search(r"The element num of metadata must be (\d+), but got 1024\b", message)
        if not message.startswith("call aclnnLightningIndexerV2MetadataGetWorkspaceSize failed") or match is None:
            raise
        required = int(match[1])
        if required <= 1024 or required % 1024:
            raise
        metadata = op(cu_query, lengths, 32, 2048, metadata_size=required)
    return metadata.numel()


def build_lightning_indexer_v2_plan(
    block_table: torch.Tensor,
    request_rows: torch.Tensor,
    local_visible_lengths: torch.Tensor,
    num_cache_pages: int,
    page_size: int,
    first_valid_page: int,
    num_heads_q: int,
    metadata_size: int = 2048,
) -> LightningIndexerV2Plan:
    """Batch equal-width windows as independent single-query TND requests."""
    from vllm_ascend.ops.triton.sharded_indexer import build_li_v2_batched_inputs

    cols = block_table.shape[1]
    max_pages = _ASCEND_950_MAX_TOPK // page_size
    windows = (cols + max_pages - 1) // max_pages
    # Balance windows to pad at most one page per window. Return all scores so
    # holes and arbitrary native tie order cannot displace a valid local ID.
    pages_per_window = (cols + windows - 1) // windows if windows else 0
    cu_query, visible, pages, valid_pages = build_li_v2_batched_inputs(
        block_table,
        request_rows,
        local_visible_lengths.reshape(-1),
        num_cache_pages,
        page_size,
        first_valid_page,
        pages_per_window,
        windows,
    )
    metadata = (
        torch.ops._C_ascend.npu_lightning_indexer_v2_metadata(
            cu_query, visible, num_heads_q, pages_per_window * page_size, metadata_size=metadata_size
        )
        if visible.numel()
        else visible
    )
    return LightningIndexerV2Plan(cu_query, visible, pages, valid_pages, metadata, windows, cols * page_size)


@lru_cache(maxsize=2)
def get_lightning_indexer_v2_unavailable_reason(*, require_seqused_q: bool = False) -> str | None:
    """Check the worker's native bindings and CANN symbols before graph capture."""
    name = "vllm_ascend.vllm_ascend_C"
    try:
        from vllm_ascend.utils import bootstrap_custom_op_env

        # The ACLNN adapter reads ASCEND_CUSTOM_OPP_PATH when its shared object loads.
        bootstrap_custom_op_env()
        extension = import_module(name)
    except (ImportError, OSError) as exc:
        spec = find_spec(name)
        origin = spec.origin if spec else name
        return f"failed to load {origin}: {exc}"

    origin = extension.__file__
    ops = torch.ops._C_ascend
    for name in ("npu_lightning_indexer_v2", "npu_lightning_indexer_v2_metadata"):
        binding = getattr(ops, name, None)
        if binding is None:
            return f"{origin} does not register _C_ascend::{name}; rebuild the vllm-ascend extension"
        operator = binding.default
        if not operator.has_kernel_for_dispatch_key("PrivateUse1"):
            return f"{origin} has no NPU kernel registered for _C_ascend::{name}"
        arguments = {arg.name for arg in operator._schema.arguments}
        if name.endswith("_metadata") and "metadata_size" not in arguments:
            return f"{origin} lacks the metadata_size argument of _C_ascend::{name}; rebuild the vllm-ascend extension"
        if require_seqused_q and "seqused_q" not in arguments:
            return f"{origin} lacks the seqused_q argument of _C_ascend::{name}; rebuild the vllm-ascend extension"

    # Older extensions expose only the boolean probe and can still run the unpadded ABI.
    try:
        details = getattr(ops, "get_lightning_indexer_v2_missing_symbols", None)
        if details is not None:
            missing = details()
            if missing:
                return f"{origin} could not resolve CANN symbols: {missing}"
        else:
            probe = getattr(ops, "is_lightning_indexer_v2_available", None)
            if probe is None:
                return f"{origin} lacks the LightningIndexerV2 availability probe; rebuild the vllm-ascend extension"
            if not probe():
                return f"{origin} could not resolve the CANN LightningIndexerV2 or LightningIndexerV2Metadata API"
    except RuntimeError as exc:
        return f"LightningIndexerV2 probe in {origin} failed: {exc}"
    logger.info_once(
        "Kimi K3.1 LightningIndexerV2 stage=probe extension=%s "
        "resolved=aclnnLightningIndexerV2,aclnnLightningIndexerV2Metadata",
        origin,
        scope="process",
    )
    return None


def has_lightning_indexer_v2() -> bool:
    return get_lightning_indexer_v2_unavailable_reason() is None


def lightning_indexer_v2_candidates(
    query: torch.Tensor,
    weights: torch.Tensor,
    key_cache: torch.Tensor,
    plan: LightningIndexerV2Plan,
    topk: int,
    *,
    for_dcp_merge: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Score every visible rank-local token with BF16 CANN LightningIndexerV2."""
    from vllm_ascend.ops.triton.sharded_indexer import pack_li_v2_window_queries, restore_li_v2_scores

    rows = query.shape[0]
    page_size = key_cache.shape[1]
    # Start in logical token order, including holes. A stable sort then selects
    # the earliest local ID on ties, independent of CANN's native Top-K order.
    dense = torch.empty((rows, max(topk, plan.capacity)), device=query.device, dtype=torch.float32)
    if rows and plan.windows:
        q, w = pack_li_v2_window_queries(query, weights, plan.windows, dense)
        width = plan.block_table.shape[1] * page_size
        indices, scores = torch.ops._C_ascend.npu_lightning_indexer_v2(
            q, key_cache, w, plan.cu_query, plan.visible_lengths, plan.block_table, plan.metadata, width
        )
        restore_li_v2_scores(indices, scores, plan.valid_pages, plan.visible_lengths, dense, 0, page_size, rows)
        del indices, scores, q, w
    else:
        dense.fill_(-torch.inf)
    best_scores, order = torch.sort(dense, dim=-1, descending=True, stable=True)
    best_scores = best_scores[:, :topk]
    best_ids = order[:, :topk]
    # The DCP pack kernel already validates scores and writes INT32 IDs.
    if not for_dcp_merge:
        best_ids = best_ids.to(torch.int32).masked_fill_(best_scores == -torch.inf, -1)
    return best_ids.unsqueeze(1), best_scores.unsqueeze(1)
