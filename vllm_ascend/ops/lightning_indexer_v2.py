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


class LightningIndexerV2Window(NamedTuple):
    block_table: torch.Tensor
    valid_pages: torch.Tensor
    visible_lengths: torch.Tensor
    metadata: torch.Tensor


class LightningIndexerV2Plan(NamedTuple):
    cu_query: torch.Tensor
    windows: tuple[LightningIndexerV2Window, ...]


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
    """Represent each query row as one request with its own rank-local causal prefix."""
    rows = request_rows.numel()
    cu_query = torch.arange(rows + 1, dtype=torch.int32, device=block_table.device)
    pages = block_table.index_select(0, request_rows)
    valid_pages = (pages >= first_valid_page) & (pages < num_cache_pages)
    # The kernel may load any page inside the visible prefix. Redirect holes
    # to allocated storage and keep their logical positions for score masking.
    pages = torch.where(valid_pages, pages, 0).to(torch.int32)
    lengths = local_visible_lengths.reshape(rows)
    windows = []
    # Request every score within each window so invalid pages and arbitrary
    # native tie order cannot displace a valid candidate at the local cutoff.
    # Ascend 950 supports score-returning Top-K up to 8192, including partial windows.
    window_pages = _ASCEND_950_MAX_TOPK // page_size
    for start in range(0, pages.shape[1], window_pages):
        stop = min(start + window_pages, pages.shape[1])
        width = (stop - start) * page_size
        visible = (lengths - start * page_size).clamp(0, width)
        metadata = torch.ops._C_ascend.npu_lightning_indexer_v2_metadata(
            cu_query, visible, num_heads_q, width, metadata_size=metadata_size
        )
        windows.append(
            LightningIndexerV2Window(pages[:, start:stop].contiguous(), valid_pages[:, start:stop], visible, metadata)
        )
    return LightningIndexerV2Plan(cu_query, tuple(windows))


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
) -> tuple[torch.Tensor, torch.Tensor]:
    """Score every visible rank-local token with BF16 CANN LightningIndexerV2."""
    from vllm_ascend.ops.triton.sharded_indexer import restore_li_v2_scores

    rows = query.shape[0]
    page_size = key_cache.shape[1]
    capacity = sum(window.block_table.shape[1] * page_size for window in plan.windows)
    # Start in logical token order, including holes. A stable sort then selects
    # the earliest local ID on ties, independent of CANN's native Top-K order.
    dense = torch.full((rows, max(topk, capacity)), -torch.inf, device=query.device, dtype=torch.float32)
    q = query.contiguous()
    w = weights.float().contiguous()
    offset = 0
    for window in plan.windows:
        width = window.block_table.shape[1] * page_size
        indices, scores = torch.ops._C_ascend.npu_lightning_indexer_v2(
            q, key_cache, w, plan.cu_query, window.visible_lengths, window.block_table, window.metadata, width
        )
        restore_li_v2_scores(indices, scores, window.valid_pages, window.visible_lengths, dense, offset, page_size)
        # Release these outputs before allocating the next pair or the final sort buffers.
        del indices, scores
        offset += width
    best_scores, order = torch.sort(dense, dim=-1, descending=True, stable=True)
    best_scores = best_scores[:, :topk]
    best_ids = order[:, :topk].to(torch.int32).masked_fill_(best_scores == -torch.inf, -1)
    return best_ids.unsqueeze(1), best_scores.unsqueeze(1)
