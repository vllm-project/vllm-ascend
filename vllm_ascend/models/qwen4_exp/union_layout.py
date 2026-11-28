# SPDX-License-Identifier: Apache-2.0
"""Step-local page layout for BF16 Union prefill over a single C8 cache."""

from dataclasses import dataclass, replace

import torch

from vllm_ascend.ops.triton.qwen4_exp.union_cache import prepare_union_pages

from .attention_dispatch import QSARequestPlan

# Bound the BF16 workspace by scheduled work. A full chunk from one or two
# requests can cover the end of a 128K prompt and the start of the next one.
_HISTORY_TO_QUERY_RATIO = 4
_LONG_PROMPT_MAX_REQUESTS = 2
_FULL_CHUNK_TOKENS = 8192
_LONG_PROMPT_TOKENS = 128 * 1024


def fits_dense_page_budget(num_requests: int, query_tokens: int, history_tokens: int) -> bool:
    budget = _HISTORY_TO_QUERY_RATIO * query_tokens
    if num_requests <= _LONG_PROMPT_MAX_REQUESTS and query_tokens >= _FULL_CHUNK_TOKENS:
        budget = max(budget, _LONG_PROMPT_TOKENS + _FULL_CHUNK_TOKENS)
    return 0 < history_tokens <= budget


@dataclass(frozen=True)
class UnionPageLayout:
    pages: torch.Tensor
    current_rows: torch.Tensor
    block_table: torch.Tensor
    request_plan: QSARequestPlan


def prepare_union_layout(
    num_pages: int,
    slots: torch.Tensor,
    indices: torch.Tensor,
    positions: torch.Tensor,
    lengths: torch.Tensor,
    requests: torch.Tensor,
    table: torch.Tensor,
    plan: QSARequestPlan,
    dense_layouts: dict[tuple[int, int, tuple[int, ...]], UnionPageLayout] | None,
) -> UnionPageLayout:
    """Share dense mappings within one attention group's step. Sparse indices vary by layer."""
    if dense_layouts is not None:
        signature = (num_pages, lengths.data_ptr(), lengths.stride())
        cached = dense_layouts.get(signature)
        if cached is not None:
            return cached

    pages, current_rows, page_map = prepare_union_pages(
        num_pages,
        slots,
        indices,
        positions,
        lengths,
        requests,
        table,
        table.shape[1] if dense_layouts is not None else 0,
    )

    def remap(source: torch.Tensor) -> torch.Tensor:
        return page_map[source.long().clamp(0, num_pages - 1)].clamp_min(0)

    batches = tuple(replace(batch, block_table=remap(batch.block_table)) for batch in plan.prepared_union_batches)
    layout = UnionPageLayout(pages, current_rows, remap(table), replace(plan, prepared_union_batches=batches))
    if dense_layouts is not None:
        dense_layouts[signature] = layout
    return layout
