# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable, Hashable
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class QLIMetadataConfig:
    """The layer-invariant CANN V2 contract for one indexer cache group."""

    num_heads_q: int
    head_dim: int
    quant_mode: int
    topk: int = 2048

    def build(self, op: Callable[..., torch.Tensor], query_start_loc: torch.Tensor, seq_lens: torch.Tensor):
        return op(
            num_heads_q=self.num_heads_q,
            num_heads_k=1,
            head_dim=self.head_dim,
            topk=self.topk,
            quant_mode=self.quant_mode,
            cu_seqlens_q=query_start_loc,
            seqused_k=seq_lens,
            batch_size=seq_lens.numel(),
            max_seqlen_q=-1,
            max_seqlen_k=-1,
            layout_q="TND",
            layout_k="PA_BBND",
            mask_mode=3,
            cmp_ratio=1,
        )


@dataclass(frozen=True)
class QLIMetadata:
    config: QLIMetadataConfig
    query_start_loc: torch.Tensor
    seq_lens: torch.Tensor
    metadata: torch.Tensor


class QLIMetadataBuffers:
    """Refresh graph-stable QLI inputs once per build, shared by all group layers.

    A key identifies storage, not cached values. Every build runs the producer,
    even when the lengths' shapes and addresses have not changed. Builders use
    the slot-mapping address to isolate concurrently prepared draft steps.
    """

    def __init__(self, config: QLIMetadataConfig, max_num_reqs: int):
        self.config = config
        self.max_num_reqs = max_num_reqs
        self._buffers: dict[Hashable, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

    def build(
        self,
        key: Hashable,
        query_start_loc: torch.Tensor,
        seq_lens: torch.Tensor,
        op: Callable[..., torch.Tensor],
    ) -> QLIMetadata:
        num_reqs = seq_lens.numel()
        if query_start_loc.ndim != 1 or seq_lens.ndim != 1 or query_start_loc.numel() != num_reqs + 1:
            raise ValueError("QLI requires one cumulative query boundary per request plus the initial zero")
        if num_reqs > self.max_num_reqs:
            raise ValueError(f"QLI request count {num_reqs} exceeds capacity {self.max_num_reqs}")
        buffers = self._buffers.get(key)
        if buffers is None:
            queries = torch.empty(self.max_num_reqs + 1, dtype=torch.int32, device=query_start_loc.device)
            keys = torch.empty(self.max_num_reqs, dtype=torch.int32, device=seq_lens.device)
        else:
            queries, keys, _ = buffers
        query_view, key_view = queries[: num_reqs + 1], keys[:num_reqs]
        query_view.copy_(query_start_loc)
        key_view.copy_(seq_lens)
        result = self.config.build(op, query_view, key_view)
        if buffers is None:
            output = torch.empty_like(result)
            self._buffers[key] = queries, keys, output
        else:
            output = buffers[2]
            if output.shape != result.shape or output.dtype != result.dtype:
                raise RuntimeError("QLI metadata output layout changed after initialization")
        # FULL graphs retain this address. Rebuilds before replay must update it
        # in place instead of returning a newly allocated operator output.
        output.copy_(result)
        return QLIMetadata(self.config, query_view, key_view, output)


def get_qli_metadata(
    prepared: QLIMetadata | None,
    config: QLIMetadataConfig,
    query_ends: torch.Tensor,
    seq_lens: torch.Tensor,
    op: Callable[..., torch.Tensor],
) -> QLIMetadata:
    if prepared is not None:
        if prepared.config != config:
            raise ValueError("QLI metadata configuration does not match the consuming indexer")
        return prepared
    # Direct adaptor callers without a builder retain their inline producer.
    query_start_loc = F.pad(query_ends, (1, 0), value=0)
    return QLIMetadata(config, query_start_loc, seq_lens, config.build(op, query_start_loc, seq_lens))
