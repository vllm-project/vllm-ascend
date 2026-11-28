# SPDX-License-Identifier: Apache-2.0
"""Synthetic A3 QLI inputs with the native paged/TND ABI."""

from dataclasses import dataclass
from itertools import accumulate
from math import ceil

import torch

HEADS = 32
WIDTH = 128
TOPK = 2048
INT64_MAX = (1 << 63) - 1
PHYSICAL_LAYOUTS = ("random", "contiguous-4")


@dataclass
class Fixture:
    query: torch.Tensor
    key: torch.Tensor
    weights: torch.Tensor
    query_scale: torch.Tensor
    key_scale: torch.Tensor
    query_ends: torch.Tensor
    key_lengths: torch.Tensor
    block_table: torch.Tensor
    cpu_pages: torch.Tensor
    cpu_query_ends: tuple[int, ...]
    cpu_key_lengths: tuple[int, ...]
    topk: int = TOPK

    def native_call(self):
        import torch_npu

        def call():
            return torch_npu.npu_quant_lightning_indexer(
                query=self.query,
                key=self.key,
                weights=self.weights,
                query_dequant_scale=self.query_scale,
                key_dequant_scale=self.key_scale,
                actual_seq_lengths_query=self.query_ends,
                actual_seq_lengths_key=self.key_lengths,
                block_table=self.block_table,
                query_quant_mode=0,
                key_quant_mode=0,
                layout_query="TND",
                layout_key="PA_BSND",
                sparse_count=self.topk,
                sparse_mode=3,
                pre_tokens=INT64_MAX,
                next_tokens=INT64_MAX,
            )

        return call

    def tilelang_call(self):
        from qli_tilelang import build_tilelang_call

        return build_tilelang_call(
            self.query,
            self.key,
            self.weights,
            self.query_scale,
            self.key_scale,
            self.query_ends,
            self.key_lengths,
            self.block_table,
            sparse_count=self.topk,
            sparse_mode=3,
        )

    def request_ranges(self):
        start = 0
        for end, length in zip(self.cpu_query_ends, self.cpu_key_lengths, strict=True):
            yield start, end, length
            start = end

    def visible_lengths(self):
        return tuple(
            max(0, length - (end - row - 1))
            for start, end, length in self.request_ranges()
            for row in range(start, end)
        )

    def logical_bytes(self):
        key_bytes = sum(self.cpu_key_lengths) * (WIDTH + self.key_scale.element_size())
        metadata_and_query = sum(
            tensor.numel() * tensor.element_size()
            for tensor in (
                self.query,
                self.query_scale,
                self.weights,
                self.query_ends,
                self.key_lengths,
                self.block_table,
            )
        )
        return key_bytes + metadata_and_query + self.query.size(0) * self.topk * 4


def case_config(name, batch, rows, context, page_size, seed, *, physical_layout="random"):
    return {
        "case": name,
        "batch": batch,
        "rows_per_request": rows,
        "context": context,
        "page_size": page_size,
        "seed": seed,
        "physical_layout": physical_layout,
        "input_source": "synthetic_paged_glm53_a3_c8",
    }


def _contiguous_four_page_table(batch, pages_per_request, seed):
    # Shuffle full four-page groups across the pool while preserving their order
    # within each group. Keep each request's short final group together too.
    full_groups, tail = divmod(pages_per_request, 4)
    generator = torch.Generator().manual_seed(seed)
    group_starts = 4 * torch.randperm(batch * full_groups, generator=generator, dtype=torch.int32)
    groups = (group_starts[:, None] + torch.arange(4, dtype=torch.int32)).reshape(batch, full_groups * 4)
    if not tail:
        return groups
    tail_starts = tail * torch.randperm(batch, generator=generator, dtype=torch.int32) + groups.numel()
    tails = tail_starts[:, None] + torch.arange(tail, dtype=torch.int32)
    return torch.cat((groups, tails), dim=1)


def make_fixture(case, device="npu"):
    batch = case["batch"]
    page_size = case["page_size"]
    query_rows = tuple(case.get("query_rows", [case["rows_per_request"]] * batch))
    key_lengths = tuple(case.get("key_lengths", [case["context"]] * batch))
    if batch < 1 or not 16 <= page_size <= 1024 or page_size % 16:
        raise ValueError("Expected a positive batch and a native page size divisible by 16")
    if len(query_rows) != batch or len(key_lengths) != batch:
        raise ValueError("Length arrays must have one entry per request")
    if min(query_rows) < 1 or min(key_lengths) < 1:
        raise ValueError("The synthetic fixture expects non-empty requests")
    physical_layout = case.get("physical_layout", "random")
    if physical_layout not in PHYSICAL_LAYOUTS:
        raise ValueError(f"Unknown QLI physical layout {physical_layout!r}")

    ends = tuple(accumulate(query_rows))
    generator = torch.Generator().manual_seed(case["seed"])
    # Generate the original page IDs first so the RNG stream and logical input
    # match between layouts, including repeated-page and all-tie stress cases.
    max_pages = ceil(max(key_lengths) / page_size)
    pages = batch * max_pages
    table = torch.randperm(pages, generator=generator, dtype=torch.int32).reshape(batch, max_pages)
    page_destination = None
    if physical_layout == "contiguous-4":
        grouped_table = _contiguous_four_page_table(batch, max_pages, case["seed"])
        page_destination = torch.empty(pages, dtype=torch.long)
        page_destination[table.flatten().long()] = grouped_table.flatten().long()
        table = grouped_table
    if case.get("repeat_pages", False):
        table[:, 1::4] = table[:, : max_pages - 1 : 4].clone()
        if max_pages > 2:
            table[:, max_pages // 2] = table[:, max_pages // 2 - 1]
    query = torch.randint(-127, 128, (ends[-1], HEADS, WIDTH), generator=generator, dtype=torch.int8)
    weights = (torch.rand((ends[-1], HEADS), generator=generator) - 0.45).half()
    query_scale = (torch.rand((ends[-1], HEADS), generator=generator) * 0.015 + 0.005).half()
    key = torch.empty((pages, page_size, 1, WIDTH), dtype=torch.int8, device=device)
    key_scale = torch.empty((pages, page_size, 1), dtype=torch.float16, device=device)
    if page_destination is not None:
        # Remap on the host so fixture setup needs only contiguous NPU copies.
        grouped_key = torch.empty(key.shape, dtype=key.dtype, device="cpu")
        grouped_scale = torch.empty(key_scale.shape, dtype=key_scale.dtype, device="cpu")
    for start in range(0, pages, 256):
        stop = min(start + 256, pages)
        key_chunk = torch.randint(-127, 128, (stop - start, page_size, 1, WIDTH), generator=generator, dtype=torch.int8)
        scale_chunk = (torch.rand((stop - start, page_size, 1), generator=generator) * 0.015 + 0.005).half()
        if page_destination is None:
            key[start:stop].copy_(key_chunk)
            key_scale[start:stop].copy_(scale_chunk)
        else:
            destinations = page_destination[start:stop]
            grouped_key.index_copy_(0, destinations, key_chunk)
            grouped_scale.index_copy_(0, destinations, scale_chunk)
    if page_destination is not None:
        key.copy_(grouped_key)
        key_scale.copy_(grouped_scale)
    if case.get("zero_scores", False):
        weights.zero_()
    return Fixture(
        query.to(device),
        key,
        weights.to(device),
        query_scale.to(device),
        key_scale,
        torch.tensor(ends, dtype=torch.int32, device=device),
        torch.tensor(key_lengths, dtype=torch.int32, device=device),
        table.to(device),
        table,
        ends,
        key_lengths,
    )


def reference_scores(fixture):
    """Replay A3 numeric stages with independent, dense CPU math a request at a time."""
    query = fixture.query.cpu().float()
    ws = (fixture.weights.cpu().float() * fixture.query_scale.cpu().float()).half().float()
    page_size = fixture.key.size(1)
    for request, (start, end, length) in enumerate(fixture.request_ranges()):
        keys = fixture.cpu_pages[request, : ceil(length / page_size)].long()
        device_keys = keys.to(fixture.key.device)
        k = fixture.key.index_select(0, device_keys).cpu().view(-1, WIDTH)[:length].float()
        scale = fixture.key_scale.index_select(0, device_keys).cpu().flatten()[:length].float()
        # The FP32 dot product is exact for signed INT8 values and 128 terms. Match
        # the original FixPipe conversion before the weighted second matmul.
        qk = (query[start:end].reshape(-1, WIDTH) @ k.T).view(end - start, HEADS, length)
        qk = (qk.relu_() * (2.0**-10)).half().float()
        score = (qk * ws[start:end, :, None]).sum(1) * scale
        for local_row in range(end - start):
            visible = max(0, length - (end - start - local_row - 1))
            yield score[local_row, :visible]
