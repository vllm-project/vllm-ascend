# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Byte layout of a paged HiSparse destination for one SFA Ready message."""

from dataclasses import dataclass


@dataclass(frozen=True)
class PageBuffer:
    base: int
    payload: int
    stride: int
    pages: int

    def address(self, page: int) -> int:
        if not 0 <= page < self.pages:
            raise ValueError(f"KV page index {page} outside the registered {self.pages} pages")
        return self.base + page * self.stride

    @classmethod
    def from_tensor(cls, tensor, pages: int):
        if pages <= 0 or tensor.shape[0] < pages or tensor.shape[0] % pages:
            raise ValueError("HiSparse logical pages do not tile the destination tensor")
        factor = tensor.shape[0] // pages
        payload = tensor[0].numel() * tensor.element_size()
        stride = tensor.stride(0) * tensor.element_size()
        if not tensor[0].is_contiguous() or stride < payload or (factor > 1 and payload != stride):
            raise ValueError("HiSparse destination has gaps inside one logical page")
        return cls(tensor.data_ptr(), payload * factor, stride * factor, pages)


def main_components(bases, sizes, shapes, latent_bytes, rope_bytes):
    """Normalize a split pair or a flat packed slot without moving Prefill data."""
    if not (len(bases) == len(sizes) == len(shapes)) or len(bases) not in (1, 2):
        raise ValueError("Inconsistent Prefill Main component metadata")
    if len(bases) == 1 and len(shapes[0]) == 1:
        if latent_bytes <= 0 or rope_bytes <= 0 or latent_bytes + rope_bytes > sizes[0]:
            raise ValueError("Packed Prefill MLA slot is smaller than its latent and RoPE payload")
        return [bases[0], bases[0] + latent_bytes], [latent_bytes, rope_bytes], [sizes[0], sizes[0]]
    # Native C8 rows include BF16 RoPE and FP32 quantization scales. For
    # a row-major tensor the destination page size is authoritative.
    if len(bases) == 2 and list(sizes) != [latent_bytes, rope_bytes]:
        raise ValueError("Prefill and HiSparse Main component widths differ")
    return bases, sizes, sizes


def main_spans(host, source_bases, source_sizes, page_tokens, page_ids, valid_tokens, *, source_strides=None):
    """Map one row-major or two component-major pages into interleaved Host rows."""
    if (
        len(source_bases) not in (1, 2)
        or len(source_sizes) != len(source_bases)
        or page_tokens <= 0
        or any(size <= 0 or size % page_tokens for size in source_sizes)
        or host.payload != sum(source_sizes)
        or not 0 < valid_tokens <= page_tokens
    ):
        raise ValueError("HiSparse layerwise Main KV sizes or valid row count disagree")
    dest_page, source_page = page_ids
    if source_page < 0:
        raise ValueError("Negative Prefill Main page ID")
    local = host.address(dest_page)
    strides = source_sizes if source_strides is None else source_strides
    chunk_bases = [addr + source_page * stride for addr, stride in zip(source_bases, strides, strict=True)]
    if len(chunk_bases) == 1:
        yield local, chunk_bases[0], host.payload // page_tokens * valid_tokens
        return
    first_base, second_base = chunk_bases
    first, second = (size // page_tokens for size in source_sizes)
    for token in range(valid_tokens):
        row = local + token * (first + second)
        yield row, first_base + token * first, first
        yield row + first, second_base + token * second, second


def indexer_spans(destination, source_bases, source_sizes, page_tokens, page_ids, valid_tokens):
    """Keep Indexer and scale in separate rank-local NPU buffers."""
    dest_page, source_page = page_ids
    if source_page < 0 or not 0 < valid_tokens <= page_tokens:
        raise ValueError("Invalid source Indexer page or valid row count")
    for dest, base, size in zip(destination, source_bases, source_sizes, strict=True):
        if size != dest.payload or size <= 0 or size % page_tokens:
            raise ValueError("HiSparse layerwise Indexer or scale page bytes disagree")
        yield dest.address(dest_page), base + source_page * size, size // page_tokens * valid_tokens


def page_offsets(start: int, count: int, rank: int, width: int) -> range:
    """Assign absolute page ordinals to TP writers sharing one Host allocation."""
    if start < 0 or count < 0 or not 0 <= rank < width:
        raise ValueError("Invalid page range or Host writer rank")
    return range(start + (rank - start) % width, start + count, width)


def check_page_range(source_ids, start, dest_count, source_pages, page_tokens, computed):
    """Reject unallocated or uncomputed rows before issuing a transfer."""
    if start < 0 or start + len(source_ids) > dest_count:
        raise ValueError("Destination page range exceeds the request allocation")
    if source_ids and (start + len(source_ids) - 1) * page_tokens >= computed:
        raise ValueError("Source pages extend beyond the computed prefix")
    if any(type(page) is not int or not 0 <= page < source_pages for page in source_ids):
        raise ValueError("Source page ID is outside its registered allocation")
