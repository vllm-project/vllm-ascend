from collections.abc import Iterable
from itertools import product as iprod
from typing import Any

import torch
from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import largest_power_of_2_divisor
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVQuantMode,
    get_kv_quant_mode,
)
from vllm.v1.worker.utils import AttentionGroup, KVBlockZeroer

from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num


def _get_spec_cache_dtype_str(spec: FullAttentionSpec, cache_dtype: str) -> str:
    """Return the cache dtype that describes this specific attention group."""
    spec_cache_dtype = getattr(spec, "cache_dtype_str", None)
    if spec_cache_dtype is not None:
        return spec_cache_dtype
    if spec.kv_quant_mode == KVQuantMode.NONE and get_kv_quant_mode(cache_dtype) != KVQuantMode.NONE:
        return "auto"
    return cache_dtype


@triton.jit
def _zero_kv_blocks_kernel(
    seg_addrs_ptr,
    seg_block_strides_ptr,
    seg_page_sizes_ptr,
    block_ids_ptr,
    n_blocks,
    N_SEGS: tl.constexpr,
    MAX_CHUNKS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    GRID_SIZE: tl.constexpr,
):
    """Zero KV cache blocks across all segments in a single launch.

    Each segment is a contiguous region of one block's data.  For backends
    where blocks are outermost (block_dim=0) there is one segment per
    buffer.  For backends where K/V is outermost (block_dim=1) there are
    two segments per buffer (one for K, one for V).

    Segments may have different block strides and page sizes, for example
    mixed BF16/FP8 attention caches plus an indexer cache. The block stride
    locates a logical block while the page size limits the bytes cleared from
    that block.

    Programs are mapped as (block_index, seg_index, chunk_index).
    """
    pid = tl.program_id(0)
    work_per_block = N_SEGS * MAX_CHUNKS
    total_work = n_blocks * work_per_block
    for work_idx in range(pid, total_work, GRID_SIZE):
        block_index = work_idx // work_per_block
        remainder = work_idx % work_per_block
        seg_index = remainder // MAX_CHUNKS
        chunk_index = remainder % MAX_CHUNKS
        block_id = tl.load(block_ids_ptr + block_index)
        seg_addr = tl.load(seg_addrs_ptr + seg_index)
        block_stride_el = tl.load(seg_block_strides_ptr + seg_index)
        page_size_el = tl.load(seg_page_sizes_ptr + seg_index)
        ptr = tl.cast(seg_addr, tl.pointer_type(tl.int32))
        offset = block_id.to(tl.int64) * block_stride_el + chunk_index.to(tl.int64) * BLOCK_SIZE
        cols = tl.arange(0, BLOCK_SIZE).to(tl.int64)
        active = chunk_index < page_size_el // BLOCK_SIZE
        tl.store(
            ptr + offset + cols,
            tl.zeros([BLOCK_SIZE], dtype=tl.int32),
            mask=active,
        )


class AscendKVBlockZeroer(KVBlockZeroer):
    """Manages efficient zeroing of KV cache blocks via a Triton kernel.

    Call :meth:`init_meta` once after KV caches are allocated to precompute
    segment addresses, then call :meth:`zero_block_ids` each step to zero
    newly-allocated blocks.
    """

    def __init__(self, device: torch.device, pin_memory: bool) -> None:
        self.device = device
        self.pin_memory = pin_memory
        self._meta: tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int, int] | None = None
        self._id_cap: int = 0
        self._ids_pinned: torch.Tensor | None = None
        self._ids_gpu: torch.Tensor | None = None

    def init_meta(
        self,
        attn_groups_iter: Iterable["AttentionGroup"],
        kernel_block_sizes: list[list[int]],
        cache_dtype: str,
        runner_only_attn_layers: set[str],
        static_forward_context: dict[str, Any],
    ) -> None:
        """One-time precomputation for zero_block_ids.

        Builds absolute-address table for the Triton zeroing kernel.
        Each entry is the absolute byte address of a segment start on the
        GPU, so segments in different CUDA allocations work correctly.

        Block IDs from the scheduler reference logical blocks whose size
        may differ from the kernel block size (virtual block splitting).
        Each virtual block is represented by an independent segment so the
        logical block stride and the page span to clear stay independent.

        Only AttentionSpec layers are processed; Mamba layers are skipped.
        """
        seen_ptrs: set[int] = set()
        seg_addrs: list[int] = []
        seg_block_strides: list[int] = []
        seg_page_sizes: list[int] = []

        for group in attn_groups_iter:
            spec = group.kv_cache_spec
            if not isinstance(spec, FullAttentionSpec):
                continue
            if group.kv_cache_group_id >= len(kernel_block_sizes):
                continue
            kernel_bs = kernel_block_sizes[group.kv_cache_group_id][0]
            assert spec.block_size % kernel_bs == 0
            ratio = spec.block_size // kernel_bs
            packed_block_dim = group.backend.get_kv_cache_block_dim(
                kernel_bs,
                spec.num_kv_heads,
                spec.head_size,
                cache_dtype_str=_get_spec_cache_dtype_str(spec, cache_dtype),
            )

            for layer_name in group.layer_names:
                if layer_name in runner_only_attn_layers:
                    continue
                kv_cache = static_forward_context[layer_name].kv_cache
                if isinstance(kv_cache, list) and len(kv_cache) == 1 and isinstance(kv_cache[0], (tuple, list)):
                    # Some model runners add a singleton virtual-engine wrapper
                    # around the separated physical cache tensors.
                    kv_cache = kv_cache[0]

                cache_tensors: tuple[tuple[torch.Tensor, int], ...]
                if isinstance(kv_cache, torch.Tensor):
                    cache_tensors = ((kv_cache, packed_block_dim),)
                elif isinstance(kv_cache, (tuple, list)) and all(isinstance(kv, torch.Tensor) for kv in kv_cache):
                    # Ascend allocates K/V (and sparse index/scale caches)
                    # separately for P/D disaggregation. Their physical block
                    # dimension is outermost even when the backend advertises
                    # a packed logical shape.
                    cache_tensors = tuple((kv, 0) for kv in kv_cache)
                else:
                    continue

                for kv, block_dim in cache_tensors:
                    dp = kv.data_ptr()
                    if dp in seen_ptrs:
                        continue
                    seen_ptrs.add(dp)

                    el = kv.element_size()
                    block_stride_bytes = kv.stride(block_dim) * el
                    assert block_stride_bytes % 4 == 0
                    assert kv.shape[block_dim] % ratio == 0

                    outer_dims = [dim for dim in range(block_dim) if kv.stride(dim) * el > block_stride_bytes]
                    outer_strides = [kv.stride(dim) * el for dim in outer_dims]
                    inner_dims = [dim for dim in range(kv.ndim) if dim != block_dim and dim not in outer_dims]
                    kernel_page_bytes = el + sum((kv.shape[dim] - 1) * kv.stride(dim) * el for dim in inner_dims)
                    assert kernel_page_bytes % 4 == 0
                    logical_block_stride_bytes = block_stride_bytes * ratio

                    for outer in iprod(*(range(kv.shape[dim]) for dim in outer_dims)):
                        off_bytes = sum(index * stride for index, stride in zip(outer, outer_strides))
                        assert (dp + off_bytes) % 4 == 0
                        for virtual_index in range(ratio):
                            seg_addrs.append(dp + off_bytes + virtual_index * block_stride_bytes)
                            seg_block_strides.append(logical_block_stride_bytes // 4)
                            seg_page_sizes.append(kernel_page_bytes // 4)

        if not seg_addrs:
            self._meta = None
            return

        # _zero_kv_blocks_kernel will use int64 zeros, to meet the UB size, we use blk_size=64B/8B=8192
        max_page_size_el = max(seg_page_sizes)
        blk_size = min(
            min(largest_power_of_2_divisor(page_size_el) for page_size_el in seg_page_sizes),
            8192,
        )
        self._id_cap = 8192
        self._ids_pinned = torch.empty(
            self._id_cap,
            dtype=torch.int64,
            pin_memory=self.pin_memory,
        )
        self._ids_gpu = torch.empty(self._id_cap, dtype=torch.int64, device=self.device)
        self._meta = (
            torch.tensor(seg_addrs, dtype=torch.uint64, device=self.device),
            torch.tensor(seg_block_strides, dtype=torch.int64, device=self.device),
            torch.tensor(seg_page_sizes, dtype=torch.int64, device=self.device),
            max_page_size_el,
            blk_size,
            len(seg_addrs),
        )

    def zero_block_ids(self, block_ids: list[int]) -> None:
        """Zero the KV cache memory for the given block IDs."""
        if not block_ids or self._meta is None:
            return
        seg_addrs, seg_block_strides, seg_page_sizes, max_page_size_el, blk_size, n_segs = self._meta
        n_blocks = len(block_ids)
        if n_blocks > self._id_cap:
            self._id_cap = n_blocks * 2
            self._ids_pinned = torch.empty(
                self._id_cap,
                dtype=torch.int64,
                pin_memory=self.pin_memory,
            )
            self._ids_gpu = torch.empty(self._id_cap, dtype=torch.int64, device=self.device)
        assert self._ids_pinned is not None and self._ids_gpu is not None
        self._ids_pinned[:n_blocks].numpy()[:] = block_ids
        idx = self._ids_gpu[:n_blocks]
        idx.copy_(self._ids_pinned[:n_blocks], non_blocking=True)
        max_chunks = max_page_size_el // blk_size
        total_work = n_blocks * n_segs * max_chunks
        grid = min(total_work, get_vectorcore_num()) if total_work > 0 else 0
        if grid == 0:
            return
        _zero_kv_blocks_kernel[(grid,)](
            seg_addrs,
            seg_block_strides,
            seg_page_sizes,
            idx,
            n_blocks,
            N_SEGS=n_segs,
            MAX_CHUNKS=max_chunks,
            BLOCK_SIZE=blk_size,
            GRID_SIZE=grid,
        )
