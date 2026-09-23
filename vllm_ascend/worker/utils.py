from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from itertools import product as iprod
from typing import Any

import numpy as np
import torch
from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import largest_power_of_2_divisor
from vllm.utils.torch_utils import async_tensor_h2d
from vllm.v1.core.kv_cache_utils import KVCacheBlockCopy
from vllm.v1.kv_cache_interface import FullAttentionSpec
from vllm.v1.worker.utils import AttentionGroup, KVBlockZeroer

from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num


def copy_kv_cache_blocks_inplace(
    kv_caches: Iterable[torch.Tensor | Sequence[torch.Tensor | None] | None],
    num_blocks: int,
    kv_cache_block_copies: Sequence[KVCacheBlockCopy],
) -> None:
    """Copy logical cache blocks for Ascend's segmented cache layout.

    Unlike the upstream block-major allocation, an Ascend cache allocation can
    contain multiple block-indexed tensor segments. For example, Mamba stores
    all convolution states before all SSM states in the same storage. Treating
    that complete storage as ``[num_blocks, page_size]`` therefore copies the
    wrong byte ranges. Copy every tensor segment as ``num_blocks`` complete
    physical pages instead. A page may span multiple kernel-level cache blocks.
    """
    if not kv_cache_block_copies:
        return

    cache_tensors: list[torch.Tensor] = []
    seen_tensors: set[int] = set()
    for entry in kv_caches:
        if entry is None:
            continue
        tensors = (entry,) if isinstance(entry, torch.Tensor) else entry
        for tensor in tensors:
            if tensor is None:
                continue
            data_ptr = tensor.data_ptr()
            if data_ptr in seen_tensors:
                continue
            seen_tensors.add(data_ptr)
            cache_tensors.append(tensor)

    if not cache_tensors:
        return

    device = cache_tensors[0].device
    indices_np = np.array(
        [[copy.src_block_id, copy.dst_block_id] for copy in kv_cache_block_copies],
        dtype=np.int64,
    )
    indices = async_tensor_h2d(indices_np, device=device)
    src_indices, dst_indices = indices.unbind(dim=1)
    for tensor in cache_tensors:
        assert tensor.device == device
        assert tensor.numel() % num_blocks == 0
        blocks = tensor.view(num_blocks, -1)
        source_blocks = torch.index_select(blocks, 0, src_indices)
        blocks.index_copy_(0, dst_indices, source_blocks)


@contextmanager
def disable_compilation(model: torch.nn.Module) -> Iterator[None]:
    compilation_model = getattr(model, "model", model)
    if not hasattr(compilation_model, "do_not_compile"):
        yield
        return

    previous = compilation_model.do_not_compile
    compilation_model.do_not_compile = True
    try:
        yield
    finally:
        compilation_model.do_not_compile = previous


@triton.jit
def _zero_kv_blocks_kernel(
    seg_addrs_ptr,
    block_ids_ptr,
    n_blocks,
    seg_strides_ptr,
    seg_sizes_ptr,
    PAGE_STRIDED: tl.constexpr,
    N_SEGS: tl.constexpr,
    PAGE_SIZE_EL: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    GRID_SIZE: tl.constexpr,
):
    """Zero KV cache blocks across all segments in a single launch.

    Each segment is a contiguous region of one block's data.  For backends
    where blocks are outermost (block_dim=0) there is one segment per
    buffer.  For backends where K/V is outermost (block_dim=1) there are
    two segments per buffer (one for K, one for V).

    seg_addrs_ptr holds absolute byte addresses (int64) for each segment,
    allowing segments to live in different CUDA allocations.

    Programs are mapped as (block_index, seg_index, chunk_index).
    """
    pid = tl.program_id(0)
    chunks = PAGE_SIZE_EL // BLOCK_SIZE
    work_per_block = N_SEGS * chunks
    total_work = n_blocks * work_per_block
    for work_idx in range(pid, total_work, GRID_SIZE):
        block_index = work_idx // work_per_block
        remainder = work_idx % work_per_block
        seg_index = remainder // chunks
        chunk_index = remainder % chunks
        block_id = tl.load(block_ids_ptr + block_index)
        seg_addr = tl.load(seg_addrs_ptr + seg_index)
        ptr = tl.cast(seg_addr, tl.pointer_type(tl.int32))
        if PAGE_STRIDED:
            page_stride = tl.load(seg_strides_ptr + seg_index)
            payload_size = tl.load(seg_sizes_ptr + seg_index)
        else:
            page_stride = PAGE_SIZE_EL
            payload_size = PAGE_SIZE_EL
        chunk_offset = chunk_index.to(tl.int64) * BLOCK_SIZE
        offset = block_id.to(tl.int64) * page_stride + chunk_offset
        cols = tl.arange(0, BLOCK_SIZE).to(tl.int64)
        tl.store(
            ptr + offset + cols,
            tl.zeros([BLOCK_SIZE], dtype=tl.int32),
            mask=chunk_offset + cols < payload_size,
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
        self._meta: tuple[torch.Tensor, int, int, int] | None = None
        self._page_strided = False
        self._seg_strides: torch.Tensor | None = None
        self._seg_sizes: torch.Tensor | None = None
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
        page_strided: bool = False,
    ) -> None:
        """One-time precomputation for zero_block_ids.

        Builds absolute-address table for the Triton zeroing kernel.
        Each entry is the absolute byte address of a segment start on the
        GPU, so segments in different CUDA allocations work correctly.

        Block IDs from the scheduler reference logical blocks whose size
        may differ from the kernel block size (virtual block splitting).
        PAGE_SIZE_EL accounts for this ratio so that
        ``block_id * PAGE_SIZE_EL`` lands at the correct offset.

        ``page_strided`` accepts FlashMLA [P,S,576] and GQA [P,2H,S,64]
        views whose trailing dimensions are dense. Page pitch and owned payload
        are separate, so gaps between layers are never cleared.

        Only AttentionSpec layers are processed; Mamba layers are skipped.
        """
        seen_ptrs: set[int] = set()
        seg_addrs: list[int] = []
        page_size_el: int | None = None
        seg_strides: list[int] = []
        seg_sizes: list[int] = []
        self._page_strided = page_strided
        self._seg_strides = self._seg_sizes = None

        for group in attn_groups_iter:
            spec = group.kv_cache_spec
            if not isinstance(spec, FullAttentionSpec):
                continue
            if group.kv_cache_group_id >= len(kernel_block_sizes):
                continue
            kernel_bs = kernel_block_sizes[group.kv_cache_group_id][0]
            ratio = spec.block_size // kernel_bs
            block_dim = 0

            for layer_name in group.layer_names:
                if layer_name in runner_only_attn_layers:
                    continue
                kv_tuple = static_forward_context[layer_name].kv_cache
                flash_layer = getattr(getattr(static_forward_context[layer_name], "impl", None), "use_flash_mla", False)
                if page_strided and (flash_layer or isinstance(kv_tuple, torch.Tensor)):
                    planes = (kv_tuple,) if isinstance(kv_tuple, torch.Tensor) else kv_tuple
                    for kv in planes:
                        assert kv.ndim in (3, 4), "Expected page-major MLA or GQA cache"
                        assert spec.block_size % kernel_bs == 0 and ratio > 0
                        payload = 1
                        for dim in range(kv.ndim - 1, 0, -1):
                            assert kv.size(dim) == 1 or kv.stride(dim) == payload, "Cache payload must be dense"
                            payload *= kv.size(dim)
                        assert payload > 0 and kv.stride(0) >= payload
                        dp = kv.data_ptr()
                        payload_bytes = payload * kv.element_size()
                        pitch_bytes = kv.stride(0) * kv.element_size()
                        assert dp % 4 == 0 and payload_bytes % 4 == 0 and pitch_bytes % 4 == 0
                        payload_el = payload_bytes // 4
                        # A scheduler block can cover several physical kernel pages.
                        # Replicated drafts store every DCP lane consecutively for
                        # that same manager block, each with its own kernel pages.
                        physical_ratio = ratio * getattr(spec, "dcp_replication_size", 1)
                        # Keep each payload separate instead of zeroing across gaps.
                        for sub_block in range(physical_ratio):
                            seg_addrs.append(dp + sub_block * pitch_bytes)
                            seg_strides.append(pitch_bytes // 4 * physical_ratio)
                            seg_sizes.append(payload_el)
                        page_size_el = max(page_size_el or 0, payload_el)
                    continue
                assert len(kv_tuple) == 2, "K and V are not stored separately"
                for kv in kv_tuple:
                    block_dim = 0
                    dp = kv.data_ptr()
                    if dp in seen_ptrs:
                        continue
                    seen_ptrs.add(dp)

                    el = kv.element_size()
                    cur_bytes = kv.stride(block_dim) * el
                    assert cur_bytes % 4 == 0
                    kernel_block_el = cur_bytes // 4
                    cur_page_el = kernel_block_el * ratio
                    if page_size_el is None or page_strided:
                        page_size_el = max(page_size_el or 0, cur_page_el)
                    else:
                        assert page_size_el == cur_page_el, f"Non-uniform page sizes: {page_size_el} vs {cur_page_el}"

                    block_stride_bytes = cur_bytes
                    outer_dims = [d for d in range(block_dim) if kv.stride(d) * el > block_stride_bytes]
                    outer_strides = [kv.stride(d) * el for d in outer_dims]
                    for outer in iprod(*(range(kv.shape[d]) for d in outer_dims)):
                        off_bytes = sum(i * s for i, s in zip(outer, outer_strides))
                        seg_addrs.append(dp + off_bytes)
                        seg_strides.append(cur_page_el)
                        seg_sizes.append(cur_page_el)

        if not seg_addrs or page_size_el is None:
            self._meta = None
            return

        if page_strided:
            self._seg_strides = torch.tensor(seg_strides, dtype=torch.int64, device=self.device)
            self._seg_sizes = torch.tensor(seg_sizes, dtype=torch.int64, device=self.device)

        # Preserve the VA tile cap: 8192 int32 values occupy 32 KiB of UB.
        blk_size = min(largest_power_of_2_divisor(page_size_el), 8192)
        self._id_cap = 8192
        self._ids_pinned = torch.empty(
            self._id_cap,
            dtype=torch.int64,
            pin_memory=self.pin_memory,
        )
        self._ids_gpu = torch.empty(self._id_cap, dtype=torch.int64, device=self.device)
        self._meta = (
            torch.tensor(seg_addrs, dtype=torch.uint64, device=self.device),
            page_size_el,
            blk_size,
            len(seg_addrs),
        )

    def zero_block_ids(self, block_ids: list[int]) -> None:
        """Zero the KV cache memory for the given block IDs."""
        if not block_ids or self._meta is None:
            return
        seg_addrs, page_size_el, blk_size, n_segs = self._meta
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
        chunks = page_size_el // blk_size
        total_work = n_blocks * n_segs * chunks
        grid = min(total_work, get_vectorcore_num()) if total_work > 0 else 0
        if grid == 0:
            return
        _zero_kv_blocks_kernel[(grid,)](
            seg_addrs,
            idx,
            n_blocks,
            self._seg_strides,
            self._seg_sizes,
            PAGE_STRIDED=self._page_strided,
            N_SEGS=n_segs,
            PAGE_SIZE_EL=page_size_el,
            BLOCK_SIZE=blk_size,
            GRID_SIZE=grid,
        )
