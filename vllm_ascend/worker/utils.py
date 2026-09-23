from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from itertools import product as iprod
from typing import Any

import numpy as np
import torch
from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import largest_power_of_2_divisor
from vllm.utils.torch_utils import async_tensor_h2d, get_dtype_size
from vllm.v1.core.kv_cache_utils import KVCacheBlockCopy
from vllm.v1.kv_cache_interface import FullAttentionSpec, MLAAttentionSpec
from vllm.v1.worker.utils import AttentionGroup, KVBlockZeroer

from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec, get_kv_cache_compression_ratio
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
        assert tensor.shape[0] % num_blocks == 0
        kernel_blocks_per_block = tensor.shape[0] // num_blocks
        # Page-strided MLA views are non-contiguous, so a flat ``view`` would
        # either fail or materialize a copy. Split only dim 0 to preserve the
        # original strided storage while copying every kernel block in a page.
        blocks = tensor.unflatten(0, (num_blocks, kernel_blocks_per_block))
        source_blocks = torch.index_select(blocks, 0, src_indices)
        blocks.index_copy_(0, dst_indices, source_blocks)


def get_single_raw_mla_backing(raw_cache: object) -> torch.Tensor | None:
    """Extract one raw MLA backing from either allocator representation.

    Pure MLA allocations use a one-element tuple so generic K/V unpacking does
    not mistake the backing for a `(key, value)` pair. Hybrid/shared pools can
    instead store a bare tensor slice for an MLA layer.
    """
    if isinstance(raw_cache, torch.Tensor):
        return raw_cache
    if (
        isinstance(raw_cache, tuple)
        and len(raw_cache) == 1
        and isinstance(raw_cache[0], torch.Tensor)
    ):
        return raw_cache[0]
    return None


def mla_spec_supports_single_raw_backing(spec: AscendMLAAttentionSpec) -> bool:
    """Whether an Ascend MLA spec can use the worker single-backing protocol.

    This is a ModelRunner allocation policy, not cache geometry, so it lives in
    the worker layer rather than on the public KV cache spec.
    """
    return (
        get_kv_cache_compression_ratio(spec) == 1
        and spec.model_version is None
        and not spec.indexes_kv_by_block_stride
    )


def row_major_strides(shape: Sequence[int]) -> tuple[int, ...]:
    """Return row-major strides for a shape without allocating tensor storage."""
    strides = [1] * len(shape)
    for dim in range(len(shape) - 2, -1, -1):
        strides[dim] = strides[dim + 1] * shape[dim + 1]
    return tuple(strides)


def make_page_strided_cache_view(
    raw_tensor: torch.Tensor,
    shape: Sequence[int],
    dtype: torch.dtype,
    page_size_bytes: int,
    offset_bytes: int = 0,
) -> torch.Tensor:
    """Create a first-axis page-strided view over a raw cache allocation."""
    dtype_size = get_dtype_size(dtype)
    strides = row_major_strides(tuple(shape))
    storage_offset_bytes = raw_tensor.storage_offset() * raw_tensor.element_size() + offset_bytes
    assert storage_offset_bytes % dtype_size == 0
    return torch.as_strided(
        raw_tensor.view(dtype),
        size=tuple(shape),
        stride=(page_size_bytes // dtype_size, *strides[1:]),
        storage_offset=storage_offset_bytes // dtype_size,
    )


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
        offset = block_id.to(tl.int64) * PAGE_SIZE_EL + chunk_index.to(tl.int64) * BLOCK_SIZE
        cols = tl.arange(0, BLOCK_SIZE).to(tl.int64)
        tl.store(ptr + offset + cols, tl.zeros([BLOCK_SIZE], dtype=tl.int32))


def _component_views_share_slot(kv_cache: object, spec: FullAttentionSpec) -> bool:
    """Whether MLA component views alias one component-major physical page."""

    if not isinstance(spec, MLAAttentionSpec) or not isinstance(kv_cache, tuple) or len(kv_cache) != 2:
        return False

    nope, rope = kv_cache
    if not isinstance(nope, torch.Tensor) or not isinstance(rope, torch.Tensor):
        return False

    storage_ptr = nope.untyped_storage().data_ptr()
    first_offset = nope.storage_offset()
    component_elements = nope[0].numel()
    return (
        rope.untyped_storage().data_ptr() == storage_ptr
        and rope.stride(0) == nope.stride(0)
        and not nope.is_contiguous()
        and not rope.is_contiguous()
        and rope.storage_offset() - first_offset == component_elements
        and component_elements < nope.stride(0)
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
        PAGE_SIZE_EL accounts for this ratio so that
        ``block_id * PAGE_SIZE_EL`` lands at the correct offset.

        Only AttentionSpec layers are processed; Mamba layers are skipped.
        """
        seen_ptrs: set[int] = set()
        seg_addrs: list[int] = []
        page_size_el: int | None = None

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
                kv_cache = static_forward_context[layer_name].kv_cache
                # Fused MLA由单一tensor表示；component-major MLA的两个view同样共享一个物理page，从nope起点清理一次即可。
                # legacy K/V协议仍逐个component清理。
                if _component_views_share_slot(kv_cache, spec):
                    kv_tensors = (kv_cache[0],)
                elif isinstance(kv_cache, torch.Tensor):
                    kv_tensors = (kv_cache,)
                else:
                    kv_tensors = kv_cache
                for kv in kv_tensors:
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
                    if page_size_el is None:
                        page_size_el = cur_page_el
                    else:
                        assert page_size_el == cur_page_el, f"Non-uniform page sizes: {page_size_el} vs {cur_page_el}"

                    block_stride_bytes = cur_bytes
                    outer_dims = [d for d in range(block_dim) if kv.stride(d) * el > block_stride_bytes]
                    outer_strides = [kv.stride(d) * el for d in outer_dims]
                    for outer in iprod(*(range(kv.shape[d]) for d in outer_dims)):
                        off_bytes = sum(i * s for i, s in zip(outer, outer_strides))
                        seg_addrs.append(dp + off_bytes)

        if not seg_addrs or page_size_el is None:
            self._meta = None
            return

        # _zero_kv_blocks_kernel will use int64 zeros, to meet the UB size, we use blk_size=64B/8B=8192
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
            N_SEGS=n_segs,
            PAGE_SIZE_EL=page_size_el,
            BLOCK_SIZE=blk_size,
            GRID_SIZE=grid,
        )
