from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from typing import Any

import torch
from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import largest_power_of_2_divisor
from vllm.utils.torch_utils import get_dtype_size
from vllm.v1.core.kv_cache_utils import KVCacheBlockCopy
from vllm.v1.kv_cache_interface import FullAttentionSpec, MLAAttentionSpec
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
        if isinstance(entry, torch.Tensor):
            tensors = entry.unbind(0) if entry.ndim == 5 and entry.shape[0] == 2 else (entry,)
        else:
            tensors = entry
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
    for tensor in cache_tensors:
        assert tensor.device == device
        assert tensor.shape[0] % num_blocks == 0
        # FP8 is storage here. NPU stack/index operators need byte views;
        # same-size bitcasts preserve the non-contiguous page geometry.
        if tensor.dtype == torch.float8_e4m3fn:
            tensor = tensor.view(torch.int8)
        kernel_blocks_per_block = tensor.shape[0] // num_blocks
        # Page-strided MLA views are non-contiguous, so a flat ``view`` would
        # either fail or materialize a copy. Split only dim 0 to preserve the
        # original strided storage while copying every kernel block in a page.
        blocks = tensor.unflatten(0, (num_blocks, kernel_blocks_per_block))
        source_blocks = torch.stack([blocks[copy.src_block_id] for copy in kv_cache_block_copies])
        for index, copy in enumerate(kv_cache_block_copies):
            blocks[copy.dst_block_id].copy_(source_blocks[index])


def get_single_raw_mla_backing(raw_cache: object) -> torch.Tensor | None:
    """Extract one raw MLA backing from either allocator representation.

    Pure MLA allocations use a one-element tuple so generic K/V unpacking does
    not mistake the backing for a `(key, value)` pair. Hybrid/shared pools can
    instead store a bare tensor slice for an MLA layer.
    """
    if isinstance(raw_cache, torch.Tensor):
        return raw_cache
    if isinstance(raw_cache, tuple) and len(raw_cache) == 1 and isinstance(raw_cache[0], torch.Tensor):
        return raw_cache[0]
    return None


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
    seg_page_sizes_ptr,
    block_ids_ptr,
    n_blocks,
    seg_page_strides_ptr,
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

    Segment payload sizes and physical page strides are independent.
    Payload sizes bound writes; page strides advance scheduler block IDs
    past neighboring caches and padding.

    seg_addrs_ptr holds absolute byte addresses (int64) for each segment,
    allowing segments to live in different CUDA allocations.

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
        page_size_el = tl.load(seg_page_sizes_ptr + seg_index)
        if chunk_index < page_size_el // BLOCK_SIZE:
            block_id = tl.load(block_ids_ptr + block_index)
            seg_addr = tl.load(seg_addrs_ptr + seg_index)
            ptr = tl.cast(seg_addr, tl.pointer_type(tl.int32))
            page_stride_el = tl.load(seg_page_strides_ptr + seg_index)
            offset = block_id.to(tl.int64) * page_stride_el.to(tl.int64) + chunk_index.to(tl.int64) * BLOCK_SIZE
            cols = tl.arange(0, BLOCK_SIZE).to(tl.int64)
            tl.store(ptr + offset + cols, tl.zeros([BLOCK_SIZE], dtype=tl.int32))


def _component_views_share_slot(kv_cache: object, spec: FullAttentionSpec) -> bool:
    """Whether already-materialized MLA nope/rope views alias one physical page.

    This is a post-reshape physical-aliasing check used by block zeroing and
    COW. It deliberately differs from single-raw-backing eligibility:
    eligibility decides how a cache may be allocated, while this function only
    recognizes component-major views that were created over the same backing.
    Legacy contiguous K/V tuples and independently backed tensors return false.
    """

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


def _is_page_strided_mla_cache(kv: torch.Tensor, spec: FullAttentionSpec, ratio: int) -> bool:
    """Whether an MLA view advances by one physical page on its first axis."""

    return (
        isinstance(spec, MLAAttentionSpec)
        and kv.ndim > 0
        and kv.shape[0] > 0
        and kv.shape[0] % ratio == 0
        and not kv.is_contiguous()
        and kv.stride(0) > kv[0].numel()
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
        self._meta: tuple[torch.Tensor, torch.Tensor, int, int, int] | None = None
        self._seg_page_strides: torch.Tensor | None = None
        self._id_cap: int = 0
        self._ids_pinned: torch.Tensor | None = None
        self._ids_gpu: torch.Tensor | None = None

    def init_meta(
        self,
        attn_groups_iter: Iterable["AttentionGroup"],
        kernel_block_sizes: list[int],
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
        Each segment's page size accounts for this ratio so that
        ``block_id * page_size_el`` lands at the correct offset.

        Only AttentionSpec layers are processed; Mamba layers are skipped.
        """
        seen_ptrs: set[int] = set()
        seg_addrs: list[int] = []
        seg_page_sizes: list[int] = []
        seg_page_strides: list[int] = []

        for group in attn_groups_iter:
            spec = group.kv_cache_spec
            if not isinstance(spec, FullAttentionSpec):
                continue
            if group.kv_cache_group_id >= len(kernel_block_sizes):
                continue
            kernel_bs = kernel_block_sizes[group.kv_cache_group_id]
            assert kernel_bs > 0 and spec.block_size % kernel_bs == 0
            ratio = spec.block_size // kernel_bs

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
                    if cache_dtype == "mxfp8" and len(kv_tensors) == 4:
                        # V scales are checkpoint constants, initialized before
                        # capture. Clearing a recycled block must preserve them.
                        kv_tensors = kv_tensors[:3]
                for kv in kv_tensors:
                    dp = kv.data_ptr()
                    if dp in seen_ptrs:
                        continue
                    seen_ptrs.add(dp)

                    el = kv.element_size()
                    if _is_page_strided_mla_cache(kv, spec, ratio):
                        physical_page_bytes = kv.stride(0) * el * ratio
                        assert physical_page_bytes % 4 == 0
                        physical_page_el = physical_page_bytes // 4
                        seg_addrs.append(dp)
                        seg_page_sizes.append(physical_page_el)
                        seg_page_strides.append(physical_page_el)
                        continue

                    payload_bytes = kv[0].numel() * el
                    stride_bytes = kv.stride(0) * el
                    assert kv[0].is_contiguous(), "KV block payload must be contiguous"
                    assert payload_bytes % 4 == 0 and stride_bytes % 4 == 0
                    # A physical stride may include other caches and padding.
                    # Only clear the payload. Contiguous subblocks can be
                    # coalesced; strided subblocks need one segment each.
                    contiguous = payload_bytes == stride_bytes
                    for subblock in range(1 if contiguous else ratio):
                        seg_addrs.append(dp + subblock * stride_bytes)
                        seg_page_sizes.append(payload_bytes * (ratio if contiguous else 1) // 4)
                        seg_page_strides.append(stride_bytes * ratio // 4)

        if not seg_addrs:
            self._meta = None
            self._seg_page_strides = None
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
        self._seg_page_strides = torch.tensor(seg_page_strides, dtype=torch.int64, device=self.device)
        self._meta = (
            torch.tensor(seg_addrs, dtype=torch.uint64, device=self.device),
            torch.tensor(seg_page_sizes, dtype=torch.int64, device=self.device),
            max_page_size_el // blk_size,
            blk_size,
            len(seg_addrs),
        )

    def zero_block_ids(self, block_ids: list[int]) -> None:
        """Zero the KV cache memory for the given block IDs."""
        if not block_ids or self._meta is None:
            return
        seg_addrs, seg_page_sizes, max_chunks, blk_size, n_segs = self._meta
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
        total_work = n_blocks * n_segs * max_chunks
        grid = min(total_work, get_vectorcore_num()) if total_work > 0 else 0
        if grid == 0:
            return
        _zero_kv_blocks_kernel[(grid,)](
            seg_addrs,
            seg_page_sizes,
            idx,
            n_blocks,
            self._seg_page_strides if self._seg_page_strides is not None else seg_page_sizes,
            N_SEGS=n_segs,
            MAX_CHUNKS=max_chunks,
            BLOCK_SIZE=blk_size,
            GRID_SIZE=grid,
        )
