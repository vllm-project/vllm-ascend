from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from math import prod
from typing import Any

import torch
from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import largest_power_of_2_divisor
from vllm.v1.kv_cache_interface import AttentionSpec
from vllm.v1.worker.utils import AttentionGroup, KVBlockZeroer

from vllm_ascend.core.kv_cache_interface import get_storage_block_size
from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num

ZERO_BLOCK_SIZE = 8192
INITIAL_BLOCK_ID_CAPACITY = 8192


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
    seg_page_strides_ptr,
    block_ids_ptr,
    n_blocks,
    N_SEGS: tl.constexpr,
    MAX_CHUNKS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    GRID_SIZE: tl.constexpr,
):
    """Zero KV cache blocks across all segments in a single launch.

    Each segment is a contiguous region of one scheduler block's data.
    Separate sizes and strides preserve padding and other hybrid cache views.

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
        block_id = tl.load(block_ids_ptr + block_index)
        seg_addr = tl.load(seg_addrs_ptr + seg_index)
        page_size_el = tl.load(seg_page_sizes_ptr + seg_index)
        page_stride_el = tl.load(seg_page_strides_ptr + seg_index)
        ptr = tl.cast(seg_addr, tl.pointer_type(tl.int32))
        offset = block_id.to(tl.int64) * page_stride_el
        cols = chunk_index.to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE).to(tl.int64)
        tl.store(ptr + offset + cols, tl.zeros([BLOCK_SIZE], dtype=tl.int32), mask=cols < page_size_el)


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
        Per-segment page strides account for this ratio. The payload size
        excludes padding, which may belong to another hybrid cache view.

        Only AttentionSpec layers are processed; Mamba layers are skipped.
        """
        seen_ptrs: set[int] = set()
        seg_addrs: list[int] = []
        seg_page_sizes: list[int] = []
        seg_page_strides: list[int] = []

        for group in attn_groups_iter:
            spec = group.kv_cache_spec
            # Match upstream's zeroing scope so installing the Ascend zeroer
            # does not silently exclude sliding-window attention groups.
            if not isinstance(spec, AttentionSpec):
                continue
            if group.kv_cache_group_id >= len(kernel_block_sizes):
                continue
            kernel_bs = kernel_block_sizes[group.kv_cache_group_id][0]
            storage_bs = get_storage_block_size(spec)
            # Compressed caches are reshaped with one physical block per
            # scheduler block, rather than the uncompressed kernel size.
            if storage_bs != spec.block_size:
                kernel_bs = storage_bs
            assert storage_bs % kernel_bs == 0
            ratio = storage_bs // kernel_bs

            for layer_name in group.layer_names:
                if layer_name in runner_only_attn_layers:
                    continue
                kv_tuple = static_forward_context[layer_name].kv_cache
                for kv in kv_tuple:
                    # No-RoPE MLA can expose an empty second component.
                    if kv.numel() == 0:
                        continue
                    dp = kv.data_ptr()
                    if dp in seen_ptrs:
                        continue
                    seen_ptrs.add(dp)

                    assert kv[0].is_contiguous(), "KV blocks must have contiguous inner dimensions"
                    stride_bytes = kv.stride(0) * kv.element_size()
                    payload_bytes = prod(kv.shape[1:]) * kv.element_size()
                    assert kv.shape[0] % ratio == 0, (
                        f"KV cache block count {kv.shape[0]} must be divisible by virtual block ratio {ratio}"
                    )
                    assert stride_bytes >= payload_bytes
                    assert stride_bytes % 4 == 0 and payload_bytes % 4 == 0 and dp % 4 == 0
                    storage = kv.untyped_storage()
                    storage_start = storage.data_ptr()
                    storage_end = storage_start + storage.nbytes()
                    payload_end = dp + (kv.shape[0] - 1) * stride_bytes + payload_bytes
                    assert storage_start <= dp and payload_end <= storage_end, (
                        "KV cache component payload exceeds its backing storage"
                    )
                    if stride_bytes == payload_bytes:
                        # Coalesce contiguous virtual blocks into one segment.
                        seg_addrs.append(dp)
                        seg_page_sizes.append(payload_bytes * ratio // 4)
                        seg_page_strides.append(stride_bytes * ratio // 4)
                    else:
                        for sub_block in range(ratio):
                            seg_addrs.append(dp + sub_block * stride_bytes)
                            seg_page_sizes.append(payload_bytes // 4)
                            seg_page_strides.append(stride_bytes * ratio // 4)

        if not seg_addrs:
            self._meta = None
            return

        # Bound each int32 store to 32 KiB of UB space.
        blk_size = min(min(largest_power_of_2_divisor(size) for size in seg_page_sizes), ZERO_BLOCK_SIZE)
        self._id_cap = INITIAL_BLOCK_ID_CAPACITY
        self._ids_pinned = torch.empty(
            self._id_cap,
            dtype=torch.int64,
            pin_memory=self.pin_memory,
        )
        self._ids_gpu = torch.empty(self._id_cap, dtype=torch.int64, device=self.device)
        self._meta = (
            torch.tensor(seg_addrs, dtype=torch.uint64, device=self.device),
            torch.tensor(seg_page_sizes, dtype=torch.int64, device=self.device),
            torch.tensor(seg_page_strides, dtype=torch.int64, device=self.device),
            max(seg_page_sizes) // blk_size,
            blk_size,
            len(seg_addrs),
        )

    def zero_block_ids(self, block_ids: list[int]) -> None:
        """Zero the KV cache memory for the given block IDs."""
        if not block_ids or self._meta is None:
            return
        seg_addrs, seg_page_sizes, seg_page_strides, max_chunks, blk_size, n_segs = self._meta
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
            seg_page_strides,
            idx,
            n_blocks,
            N_SEGS=n_segs,
            MAX_CHUNKS=max_chunks,
            BLOCK_SIZE=blk_size,
            GRID_SIZE=grid,
        )
