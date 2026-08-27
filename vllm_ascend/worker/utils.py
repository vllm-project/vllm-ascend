from collections.abc import Iterable
from types import SimpleNamespace
from typing import Any

import torch
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import async_tensor_h2d
from vllm.v1.kv_cache_interface import AttentionSpec
from vllm.v1.worker.utils import AttentionGroup, KVBlockZeroer

from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num


@triton.jit(do_not_specialize=["n_blocks"])
def _zero_kv_blocks_kernel(
    seg_addrs_ptr,
    seg_block_strides_ptr,
    seg_page_sizes_ptr,
    block_ids_ptr,
    n_blocks,
    N_SEGS: tl.constexpr,
    MAX_CHUNKS: tl.constexpr,
    GRID_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Zero KV cache blocks across all segments in a single launch.

    Each segment is a contiguous region of one block's data.  For backends
    where blocks are outermost (block_dim=0) there is one segment per
    buffer.  For backends where K/V is outermost (block_dim=1) there are
    two segments per buffer (one for K, one for V).

    seg_addrs_ptr holds absolute byte addresses (int64) for each segment,
    allowing segments to live in different device allocations.

    Programs are mapped as (block_index, seg_index, chunk_index).
    """
    pid = tl.program_id(0)
    work_per_block = N_SEGS * MAX_CHUNKS
    total_works = n_blocks.to(tl.int64) * work_per_block
    for work_index in range(pid, total_works, GRID_SIZE):
        block_index = work_index // work_per_block
        remainder = work_index % work_per_block
        seg_index = remainder // MAX_CHUNKS
        chunk_index = remainder % MAX_CHUNKS
        block_stride_el = tl.load(seg_block_strides_ptr + seg_index)
        page_size_el = tl.load(seg_page_sizes_ptr + seg_index)
        chunk_offset = chunk_index.to(tl.int64) * BLOCK_SIZE
        block_id = tl.load(block_ids_ptr + block_index)
        seg_addr = tl.load(seg_addrs_ptr + seg_index)
        ptr = tl.cast(seg_addr, tl.pointer_type(tl.int32))
        block_offset = block_id.to(tl.int64) * block_stride_el.to(tl.int64)
        cols = chunk_offset + tl.arange(0, BLOCK_SIZE).to(tl.int64)
        tl.store(
            ptr + block_offset + cols,
            tl.zeros([BLOCK_SIZE], dtype=tl.int32),
            mask=cols < page_size_el,
        )


class AscendKVBlockZeroer(KVBlockZeroer):
    """Manages efficient zeroing of KV cache blocks via a Triton kernel.

    Adapt ascend's separate K/V tensors to vllm upstream metadata planner,
    while retaining specific triton launch strategy.
    """

    class _BlockFirstBackend:
        """Expose ascend's separate K/V tensors as block-first caches."""

        @staticmethod
        def get_kv_cache_block_dim(*args, **kwargs) -> int:
            return 0

    def __init__(
        self,
        device: torch.device,
        attn_groups_iter: Iterable["AttentionGroup"],
        kernel_block_sizes: list[list[int]],
        cache_dtype: str,
        static_forward_context: dict[str, Any],
        runner_only_attn_layers: set[str] | None = None,
    ) -> None:
        """Adapt ascend's separate K/V tensors for creating metadata."""
        adapted_attn_groups: list[Any] = []
        adapted_forward_context: dict[str, Any] = {}

        for group in attn_groups_iter:
            spec = group.kv_cache_spec
            group_id = group.kv_cache_group_id
            if not isinstance(spec, AttentionSpec) or group_id >= len(kernel_block_sizes):
                adapted_attn_groups.append(group)
                continue
            pseudo_layer_names: list[str] = []
            for layer_name in group.layer_names:
                if runner_only_attn_layers is not None and layer_name in runner_only_attn_layers:
                    pseudo_layer_names.append(layer_name)
                    continue
                kv_tuple = static_forward_context[layer_name].kv_cache
                assert isinstance(kv_tuple, tuple) and len(kv_tuple) > 0
                for tensor_index, kv in enumerate(kv_tuple):
                    pseudo_name = f"{layer_name}.kv_zeroer.{tensor_index}"
                    pseudo_layer_names.append(pseudo_name)
                    adapted_forward_context[pseudo_name] = SimpleNamespace(kv_cache=kv)
            adapted_attn_groups.append(
                SimpleNamespace(
                    backend=self._BlockFirstBackend,
                    layer_names=pseudo_layer_names,
                    kv_cache_group_id=group_id,
                    kv_cache_spec=spec,
                )
            )

        super().__init__(
            device=device,
            attn_groups_iter=adapted_attn_groups,
            kernel_block_sizes=[size[0] for size in kernel_block_sizes],
            cache_dtype=cache_dtype,
            static_forward_context=adapted_forward_context,
            runner_only_attn_layers=runner_only_attn_layers,
        )

    def zero_block_ids(self, block_ids: list[int]) -> None:
        """Zero the KV cache memory for the given block IDs."""
        if not block_ids or self._meta is None:
            return
        (
            seg_addrs,
            seg_block_strides,
            seg_page_sizes,
            max_chunks,
            blk_size,
            n_segs,
        ) = self._meta
        n_blocks = len(block_ids)
        idx = async_tensor_h2d(block_ids, device=self.device, dtype=torch.int64)
        total_works = n_blocks * n_segs * max_chunks
        if total_works == 0:
            return
        grid_size = get_vectorcore_num()
        _zero_kv_blocks_kernel[(grid_size,)](
            seg_addrs,
            seg_block_strides,
            seg_page_sizes,
            idx,
            n_blocks,
            MAX_CHUNKS=max_chunks,
            N_SEGS=n_segs,
            GRID_SIZE=grid_size,
            BLOCK_SIZE=blk_size,
        )
