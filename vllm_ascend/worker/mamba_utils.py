# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ascend-owned Mamba state-copy integration.

This module keeps the NPU-specific dispatch local to the Ascend runners.  It
intentionally does not replace objects in :mod:`vllm.v1.worker.mamba_utils`.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable
from typing import Any

import torch
from vllm.config import CacheConfig
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateCopyFunc,
    is_conv_state_dim_first,
)
from vllm.utils.math_utils import cdiv
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.utils import CpuGpuBuffer
from vllm.v1.worker import mamba_utils as upstream_mamba_utils
from vllm.v1.worker.gpu_input_batch import CachedRequestState
from vllm.v1.worker.lora_model_runner_mixin import GPUInputBatch

from vllm_ascend.ops.triton.batch_memcpy import batch_memcpy_kernel
from vllm_ascend.ops.triton.mamba.postprocess import postprocess_mamba_fused_kernel
from vllm_ascend.utils import is_310p

MambaBuffers = upstream_mamba_utils.MambaBuffers
MambaCopyBuffers = upstream_mamba_utils.MambaCopyBuffers
postprocess_mamba_all = upstream_mamba_utils.postprocess_mamba_all
stage_postprocess_inputs_to_gpu = upstream_mamba_utils.stage_postprocess_inputs_to_gpu


class AscendMambaSpecDecodeGPUContext(upstream_mamba_utils.MambaSpecDecodeGPUContext):
    """Use the Ascend precision-safe postprocess kernel without global patching."""

    def run_fused_postprocess(
        self,
        num_reqs: int,
        num_accepted_tokens_gpu: torch.Tensor,
        mamba_state_idx_gpu: torch.Tensor,
        num_scheduled_tokens_gpu: torch.Tensor,
        num_computed_tokens_gpu: torch.Tensor,
        num_draft_tokens_gpu: torch.Tensor,
    ) -> None:
        if num_reqs == 0 or not self.is_initialized:
            return

        self.num_accepted_tokens_out[:num_reqs].copy_(num_accepted_tokens_gpu[:num_reqs])
        grid = (num_reqs, self.num_layers * self.num_state_types)
        postprocess_mamba_fused_kernel[grid](
            num_accepted_tokens_gpu,
            mamba_state_idx_gpu,
            num_scheduled_tokens_gpu,
            num_computed_tokens_gpu,
            num_draft_tokens_gpu,
            self.block_table_ptrs,
            self.block_table_stride_req,
            self.state_base_addrs,
            self.state_block_strides,
            self.state_elem_sizes,
            self.state_inner_sizes,
            self.state_conv_widths,
            self.state_group_indices,
            self.state_dim_row_count,
            self.state_dim_row_stride,
            self.num_accepted_tokens_out,
            None,
            num_reqs,
            block_size=self.block_size,
            COPY_BLOCK_SIZE=1024,
            CONV_STATE_DIM_FIRST=is_conv_state_dim_first(),
        )

    def run_fused_postprocess_align(
        self,
        num_reqs: int,
        num_accepted_tokens_gpu: torch.Tensor,
        state_idx_gpu: torch.Tensor,
        new_num_computed_tokens_gpu: torch.Tensor,
        idx_mapping: torch.Tensor,
    ) -> None:
        if num_reqs == 0 or not self.is_initialized:
            return

        num_accepted_tokens_snapshot = self.num_accepted_tokens_out
        num_accepted_tokens_snapshot.copy_(num_accepted_tokens_gpu)
        grid = (num_reqs, self.num_layers * self.num_state_types)
        postprocess_mamba_fused_kernel[grid](
            num_accepted_tokens_snapshot,
            state_idx_gpu,
            None,
            new_num_computed_tokens_gpu,
            None,
            self.block_table_ptrs,
            self.block_table_stride_req,
            self.state_base_addrs,
            self.state_block_strides,
            self.state_elem_sizes,
            self.state_inner_sizes,
            self.state_conv_widths,
            self.state_group_indices,
            self.state_dim_row_count,
            self.state_dim_row_stride,
            num_accepted_tokens_gpu,
            idx_mapping,
            num_reqs,
            block_size=self.block_size,
            COPY_BLOCK_SIZE=1024,
            CONV_STATE_DIM_FIRST=is_conv_state_dim_first(),
            HAS_IDX_MAPPING=True,
            PRECOMPUTED_NEW_COMPUTED=True,
        )


def create_mamba_buffers(
    max_num_reqs: int,
    kv_cache_config: KVCacheConfig,
    copy_funcs: tuple[MambaStateCopyFunc, ...],
    make_buffer: Callable[..., CpuGpuBuffer],
    device: torch.device,
    with_postprocess_align: bool,
) -> MambaBuffers:
    """Create upstream-compatible buffers with Ascend-owned implementations."""
    mamba_group_ids, mamba_spec = upstream_mamba_utils.get_mamba_groups(kv_cache_config)
    entries_per_req = sum(len(kv_cache_config.kv_cache_groups[gid].layer_names) for gid in mamba_group_ids) * len(
        copy_funcs
    )
    num_entries = max_num_reqs * entries_per_req

    # aclnnInplaceZero does not support uint64 on Ascend. Pointer bit patterns
    # are unchanged when stored in signed int64 tensors.
    preprocess = MambaCopyBuffers(
        src_ptrs=make_buffer(num_entries, dtype=torch.int64),
        dst_ptrs=make_buffer(num_entries, dtype=torch.int64),
        sizes=make_buffer(num_entries, dtype=torch.int32),
        mamba_group_ids=mamba_group_ids,
        mamba_spec=mamba_spec,
    )
    postprocess_align = (
        AscendMambaSpecDecodeGPUContext.create(
            max_num_reqs=max_num_reqs,
            kv_cache_config=kv_cache_config,
            num_state_types=len(copy_funcs),
            device=device,
            make_buffer=make_buffer,
        )
        if with_postprocess_align
        else None
    )
    return MambaBuffers(
        preprocess=preprocess,
        postprocess_align=postprocess_align,
    )


def _batch_memcpy_triton(
    src_ptrs: torch.Tensor,
    dst_ptrs: torch.Tensor,
    sizes: torch.Tensor,
) -> None:
    batch = src_ptrs.shape[0]
    assert dst_ptrs.shape[0] == batch
    assert sizes.shape[0] == batch
    batch_memcpy_kernel[(batch,)](
        src_ptrs,
        dst_ptrs,
        sizes,
        BLOCK_SIZE=8192,
    )


def _tensor_view_from_data_ptr(
    state: torch.Tensor,
    start_addr: int,
    num_elements: int,
) -> torch.Tensor:
    byte_offset = start_addr - state.data_ptr()
    element_size = state.element_size()
    if byte_offset < 0 or byte_offset % element_size != 0:
        raise RuntimeError("Invalid Mamba state copy pointer.")

    element_offset = byte_offset // element_size
    storage_offset = state.storage_offset()
    storage_numel = state.untyped_storage().nbytes() // element_size
    flat_state = state.as_strided(
        (storage_numel - storage_offset,),
        (1,),
        storage_offset=storage_offset,
    )
    if element_offset + num_elements > flat_state.numel():
        raise RuntimeError("Mamba state copy range exceeds tensor storage.")
    return flat_state.narrow(0, element_offset, num_elements)


def _get_tensor_copy_pairs(
    copy_bufs: MambaCopyBuffers,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    if copy_bufs.offset == 0 or not hasattr(copy_bufs, "_tensor_copy_pairs"):
        copy_bufs._tensor_copy_pairs = []
    return copy_bufs._tensor_copy_pairs


def _collect_mamba_copy_meta_torch(
    copy_bufs: MambaCopyBuffers,
    kv_cache_config: KVCacheConfig,
    mamba_state_copy_funcs: tuple[MambaStateCopyFunc, ...],
    mamba_group_ids: list[int],
    src_block_idx: int,
    dest_block_idx: int,
    accept_token_bias: int,
    req_state: CachedRequestState,
    forward_context: dict[str, Any],
) -> None:
    if src_block_idx == dest_block_idx and accept_token_bias == 0:
        return

    tensor_copy_pairs = _get_tensor_copy_pairs(copy_bufs)
    sizes_np = copy_bufs.sizes.np
    offset = copy_bufs.offset
    for mamba_group_id in mamba_group_ids:
        block_ids = req_state.block_ids[mamba_group_id]
        dest_block_id = block_ids[dest_block_idx]
        layer_names = kv_cache_config.kv_cache_groups[mamba_group_id].layer_names
        for layer_name in layer_names:
            attention = forward_context[layer_name]
            kv_caches: list[torch.Tensor] = attention.kv_cache
            for state, state_copy_func in zip(kv_caches, mamba_state_copy_funcs):
                copy_spec = state_copy_func(
                    state,
                    block_ids,
                    src_block_idx,
                    accept_token_bias + 1,
                )
                src_state = _tensor_view_from_data_ptr(state, copy_spec.start_addr, copy_spec.num_elements)
                dst_state = _tensor_view_from_data_ptr(
                    state,
                    state[dest_block_id].data_ptr(),
                    copy_spec.num_elements,
                )
                tensor_copy_pairs.append((src_state, dst_state))
                sizes_np[offset] = copy_spec.num_elements * state.element_size()
                offset += 1
    copy_bufs.offset = offset


def collect_mamba_copy_meta(
    copy_bufs: MambaCopyBuffers,
    kv_cache_config: KVCacheConfig,
    mamba_state_copy_funcs: tuple[MambaStateCopyFunc, ...],
    mamba_group_ids: list[int],
    src_block_idx: int,
    dest_block_idx: int,
    accept_token_bias: int,
    req_state: CachedRequestState,
    forward_context: dict[str, Any],
) -> None:
    collector = _collect_mamba_copy_meta_torch if is_310p() else upstream_mamba_utils.collect_mamba_copy_meta
    collector(
        copy_bufs,
        kv_cache_config,
        mamba_state_copy_funcs,
        mamba_group_ids,
        src_block_idx,
        dest_block_idx,
        accept_token_bias,
        req_state,
        forward_context,
    )


def _do_mamba_copy_block_torch(copy_bufs: MambaCopyBuffers) -> None:
    count = copy_bufs.offset
    if count == 0:
        if hasattr(copy_bufs, "_tensor_copy_pairs"):
            copy_bufs._tensor_copy_pairs = []
        return

    tensor_copy_pairs = getattr(copy_bufs, "_tensor_copy_pairs", None)
    if tensor_copy_pairs is None or len(tensor_copy_pairs) != count:
        raise RuntimeError("Mamba tensor copy metadata is incomplete.")
    for src_state, dst_state in tensor_copy_pairs:
        dst_state.copy_(src_state.clone())
    copy_bufs._tensor_copy_pairs = []


def do_mamba_copy_block(copy_bufs: MambaCopyBuffers) -> None:
    if is_310p():
        _do_mamba_copy_block_torch(copy_bufs)
        return
    count = copy_bufs.offset
    if count == 0:
        return
    _batch_memcpy_triton(
        copy_bufs.src_ptrs.copy_to_gpu(count),
        copy_bufs.dst_ptrs.copy_to_gpu(count),
        copy_bufs.sizes.copy_to_gpu(count),
    )


def preprocess_mamba(
    scheduler_output: SchedulerOutput,
    kv_cache_config: KVCacheConfig,
    cache_config: CacheConfig,
    mamba_state_idx: dict[str, int],
    input_batch: GPUInputBatch,
    requests: dict[str, CachedRequestState],
    forward_context: dict[str, Any],
    mamba_state_copy_funcs: tuple[MambaStateCopyFunc, ...],
    copy_bufs: MambaCopyBuffers,
) -> None:
    """Collect pre-copy metadata after scheduling and defer the actual copy.

    Ascend KV connectors finish loading cache blocks inside the model-forward
    context, so copying before that point can be overwritten by KV Transfer.
    """
    del cache_config
    mamba_group_ids = copy_bufs.mamba_group_ids
    mamba_spec = copy_bufs.mamba_spec
    num_speculative_blocks = mamba_spec.num_speculative_blocks
    block_size = mamba_spec.block_size

    finished_req_ids = scheduler_output.finished_req_ids
    preempted_req_ids = scheduler_output.preempted_req_ids or set()
    resumed_req_ids = scheduler_output.scheduled_cached_reqs.resumed_req_ids
    for req_id in itertools.chain(finished_req_ids, preempted_req_ids, resumed_req_ids):
        mamba_state_idx.pop(req_id, None)

    copy_bufs.offset = 0
    for index, req_id in enumerate(input_batch.req_ids):
        req_state = requests[req_id]
        prev_state_idx = mamba_state_idx.get(req_id)
        if prev_state_idx is None:
            prev_state_idx = (req_state.num_computed_tokens - 1) // block_size

        num_scheduled_tokens = scheduler_output.num_scheduled_tokens[req_id]
        num_blocks = (
            cdiv(
                req_state.num_computed_tokens + num_scheduled_tokens,
                block_size,
            )
            + num_speculative_blocks
        )
        curr_state_idx = num_blocks - 1 - num_speculative_blocks
        mamba_state_idx[req_id] = curr_state_idx
        if prev_state_idx != -1 and prev_state_idx != curr_state_idx:
            collect_mamba_copy_meta(
                copy_bufs,
                kv_cache_config,
                mamba_state_copy_funcs,
                mamba_group_ids,
                prev_state_idx,
                curr_state_idx,
                input_batch.num_accepted_tokens_cpu[index] - 1,
                req_state,
                forward_context,
            )
            input_batch.num_accepted_tokens_cpu[index] = 1


def _postprocess_mamba_align_gpu_cpu_fallback(
    *,
    bufs: MambaBuffers,
    num_reqs: int,
    num_accepted_tokens_gpu: torch.Tensor,
    num_accepted_tokens_cpu_tensor: torch.Tensor,
    input_batch: GPUInputBatch,
    kv_cache_config: KVCacheConfig,
    forward_context: dict[str, Any],
    mamba_state_copy_funcs: tuple[MambaStateCopyFunc, ...],
) -> None:
    ctx = bufs.postprocess_align
    assert ctx is not None
    assert ctx.mamba_state_idx_buf is not None
    assert ctx.num_scheduled_tokens_buf is not None
    assert ctx.num_computed_tokens_buf is not None
    assert ctx.num_draft_tokens_buf is not None

    mamba_state_idx = ctx.mamba_state_idx_buf.np
    num_scheduled_tokens = ctx.num_scheduled_tokens_buf.np
    num_computed_tokens = ctx.num_computed_tokens_buf.np
    num_draft_tokens = ctx.num_draft_tokens_buf.np
    block_size = ctx.block_size
    num_accepted_tokens_cpu_tensor[:num_reqs].copy_(num_accepted_tokens_gpu[:num_reqs])
    num_accepted_tokens = input_batch.num_accepted_tokens_cpu
    for index in range(num_reqs):
        num_tokens_running_state = num_computed_tokens[index] + num_scheduled_tokens[index] - num_draft_tokens[index]
        new_num_computed_tokens = num_tokens_running_state + num_accepted_tokens[index] - 1
        aligned_new_computed_tokens = new_num_computed_tokens // block_size * block_size
        if aligned_new_computed_tokens < num_tokens_running_state:
            continue

        src_block_idx = mamba_state_idx[index]
        dest_block_idx = aligned_new_computed_tokens // block_size - 1
        accept_token_bias = aligned_new_computed_tokens - num_tokens_running_state
        if src_block_idx == dest_block_idx:
            num_accepted_tokens_cpu_tensor[index] = 1
            if accept_token_bias == 0:
                continue

        for mamba_group_id in ctx.mamba_group_ids:
            block_ids = input_batch.block_table[mamba_group_id].get_numpy_array()[index]
            dest_block_id = block_ids[dest_block_idx]
            layer_names = kv_cache_config.kv_cache_groups[mamba_group_id].layer_names
            for layer_name in layer_names:
                attention = forward_context[layer_name]
                kv_caches: list[torch.Tensor] = attention.kv_cache
                for state, state_copy_func in zip(kv_caches, mamba_state_copy_funcs):
                    copy_spec = state_copy_func(
                        state,
                        block_ids,
                        src_block_idx,
                        accept_token_bias + 1,
                    )
                    src_state = _tensor_view_from_data_ptr(state, copy_spec.start_addr, copy_spec.num_elements)
                    dst_state = _tensor_view_from_data_ptr(
                        state,
                        state[dest_block_id].data_ptr(),
                        copy_spec.num_elements,
                    )
                    dst_state.copy_(src_state.clone())


def postprocess_mamba_align_gpu(
    *,
    bufs: MambaBuffers,
    num_reqs: int,
    num_accepted_tokens_gpu: torch.Tensor,
    num_accepted_tokens_cpu_tensor: torch.Tensor,
    input_batch: GPUInputBatch,
    kv_cache_config: KVCacheConfig,
    forward_context: dict[str, Any],
    mamba_state_copy_funcs: tuple[MambaStateCopyFunc, ...],
) -> None:
    if is_310p():
        _postprocess_mamba_align_gpu_cpu_fallback(
            bufs=bufs,
            num_reqs=num_reqs,
            num_accepted_tokens_gpu=num_accepted_tokens_gpu,
            num_accepted_tokens_cpu_tensor=num_accepted_tokens_cpu_tensor,
            input_batch=input_batch,
            kv_cache_config=kv_cache_config,
            forward_context=forward_context,
            mamba_state_copy_funcs=mamba_state_copy_funcs,
        )
        return

    ctx = bufs.postprocess_align
    assert ctx is not None
    assert ctx.mamba_state_idx_buf is not None
    assert ctx.num_scheduled_tokens_buf is not None
    assert ctx.num_computed_tokens_buf is not None
    assert ctx.num_draft_tokens_buf is not None
    if not ctx.is_initialized:
        ctx.initialize_from_forward_context(
            kv_cache_config,
            forward_context,
            mamba_state_copy_funcs,
            [input_batch.block_table[group_id].get_device_tensor(num_reqs) for group_id in ctx.mamba_group_ids],
        )
    ctx.run_fused_postprocess(
        num_reqs=num_reqs,
        num_accepted_tokens_gpu=num_accepted_tokens_gpu,
        mamba_state_idx_gpu=ctx.mamba_state_idx_buf.gpu,
        num_scheduled_tokens_gpu=ctx.num_scheduled_tokens_buf.gpu,
        num_computed_tokens_gpu=ctx.num_computed_tokens_buf.gpu,
        num_draft_tokens_gpu=ctx.num_draft_tokens_buf.gpu,
    )
    num_accepted_tokens_cpu_tensor[:num_reqs].copy_(ctx.num_accepted_tokens_out[:num_reqs], non_blocking=True)
