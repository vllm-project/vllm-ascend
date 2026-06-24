# mypy: ignore-errors

import itertools
from typing import Any

import torch
from vllm.config import CacheConfig
from vllm.model_executor.layers.mamba.mamba_utils import MambaStateCopyFunc
from vllm.utils.math_utils import cdiv
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker import mamba_utils
from vllm.v1.worker.gpu_input_batch import CachedRequestState
from vllm.v1.worker.lora_model_runner_mixin import GPUInputBatch
from vllm.v1.worker.mamba_utils import MambaCopyBuffers

from vllm_ascend.ops.triton.batch_memcpy import batch_memcpy_kernel
from vllm_ascend.utils import is_310p


def _can_launch_triton_batch_memcpy() -> bool:
    return not is_310p()


def _batch_memcpy_triton(src_ptrs, dst_ptrs, sizes):
    batch = src_ptrs.shape[0]
    assert dst_ptrs.shape[0] == batch
    assert sizes.shape[0] == batch

    grid = (batch,)
    # using larger block_size to accelerate copy.
    BLOCK_SIZE = 8192
    batch_memcpy_kernel[grid](src_ptrs, dst_ptrs, sizes, BLOCK_SIZE=BLOCK_SIZE)


def _tensor_view_from_data_ptr(state: torch.Tensor, start_addr: int, num_elements: int) -> torch.Tensor:
    byte_offset = start_addr - state.data_ptr()
    element_size = state.element_size()
    if byte_offset < 0 or byte_offset % element_size != 0:
        raise RuntimeError("Invalid Mamba state copy pointer.")

    element_offset = byte_offset // element_size
    flat_state = state.view(-1)
    if element_offset + num_elements > flat_state.numel():
        raise RuntimeError("Mamba state copy range exceeds tensor storage.")
    return flat_state.narrow(0, element_offset, num_elements)


def _get_tensor_copy_pairs(copy_bufs: mamba_utils.MambaCopyBuffers) -> list[tuple[torch.Tensor, torch.Tensor]]:
    if copy_bufs.offset == 0 or not hasattr(copy_bufs, "_tensor_copy_pairs"):
        copy_bufs._tensor_copy_pairs = []
    return copy_bufs._tensor_copy_pairs


def _collect_mamba_copy_meta_torch(
    copy_bufs: mamba_utils.MambaCopyBuffers,
    kv_cache_config,
    mamba_state_copy_funcs,
    mamba_group_ids: list[int],
    src_block_idx: int,
    dest_block_idx: int,
    accept_token_bias: int,
    req_state,
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
                copy_spec = state_copy_func(state, block_ids, src_block_idx, accept_token_bias + 1)
                src_state = _tensor_view_from_data_ptr(state, copy_spec.start_addr, copy_spec.num_elements)
                dst_state = _tensor_view_from_data_ptr(state, state[dest_block_id].data_ptr(), copy_spec.num_elements)
                tensor_copy_pairs.append((src_state, dst_state))
                sizes_np[offset] = copy_spec.num_elements * state.element_size()
                offset += 1

    copy_bufs.offset = offset


def _do_mamba_copy_block_torch(copy_bufs: mamba_utils.MambaCopyBuffers):
    n = copy_bufs.offset
    if n == 0:
        if hasattr(copy_bufs, "_tensor_copy_pairs"):
            copy_bufs._tensor_copy_pairs = []
        return

    tensor_copy_pairs = getattr(copy_bufs, "_tensor_copy_pairs", None)
    if tensor_copy_pairs is None or len(tensor_copy_pairs) != n:
        raise RuntimeError("Mamba tensor copy metadata is incomplete.")

    for src_state, dst_state in tensor_copy_pairs:
        dst_state.copy_(src_state.clone())
    copy_bufs._tensor_copy_pairs = []


def _batch_memcpy_unavailable(src_ptrs, dst_ptrs, sizes):
    raise RuntimeError(
        "Pointer-based Mamba batch memcpy requires Triton and is not available "
        "on 310P. Use the tensor-copy fallback path instead."
    )


if _can_launch_triton_batch_memcpy():
    mamba_utils.batch_memcpy_kernel = batch_memcpy_kernel
    mamba_utils.batch_memcpy = _batch_memcpy_triton
else:
    mamba_utils.batch_memcpy = _batch_memcpy_unavailable
    mamba_utils.collect_mamba_copy_meta = _collect_mamba_copy_meta_torch
    mamba_utils.do_mamba_copy_block = _do_mamba_copy_block_torch


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
):
    """
    Copy the mamba state of previous step to the last
    (1 + num_speculative_blocks) block.
    """
    mamba_group_ids = copy_bufs.mamba_group_ids
    mamba_spec = copy_bufs.mamba_spec
    num_speculative_blocks = mamba_spec.num_speculative_blocks
    # TODO(Chen): we need to optimize this function a lot
    # assert cache_config.enable_prefix_caching
    block_size = mamba_spec.block_size
    finished_req_ids = scheduler_output.finished_req_ids
    preempted_req_ids = scheduler_output.preempted_req_ids or set()
    resumed_req_ids = scheduler_output.scheduled_cached_reqs.resumed_req_ids
    for req_id in itertools.chain(finished_req_ids, preempted_req_ids, resumed_req_ids):
        mamba_state_idx.pop(req_id, None)

    copy_bufs.offset = 0
    for i, req_id in enumerate(input_batch.req_ids):
        req_state = requests[req_id]
        prev_state_idx = mamba_state_idx.get(req_id)
        if prev_state_idx is None:
            # new / resumed request, no previous state
            # if num_computed_tokens is 0, prev_state_idx will be -1
            prev_state_idx = (req_state.num_computed_tokens - 1) // block_size

        num_scheduled_tokens = scheduler_output.num_scheduled_tokens[req_id]
        num_blocks: int = (
            cdiv(req_state.num_computed_tokens + num_scheduled_tokens, block_size) + num_speculative_blocks
        )

        # We always save the current running state at the last
        # (1 + num_speculative_blocks) block.
        # A corner case worth mention here: assume we have block_size = 4 and
        # num_speculative_tokens = 2. The request is [A, B, C] and contains 2 draft
        # tokens [draft 1, draft 2]. Then we will have:
        # Block 0: [A, B, C, draft 1]
        # Block 1: [draft 2, TOFILL, TOFILL, TOFILL]
        # Block 2: speculative block
        # Block 3: speculative block
        # And use block 1 to save the running state.
        curr_state_idx = num_blocks - 1 - num_speculative_blocks
        mamba_state_idx[req_id] = curr_state_idx
        if prev_state_idx != -1 and prev_state_idx != curr_state_idx:
            mamba_utils.collect_mamba_copy_meta(
                copy_bufs,
                kv_cache_config,
                mamba_state_copy_funcs,
                mamba_group_ids,
                prev_state_idx,
                curr_state_idx,
                input_batch.num_accepted_tokens_cpu[i] - 1,
                req_state,
                forward_context,
            )
            input_batch.num_accepted_tokens_cpu[i] = 1
    # do not copy here, since kv_transfer still not load
    # do_mamba_copy_block(copy_bufs)


mamba_utils.preprocess_mamba = preprocess_mamba
