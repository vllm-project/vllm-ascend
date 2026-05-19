# mypy: ignore-errors

import itertools

from vllm.distributed.parallel_state import get_pcp_group
from vllm.v1.worker import mamba_utils

from vllm_ascend.ops.triton.batch_memcpy import batch_memcpy_kernel


def _get_effective_block_size(block_size: int) -> int:
    pcp_world_size = get_pcp_group().world_size
    if pcp_world_size > 1:
        block_size *= pcp_world_size
    return block_size


def preprocess_mamba(
    scheduler_output,
    kv_cache_config,
    cache_config,
    mamba_state_idx,
    input_batch,
    requests,
    forward_context,
    mamba_state_copy_funcs,
    copy_bufs,
):
    """Copy mamba state using CP-aware block indexing."""
    mamba_group_ids = copy_bufs.mamba_group_ids
    mamba_spec = copy_bufs.mamba_spec
    num_speculative_blocks = mamba_spec.num_speculative_blocks
    if not cache_config.enable_prefix_caching:
        return
    block_size = _get_effective_block_size(mamba_spec.block_size)
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
            prev_state_idx = (req_state.num_computed_tokens - 1) // block_size

        num_scheduled_tokens = scheduler_output.num_scheduled_tokens[req_id]
        num_blocks = (
            mamba_utils.cdiv(req_state.num_computed_tokens + num_scheduled_tokens, block_size) + num_speculative_blocks
        )

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
    mamba_utils.do_mamba_copy_block(copy_bufs)


def postprocess_mamba(
    scheduler_output,
    kv_cache_config,
    input_batch,
    requests,
    mamba_state_idx,
    forward_context,
    mamba_state_copy_funcs,
    copy_bufs,
):
    """Finalize mamba state copies using CP-aware block indexing."""
    num_scheduled_tokens_dict = scheduler_output.num_scheduled_tokens
    scheduled_spec_decode_tokens_dict = scheduler_output.scheduled_spec_decode_tokens
    num_accepted_tokens_cpu = input_batch.num_accepted_tokens_cpu
    mamba_group_ids = copy_bufs.mamba_group_ids
    mamba_spec = copy_bufs.mamba_spec
    block_size = _get_effective_block_size(mamba_spec.block_size)
    copy_bufs.offset = 0
    for i, req_id in enumerate(input_batch.req_ids):
        req_state = requests[req_id]
        num_computed_tokens = req_state.num_computed_tokens
        num_draft_tokens = len(scheduled_spec_decode_tokens_dict.get(req_id, []))
        num_scheduled_tokens = num_scheduled_tokens_dict[req_id]
        num_accepted_tokens = num_accepted_tokens_cpu[i]
        num_tokens_running_state = num_computed_tokens + num_scheduled_tokens - num_draft_tokens
        new_num_computed_tokens = num_tokens_running_state + num_accepted_tokens - 1
        aligned_new_computed_tokens = new_num_computed_tokens // block_size * block_size
        if aligned_new_computed_tokens >= num_tokens_running_state:
            accept_token_bias = aligned_new_computed_tokens - num_tokens_running_state
            src_block_idx = mamba_state_idx[req_id]
            dest_block_idx = aligned_new_computed_tokens // block_size - 1
            mamba_utils.collect_mamba_copy_meta(
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
            if src_block_idx == dest_block_idx:
                num_accepted_tokens_cpu[i] = 1
    mamba_utils.do_mamba_copy_block(copy_bufs)


def batch_memcpy(src_ptrs, dst_ptrs, sizes):
    batch = src_ptrs.shape[0]
    assert dst_ptrs.shape[0] == batch
    assert sizes.shape[0] == batch

    grid = (batch,)
    # using larger block_size to accelerate copy.
    BLOCK_SIZE = 8192
    batch_memcpy_kernel[grid](src_ptrs, dst_ptrs, sizes, BLOCK_SIZE=BLOCK_SIZE)


mamba_utils.batch_memcpy_kernel = batch_memcpy_kernel
mamba_utils.batch_memcpy = batch_memcpy
mamba_utils.preprocess_mamba = preprocess_mamba
mamba_utils.postprocess_mamba = postprocess_mamba
