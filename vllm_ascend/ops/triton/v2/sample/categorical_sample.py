# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton categorical sampling operator for Ascend NPU."""

import torch
from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num, init_device_properties_triton

# Hierarchical sampling: prepare 8K coarse-block masses once, then sample
# from one 1K fine block. The two granularities are tuned independently.
_COARSE_BLOCK_SIZE = 8192
_FINE_BLOCK_SIZE = 1024
_NUM_FINE_BLOCKS = _COARSE_BLOCK_SIZE // _FINE_BLOCK_SIZE


def _get_vectorcore_num() -> int:
    try:
        return int(get_vectorcore_num())
    except AssertionError:
        init_device_properties_triton()
        return int(get_vectorcore_num())


# Stage 1: scan logits and build hierarchical probability-mass metadata.
# Stage 2: select coarse/fine blocks from metadata and sample one token.
@triton.jit(
    do_not_specialize=[
        "block_argmax_stride",
        "block_max_stride",
        "block_mass_stride",
        "fine_mass_stride_0",
        "fine_mass_stride_1",
        "logits_cache_stride_0",
        "logits_cache_stride_1",
        "logits_stride",
        "num_tokens",
        "vocab_size",
        "num_blocks",
    ]
)
def _categorical_prepare_mass_kernel(
    block_argmax_ptr,
    block_argmax_stride,
    block_max_ptr,
    block_max_stride,
    block_mass_ptr,
    block_mass_stride,
    fine_mass_ptr,
    fine_mass_stride_0,
    fine_mass_stride_1,
    logits_cache_ptr,
    logits_cache_stride_0,
    logits_cache_stride_1,
    logits_cache_col_ptr,
    logits_ptr,
    logits_stride,
    expanded_idx_mapping_ptr,
    temp_ptr,
    num_tokens,
    vocab_size,
    num_blocks,
    COARSE_BLOCK_SIZE: tl.constexpr,
    FINE_BLOCK_SIZE: tl.constexpr,
    NUM_FINE_BLOCKS: tl.constexpr,
    APPLY_TEMPERATURE: tl.constexpr,
    PER_TOKEN_COL: tl.constexpr,
):
    worker_id = tl.program_id(0).to(tl.int64)
    num_workers = tl.num_programs(0).to(tl.int64)
    num_tokens_i64 = num_tokens.to(tl.int64)
    num_blocks_i64 = num_blocks.to(tl.int64)
    total_tasks = num_tokens_i64 * num_blocks_i64
    tasks_per_worker = total_tasks // num_workers
    extra_tasks = total_tasks % num_workers
    task_start = worker_id * tasks_per_worker + tl.minimum(worker_id, extra_tasks)
    task_count = tasks_per_worker + (worker_id < extra_tasks)
    lanes = tl.arange(0, COARSE_BLOCK_SIZE)
    fine_block_ids = tl.arange(0, NUM_FINE_BLOCKS)

    for task_idx in tl.range(task_start, task_start + task_count):
        token_idx = task_idx // num_blocks_i64
        block_idx = task_idx - token_idx * num_blocks_i64
        req_state_idx = tl.load(expanded_idx_mapping_ptr + token_idx).to(tl.int64)
        is_valid_req = req_state_idx >= 0
        temp = tl.load(temp_ptr + req_state_idx, mask=is_valid_req, other=0.0).to(tl.float32)

        offsets = block_idx * COARSE_BLOCK_SIZE + lanes
        mask = offsets < vocab_size
        logits = tl.load(logits_ptr + token_idx * logits_stride + offsets, mask=mask, other=float("-inf")).to(
            tl.float32
        )

        if logits_cache_ptr is not None:
            if logits_cache_col_ptr is not None:
                if PER_TOKEN_COL:
                    col = tl.load(logits_cache_col_ptr + token_idx)
                else:
                    col = tl.load(logits_cache_col_ptr)
            else:
                col = 0
            tl.store(
                logits_cache_ptr + req_state_idx * logits_cache_stride_0 + col * logits_cache_stride_1 + offsets,
                logits,
                mask=mask & is_valid_req,
            )

        if temp != 0.0 and APPLY_TEMPERATURE:
            logits = logits / temp

        block_max, block_argmax = tl.max(logits, axis=0, return_indices=True)
        has_mass = block_max > float("-inf")
        safe_block_max = tl.where(has_mass, block_max, 0.0)

        weights = tl.where(mask & (temp != 0.0) & has_mass, tl.exp(logits - safe_block_max), 0.0)
        weights_2d = tl.reshape(weights, (NUM_FINE_BLOCKS, FINE_BLOCK_SIZE))
        fine_mass_values = tl.sum(weights_2d, axis=1)
        block_mass = tl.sum(fine_mass_values, axis=0)

        tl.store(
            block_argmax_ptr + token_idx * block_argmax_stride + block_idx,
            block_idx * COARSE_BLOCK_SIZE + block_argmax,
        )
        tl.store(block_max_ptr + token_idx * block_max_stride + block_idx, block_max)
        tl.store(block_mass_ptr + token_idx * block_mass_stride + block_idx, block_mass)
        tl.store(
            fine_mass_ptr + token_idx * fine_mass_stride_0 + block_idx * fine_mass_stride_1 + fine_block_ids,
            fine_mass_values,
        )


@triton.jit(
    do_not_specialize=[
        "block_argmax_stride",
        "block_max_stride",
        "block_mass_stride",
        "fine_mass_stride_0",
        "fine_mass_stride_1",
        "logits_stride",
        "num_tokens",
        "vocab_size",
        "num_blocks",
    ]
)
def _categorical_sample_kernel(
    sampled_ptr,
    block_argmax_ptr,
    block_argmax_stride,
    block_max_ptr,
    block_max_stride,
    block_mass_ptr,
    block_mass_stride,
    fine_mass_ptr,
    fine_mass_stride_0,
    fine_mass_stride_1,
    logits_ptr,
    logits_stride,
    expanded_idx_mapping_ptr,
    seeds_ptr,
    pos_ptr,
    temp_ptr,
    num_tokens,
    vocab_size,
    num_blocks,
    COARSE_BLOCK_SIZE: tl.constexpr,
    FINE_BLOCK_SIZE: tl.constexpr,
    NUM_FINE_BLOCKS: tl.constexpr,
    PADDED_NUM_BLOCKS: tl.constexpr,
    APPLY_TEMPERATURE: tl.constexpr,
    IS_DRAFTING: tl.constexpr,
):
    worker_id = tl.program_id(0).to(tl.int64)
    num_workers = tl.num_programs(0).to(tl.int64)
    num_tokens_i64 = num_tokens.to(tl.int64)
    tokens_per_worker = num_tokens_i64 // num_workers
    extra_tokens = num_tokens_i64 % num_workers
    token_start = worker_id * tokens_per_worker + tl.minimum(worker_id, extra_tokens)
    token_count = tokens_per_worker + (worker_id < extra_tokens)

    for token_idx in tl.range(token_start, token_start + token_count):
        req_state_idx = tl.load(expanded_idx_mapping_ptr + token_idx).to(tl.int64)
        is_valid_req = req_state_idx >= 0
        temp = tl.load(temp_ptr + req_state_idx, mask=is_valid_req, other=0.0).to(tl.float32)
        is_random = temp != 0.0

        block_ids = tl.arange(0, PADDED_NUM_BLOCKS)
        valid_block_mask = block_ids < num_blocks
        block_max = tl.load(
            block_max_ptr + token_idx * block_max_stride + block_ids,
            mask=valid_block_mask,
            other=float("-inf"),
        ).to(tl.float32)
        greedy_block = tl.argmax(block_max, axis=0)
        greedy_token = tl.load(block_argmax_ptr + token_idx * block_argmax_stride + greedy_block)

        stored_mass = tl.load(
            block_mass_ptr + token_idx * block_mass_stride + block_ids,
            mask=valid_block_mask & is_random,
            other=0.0,
        ).to(tl.float32)
        global_max = tl.max(
            tl.where(valid_block_mask & is_random, block_max, float("-inf")),
            axis=0,
        )
        safe_global_max = tl.where(global_max > float("-inf"), global_max, 0.0)
        block_mass = stored_mass * tl.exp(block_max - safe_global_max)
        block_mass = tl.where(valid_block_mask & is_random, block_mass, 0.0)
        total_mass = tl.sum(block_mass, axis=0)
        has_total_mass = total_mass > 0.0

        seed = tl.load(seeds_ptr + req_state_idx, mask=is_valid_req & is_random, other=0)
        position = tl.load(pos_ptr + token_idx).to(tl.int32)
        if IS_DRAFTING:
            DRAFT_NOISE_SALT: tl.constexpr = 1 << 30
            position += DRAFT_NOISE_SALT
        uniform = tl.max(
            tl.rand(tl.randint(seed, position), tl.arange(0, 1)).to(tl.float32),
            axis=0,
        )
        threshold = uniform * total_mass

        block_prefix = tl.cumsum(block_mass, axis=0)
        candidate_blocks = tl.where(
            (block_prefix > threshold) & valid_block_mask & has_total_mass,
            block_ids,
            PADDED_NUM_BLOCKS,
        )
        selected_block = tl.minimum(tl.min(candidate_blocks, axis=0), num_blocks - 1)
        prefix_before_block = tl.sum(
            tl.where(valid_block_mask & (block_ids < selected_block), block_mass, 0.0),
            axis=0,
        )
        block_threshold = threshold - prefix_before_block

        # Use prepared 1K fine-block masses to avoid rescanning the selected 8K block.
        fine_block_ids = tl.arange(0, NUM_FINE_BLOCKS)
        stored_fine_mass = tl.load(
            fine_mass_ptr + token_idx * fine_mass_stride_0 + selected_block * fine_mass_stride_1 + fine_block_ids,
            mask=is_random & has_total_mass,
            other=0.0,
        ).to(tl.float32)
        selected_block_max = tl.load(block_max_ptr + token_idx * block_max_stride + selected_block).to(tl.float32)
        block_scale = tl.exp(selected_block_max - safe_global_max)
        fine_mass_values = tl.where(
            is_random & has_total_mass,
            stored_fine_mass * block_scale,
            0.0,
        )
        fine_prefix = tl.cumsum(fine_mass_values, axis=0)
        candidate_fine_blocks = tl.where(
            (fine_prefix > block_threshold) & is_random & has_total_mass,
            fine_block_ids,
            NUM_FINE_BLOCKS,
        )
        selected_fine_block = tl.minimum(tl.min(candidate_fine_blocks, axis=0), NUM_FINE_BLOCKS - 1)
        prefix_before_fine = tl.sum(
            tl.where(fine_block_ids < selected_fine_block, fine_mass_values, 0.0),
            axis=0,
        )
        token_threshold = block_threshold - prefix_before_fine

        fine_offsets = tl.arange(0, FINE_BLOCK_SIZE)
        fine_base = selected_block * COARSE_BLOCK_SIZE + selected_fine_block * FINE_BLOCK_SIZE
        token_ids = fine_base + fine_offsets
        token_mask = token_ids < vocab_size
        active_token_mask = token_mask & is_random & has_total_mass
        logits = tl.load(
            logits_ptr + token_idx * logits_stride + token_ids, mask=active_token_mask, other=float("-inf")
        ).to(tl.float32)
        if APPLY_TEMPERATURE:
            safe_temp = tl.where(is_random, temp, 1.0)
            logits = logits / safe_temp

        token_mass = tl.where(
            active_token_mask,
            tl.exp(logits - safe_global_max),
            0.0,
        )
        token_prefix = tl.cumsum(token_mass, axis=0)
        candidate_offsets = tl.where(
            (token_prefix > token_threshold) & token_mask & has_total_mass,
            fine_offsets,
            FINE_BLOCK_SIZE,
        )
        selected_offset = tl.min(candidate_offsets, axis=0)
        fallback_offset = tl.max(
            tl.where(token_mask & (token_mass > 0.0), fine_offsets, 0),
            axis=0,
        )
        selected_offset = tl.where(
            selected_offset < FINE_BLOCK_SIZE,
            selected_offset,
            fallback_offset,
        )
        categorical_token = fine_base + selected_offset
        categorical_token = tl.where(has_total_mass, categorical_token, 0)
        sampled_token = tl.where(is_random, categorical_token, greedy_token)
        tl.store(sampled_ptr + token_idx, sampled_token)


def categorical_sample(
    logits: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    temperature: torch.Tensor,
    seed: torch.Tensor,
    pos: torch.Tensor,
    apply_temperature: bool,
    is_drafting: bool = False,
    logits_cache: torch.Tensor | None = None,
    logits_cache_col: torch.Tensor | None = None,
    use_fp64: bool = False,
) -> torch.Tensor:
    """Sample token ids from logits with categorical sampling.

    Args:
        logits: Logits tensor with shape [num_tokens, vocab_size].
        expanded_idx_mapping: Mapping from token index to request index,
            shape [num_tokens].
        temperature: Per-request temperature, shape [max_num_reqs].
        seed: Per-request random seed, shape [max_num_reqs].
        pos: Per-token random position, shape [num_tokens].
        apply_temperature: Whether to divide logits by non-zero temperature.
        is_drafting: Whether to salt the random position for draft sampling.
        logits_cache: Optional raw-logits cache with shape
            [max_num_reqs, num_cols, vocab_size].
        logits_cache_col: Optional scalar or per-token cache column index.
        use_fp64: Reserved compatibility argument. FP64 sampling is not
            supported on NPU.

    Returns:
        int64 tensor with shape [num_tokens] containing sampled token ids.
    """
    if use_fp64:
        raise NotImplementedError("FP64 categorical sampling is not supported on NPU.")

    expanded_idx_mapping = expanded_idx_mapping.contiguous()
    pos = pos.contiguous()
    if logits_cache_col is not None:
        logits_cache_col = logits_cache_col.contiguous()

    num_tokens, vocab_size = logits.shape
    if logits_cache is not None:
        assert logits_cache.size(-1) >= vocab_size, (
            f"draft logits cache vocab dim ({logits_cache.size(-1)}) is narrower "
            f"than the sampled logits ({vocab_size}). Cached logits would be truncated."
        )
    if num_tokens == 0:
        return torch.empty(0, dtype=torch.int64, device=logits.device)

    num_blocks = triton.cdiv(vocab_size, _COARSE_BLOCK_SIZE)
    padded_num_blocks = triton.next_power_of_2(num_blocks)
    # Metadata produced once by the prepare kernel and consumed by the sample kernel.
    block_argmax_workspace = torch.empty(num_tokens, num_blocks, dtype=torch.int64, device=logits.device)
    block_max_workspace = torch.empty(num_tokens, num_blocks, dtype=torch.float32, device=logits.device)
    block_mass_workspace = torch.empty(num_tokens, num_blocks, dtype=torch.float32, device=logits.device)
    fine_mass = torch.empty(num_tokens, num_blocks, _NUM_FINE_BLOCKS, dtype=torch.float32, device=logits.device)
    sampled = torch.empty(num_tokens, dtype=torch.int64, device=logits.device)
    per_token_col = logits_cache_col is not None and logits_cache_col.dim() > 0

    total_tasks = num_tokens * num_blocks
    num_workers = min(_get_vectorcore_num(), total_tasks)

    _categorical_prepare_mass_kernel[(num_workers,)](
        block_argmax_workspace,
        block_argmax_workspace.stride(0),
        block_max_workspace,
        block_max_workspace.stride(0),
        block_mass_workspace,
        block_mass_workspace.stride(0),
        fine_mass,
        fine_mass.stride(0),
        fine_mass.stride(1),
        logits_cache,
        logits_cache.stride(0) if logits_cache is not None else 0,
        logits_cache.stride(1) if logits_cache is not None else 0,
        logits_cache_col,
        logits,
        logits.stride(0),
        expanded_idx_mapping,
        temperature,
        num_tokens,
        vocab_size,
        num_blocks,
        COARSE_BLOCK_SIZE=_COARSE_BLOCK_SIZE,
        FINE_BLOCK_SIZE=_FINE_BLOCK_SIZE,
        NUM_FINE_BLOCKS=_NUM_FINE_BLOCKS,
        APPLY_TEMPERATURE=apply_temperature,
        PER_TOKEN_COL=per_token_col,
    )

    sample_workers = min(_get_vectorcore_num(), num_tokens)
    _categorical_sample_kernel[(sample_workers,)](
        sampled,
        block_argmax_workspace,
        block_argmax_workspace.stride(0),
        block_max_workspace,
        block_max_workspace.stride(0),
        block_mass_workspace,
        block_mass_workspace.stride(0),
        fine_mass,
        fine_mass.stride(0),
        fine_mass.stride(1),
        logits,
        logits.stride(0),
        expanded_idx_mapping,
        seed,
        pos,
        temperature,
        num_tokens,
        vocab_size,
        num_blocks,
        COARSE_BLOCK_SIZE=_COARSE_BLOCK_SIZE,
        FINE_BLOCK_SIZE=_FINE_BLOCK_SIZE,
        NUM_FINE_BLOCKS=_NUM_FINE_BLOCKS,
        PADDED_NUM_BLOCKS=padded_num_blocks,
        APPLY_TEMPERATURE=apply_temperature,
        IS_DRAFTING=is_drafting,
    )
    return sampled
