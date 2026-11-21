"""Opt-in all-greedy MTP verification using blockwise target argmax."""

import inspect

import numpy as np
import torch
from vllm.triton_utils import tl, triton

from vllm_ascend.ascend_config import get_ascend_config

VOCAB_BLOCK_SIZE = 8192


@triton.jit
def _greedy_block_stats(
    logits,
    local_max,
    local_idx,
    row_stride,
    block_stride,
    vocab,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    block = tl.program_id(1)
    offsets = block * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(
        logits + row * row_stride + offsets,
        mask=offsets < vocab,
        other=float("-inf"),
    ).to(tl.float32)
    value, index = tl.max(values, axis=0, return_indices=True)
    tl.store(local_max + row * block_stride + block, value)
    tl.store(local_idx + row * block_stride + block, block * BLOCK + index)


@triton.jit
def _greedy_row_argmax(
    max_ptr,
    idx_ptr,
    row,
    stride,
    blocks,
    PB: tl.constexpr,
):
    block_offsets = tl.arange(0, PB)
    values = tl.load(
        max_ptr + row * stride + block_offsets,
        mask=block_offsets < blocks,
        other=float("-inf"),
    )
    # Keep tl.argmax inside the valid block range when a block contains NaNs.
    values = tl.where(values != values, float("-inf"), values)
    selected = tl.argmax(values, axis=0)
    return tl.load(idx_ptr + row * stride + selected)


@triton.jit
def _verify_greedy(
    max_ptr,
    idx_ptr,
    draft_ptr,
    cu_ptr,
    sampled_ptr,
    counts_ptr,
    stride,
    blocks,
    stats_stride,
    PB: tl.constexpr,
):
    request_idx = tl.program_id(0)
    start = tl.load(cu_ptr + request_idx).to(tl.int64)
    end = tl.load(cu_ptr + request_idx + 1).to(tl.int64)
    num_draft_tokens = end - start - 1
    accepted = True
    count = 0

    for draft_idx in range(num_draft_tokens):
        if accepted:
            target = _greedy_row_argmax(
                max_ptr,
                idx_ptr,
                start + draft_idx,
                stats_stride,
                blocks,
                PB,
            )
            draft = tl.load(draft_ptr + start + draft_idx + 1)
            tl.store(sampled_ptr + request_idx * stride + draft_idx, target)
            accepted = (draft >= 0) & (draft == target)
            count += 1

    if accepted:
        target = _greedy_row_argmax(
            max_ptr,
            idx_ptr,
            end - 1,
            stats_stride,
            blocks,
            PB,
        )
        tl.store(
            sampled_ptr + request_idx * stride + num_draft_tokens,
            target,
        )
        count += 1

    tl.store(counts_ptr + request_idx, count)


def greedy_verify(
    processed_logits,
    draft_sampled,
    cu_num_logits,
    num_steps,
):
    num_requests = cu_num_logits.shape[0] - 1
    num_rows, vocab_size = processed_logits.shape
    num_blocks = triton.cdiv(vocab_size, VOCAB_BLOCK_SIZE)
    local_max = processed_logits.new_empty(
        (num_rows, num_blocks),
        dtype=torch.float32,
    )
    local_idx = processed_logits.new_empty(
        (num_rows, num_blocks),
        dtype=torch.int64,
    )
    _greedy_block_stats[(num_rows, num_blocks)](
        processed_logits,
        local_max,
        local_idx,
        processed_logits.stride(0),
        num_blocks,
        vocab_size,
        BLOCK=VOCAB_BLOCK_SIZE,
        has_auto_blockify_blacklist_op=True,
    )
    sampled = draft_sampled.new_empty(
        (num_requests, num_steps + 1),
        dtype=torch.int64,
    )
    counts = draft_sampled.new_empty(
        (num_requests,),
        dtype=torch.int32,
    )
    _verify_greedy[(num_requests,)](
        local_max,
        local_idx,
        draft_sampled,
        cu_num_logits,
        sampled,
        counts,
        sampled.stride(0),
        num_blocks,
        num_blocks,
        PB=triton.next_power_of_2(num_blocks),
        num_warps=1,
    )
    return sampled, counts


def install(rejection_sampler_cls):
    if getattr(rejection_sampler_cls, "_ascend_greedy_verify_installed", False):
        return
    original_init = rejection_sampler_cls.__init__
    init_signature = inspect.signature(original_init)
    original_verify = rejection_sampler_cls._verify

    def _init(self, *args, **kwargs):
        bound = init_signature.bind(self, *args, **kwargs)
        config = bound.arguments.get("spec_config")
        original_init(self, *args, **kwargs)
        self._ascend_greedy_mtp = getattr(config, "method", None) == "mtp"

    def _verify(
        self,
        logits,
        draft_logits,
        draft_sampled,
        pos,
        cu_num_logits,
        idx_mapping,
        idx_mapping_np,
        expanded_idx_mapping,
        expanded_local_pos,
        seq_lens_upper_bound_np,
        verify_draft_sampled,
    ):
        if (
            not getattr(self, "_ascend_greedy_mtp", False)
            or get_ascend_config().rejection_sampler_config.enable_entropy_verify
            or self.watermark_key is not None
            or self.synthetic_conditional_rates is not None
            or self.use_block_verification
            or self.fly_window_size > 0
            or (draft_logits is not None and draft_logits.shape[-1] < logits.shape[-1])
            or not np.all(self.sampler.sampling_states.temperature.np[idx_mapping_np] == 0.0)
        ):
            return original_verify(
                self,
                logits,
                draft_logits,
                draft_sampled,
                pos,
                cu_num_logits,
                idx_mapping,
                idx_mapping_np,
                expanded_idx_mapping,
                expanded_local_pos,
                seq_lens_upper_bound_np,
                verify_draft_sampled,
            )

        processed_logits = self.sampler.apply_sampling_params(
            logits,
            expanded_idx_mapping,
            idx_mapping,
            idx_mapping_np,
            pos,
            draft_sampled,
            expanded_local_pos,
            seq_lens_upper_bound_np,
        )
        sampled, counts = greedy_verify(
            processed_logits,
            verify_draft_sampled,
            cu_num_logits,
            self.num_speculative_steps,
        )
        return processed_logits, sampled, counts

    rejection_sampler_cls.__init__ = _init
    rejection_sampler_cls._verify = _verify
    rejection_sampler_cls._ascend_greedy_verify_installed = True
