# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable
from functools import wraps

import numpy as np
import torch
from vllm.config.model import PROCESSED_LOGPROBS_MODES
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.sample.logits_processor.interface import LogitsContext
from vllm.v1.worker.gpu.sample.sampler import Sampler
from vllm.v1.worker.gpu.spec_decode.rejection_sampler import RejectionSampler, get_max_chunk_logits

from vllm_ascend.ascend_forward_context import WRITABLE_TARGET_LOGITS_PTR


def wants_fp32_target_logits(
    sampler: Sampler | None,
    rejection_sampler: RejectionSampler | None,
    input_batch: InputBatch,
    vocab_size: int,
) -> bool:
    """Match the stock V2 sampling rules for a writable FP32 target buffer."""
    if type(sampler) is not Sampler:
        return False
    is_verifying = input_batch.num_draft_tokens > 0 and rejection_sampler is not None
    if is_verifying and (
        type(rejection_sampler) is not RejectionSampler
        or input_batch.logits_indices.numel() > get_max_chunk_logits(vocab_size)
    ):
        return False

    requests = input_batch.idx_mapping_np
    return bool(np.any(sampler.needs_logits_processing[requests])) and (
        sampler.logprobs_mode in PROCESSED_LOGPROBS_MODES
        or sampler.get_logprobs_dims(requests, include_token_ids=not is_verifying) is None
    )


def reuse_writable_target_logits(original: Callable) -> Callable:
    """Skip the stock FP32 copy for the buffer produced by the fused gather."""

    @wraps(original)
    def apply_sampling_params(
        self: Sampler,
        logits: torch.Tensor,
        expanded_idx_mapping: torch.Tensor,
        idx_mapping: torch.Tensor,
        idx_mapping_np: np.ndarray,
        pos: torch.Tensor,
        input_ids: torch.Tensor,
        expanded_local_pos: torch.Tensor,
        seq_lens_upper_bound_np: np.ndarray,
        skip_top_k_top_p: bool = False,
    ) -> torch.Tensor:
        writable_ptr = (
            get_forward_context().additional_kwargs.get(WRITABLE_TARGET_LOGITS_PTR)
            if is_forward_context_available()
            else None
        )
        if writable_ptr is None or logits.data_ptr() != writable_ptr:
            return original(
                self,
                logits,
                expanded_idx_mapping,
                idx_mapping,
                idx_mapping_np,
                pos,
                input_ids,
                expanded_local_pos,
                seq_lens_upper_bound_np,
                skip_top_k_top_p,
            )

        # Mirror the stock processing order. The gather already wrote FP32,
        # and the runner has checked that no returned logprobs read raw logits.
        ctx = LogitsContext(
            expanded_idx_mapping=expanded_idx_mapping,
            idx_mapping=idx_mapping,
            idx_mapping_np=idx_mapping_np,
            expanded_local_pos=expanded_local_pos,
            input_ids=input_ids,
            pos=pos,
            seq_lens_upper_bound_np=seq_lens_upper_bound_np,
        )
        for processor in self.logits_processors:
            logits = processor.apply(logits, ctx)
        self.thinking_budget_state.apply(logits, ctx)
        self.sampling_states.apply_temperature(logits, expanded_idx_mapping, idx_mapping_np)
        self.sampling_states.apply_min_p(logits, expanded_idx_mapping, idx_mapping_np)
        if skip_top_k_top_p:
            return logits
        return self.sampling_states.apply_top_k_top_p(logits, expanded_idx_mapping, idx_mapping_np)

    return apply_sampling_params
