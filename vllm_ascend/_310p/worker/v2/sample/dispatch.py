# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""310P PyTorch implementations of upstream sampling kernel launch ABIs."""

from functools import cache

import torch
from vllm.triton_utils import triton
from vllm.triton_utils.dispatcher import register_kernels
from vllm.utils.math_utils import next_power_of_2

from vllm_ascend._310p.worker.v2.sample.min_p import apply_min_p
from vllm_ascend._310p.worker.v2.sample.temperature import apply_temperature


def temperature_kernel(logits_ptr, logits_stride, expanded_idx_mapping_ptr, temperature_ptr, vocab_size, BLOCK_SIZE):
    del logits_stride, BLOCK_SIZE
    apply_temperature(logits_ptr[:, :vocab_size], expanded_idx_mapping_ptr, temperature_ptr)


def min_p_kernel(logits_ptr, logits_stride, expanded_idx_mapping_ptr, min_p_ptr, vocab_size, BLOCK_SIZE):
    del logits_stride, BLOCK_SIZE
    apply_min_p(logits_ptr[:, :vocab_size], expanded_idx_mapping_ptr, min_p_ptr)


def bias_kernel(
    logits_ptr,
    logits_stride,
    vocab_size,
    expanded_idx_mapping_ptr,
    num_allowed_token_ids_ptr,
    allowed_token_ids_ptr,
    allowed_token_ids_stride,
    num_logit_bias_ptr,
    bias_token_ids_ptr,
    bias_token_ids_stride,
    bias_ptr,
    bias_stride,
    pos_ptr,
    min_lens_ptr,
    num_stop_token_ids_ptr,
    restore_when_all_masked_ptr,
    stop_token_ids_ptr,
    stop_token_ids_stride,
    BLOCK_SIZE,
    LOGITS_BLOCK_SIZE,
    CHECK_ALL_MASKED_ROWS,
):
    """Preserve upstream whitelist -> bias -> min_tokens ordering and writes."""
    del logits_stride, allowed_token_ids_stride, bias_token_ids_stride, bias_stride, stop_token_ids_stride
    del BLOCK_SIZE, LOGITS_BLOCK_SIZE
    mapping = expanded_idx_mapping_ptr.detach().cpu().tolist()
    allowed_counts = num_allowed_token_ids_ptr.detach().cpu().tolist()
    bias_counts = num_logit_bias_ptr.detach().cpu().tolist()
    stop_counts = num_stop_token_ids_ptr.detach().cpu().tolist()
    positions = pos_ptr.detach().cpu().tolist()
    min_lens = min_lens_ptr.detach().cpu().tolist()
    for row, req_idx in enumerate(mapping):
        logits = logits_ptr[row, :vocab_size]
        if allowed_counts[req_idx]:
            ids = allowed_token_ids_ptr[req_idx, : allowed_counts[req_idx]].long()
            ids = ids[(ids >= 0) & (ids < vocab_size)]
            saved = logits[ids].clone()
            logits.fill_(float("-inf"))
            logits[ids] = saved
        if bias_counts[req_idx]:
            count = bias_counts[req_idx]
            ids = bias_token_ids_ptr[req_idx, :count].long()
            valid = (ids >= 0) & (ids < vocab_size)
            logits[ids[valid]] += bias_ptr[req_idx, :count][valid]
        if stop_counts[req_idx] and positions[row] + 1 < min_lens[req_idx]:
            ids = stop_token_ids_ptr[req_idx, : stop_counts[req_idx]].long()
            saved = logits[ids].clone()
            logits[ids] = float("-inf")
            if CHECK_ALL_MASKED_ROWS and bool(restore_when_all_masked_ptr[req_idx]):
                if bool((logits == float("-inf")).all()):
                    finite = torch.isfinite(saved)
                    logits[ids[finite]] = saved[finite]


@cache
def register_sampling_kernel_overrides() -> None:
    """Register once per 310P worker, before upstream sampling helpers launch."""
    # The upstream bias wrapper calculates its launch block size on the host.
    # TritonPlaceholder has cdiv but omits this equally device-independent helper.
    if not hasattr(triton, "next_power_of_2"):
        triton.next_power_of_2 = next_power_of_2
    register_kernels(
        {
            "vllm.v1.worker.gpu.sample.gumbel._temperature_kernel": temperature_kernel,
            "vllm.v1.worker.gpu.sample.min_p._min_p_kernel": min_p_kernel,
            "vllm.v1.worker.gpu.sample.logit_bias._bias_kernel": bias_kernel,
        }
    )
