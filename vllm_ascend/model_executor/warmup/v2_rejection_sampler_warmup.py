# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Compile the model-dtype V2 rejection sampler variant before serving."""

from __future__ import annotations

import torch
from vllm.logger import logger

from vllm_ascend.worker.v2.spec_decode.rejection_sampler_utils import rejection_sample


@torch.inference_mode()
def warmup_v2_model_dtype_rejection_sampler(model_runner) -> None:
    """Cover logits dtypes missed by the scheduler's FP32 sampler warmup.

    The actual decode model can emit BF16/FP16 logits even though the generic
    V2 warmup produces FP32 logits. Triton specializes on the pointer dtype,
    which otherwise forces four rejection/resampling compilations on request.
    """
    if not model_runner.is_last_pp_rank or model_runner.num_speculative_steps <= 0:
        return

    target_dtype = model_runner.model_config.dtype
    if target_dtype == torch.float32:
        return

    draft_logits = model_runner.speculator.draft_logits
    if draft_logits is None:
        return

    num_spec_steps = model_runner.num_speculative_steps
    vocab_size = model_runner.model_config.get_vocab_size()
    draft_vocab_size = draft_logits.shape[-1]
    device = model_runner.device
    num_logits = num_spec_steps + 1

    target = torch.zeros((num_logits, vocab_size), dtype=target_dtype, device=device)
    draft = torch.zeros((1, num_spec_steps, draft_vocab_size), dtype=draft_logits.dtype, device=device)
    sampled = torch.zeros(num_logits, dtype=torch.int32, device=device)
    cu_num_logits = torch.tensor([0, num_logits], dtype=torch.int32, device=device)
    positions = torch.arange(num_logits, dtype=torch.int64, device=device)
    idx_mapping = torch.zeros(1, dtype=torch.int64, device=device)
    expanded_idx_mapping = torch.zeros(num_logits, dtype=torch.int64, device=device)
    expanded_local_pos = torch.arange(num_logits, dtype=torch.int32, device=device)
    temperature = torch.ones(1, dtype=torch.float32, device=device)
    seed = torch.zeros(1, dtype=torch.int64, device=device)

    rejection_sample(
        target,
        draft,
        sampled,
        cu_num_logits,
        positions,
        idx_mapping,
        expanded_idx_mapping,
        expanded_local_pos,
        temperature,
        seed,
        num_spec_steps,
    )
    logger.info("Warmed V2 rejection sampler for %s logits.", target_dtype)
