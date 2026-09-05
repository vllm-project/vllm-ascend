# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Greedy rejection sampler for 310P MRv2 MTP (no Triton)."""

from __future__ import annotations

import torch
from vllm.config import SpeculativeConfig
from vllm.v1.worker.gpu.sample.output import SamplerOutput

from vllm_ascend._310p.worker.v2.spec_utils import (
    get_num_sampled_and_rejected_cpu,
    greedy_rejection_sample_cpu,
)
from vllm_ascend.worker.v2.input_batch import AscendInputBatch


class RejectionSampler310V2:
    """Greedy MTP rejection sampler for 310P MRv2."""

    def __init__(
        self,
        sampler,
        spec_config: SpeculativeConfig,
        device: torch.device,
    ) -> None:
        del device
        self.sampler = sampler
        self.num_speculative_steps = spec_config.num_speculative_tokens

    def __call__(
        self,
        logits: torch.Tensor,
        input_batch: AscendInputBatch,
        draft_logits: torch.Tensor | None,
    ) -> SamplerOutput:
        del draft_logits
        draft_sampled = input_batch.input_ids[input_batch.logits_indices]
        sampled, num_sampled = greedy_rejection_sample_cpu(
            logits,
            draft_sampled,
            input_batch.cu_num_logits,
            self.num_speculative_steps,
        )
        num_sampled, num_rejected = get_num_sampled_and_rejected_cpu(
            num_sampled,
            input_batch.seq_lens,
            input_batch.cu_num_logits,
            input_batch.idx_mapping_np,
            input_batch.prefill_len_np,
        )
        return SamplerOutput(
            sampled_token_ids=sampled,
            logprobs_tensors=None,
            num_nans=None,
            num_sampled=num_sampled,
            num_rejected=num_rejected,
        )
