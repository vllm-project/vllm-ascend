# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# mypy: ignore-errors

"""Greedy rejection sampler for 310P MRv2 MTP (no Triton)."""

from __future__ import annotations

import torch
from vllm.config import SpeculativeConfig
from vllm.v1.worker.gpu.sample.output import SamplerOutput

from vllm_ascend._310p.worker.v2.input_batch import Ascend310PInputBatch
from vllm_ascend._310p.worker.v2.spec_utils import (
    get_num_sampled_and_rejected_cpu,
    greedy_rejection_sample_cpu,
)


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
        input_batch: Ascend310PInputBatch,
        draft_logits: torch.Tensor | None,
    ) -> SamplerOutput:
        del draft_logits
        assert input_batch.input_ids_cpu is not None
        assert input_batch.logits_indices_np is not None
        draft_sampled_cpu = input_batch.input_ids_cpu[torch.from_numpy(input_batch.logits_indices_np)]
        sampled, num_sampled, sampled_cpu, num_sampled_cpu = greedy_rejection_sample_cpu(
            logits,
            draft_sampled_cpu,
            input_batch.cu_num_logits_np,
            self.num_speculative_steps,
        )
        num_sampled, num_rejected, num_sampled_cpu, num_rejected_cpu = get_num_sampled_and_rejected_cpu(
            num_sampled_cpu,
            input_batch.seq_lens_np,
            input_batch.cu_num_logits_np,
            input_batch.idx_mapping_np,
            input_batch.prefill_len_np,
            logits.device,
        )
        # Publish host bookkeeping produced by rejection sampling.
        # postprocess_sampled must not copy these tensors back from NPU again.
        self.sampled_tokens_cpu = sampled_cpu
        self.num_sampled_cpu = num_sampled_cpu
        self.num_rejected_cpu = num_rejected_cpu
        return SamplerOutput(
            sampled_token_ids=sampled,
            logprobs_tensors=None,
            num_nans=None,
            num_sampled=num_sampled,
            num_rejected=num_rejected,
        )
