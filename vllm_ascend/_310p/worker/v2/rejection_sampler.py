# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# mypy: ignore-errors

"""MTP rejection sampler for 310P MRv2 (greedy + temperature, no Triton)."""

from __future__ import annotations

import numpy as np
import torch
from vllm.config import SpeculativeConfig
from vllm.v1.worker.gpu.sample.output import SamplerOutput

from vllm_ascend._310p.worker.v2.input_batch import Ascend310PInputBatch
from vllm_ascend._310p.worker.v2.spec_utils import (
    get_num_sampled_and_rejected_cpu,
    greedy_rejection_sample_cpu,
    probabilistic_rejection_sample_cpu,
)

_SAMPLING_EPS = 1e-5


class RejectionSampler310V2:
    """MTP rejection sampler for 310P MRv2.

    - ``temperature≈0``: greedy argmax verify (``greedy_rejection_sample_cpu``).
    - ``temperature>0``: Leviathan / IS_NGRAM path aligned with MRV1
      ``AscendRejectionSampler310`` (accept iff ``u < p(draft)``, else recovered
      token from residual; bonus via inverse-CDF).
    """

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
        idx_mapping_np = input_batch.idx_mapping_np
        expanded_idx_mapping = input_batch.expanded_idx_mapping

        processed = self.sampler.apply_sampling_params(
            logits,
            expanded_idx_mapping,
            idx_mapping_np,
        )

        temperature_np = getattr(self.sampler, "_temperature_np", None)
        if temperature_np is None:
            temperature_np = np.zeros(int(idx_mapping_np.max(initial=0)) + 1, dtype=np.float32)

        any_random = bool(np.any(temperature_np[idx_mapping_np] >= _SAMPLING_EPS))
        if not any_random:
            sampled, num_sampled, sampled_cpu, num_sampled_cpu = greedy_rejection_sample_cpu(
                processed,
                draft_sampled_cpu,
                input_batch.cu_num_logits_np,
                self.num_speculative_steps,
            )
        else:
            source_generators = getattr(self.sampler, "_source_generators", {})
            sampled, num_sampled, sampled_cpu, num_sampled_cpu = probabilistic_rejection_sample_cpu(
                processed,
                draft_sampled_cpu,
                input_batch.cu_num_logits_np,
                self.num_speculative_steps,
                temperature_np,
                idx_mapping_np,
                source_generators,
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
