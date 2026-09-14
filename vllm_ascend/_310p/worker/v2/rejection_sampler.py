# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# mypy: ignore-errors

"""MTP rejection sampler for 310P MRv2 (greedy + temperature, no Triton)."""

from __future__ import annotations

import numpy as np
import torch
from vllm.config import SpeculativeConfig
from vllm.utils.platform_utils import is_pin_memory_available
from vllm.v1.sample.rejection_sampler import PLACEHOLDER_TOKEN_ID
from vllm.v1.utils import CpuGpuBuffer
from vllm.v1.worker.gpu.sample.output import SamplerOutput

from vllm_ascend._310p.worker.v2.input_batch import Ascend310PInputBatch
from vllm_ascend._310p.worker.v2.spec_utils import (
    get_num_sampled_and_rejected_cpu,
    probabilistic_rejection_sample_cpu,
)
from vllm_ascend.sample.rejection_sampler import (
    rejection_greedy_sample_pytorch,
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
        self.sampler = sampler
        self.num_speculative_steps = spec_config.num_speculative_tokens
        max_num_reqs = sampler.max_num_reqs
        max_num_drafts = max_num_reqs * self.num_speculative_steps
        self._target_rows = CpuGpuBuffer(
            max_num_drafts, dtype=torch.int64, device=device
        )
        self._draft_rows = CpuGpuBuffer(
            max_num_drafts, dtype=torch.int64, device=device
        )
        self._bonus_rows = CpuGpuBuffer(
            max_num_reqs, dtype=torch.int64, device=device
        )
        self._is_chunked_prefill = CpuGpuBuffer(
            max_num_reqs, dtype=torch.bool, device=device
        )
        pin_memory = is_pin_memory_available()
        self.sampled_tokens_cpu = torch.empty(
            (max_num_reqs, self.num_speculative_steps + 1),
            dtype=torch.int32,
            device="cpu",
            pin_memory=pin_memory,
        )
        self.num_sampled_cpu = torch.empty(
            max_num_reqs,
            dtype=torch.int32,
            device="cpu",
            pin_memory=pin_memory,
        )
        self.num_rejected_cpu = torch.empty_like(self.num_sampled_cpu)
        self._copy_stream = torch.npu.Stream()
        self._copy_event = torch.npu.Event()
        self._copy_pending = False

    def __call__(
        self,
        logits: torch.Tensor,
        input_batch: Ascend310PInputBatch,
        draft_logits: torch.Tensor | None,
    ) -> SamplerOutput:
        del draft_logits
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
        if any_random:
            assert input_batch.input_ids_cpu is not None
            assert input_batch.logits_indices_np is not None
            draft_sampled_cpu = input_batch.input_ids_cpu[torch.from_numpy(input_batch.logits_indices_np)]
            source_generators = getattr(self.sampler, "_source_generators", {})
            sampled, _, sampled_cpu, num_sampled_cpu = probabilistic_rejection_sample_cpu(
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
                idx_mapping_np,
                input_batch.prefill_len_np,
                logits.device,
            )
            self.sampled_tokens_cpu = sampled_cpu
            self.num_sampled_cpu = num_sampled_cpu
            self.num_rejected_cpu = num_rejected_cpu
            self._copy_pending = False
            return SamplerOutput(
                sampled_token_ids=sampled,
                logprobs_tensors=None,
                num_nans=None,
                num_sampled=num_sampled,
                num_rejected=num_rejected,
            )

        num_reqs = input_batch.num_reqs
        cu_num_logits_np = input_batch.cu_num_logits_np
        target_rows_np = self._target_rows.np
        draft_rows_np = self._draft_rows.np
        bonus_rows_np = self._bonus_rows.np
        num_draft_tokens: list[int] = []
        offset = 0
        for req_idx in range(num_reqs):
            start = int(cu_num_logits_np[req_idx])
            end = int(cu_num_logits_np[req_idx + 1])
            num_drafts = max(end - start - 1, 0)
            num_draft_tokens.append(num_drafts)
            target_rows_np[offset : offset + num_drafts] = range(start, end - 1)
            draft_rows_np[offset : offset + num_drafts] = range(start + 1, end)
            bonus_rows_np[req_idx] = end - 1
            offset += num_drafts

        self._target_rows.copy_to_gpu(offset)
        self._draft_rows.copy_to_gpu(offset)
        self._bonus_rows.copy_to_gpu(num_reqs)

        target_argmax = logits.argmax(dim=-1).to(dtype=torch.int32)
        sampled_inputs = input_batch.input_ids[input_batch.logits_indices]
        draft_token_ids = sampled_inputs.index_select(
            0, self._draft_rows.gpu[:offset]
        )
        draft_target_argmax = target_argmax.index_select(
            0, self._target_rows.gpu[:offset]
        )
        bonus_token_ids = target_argmax.index_select(
            0, self._bonus_rows.gpu[:num_reqs]
        ).view(-1, 1)

        sampled = torch.full(
            (num_reqs, self.num_speculative_steps + 1),
            PLACEHOLDER_TOKEN_ID,
            dtype=torch.int32,
            device=logits.device,
        )
        cu_num_draft_tokens = (
            input_batch.cu_num_logits[1 : num_reqs + 1]
            - torch.arange(1, num_reqs + 1, dtype=torch.int32, device=logits.device)
        )
        rejection_greedy_sample_pytorch(
            sampled,
            cu_num_draft_tokens,
            draft_token_ids,
            draft_target_argmax,
            bonus_token_ids,
            num_draft_tokens,
            self.num_speculative_steps,
        )

        num_sampled = (sampled != PLACEHOLDER_TOKEN_ID).sum(dim=1).to(torch.int32)
        num_logits = input_batch.cu_num_logits[1 : num_reqs + 1] - input_batch.cu_num_logits[:num_reqs]
        self._is_chunked_prefill.np[:num_reqs] = (
            input_batch.seq_lens_np[:num_reqs] < input_batch.prefill_len_np
        )
        self._is_chunked_prefill.copy_to_gpu(num_reqs)
        is_chunked_prefill = self._is_chunked_prefill.gpu[:num_reqs]
        num_sampled = torch.where(is_chunked_prefill, 0, num_sampled)
        num_rejected = torch.where(
            is_chunked_prefill,
            0,
            num_logits.to(torch.int32) - num_sampled,
        )

        main_stream = torch.npu.current_stream()
        with torch.npu.stream(self._copy_stream):
            self._copy_stream.wait_stream(main_stream)
            self.sampled_tokens_cpu[:num_reqs].copy_(sampled, non_blocking=True)
            self.num_sampled_cpu[:num_reqs].copy_(num_sampled, non_blocking=True)
            self.num_rejected_cpu[:num_reqs].copy_(num_rejected, non_blocking=True)
            self._copy_event.record()
            self._copy_pending = True
        return SamplerOutput(
            sampled_token_ids=sampled,
            logprobs_tensors=None,
            num_nans=None,
            num_sampled=num_sampled,
            num_rejected=num_rejected,
        )

    def synchronize_cpu(self) -> None:
        """Wait only when CPU request bookkeeping consumes sampled results."""
        if self._copy_pending:
            self._copy_event.synchronize()
            self._copy_pending = False
