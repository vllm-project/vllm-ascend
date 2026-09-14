# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# mypy: ignore-errors

"""Greedy rejection sampler for 310P MRv2 MTP (no Triton)."""

from __future__ import annotations

import torch
from vllm.config import SpeculativeConfig
from vllm.utils.platform_utils import is_pin_memory_available
from vllm.v1.sample.rejection_sampler import PLACEHOLDER_TOKEN_ID
from vllm.v1.utils import CpuGpuBuffer
from vllm.v1.worker.gpu.sample.output import SamplerOutput

from vllm_ascend._310p.worker.v2.input_batch import Ascend310PInputBatch
from vllm_ascend.sample.rejection_sampler import (
    rejection_greedy_sample_pytorch,
)


class RejectionSampler310V2:
    """Greedy MTP rejection sampler for 310P MRv2."""

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

    def __call__(
        self,
        logits: torch.Tensor,
        input_batch: Ascend310PInputBatch,
        draft_logits: torch.Tensor | None,
    ) -> SamplerOutput:
        del draft_logits
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
        return SamplerOutput(
            sampled_token_ids=sampled,
            logprobs_tensors=None,
            num_nans=None,
            num_sampled=num_sampled,
            num_rejected=num_rejected,
        )

    def synchronize_cpu(self) -> None:
        """Wait only when CPU request bookkeeping consumes sampled results."""
        self._copy_event.synchronize()
