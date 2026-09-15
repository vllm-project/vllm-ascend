# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# mypy: ignore-errors

"""Greedy rejection sampler for 310P MRv2 MTP (no Triton)."""

from __future__ import annotations

import torch
from vllm.config import SpeculativeConfig
from vllm.utils.platform_utils import is_pin_memory_available
from vllm.v1.sample.rejection_sampler import PLACEHOLDER_TOKEN_ID
from vllm.v1.worker.gpu.sample.output import SamplerOutput
from vllm.v1.worker.gpu.spec_decode.rejection_sampler import RejectionSampler

from vllm_ascend._310p.worker.v2.input_batch import Ascend310PInputBatch
from vllm_ascend.sample.rejection_sampler import (
    rejection_greedy_sample_pytorch,
    rejection_greedy_sample_spec_len_1_pytorch,
)


class RejectionSampler310V2(RejectionSampler):
    """Greedy MTP rejection sampler for 310P MRv2."""

    def __init__(
        self,
        sampler,
        spec_config: SpeculativeConfig,
        device: torch.device,
    ) -> None:
        super().__init__(sampler, spec_config, device)
        max_num_reqs = sampler.max_num_reqs
        self.device = device
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
        num_draft_tokens = (
            cu_num_logits_np[1 : num_reqs + 1]
            - cu_num_logits_np[:num_reqs]
            - 1
        ).tolist()
        cu_num_logits = input_batch.cu_num_logits[: num_reqs + 1]
        draft_counts = cu_num_logits[1:] - cu_num_logits[:-1] - 1
        cu_num_draft_tokens = cu_num_logits[1:] - torch.arange(
            1, num_reqs + 1, dtype=torch.int32, device=logits.device
        )

        if min(num_draft_tokens) == 1 and max(num_draft_tokens) == 1:
            # MTP1: every request contributes one target row and one draft row.
            target_rows = cu_num_logits[:-1].to(torch.int64)
        else:
            # Variable/MTP2+: remove one interleaved bonus row per request.
            token_req_ids = torch.repeat_interleave(
                torch.arange(num_reqs, device=logits.device), draft_counts
            )
            target_rows = torch.arange(
                int(sum(num_draft_tokens)), device=logits.device
            ) + token_req_ids
        draft_rows = target_rows + 1
        bonus_rows = (cu_num_logits[1:] - 1).to(torch.int64)

        sampled_inputs = input_batch.input_ids[input_batch.logits_indices]
        draft_token_ids = sampled_inputs.index_select(0, draft_rows)
        target_logits = logits.index_select(0, target_rows)
        bonus_token_ids = logits.index_select(0, bonus_rows).argmax(
            dim=-1
        ).to(torch.int32).view(-1, 1)

        target_argmax = target_logits.argmax(dim=-1).to(torch.int32)
        sampled = torch.full(
            (num_reqs, self.num_speculative_steps + 1),
            PLACEHOLDER_TOKEN_ID,
            dtype=torch.int32,
            device=logits.device,
        )
        if min(num_draft_tokens) == 1 and max(num_draft_tokens) == 1:
            rejection_greedy_sample_spec_len_1_pytorch(
                sampled,
                draft_token_ids,
                target_argmax,
                bonus_token_ids,
            )
        else:
            rejection_greedy_sample_pytorch(
                sampled,
                cu_num_draft_tokens,
                draft_token_ids,
                target_argmax,
                bonus_token_ids,
                num_draft_tokens,
                self.num_speculative_steps,
            )

        num_sampled = (sampled != -1).sum(dim=1).to(torch.int32)
        num_logits = cu_num_logits[1:] - cu_num_logits[:-1]
        is_chunked_prefill = torch.from_numpy(
            input_batch.seq_lens_np[:num_reqs] < input_batch.prefill_len_np
        ).to(device=self.device, non_blocking=True)
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
