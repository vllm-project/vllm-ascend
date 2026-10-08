# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from typing import TYPE_CHECKING, Any

import torch
from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.spec_decode.speculator import BaseSpeculator

from vllm_ascend.ops.triton.spec_decode.ngram import triton_ngram_spec_decode

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.dp_utils import DPSyncState
    from vllm.v1.worker.gpu.input_batch import InputBatch
    from vllm.v1.worker.gpu.states import RequestState


class AscendNgramSpeculator(BaseSpeculator):
    """NGram proposal backed by the existing Ascend NPU kernel."""

    supports_mm_inputs = False
    draft_logits = None

    def __init__(self, vllm_config: VllmConfig, device: torch.device, req_states: "RequestState"):
        spec = vllm_config.speculative_config
        assert spec is not None
        self.req_states = req_states
        assert spec.prompt_lookup_min is not None
        assert spec.prompt_lookup_max is not None
        self.min_n = spec.prompt_lookup_min
        self.max_n = spec.prompt_lookup_max
        self.num_speculative_steps = spec.num_speculative_tokens
        self.vocab_size = vllm_config.model_config.get_vocab_size()
        self.drafts = torch.zeros(
            (vllm_config.scheduler_config.max_num_seqs, self.num_speculative_steps),
            dtype=torch.int64,
            device=device,
        )
        self.draft_offsets = torch.arange(self.num_speculative_steps, device=device)

    def init_cudagraph_manager(self, cudagraph_mode: CUDAGraphMode) -> None:
        del cudagraph_mode

    def capture(self) -> None:
        return None

    @torch.inference_mode()
    def propose(
        self,
        input_batch: "InputBatch",
        attn_metadata: Any,
        slot_mappings: Any,
        last_hidden_states: torch.Tensor,
        aux_hidden_states: list[torch.Tensor] | None,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        last_sampled: torch.Tensor,
        next_prefill_tokens: torch.Tensor,
        temperature: torch.Tensor,
        seeds: torch.Tensor,
        dp_sync: "DPSyncState | None" = None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
        mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
        is_profile: bool = False,
        num_speculative_tokens: int | None = None,
    ) -> torch.Tensor:
        batch_size = input_batch.num_reqs
        drafts = self.drafts[:batch_size]
        if dummy_run or batch_size == 0:
            drafts.zero_()
            return drafts
        k = (
            self.num_speculative_steps
            if num_speculative_tokens is None
            else min(num_speculative_tokens, self.num_speculative_steps)
        )
        drafts.fill_(-1)
        if k == 0:
            return drafts

        indices = input_batch.idx_mapping[:batch_size]
        device = self.drafts.device
        indices = indices.to(device=device)
        safe_indices = indices.clamp(min=0).long()
        # index_select owns its storage: the V1 kernel must not append to the
        # authoritative history, which MRV2 post_update has already committed.
        history = self.req_states.all_token_ids.gpu.index_select(0, safe_indices)
        total_len = self.req_states.total_len.gpu.index_select(0, safe_indices)
        counts = num_sampled[:batch_size].to(device=device).clamp(min=0, max=self.num_speculative_steps + 1)
        eligible = (indices >= 0) & (counts > 0)
        prefix_len = (total_len - counts).clamp(min=0).to(torch.int32)
        sample_width = self.num_speculative_steps + 1
        offsets = torch.arange(sample_width, dtype=torch.int32, device=device)
        positions = prefix_len[:, None] + offsets[None, :]
        positions = positions.clamp(min=0, max=max(history.shape[1] - 1, 0))
        sampled = history.gather(1, positions.long()).to(torch.int32)
        sampled = sampled.masked_fill(offsets[None, :] >= counts[:, None], -1)
        sampled = sampled.masked_fill(~eligible[:, None], -1)
        _, proposed, valid_len, _ = triton_ngram_spec_decode(
            history,
            prefix_len,
            sampled,
            ~eligible,
            self.vocab_size,
            self.min_n,
            self.max_n,
            k,
        )
        # Unmatched slots repeat the last token, as in upstream #40704; the V1
        # valid length is not sent to the scheduler.
        last_sampled_batch = last_sampled.view(-1).to(device=device).index_select(0, safe_indices)
        fallback = last_sampled_batch[:, None].expand(-1, k)
        valid = self.draft_offsets[:k][None, :] < valid_len[:, None]
        drafts[:, :k].copy_(torch.where(valid, proposed.to(drafts.dtype), fallback))
        drafts.masked_fill_((indices < 0)[:, None], -1)
        return drafts
