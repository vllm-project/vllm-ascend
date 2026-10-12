# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ascend v2 execution adapter for the upstream PARD-2 speculator."""

from typing import Any

import torch
from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.spec_decode.pard2.speculator import Pard2Speculator

from vllm_ascend.worker.v2.spec_decode.pard2.metadata import pard2_attn_metadata_context


class AscendPard2Speculator(Pard2Speculator):
    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        parallel = vllm_config.parallel_config
        if (
            parallel.pipeline_parallel_size != 1
            or parallel.prefill_context_parallel_size != 1
            or parallel.decode_context_parallel_size != 1
        ):
            raise ValueError("PARD-2 on Ascend currently requires PP=PCP=DCP=1.")
        # Both draft passes remain eager until ACL graph replay is validated.
        vllm_config.speculative_config.enforce_eager = True
        super().__init__(vllm_config, device)

    def init_cudagraph_manager(self, cudagraph_mode: CUDAGraphMode) -> None:
        # NONE keeps upstream's batch dispatch without allocating a graph pool
        # or capture buckets. DFlash's ACL replay requires DFlash-specific
        # metadata callbacks that PARD-2 does not implement. Do not wire that
        # replay path into this eager-only adapter, even if the target uses graphs.
        super().init_cudagraph_manager(CUDAGraphMode.NONE)

    def capture(self) -> None:
        # The draft always runs eagerly, even when the target uses graphs.
        return

    def _build_uniform_attn_metadata(
        self,
        batch_desc,
        num_reqs: int,
        num_query_per_req: int,
        seq_lens_cpu_upper_bound: torch.Tensor,
        step: int,
        causal: bool = True,
        dcp_local_seq_lens: torch.Tensor | None = None,
    ) -> dict[str, Any] | None:
        with pard2_attn_metadata_context(
            positions=self.input_buffers.positions,
            num_tokens=batch_desc.num_tokens,
            is_prefilling=self.draft_is_prefilling[:num_reqs],
            parallel_config=self.attn_vllm_config.parallel_config,
        ):
            return super()._build_uniform_attn_metadata(
                batch_desc=batch_desc,
                num_reqs=num_reqs,
                num_query_per_req=num_query_per_req,
                seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,
                step=step,
                causal=causal,
                dcp_local_seq_lens=dcp_local_seq_lens,
            )

    def _fill_context_hidden_states(
        self,
        input_batch: InputBatch,
        target_hidden_states: torch.Tensor,
        last_accepted_rows: torch.Tensor,
        num_reqs: int,
        num_tokens: int,
    ) -> None:
        super()._fill_context_hidden_states(input_batch, target_hidden_states, last_accepted_rows, num_reqs, num_tokens)
        # The context forward uses the target's padded batch shape. Clear stale
        # features left by a previous, larger batch without changing carry state.
        self.hidden_states[num_tokens : input_batch.num_tokens_after_padding].zero_()

    def _generate_draft(
        self,
        num_reqs: int,
        num_tokens_padded: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    ) -> None:
        # DP synchronization may pad an eager query batch as well.
        num_query_tokens = num_reqs * self.num_query_per_req
        self.hidden_states[num_query_tokens:num_tokens_padded].zero_()
        super()._generate_draft(
            num_reqs,
            num_tokens_padded,
            attn_metadata,
            slot_mappings,
            num_tokens_across_dp,
            cudagraph_runtime_mode,
        )
