# Adapt from https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu/model_runner.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#

from dataclasses import replace

import numpy as np
import torch
from vllm.config import VllmConfig
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.buffer_utils import async_copy_to_gpu
from vllm.v1.worker.gpu.pcp_manager import PCPManager
from vllm.v1.worker.gpu.states import RequestState

from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.worker.v2.attn_utils import build_attn_state
from vllm_ascend.worker.v2.input_batch import AscendInputBatch


class AscendPCPManager(PCPManager):
    """PCP manager that refreshes Ascend-only local-batch metadata."""

    @staticmethod
    def validate_config(
        vllm_config: VllmConfig,
        supports_mm_inputs: bool,
    ) -> None:
        """Validate the Ascend MRV2 MLA and GQA PCP implementations."""
        parallel_config = vllm_config.parallel_config
        model_config = vllm_config.model_config
        if parallel_config.prefill_context_parallel_size <= 1:
            return

        if parallel_config.decode_context_parallel_size > 1:
            raise NotImplementedError("Ascend MRV2 does not support PCP and DCP simultaneously yet.")
        if parallel_config.pipeline_parallel_size > 1:
            raise NotImplementedError("Ascend MRV2 PCP does not support PP yet.")
        if model_config.is_encoder_decoder:
            raise NotImplementedError("Ascend MRV2 PCP does not support encoder-decoder models yet.")
        if supports_mm_inputs:
            raise NotImplementedError("Ascend MRV2 PCP does not support MM inputs yet.")
        if vllm_config.lora_config is not None:
            raise NotImplementedError("Ascend MRV2 PCP does not support LoRA yet.")

    def __init__(
        self,
        pcp_world_size: int,
        pcp_rank: int,
        device: torch.device,
        vllm_config: VllmConfig | None = None,
        req_states: RequestState | None = None,
        max_num_reqs: int | None = None,
        max_num_tokens: int | None = None,
        block_tables: BlockTables | None = None,
        dcp_world_size: int = 1,
        dcp_rank: int = 0,
        cp_interleave: int = 1,
    ) -> None:
        super().__init__(
            pcp_world_size,
            pcp_rank,
            device,
            req_states=req_states,
            max_num_reqs=max_num_reqs,
            max_num_tokens=max_num_tokens,
            block_tables=block_tables,
            dcp_world_size=dcp_world_size,
            dcp_rank=dcp_rank,
            cp_interleave=cp_interleave,
        )
        self.vllm_config = vllm_config

    def partition_batch(
        self,
        input_batch: AscendInputBatch,
    ) -> AscendInputBatch:
        """Reuse upstream PCP partitioning and refresh Ascend-only metadata."""
        assert self.vllm_config is not None
        graph_num_reqs = input_batch.num_reqs_after_padding
        graph_num_tokens = input_batch.num_tokens_after_padding
        is_decode_only = not np.any(input_batch.is_prefilling_np)

        upstream_input_batch = input_batch
        if input_batch.num_draft_tokens > 0:
            # The upstream implementation only uses num_draft_tokens to reject
            # speculative decoding, then clears the two draft fields on the
            # returned rank-local batch. Present the same post-guard view while
            # retaining the real global batch for proposal and sampling.
            upstream_input_batch = replace(
                input_batch,
                num_draft_tokens=0,
                num_draft_tokens_per_req=None,
            )

        try:
            local_batch = super().partition_batch(upstream_input_batch)
        finally:
            self._global_batch = input_batch

        assert isinstance(local_batch, AscendInputBatch)
        local_seq_lens_np = local_batch.num_computed_tokens_np + local_batch.num_scheduled_tokens
        if is_decode_only:
            local_batch.attn_state = input_batch.attn_state
            if local_batch.attn_state is None:
                local_batch.attn_state = AscendAttentionState.DecodeOnly

            return self._pad_decode_batch_for_full_graph(
                local_batch,
                graph_num_reqs,
                graph_num_tokens,
                local_seq_lens_np,
            )

        local_num_valid_tokens = local_batch.num_scheduled_tokens
        if local_batch.num_draft_tokens_per_req is not None:
            local_num_valid_tokens = local_batch.num_scheduled_tokens - local_batch.num_draft_tokens_per_req
        local_batch.attn_state = build_attn_state(
            self.vllm_config,
            local_seq_lens_np,
            local_batch.num_reqs,
            local_batch.num_scheduled_tokens,
            local_num_valid_tokens,
        )
        local_batch.seq_lens_np = local_seq_lens_np
        return local_batch

    def prepare_speculator_attn(
        self,
        input_batch: AscendInputBatch,
    ) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
        """Build the global KV layout consumed by a PCP speculator."""
        if input_batch is not self._global_batch:
            raise RuntimeError("PCP speculative proposal must use the restored global batch.")
        assert self._block_tables is not None
        block_tables = self._block_tables.gather_block_tables(
            input_batch.idx_mapping,
            num_reqs_padded=input_batch.num_reqs_after_padding,
        )
        assert self._global_batch_slot_mappings is not None
        num_tokens = input_batch.num_tokens_after_padding
        if num_tokens > input_batch.num_tokens:
            self._global_batch_slot_mappings[
                :,
                input_batch.num_tokens : num_tokens,
            ].fill_(PAD_SLOT_ID)
        global_slot_mappings = self._global_batch_slot_mappings[:, :num_tokens]

        assert self._gathered_kv_slot_mappings is not None
        num_expanded_tokens = num_tokens * self.pcp_world_size
        if num_expanded_tokens > self._gathered_kv_slot_mappings.shape[1]:
            raise RuntimeError(
                "PCP speculator slot mapping exceeds the persistent buffer "
                f"capacity: {num_expanded_tokens} > "
                f"{self._gathered_kv_slot_mappings.shape[1]}."
            )
        expanded_slot_mappings = self._gathered_kv_slot_mappings[:, :num_expanded_tokens]
        for rank in range(self.pcp_world_size):
            rank_start = rank * num_tokens
            expanded_slot_mappings[
                :,
                rank_start : rank_start + num_tokens,
            ].copy_(global_slot_mappings)
        return block_tables, expanded_slot_mappings

    def _pad_decode_batch_for_full_graph(
        self,
        local_batch: AscendInputBatch,
        graph_num_reqs: int,
        graph_num_tokens: int,
        local_seq_lens_np: np.ndarray,
    ) -> AscendInputBatch:
        """Pad a rank-local decode batch to the selected full-graph shape."""
        num_reqs = local_batch.num_reqs
        num_tokens = local_batch.num_tokens
        if graph_num_reqs < num_reqs or graph_num_tokens < num_tokens:
            raise RuntimeError(
                "PCP graph shape is smaller than the rank-local decode batch: "
                f"requests {graph_num_reqs} < {num_reqs} or "
                f"tokens {graph_num_tokens} < {num_tokens}."
            )

        num_padding_reqs = graph_num_reqs - num_reqs
        num_padding_tokens = graph_num_tokens - num_tokens
        decode_query_len = getattr(self.vllm_config, "num_speculative_tokens", 0) + 1
        uses_per_request_padding = num_padding_tokens == num_padding_reqs * decode_query_len
        uses_single_fia_dummy = num_padding_reqs == 1 and num_padding_tokens > 0
        if not (uses_per_request_padding or uses_single_fia_dummy):
            raise RuntimeError(
                "PCP FULL_DECODE_ONLY requires either a uniform decode query "
                "per padded request or one FIA dummy request for all padding "
                "tokens: "
                f"{num_padding_tokens} tokens for {num_padding_reqs} requests."
            )

        # Full-graph model inputs keep using the model runner's original
        # buffers, while PCP attention metadata uses the rank-local buffers.
        # Clear both views so a smaller replay cannot observe stale padding
        # left by a previously larger decode batch.
        # Clear the full image buffer to prevent dirty data from polluting the next image.
        global_batch = self._global_batch
        assert global_batch is not None
        global_batch.input_ids[num_tokens:graph_num_tokens].zero_()
        global_batch.positions[num_tokens:graph_num_tokens].zero_()
        global_batch.is_padding[:num_tokens].fill_(False)
        global_batch.is_padding[num_tokens:graph_num_tokens].fill_(True)

        if num_padding_reqs == 0:
            local_batch.seq_lens_np = local_seq_lens_np
            return local_batch

        input_buffers = self._input_buffers
        assert input_buffers is not None
        if graph_num_reqs > input_buffers.max_num_reqs:
            raise RuntimeError(
                "PCP graph request count exceeds the local input buffer capacity: "
                f"{graph_num_reqs} > {input_buffers.max_num_reqs}."
            )
        if graph_num_tokens > input_buffers.max_num_tokens:
            raise RuntimeError(
                "PCP graph token count exceeds the local input buffer capacity: "
                f"{graph_num_tokens} > {input_buffers.max_num_tokens}."
            )

        input_buffers.input_ids[num_tokens:graph_num_tokens].zero_()
        input_buffers.positions[num_tokens:graph_num_tokens].zero_()
        input_buffers.seq_lens[num_reqs:graph_num_reqs].zero_()
        input_buffers.is_padding[:num_tokens].fill_(False)
        input_buffers.is_padding[num_tokens:graph_num_tokens].fill_(True)

        query_start_loc_buffer_np = np.full(
            input_buffers.max_num_reqs + 1,
            graph_num_tokens,
            dtype=np.int32,
        )
        query_start_loc_buffer_np[: num_reqs + 1] = local_batch.query_start_loc_np
        if uses_per_request_padding:
            query_start_loc_buffer_np[num_reqs + 1 : graph_num_reqs + 1] = num_tokens + decode_query_len * np.arange(
                1,
                num_padding_reqs + 1,
                dtype=np.int32,
            )
        else:
            query_start_loc_buffer_np[num_reqs + 1 : graph_num_reqs + 1] = graph_num_tokens
        async_copy_to_gpu(
            query_start_loc_buffer_np,
            out=input_buffers.query_start_loc,
        )
        query_start_loc_np = query_start_loc_buffer_np[: graph_num_reqs + 1]

        padded_seq_lens_np = np.zeros(graph_num_reqs, dtype=np.int32)
        padded_seq_lens_np[:num_reqs] = local_seq_lens_np
        seq_lens_cpu_upper_bound = torch.zeros(graph_num_reqs, dtype=torch.int32)
        seq_lens_cpu_upper_bound[:num_reqs].copy_(local_batch.seq_lens_cpu_upper_bound)

        return replace(
            local_batch,
            num_reqs_after_padding=graph_num_reqs,
            num_tokens_after_padding=graph_num_tokens,
            query_start_loc=input_buffers.query_start_loc[: graph_num_reqs + 1],
            query_start_loc_np=query_start_loc_np,
            seq_lens=input_buffers.seq_lens[:graph_num_reqs],
            seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,
            input_ids=input_buffers.input_ids[:graph_num_tokens],
            positions=input_buffers.positions[:graph_num_tokens],
            is_padding=input_buffers.is_padding[:graph_num_tokens],
            seq_lens_np=padded_seq_lens_np,
        )

    def prepare_dummy_attn(
        self,
        num_reqs: int,
        num_tokens: int,
    ) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
        """Prepare capture inputs using the same buffers as graph replay."""
        assert self._local_block_tables is not None
        for block_table in self._local_block_tables:
            block_table[:num_reqs].zero_()
        return (
            tuple(block_table[:num_reqs] for block_table in self._local_block_tables),
            self.get_dummy_slot_mappings(num_tokens),
        )

    def prepare_slot_mappings(self) -> torch.Tensor:
        slot_mappings = super().prepare_slot_mappings()
        assert self._global_batch is not None
        assert self._gathered_kv_slot_mappings is not None

        global_batch = self._global_batch
        if np.any(global_batch.is_prefilling_np):
            return slot_mappings

        graph_num_tokens = global_batch.num_tokens_after_padding
        if graph_num_tokens <= global_batch.num_tokens:
            return slot_mappings

        num_expanded_tokens = slot_mappings.shape[1]
        graph_num_expanded_tokens = graph_num_tokens * self.pcp_world_size
        if graph_num_expanded_tokens > self._gathered_kv_slot_mappings.shape[1]:
            raise RuntimeError(
                "PCP graph slot mapping exceeds the persistent buffer capacity: "
                f"{graph_num_expanded_tokens} > "
                f"{self._gathered_kv_slot_mappings.shape[1]}."
            )
        self._gathered_kv_slot_mappings[:, num_expanded_tokens:graph_num_expanded_tokens].fill_(PAD_SLOT_ID)
        return self._gathered_kv_slot_mappings[:, :graph_num_expanded_tokens]
