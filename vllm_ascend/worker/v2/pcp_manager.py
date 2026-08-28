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

from dataclasses import dataclass, replace

import torch
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.pcp_manager import PCPManager
from vllm.v1.worker.gpu.states import RequestState

from vllm_ascend.worker.v2.attn_utils import build_attn_state
from vllm_ascend.worker.v2.input_batch import AscendInputBatch, AscendInputBuffers


@dataclass(frozen=True)
class AscendPCPAttentionContext:
    """Canonical global PCP view for one attention step."""

    # The global batch and its associated metadata, used to build DSA attention metadata.
    global_batch: AscendInputBatch
    global_block_tables: tuple[torch.Tensor, ...]
    global_slot_mappings: torch.Tensor
    hidden_restore_idx: torch.Tensor
    local_num_tokens_after_padding: int


class AscendPCPManager(PCPManager):
    """PCP manager that refreshes Ascend-only local-batch metadata."""

    vllm_config: VllmConfig

    def __init__(
        self,
        pcp_world_size: int,
        pcp_rank: int,
        device: torch.device,
        req_states: RequestState | None = None,
        max_num_reqs: int | None = None,
        max_num_tokens: int | None = None,
        block_tables: BlockTables | None = None,
        dcp_world_size: int = 1,
        dcp_rank: int = 0,
        cp_interleave: int = 1,
    ) -> None:
        super().__init__(
            pcp_world_size=pcp_world_size,
            pcp_rank=pcp_rank,
            device=device,
            req_states=req_states,
            max_num_reqs=max_num_reqs,
            max_num_tokens=max_num_tokens,
            block_tables=block_tables,
            dcp_world_size=dcp_world_size,
            dcp_rank=dcp_rank,
            cp_interleave=cp_interleave,
        )

        # vLLM #53515 made the PCP-local buffers persistent and uses them for
        # graph capture. Preserve that ownership while providing the extra CPU
        # and NumPy sequence-length views required by AscendInputBatch.
        if max_num_reqs is not None and max_num_tokens is not None:
            self._input_buffers = AscendInputBuffers(
                max_num_reqs=2 * max_num_reqs,
                max_num_tokens=max_num_tokens,
                device=device,
            )
            # The upstream PCP manager copies exactly max_num_reqs + 1
            # offsets into the whole query_start_loc tensor. AscendInputBuffers
            # normally reserves one additional FIA padding slot, but PCP never
            # uses that slot; expose the exact upstream-sized view here.
            self._input_buffers.query_start_loc = self._input_buffers.query_start_loc[:-1]

    @staticmethod
    def validate_config(
        vllm_config: VllmConfig,
        supports_mm_inputs: bool,
    ) -> None:
        """Validate the graph-safe Ascend MRV2 PCP configuration."""
        parallel_config = vllm_config.parallel_config
        model_config = vllm_config.model_config
        pcp_size = parallel_config.prefill_context_parallel_size
        if pcp_size <= 1:
            return

        if parallel_config.pipeline_parallel_size > 1:
            raise NotImplementedError("MRV2 PCP does not support PP yet.")
        if model_config.is_encoder_decoder:
            raise NotImplementedError("MRV2 PCP does not support encoder-decoder models yet.")
        if supports_mm_inputs:
            raise NotImplementedError("MRV2 PCP does not support MM inputs yet.")
        if vllm_config.lora_config is not None:
            raise NotImplementedError("MRV2 PCP does not support LoRA yet.")
        if vllm_config.speculative_config is not None:
            raise NotImplementedError("MRV2 PCP does not support speculative decoding yet.")

        is_sparse_mla = hasattr(model_config.hf_text_config, "index_topk")
        cudagraph_mode = vllm_config.compilation_config.cudagraph_mode
        if is_sparse_mla and cudagraph_mode not in {
            CUDAGraphMode.NONE,
            CUDAGraphMode.FULL_DECODE_ONLY,
        }:
            raise NotImplementedError("MRV2 sparse MLA PCP supports eager mode or FULL_DECODE_ONLY CUDA graphs only.")
        if cudagraph_mode.has_full_cudagraphs() and cudagraph_mode != CUDAGraphMode.FULL_DECODE_ONLY:
            raise NotImplementedError("MRV2 PCP supports FULL_DECODE_ONLY CUDA graphs only.")

    def partition_batch(self, input_batch: AscendInputBatch) -> AscendInputBatch:
        """Partition the batch and update Ascend-specific local metadata."""
        local_batch = super().partition_batch(input_batch)
        assert isinstance(local_batch, AscendInputBatch)

        # PCP builds the local layout from actual tokens, but a FULL decode
        # graph replays a fixed padded layout on every rank.
        graph_num_tokens = input_batch.num_tokens_after_padding
        is_decode_only = not bool(input_batch.is_prefilling_np.any())
        # FULL_DECODE_ONLY graphs capture one token for every padded request.
        # Keep the request-shaped metadata at that same fixed graph extent.
        graph_num_reqs = graph_num_tokens if is_decode_only else input_batch.num_reqs_after_padding
        if is_decode_only and graph_num_tokens > local_batch.num_tokens_after_padding:
            assert self._input_buffers is not None
            input_buffers = self._input_buffers
            actual_tokens = local_batch.num_tokens
            actual_reqs = local_batch.num_reqs
            if graph_num_tokens > input_buffers.max_num_tokens:
                raise RuntimeError(
                    "PCP graph token count exceeds the local input buffer: "
                    f"{graph_num_tokens} > {input_buffers.max_num_tokens}."
                )
            if graph_num_reqs > input_buffers.max_num_reqs:
                raise RuntimeError(
                    "PCP graph request count exceeds the local input buffer: "
                    f"{graph_num_reqs} > {input_buffers.max_num_reqs}."
                )
            input_buffers.input_ids[actual_tokens:graph_num_tokens].zero_()
            input_buffers.positions[actual_tokens:graph_num_tokens].zero_()
            input_buffers.is_padding[actual_tokens:graph_num_tokens].fill_(True)
            input_buffers.seq_lens[actual_reqs:graph_num_reqs].zero_()
            input_buffers.query_start_loc[actual_reqs + 1 : graph_num_reqs + 1].fill_(actual_tokens)
            seq_lens_cpu_upper_bound = torch.zeros(
                graph_num_reqs,
                dtype=local_batch.seq_lens_cpu_upper_bound.dtype,
            )
            seq_lens_cpu_upper_bound[:actual_reqs].copy_(local_batch.seq_lens_cpu_upper_bound[:actual_reqs])
            local_batch = replace(  # type: ignore[call-arg]
                local_batch,
                num_reqs_after_padding=graph_num_reqs,
                num_tokens_after_padding=graph_num_tokens,
                query_start_loc=input_buffers.query_start_loc[: graph_num_reqs + 1],
                seq_lens=input_buffers.seq_lens[:graph_num_reqs],
                seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,
                input_ids=input_buffers.input_ids[:graph_num_tokens],
                positions=input_buffers.positions[:graph_num_tokens],
                is_padding=input_buffers.is_padding[:graph_num_tokens],
            )

        local_batch.seq_lens_np = local_batch.num_computed_tokens_np + local_batch.num_scheduled_tokens
        num_valid_tokens = local_batch.num_scheduled_tokens
        if local_batch.num_draft_tokens_per_req is not None:
            num_valid_tokens = num_valid_tokens - local_batch.num_draft_tokens_per_req
        local_batch.attn_state = build_attn_state(
            self.vllm_config,
            local_batch.seq_lens_np,
            local_batch.num_reqs,
            local_batch.num_scheduled_tokens,
            num_valid_tokens,
        )
        return local_batch

    def prepare_slot_mappings(self) -> torch.Tensor:
        """Pad PCP slot mappings to the fixed FULL-decode graph layout.

        The upstream manager packs current local rows as
        [rank 0 rows | rank 1 rows | ...]. A full-decode graph pads the model
        input of each PCP rank to graph_num_tokens. Preserve that rank-major
        layout when expanding the slot mapping:
        [rank 0 rows | rank 0 padding | rank 1 rows | rank 1 padding | ...].
        """
        slot_mappings = super().prepare_slot_mappings()
        assert self._global_batch is not None
        graph_num_tokens = self._global_batch.num_tokens_after_padding
        is_decode_only = not bool(self._global_batch.is_prefilling_np.any())
        if not is_decode_only or graph_num_tokens <= self._global_batch.num_tokens:
            return slot_mappings

        assert self._gathered_kv_slot_mappings is not None
        graph_num_slots = graph_num_tokens * self.pcp_world_size
        local_num_tokens = slot_mappings.shape[1] // self.pcp_world_size
        if local_num_tokens * self.pcp_world_size != slot_mappings.shape[1]:
            raise RuntimeError(
                "PCP gathered slot mappings must contain an equal local span "
                f"for every rank, got {slot_mappings.shape[1]} slots for "
                f"pcp_world_size={self.pcp_world_size}."
            )

        graph_slot_mappings = self._gathered_kv_slot_mappings[:, :graph_num_slots]
        # The compact source is a view of this reusable destination buffer.
        # Snapshot it first: a graph stride can overlap the next compact rank
        # span (for example, 3 compact tokens expanded to a 4-token graph).
        compact_slot_mappings = slot_mappings.clone()
        for pcp_rank in range(self.pcp_world_size):
            source_start = pcp_rank * local_num_tokens
            target_start = pcp_rank * graph_num_tokens
            graph_slot_mappings[:, target_start : target_start + local_num_tokens].copy_(
                compact_slot_mappings[:, source_start : source_start + local_num_tokens]
            )
            graph_slot_mappings[:, target_start + local_num_tokens : target_start + graph_num_tokens].fill_(-1)
        return graph_slot_mappings

    def build_attention_context(
        self,
        input_batch: AscendInputBatch,
        block_tables: tuple[torch.Tensor, ...],
        slot_mappings: torch.Tensor,
    ) -> AscendPCPAttentionContext:
        """Build the PCP context consumed by attention metadata builders."""
        if input_batch.is_dummy:
            local_num_tokens_after_padding = input_batch.num_tokens
            restore_start = self.pcp_rank * local_num_tokens_after_padding
            return AscendPCPAttentionContext(
                global_batch=input_batch,
                global_block_tables=block_tables,
                global_slot_mappings=slot_mappings.view(
                    slot_mappings.shape[0],
                    self.pcp_world_size,
                    local_num_tokens_after_padding,
                )[:, self.pcp_rank],
                hidden_restore_idx=torch.arange(
                    restore_start,
                    restore_start + local_num_tokens_after_padding,
                    device=self.device,
                ),
                local_num_tokens_after_padding=local_num_tokens_after_padding,
            )

        global_batch = self._global_batch
        return AscendPCPAttentionContext(
            global_batch=global_batch,
            global_block_tables=self._block_tables.gather_block_tables(
                global_batch.idx_mapping,
                global_batch.num_reqs_after_padding,
            ),
            global_slot_mappings=self._global_batch_slot_mappings[:, : global_batch.num_tokens],
            hidden_restore_idx=self._hidden_restore_idx,
            local_num_tokens_after_padding=input_batch.num_tokens_after_padding,
        )
