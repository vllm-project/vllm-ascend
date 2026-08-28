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

import numpy as np
import torch
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.distributed import get_pp_group
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.buffer_utils import async_copy_to_gpu
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

    @property
    def global_batch(self) -> AscendInputBatch:
        """Return the scheduled batch retained before PCP partitioning."""
        global_batch = self._global_batch
        if not isinstance(global_batch, AscendInputBatch):
            raise RuntimeError("PCP global batch is unavailable before partition_batch().")
        return global_batch

    @property
    def is_last_pp_rank(self) -> bool:
        """Whether this PCP manager belongs to the last PP stage."""
        return get_pp_group().is_last_rank

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

        if model_config.is_encoder_decoder:
            raise NotImplementedError("MRV2 PCP does not support encoder-decoder models yet.")
        if supports_mm_inputs:
            raise NotImplementedError("MRV2 PCP does not support MM inputs yet.")
        if vllm_config.lora_config is not None:
            raise NotImplementedError("MRV2 PCP does not support LoRA yet.")
        speculative_config = vllm_config.speculative_config
        if speculative_config is not None:
            if speculative_config.method not in ("mtp", "eagle3"):
                raise NotImplementedError("Ascend MRV2 PCP supports speculative decoding only with MTP and Eagle3.")
            if speculative_config.draft_sample_method != "greedy":
                raise NotImplementedError(
                    "Ascend MRV2 PCP speculative decoding currently requires greedy draft sampling."
                )
        is_sparse_mla = hasattr(model_config.hf_text_config, "index_topk")
        cudagraph_mode = vllm_config.compilation_config.cudagraph_mode
        if is_sparse_mla and cudagraph_mode not in {
            CUDAGraphMode.NONE,
            CUDAGraphMode.FULL_DECODE_ONLY,
        }:
            raise NotImplementedError("MRV2 sparse MLA PCP supports eager mode or FULL_DECODE_ONLY CUDA graphs only.")
        if cudagraph_mode.has_full_cudagraphs() and cudagraph_mode != CUDAGraphMode.FULL_DECODE_ONLY:
            raise NotImplementedError("MRV2 PCP supports FULL_DECODE_ONLY CUDA graphs only.")

    # TODO To bypass the upstream verification, a pseudo-batch method is used to perform reconstruction after bypassing,
    # and the changes will be deleted after the upstream is merged.
    def _partition_speculative_batch_compat(
        self,
        global_batch: AscendInputBatch,
    ) -> AscendInputBatch:
        """Adapt spec decode until upstream PCP supports it natively."""
        global_draft_counts = global_batch.num_draft_tokens_per_req
        if global_draft_counts is None:
            raise RuntimeError("PCP speculative decoding requires per-request draft token counts.")
        if np.any(global_draft_counts[global_batch.is_prefilling_np] != 0):
            raise NotImplementedError("PCP speculative decoding does not support draft tokens on prefill requests.")

        # Upstream currently rejects speculative batches before building the
        # ordinary PCP rank-local layout. Temporarily clear only its spec
        # indicators, then restore the authoritative speculative state below.
        non_spec_batch = replace(  # type: ignore[call-arg]
            global_batch,
            num_draft_tokens=0,
            num_draft_tokens_per_req=None,
        )
        try:
            local_batch = super().partition_batch(non_spec_batch)
        finally:
            self._global_batch = global_batch
        assert isinstance(local_batch, AscendInputBatch)

        # Upstream rewrites the local decode tokens while constructing its
        # non-spec logits layout. Restore the already prepared K+1 target
        # inputs from the authoritative global batch.
        assert self._padded_gather_idx is not None
        local_num_tokens_padded = local_batch.num_tokens_after_padding
        rank_token_start = self.pcp_rank * local_num_tokens_padded
        local_gather_idx = self._padded_gather_idx[rank_token_start : rank_token_start + local_num_tokens_padded]
        torch.index_select(
            global_batch.input_ids,
            0,
            local_gather_idx,
            out=local_batch.input_ids[:local_num_tokens_padded],
        )
        draft_count_by_req = dict(zip(global_batch.req_ids, global_draft_counts, strict=True))
        local_draft_counts = np.fromiter(
            (draft_count_by_req[req_id] for req_id in local_batch.req_ids),
            dtype=np.int32,
            count=local_batch.num_reqs,
        )
        return replace(  # type: ignore[call-arg]
            local_batch,
            num_draft_tokens=int(local_draft_counts.sum()),
            num_draft_tokens_per_req=local_draft_counts,
        )

    def partition_batch(self, input_batch: AscendInputBatch) -> AscendInputBatch:
        """Partition the batch and update Ascend-specific local metadata."""
        global_batch = input_batch
        if global_batch.num_draft_tokens > 0:
            local_batch = self._partition_speculative_batch_compat(global_batch)
        else:
            local_batch = super().partition_batch(global_batch)
            assert isinstance(local_batch, AscendInputBatch)

        # PCP builds the local layout from actual tokens, but a FULL decode
        # graph replays a fixed padded layout on every rank.
        graph_num_tokens = global_batch.num_tokens_after_padding
        graph_num_reqs = global_batch.num_reqs_after_padding
        is_decode_only = not bool(global_batch.is_prefilling_np.any())
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

            # Decode requests are replicated on every PCP rank, so the global
            # FULL-graph query layout is also the authoritative rank-local
            # layout, including any FIA dummy request.
            graph_query_start_loc_np = global_batch.query_start_loc_np[: graph_num_reqs + 1]
            async_copy_to_gpu(
                graph_query_start_loc_np,
                out=input_buffers.query_start_loc[: graph_num_reqs + 1],
            )

            # Graph padding has no RankSegment, so _build_batch_layout does
            # not initialize the corresponding hidden restore indices.
            assert self._hidden_restore_idx is not None
            self._hidden_restore_idx[global_batch.num_tokens : graph_num_tokens].zero_()
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
                query_start_loc_np=graph_query_start_loc_np,
                seq_lens=input_buffers.seq_lens[:graph_num_reqs],
                seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,
                input_ids=input_buffers.input_ids[:graph_num_tokens],
                positions=input_buffers.positions[:graph_num_tokens],
                is_padding=input_buffers.is_padding[:graph_num_tokens],
            )

        local_seq_lens_np = local_batch.num_computed_tokens_np + local_batch.num_scheduled_tokens
        if local_batch.num_reqs_after_padding > local_batch.num_reqs:
            padded_seq_lens_np = np.zeros(
                local_batch.num_reqs_after_padding,
                dtype=local_seq_lens_np.dtype,
            )
            padded_seq_lens_np[: local_batch.num_reqs] = local_seq_lens_np
            local_seq_lens_np = padded_seq_lens_np

        local_batch.seq_lens_np = local_seq_lens_np
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

    def restore_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Restore active tokens and zero any fixed-graph padding rows."""
        if not self.is_last_pp_rank:
            return hidden_states

        restored_hidden_states = super().restore_hidden_states(hidden_states)
        if self._global_batch is None:
            return restored_hidden_states

        num_tokens = self._global_batch.num_tokens
        num_tokens_after_padding = self._global_batch.num_tokens_after_padding
        if num_tokens == num_tokens_after_padding:
            return restored_hidden_states
        if restored_hidden_states.shape[0] != num_tokens_after_padding:
            raise RuntimeError(
                "PCP restored hidden-state length does not match the global "
                "graph layout: "
                f"{restored_hidden_states.shape[0]} != {num_tokens_after_padding}."
            )

        restored_hidden_states[num_tokens:num_tokens_after_padding].zero_()
        return restored_hidden_states

    def restore_hidden_state_buffer(self, hidden_states: torch.Tensor) -> None:
        """Restore a model-owned rank-local buffer to the global PCP layout."""
        if not self.is_last_pp_rank:
            return

        assert self._padded_gather_idx is not None
        local_num_tokens_padded = self._padded_gather_idx.shape[0] // self.pcp_world_size
        restored_hidden_states = self.restore_hidden_states(hidden_states[:local_num_tokens_padded])
        hidden_states[: restored_hidden_states.shape[0]].copy_(restored_hidden_states)

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

    def build_attention_context(self) -> AscendPCPAttentionContext:
        """Build the PCP context consumed by attention metadata builders."""
        global_batch = self._global_batch
        hidden_restore_idx = self._hidden_restore_idx
        assert global_batch is not None
        assert self._block_tables is not None
        assert self._global_batch_slot_mappings is not None
        assert hidden_restore_idx is not None
        return AscendPCPAttentionContext(
            global_batch=global_batch,
            global_block_tables=self._block_tables.gather_block_tables(
                global_batch.idx_mapping,
                global_batch.num_reqs_after_padding,
            ),
            global_slot_mappings=self._global_batch_slot_mappings[:, : global_batch.num_tokens_after_padding],
            hidden_restore_idx=hidden_restore_idx,
        )
