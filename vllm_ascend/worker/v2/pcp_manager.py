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

from collections.abc import Iterator
from dataclasses import dataclass, replace

import numpy as np
import torch
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.distributed import get_pcp_group, get_pp_group

# vLLM main (#56888) replaced buffer_utils.async_copy_to_gpu with
# torch_utils.async_tensor_h2d (gaining out=/device=None support).
from vllm.utils.torch_utils import async_tensor_h2d as async_copy_to_gpu
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.pcp_manager import PCPManager
from vllm.v1.worker.gpu.states import RequestState

from vllm_ascend.ops.rotary_embedding import update_cos_sin
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
    padded_gather_idx: torch.Tensor | None = None
    gathered_kv_write_mask: torch.Tensor | None = None
    # Device snapshot of allocated kernel-block counts in global request order.
    global_block_table_num_blocks: torch.Tensor | None = None


class AscendPCPManager(PCPManager):
    """PCP manager that refreshes Ascend-only local-batch metadata."""

    vllm_config: VllmConfig
    kv_cache_config: KVCacheConfig | None = None
    _hidden_restore_idx: torch.Tensor | None
    _global_batch_slot_mappings: torch.Tensor | None
    _gathered_kv_slot_mappings: torch.Tensor | None
    _pad_slot_id: torch.Tensor
    _sampling_hidden_restored: bool = False
    # The runner's global input buffers, which back global graph metadata.
    global_input_buffers: AscendInputBuffers | None = None

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
            max_num_reqs=max_num_reqs,
            max_num_tokens=max_num_tokens,
            block_tables=block_tables,
            dcp_world_size=dcp_world_size,
            dcp_rank=dcp_rank,
            cp_interleave=cp_interleave,
        )
        # PCP supplies its own output buffers to compute_slot_mappings, so their
        # dtype must match Ascend block-table slots for cache-write operators.
        if block_tables is not None:
            slot_dtype = block_tables.slot_mappings.dtype
            if self._global_batch_slot_mappings is not None:
                self._global_batch_slot_mappings = torch.empty_like(self._global_batch_slot_mappings, dtype=slot_dtype)
            if self._gathered_kv_slot_mappings is not None:
                self._gathered_kv_slot_mappings = torch.empty_like(self._gathered_kv_slot_mappings, dtype=slot_dtype)
            self._pad_slot_id = self._pad_slot_id.to(slot_dtype)

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
            # Restore indices cover the global graph extent, which can exceed
            # the scheduled tokens that upstream assigns to rank segments.
            self._hidden_restore_idx_buffer = torch.zeros(max_num_tokens, dtype=torch.int64, device=device)

        # Whether the runner already restored the target hidden states to the
        # global PCP layout for the current sampling step (replicated PCP
        # draft). A second all-gather in restore_for_sampling would reorder
        # them, so the sampling path consumes this flag instead of inferring
        # the layout from the tensor length, which is ambiguous under
        # piecewise/FULL graphs where each rank pads its local batch to the
        # same global padded length.
        self._sampling_hidden_restored = False

    @property
    def is_decode_sharded(self) -> bool:
        """Use the shared policy after the runner injects vllm_config."""
        return self.vllm_config.parallel_config.pcp_shard_decode_requests

    # TODO: Remove replicated_requests and _iter_rank_chunks once the paired
    # vLLM includes #52162, which shards decode requests upstream.
    def replicated_requests(self, num_scheduled_tokens: np.ndarray, is_prefilling: np.ndarray) -> np.ndarray:
        """Sharded decodes have a single owner, so none of them is replicated."""
        replicated = super().replicated_requests(num_scheduled_tokens, is_prefilling)
        if self.is_decode_sharded:
            replicated &= np.asarray(is_prefilling, dtype=np.bool_)
        return replicated

    def _iter_rank_chunks(
        self,
        rank: int,
        num_scheduled_tokens: np.ndarray,
        is_prefilling: np.ndarray,
    ) -> Iterator[tuple[int, int, int]]:
        """Split prefills as upstream does and assign decodes round-robin.

        Only scheduled decodes count toward the round-robin order, which keeps
        every step balanced. Ownership may change between steps because KV
        and hidden states are gathered back to every PCP rank.
        """
        if not self.is_decode_sharded:
            yield from super()._iter_rank_chunks(rank, num_scheduled_tokens, is_prefilling)
            return

        decode_ordinal = 0
        num_chunks = 2 * self.pcp_world_size
        for req_idx, num_tokens in enumerate(num_scheduled_tokens):
            query_len = int(num_tokens)
            if query_len == 0:
                continue

            if not is_prefilling[req_idx]:
                owner_rank = decode_ordinal % self.pcp_world_size
                decode_ordinal += 1
                if rank == owner_rank:
                    yield req_idx, 0, query_len
                continue

            # DCP == 1 here, so prefills use upstream's DualChunkSwap split.
            chunk_size = (query_len + num_chunks - 1) // num_chunks
            for chunk_idx in (rank, num_chunks - 1 - rank):
                chunk_offset = chunk_idx * chunk_size
                chunk_len = min(chunk_size, query_len - chunk_offset)
                if chunk_len > 0:
                    yield req_idx, chunk_offset, chunk_len

    @staticmethod
    def broadcast_replicated_hidden_states(
        last_hidden_states: torch.Tensor,
        hidden_states: torch.Tensor,
        num_tokens: int,
        replicated_pcp: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Broadcast actual draft hidden states only when target PCP is replicated."""
        if not replicated_pcp:
            return last_hidden_states, hidden_states

        # Draft tokens are replicated: broadcast PCP rank 0 directly,
        # matching target decode's gather-then-select behavior.
        pcp_group = get_pcp_group()
        shared_hidden_states = hidden_states is last_hidden_states
        last_hidden_states = pcp_group.broadcast(last_hidden_states[:num_tokens].contiguous(), src=0)
        if shared_hidden_states:
            hidden_states = last_hidden_states
        else:
            hidden_states = pcp_group.broadcast(hidden_states[:num_tokens].contiguous(), src=0)
        return last_hidden_states, hidden_states

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
            if speculative_config.method not in ("mtp", "eagle3", "dspark"):
                raise NotImplementedError(
                    "Ascend MRV2 PCP supports speculative decoding only with MTP, Eagle3 and DSpark."
                )
            if speculative_config.draft_sample_method != "greedy":
                raise NotImplementedError(
                    "Ascend MRV2 PCP speculative decoding currently requires greedy draft sampling."
                )
        is_sparse_mla = hasattr(model_config.hf_text_config, "index_topk")
        cudagraph_mode = vllm_config.compilation_config.cudagraph_mode
        if parallel_config.data_parallel_size > 1 and cudagraph_mode not in {
            CUDAGraphMode.NONE,
            CUDAGraphMode.FULL_DECODE_ONLY,
        }:
            raise NotImplementedError("MRV2 PCP+DP supports eager mode or FULL_DECODE_ONLY CUDA graphs only.")
        if is_sparse_mla and cudagraph_mode not in {
            CUDAGraphMode.NONE,
            CUDAGraphMode.FULL_DECODE_ONLY,
        }:
            raise NotImplementedError("MRV2 sparse MLA PCP supports eager mode or FULL_DECODE_ONLY CUDA graphs only.")
        if cudagraph_mode.has_full_cudagraphs() and cudagraph_mode != CUDAGraphMode.FULL_DECODE_ONLY:
            raise NotImplementedError("MRV2 PCP supports FULL_DECODE_ONLY CUDA graphs only.")

    # TODO: Remove once upstream PCP keeps per-request draft counts in the
    # rank-local batch.
    @staticmethod
    def _restore_draft_token_counts(
        global_batch: AscendInputBatch,
        local_batch: AscendInputBatch,
    ) -> AscendInputBatch:
        """Carry the scheduled draft counts over to the rank-local requests.

        Upstream already gathers the prepared K+1 input IDs; it only clears the
        speculative counters of the local batch.
        """
        global_draft_counts = global_batch.num_draft_tokens_per_req
        if global_draft_counts is None:
            raise RuntimeError("PCP speculative decoding requires per-request draft token counts.")
        if np.any(global_draft_counts[global_batch.is_prefilling_np] != 0):
            raise NotImplementedError("PCP speculative decoding does not support draft tokens on prefill requests.")

        draft_count_by_req = dict(zip(global_batch.req_ids, global_draft_counts, strict=True))
        local_draft_counts = np.fromiter(
            (draft_count_by_req[req_id] for req_id in local_batch.req_ids),
            dtype=np.int32,
            count=local_batch.num_reqs,
        )
        # The placeholder request of an empty rank schedules no drafts.
        local_draft_counts[local_batch.num_scheduled_tokens == 0] = 0
        return replace(  # type: ignore[call-arg]
            local_batch,
            num_draft_tokens=int(local_draft_counts.sum()),
            num_draft_tokens_per_req=local_draft_counts,
        )

    def get_global_graph_num_reqs(self, batch_desc: BatchExecutionDescriptor) -> int | None:
        """Global request capacity of a sharded FULL decode graph, if any.

        Dispatch sizes the graph for the largest rank-local batch, while the
        global batch keeps every request for cache updates and sampling.
        """
        if not self.is_decode_sharded or batch_desc.cg_mode != CUDAGraphMode.FULL:
            return None
        # validate_config only allows FULL_DECODE_ONLY, whose batches are uniform.
        assert batch_desc.num_reqs is not None and batch_desc.uniform_token_count is not None
        return self._get_global_graph_num_reqs(batch_desc.num_reqs, batch_desc.uniform_token_count)

    def _get_global_graph_num_reqs(self, num_local_reqs: int, query_len: int) -> int:
        assert self.global_input_buffers is not None
        return min(
            num_local_reqs * self.pcp_world_size,
            self.global_input_buffers.max_num_reqs,
            self.global_input_buffers.max_num_tokens // query_len,
        )

    def _full_decode_requests_are_token_sized(self, global_batch: AscendInputBatch) -> bool:
        """Whether a FULL_DECODE_ONLY graph replays exactly one token per padded request.

        When that holds, request-shaped metadata must be padded to the token
        extent. Speculative (MTP/Eagle3) decode slots may carry more than one
        token, so their request metadata is kept at the request extent instead.
        """
        return (
            not bool(global_batch.is_prefilling_np.any())
            and self.vllm_config.compilation_config.cudagraph_mode == CUDAGraphMode.FULL_DECODE_ONLY
            and global_batch.num_draft_tokens == 0
        )

    def partition_batch(
        self,
        input_batch: AscendInputBatch,
        padded_num_tokens: int | None = None,
        padded_num_reqs: int | None = None,
    ) -> AscendInputBatch:
        """Partition the batch and update Ascend-specific local metadata."""
        global_batch = input_batch
        # A sharded decode graph is sized per rank, so upstream can pad the
        # local requests. Replicated decode graphs are padded below instead.
        local_batch = super().partition_batch(
            global_batch,
            padded_num_tokens=padded_num_tokens,
            padded_num_reqs=padded_num_reqs if self.is_decode_sharded else None,
        )
        assert isinstance(local_batch, AscendInputBatch)
        if global_batch.num_draft_tokens > 0:
            local_batch = self._restore_draft_token_counts(global_batch, local_batch)
        self._pad_hidden_restore_idx(global_batch)
        if not self.is_decode_sharded:
            local_batch = self._pad_replicated_decode_graph(global_batch, local_batch)
        elif padded_num_reqs is not None:
            self._pad_uniform_query_start_loc(global_batch, local_batch)

        actual_seq_lens_np = local_batch.num_computed_tokens_np + local_batch.num_scheduled_tokens
        if local_batch.num_tokens == 0:
            # An empty rank keeps one zero-token placeholder request whose
            # device sequence length upstream already sets to zero.
            actual_seq_lens_np[:] = 0
            local_batch.seq_lens_cpu_upper_bound.zero_()
        if local_batch.num_reqs_after_padding > local_batch.num_reqs:
            assert self._input_buffers is not None
            seq_lens_np = self._input_buffers.seq_lens_np
            seq_lens_np[: local_batch.num_reqs] = actual_seq_lens_np
            seq_lens_np[local_batch.num_reqs : local_batch.num_reqs_after_padding] = 0
            local_batch.seq_lens_np = seq_lens_np[: local_batch.num_reqs_after_padding]
        else:
            local_batch.seq_lens_np = actual_seq_lens_np
        # Decode-only batches, including empty ranks, keep the attention state
        # computed from the global batch before partitioning.
        if not bool(global_batch.is_prefilling_np.any()):
            return local_batch

        num_valid_tokens = local_batch.num_scheduled_tokens
        if local_batch.num_draft_tokens_per_req is not None:
            num_valid_tokens = num_valid_tokens - local_batch.num_draft_tokens_per_req
        local_batch.attn_state = build_attn_state(
            self.vllm_config,
            actual_seq_lens_np,
            local_batch.num_reqs,
            local_batch.num_scheduled_tokens,
            num_valid_tokens,
            kv_cache_config=self.kv_cache_config,
        )
        return local_batch

    def _pad_hidden_restore_idx(self, global_batch: AscendInputBatch) -> None:
        """Extend restore indices to the global padded extent.

        Rows past the scheduled tokens restore gathered row 0 and are zeroed
        by ``restore_hidden_states``.
        """
        assert self._hidden_restore_idx is not None
        num_tokens = global_batch.num_tokens
        restore_idx = self._hidden_restore_idx_buffer[: global_batch.num_tokens_after_padding]
        restore_idx[:num_tokens].copy_(self._hidden_restore_idx[:num_tokens])
        restore_idx[num_tokens:].zero_()
        self._hidden_restore_idx = restore_idx

    @staticmethod
    def _pad_uniform_query_start_loc(global_batch: AscendInputBatch, local_batch: AscendInputBatch) -> None:
        """Give every row of a uniform FULL graph the captured query width.

        FIA needs query offsets that cover the whole padded input, including
        padding rows and the placeholder of an empty rank. Their sequence
        lengths and slots stay invalid.
        """
        num_reqs = local_batch.num_reqs_after_padding
        query_len = local_batch.num_tokens_after_padding // num_reqs
        if np.any(global_batch.num_scheduled_tokens != query_len):
            return
        local_batch.query_start_loc_np[:] = np.arange(num_reqs + 1, dtype=np.int32) * query_len
        async_copy_to_gpu(local_batch.query_start_loc_np, out=local_batch.query_start_loc)

    def _pad_replicated_decode_graph(
        self,
        global_batch: AscendInputBatch,
        local_batch: AscendInputBatch,
    ) -> AscendInputBatch:
        """Pad a replicated decode batch to the fixed FULL-graph layout."""
        graph_num_tokens = global_batch.num_tokens_after_padding
        is_decode_only = not bool(global_batch.is_prefilling_np.any())
        # FULL_DECODE_ONLY graphs capture one token for every padded request.
        # Other graph modes may pad tokens without padding request metadata.
        is_full_decode_graph = self._full_decode_requests_are_token_sized(global_batch)
        graph_num_reqs = (
            global_batch.num_tokens_after_padding if is_full_decode_graph else global_batch.num_reqs_after_padding
        )
        # The base PCP manager may already honor ``padded_num_tokens`` while
        # leaving request-shaped metadata at the actual request count. Pad when
        # either extent is still short so the runtime metadata matches the fixed
        # graph capture layout.
        needs_token_padding = graph_num_tokens > local_batch.num_tokens_after_padding
        needs_request_padding = graph_num_reqs > local_batch.num_reqs_after_padding
        if not is_decode_only or not (needs_token_padding or needs_request_padding):
            return local_batch

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

        if local_batch.dcp_local_seq_lens is not None:
            input_buffers.dcp_local_seq_lens[actual_reqs:graph_num_reqs].zero_()
        dcp_local_seq_lens = (
            input_buffers.dcp_local_seq_lens[:graph_num_reqs] if local_batch.dcp_local_seq_lens is not None else None
        )
        seq_lens_cpu_upper_bound = torch.zeros(
            graph_num_reqs,
            dtype=local_batch.seq_lens_cpu_upper_bound.dtype,
        )
        seq_lens_cpu_upper_bound[:actual_reqs].copy_(local_batch.seq_lens_cpu_upper_bound[:actual_reqs])
        return replace(  # type: ignore[call-arg]
            local_batch,
            num_reqs_after_padding=graph_num_reqs,
            num_tokens_after_padding=graph_num_tokens,
            query_start_loc=input_buffers.query_start_loc[: graph_num_reqs + 1],
            query_start_loc_np=graph_query_start_loc_np,
            seq_lens=input_buffers.seq_lens[:graph_num_reqs],
            seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,
            dcp_local_seq_lens=dcp_local_seq_lens,
            input_ids=input_buffers.input_ids[:graph_num_tokens],
            positions=input_buffers.positions[:graph_num_tokens],
            is_padding=input_buffers.is_padding[:graph_num_tokens],
        )

    def prepare_draft_prefill(self, input_batch: InputBatch, input_ids: torch.Tensor) -> None:
        """Keep the replicated PCP draft on the global batch.

        vLLM main (#56181) routes the draft through the target PCP partition and
        shrinks the model input to the rank-local batch, desyncing it from the
        global metadata (FIA rejects `last(actual_seq_lengths_q) != T` in TND
        layout). Ascend PCP spec decode is always replicated, so skip the
        partition.
        """
        return

    def restore_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Restore active tokens and zero any fixed-graph padding rows."""
        if not self.is_last_pp_rank:
            return hidden_states

        restored_hidden_states = super().restore_hidden_states(hidden_states)
        if self._global_batch is not None:
            restored_hidden_states[self._global_batch.num_tokens :].zero_()
        return restored_hidden_states

    def restore_for_sampling(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, InputBatch]:
        """Return the global batch and, when already global, skip re-gathering.

        On vLLM main the Ascend runner restores the target hidden states to the
        global PCP layout before sampling (draft_hidden_states is captured
        before the upstream restore), so a second all-gather here would reorder
        them. Consume the runner's explicit pre-restore marker instead of
        inferring the layout from the tensor length: under piecewise/FULL
        graphs each rank pads its local batch to the same global padded length,
        so a length match does not distinguish local from restored layouts.
        """
        assert self._global_batch is not None
        if self._sampling_hidden_restored:
            self._sampling_hidden_restored = False
            return hidden_states, self._global_batch
        return super().restore_for_sampling(hidden_states)

    def restore_hidden_state_buffer(self, hidden_states: torch.Tensor) -> None:
        """Restore a model-owned rank-local buffer to the global PCP layout."""
        if not self.is_last_pp_rank:
            return

        assert self._padded_gather_idx is not None
        local_num_tokens_padded = self._padded_gather_idx.shape[0] // self.pcp_world_size
        restored_hidden_states = self.restore_hidden_states(hidden_states[:local_num_tokens_padded])
        hidden_states[: restored_hidden_states.shape[0]].copy_(restored_hidden_states)

    def get_dummy_block_tables(self, num_reqs: int) -> tuple[torch.Tensor, ...]:
        """Return capture views backed by the persistent PCP-local tables.

        FULL graph replay cannot rebind SFA's captured block-table pointer, so
        capture must use the same storage that ``prepare_attn`` updates at
        runtime instead of the model runner's global block-table buffers.
        """
        if self._local_block_tables is None:
            raise RuntimeError("PCP-local block tables are not initialized.")

        dummy_block_tables = []
        for block_table in self._local_block_tables:
            if num_reqs > block_table.shape[0]:
                raise RuntimeError(
                    f"PCP graph request count exceeds the local block table: {num_reqs} > {block_table.shape[0]}."
                )
            dummy_block_tables.append(block_table[:num_reqs].zero_())
        return tuple(dummy_block_tables)

    def prepare_slot_mappings(self) -> torch.Tensor:
        """Pad PCP slot mappings to the fixed FULL-decode graph layout.

        The upstream manager packs current local rows as
        [rank 0 rows | rank 1 rows | ...]. A full-decode graph pads the model
        input of each PCP rank to graph_num_tokens. Preserve that rank-major
        layout when expanding the slot mapping:
        [rank 0 rows | rank 0 padding | rank 1 rows | rank 1 padding | ...].

        Sharded decode already uses the rank-local graph stride upstream.
        """
        slot_mappings = super().prepare_slot_mappings()
        assert self._global_batch is not None
        graph_num_tokens = self._global_batch.num_tokens_after_padding
        is_decode_only = not bool(self._global_batch.is_prefilling_np.any())
        if self.is_decode_sharded or not is_decode_only or graph_num_tokens <= self._global_batch.num_tokens:
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

    def _build_sharded_dummy_attention_context(
        self,
        input_batch: AscendInputBatch,
        query_len: int,
    ) -> AscendPCPAttentionContext:
        """Mirror the global view that prepare_inputs builds for a uniform graph.

        ``make_dummy`` spreads tokens evenly, so a dummy batch is uniform
        exactly when its tokens divide by its requests.
        """
        assert self.global_input_buffers is not None
        assert self._block_tables is not None
        assert self._global_batch_slot_mappings is not None
        num_reqs = self._get_global_graph_num_reqs(input_batch.num_reqs_after_padding, query_len)
        num_tokens = num_reqs * query_len
        global_batch = AscendInputBatch.make_dummy(
            num_reqs,
            num_tokens,
            self.global_input_buffers,
            max_query_len=query_len,
        )
        # make_dummy refreshes the shared RoPE tables from the global
        # positions, but the model consumes the rank-local ones.
        update_cos_sin(input_batch.positions)
        return AscendPCPAttentionContext(
            global_batch=global_batch,
            global_block_tables=self._block_tables.get_dummy_block_tables(num_reqs),
            global_slot_mappings=self._global_batch_slot_mappings[:, :num_tokens].fill_(-1),
            hidden_restore_idx=torch.arange(num_tokens, device=self.device),
        )

    def build_attention_context(
        self,
        input_batch: AscendInputBatch | None = None,
        block_tables: tuple[torch.Tensor, ...] | None = None,
        slot_mappings: torch.Tensor | None = None,
    ) -> AscendPCPAttentionContext:
        """Build PCP context for the current real, capture, or idle DP batch."""
        if input_batch is not None and input_batch.is_dummy:
            # Both capture and runtime dummy batches bypass partition_batch().
            # Saved layout state may be absent or belong to a previous request.
            assert block_tables is not None
            assert slot_mappings is not None
            num_tokens = input_batch.num_tokens_after_padding
            query_len, remainder = divmod(num_tokens, input_batch.num_reqs_after_padding)
            if self.is_decode_sharded and remainder == 0:
                return self._build_sharded_dummy_attention_context(input_batch, query_len)
            restore_start = self.pcp_rank * num_tokens
            return AscendPCPAttentionContext(
                global_batch=input_batch,
                global_block_tables=block_tables,
                global_slot_mappings=slot_mappings.view(slot_mappings.shape[0], self.pcp_world_size, num_tokens)[
                    :, self.pcp_rank
                ],
                hidden_restore_idx=torch.arange(restore_start, restore_start + num_tokens, device=self.device),
            )

        global_batch = self._global_batch
        hidden_restore_idx = self._hidden_restore_idx
        assert global_batch is not None
        assert self._block_tables is not None
        assert self._global_batch_slot_mappings is not None
        assert hidden_restore_idx is not None
        global_block_table_num_blocks = None
        if self.dcp_world_size > 1 and bool(global_batch.is_prefilling_np.any()):
            global_block_table_num_blocks = torch.from_numpy(
                self._block_tables.num_blocks.np[:, global_batch.idx_mapping_np[: global_batch.num_reqs]]
            ).to(device=self.device, non_blocking=True)
        return AscendPCPAttentionContext(
            global_batch=global_batch,
            global_block_tables=self._block_tables.gather_block_tables(
                global_batch.idx_mapping,
                global_batch.num_reqs_after_padding,
            ),
            global_slot_mappings=self._global_batch_slot_mappings[:, : global_batch.num_tokens_after_padding],
            hidden_restore_idx=hidden_restore_idx,
            padded_gather_idx=self._padded_gather_idx,
            gathered_kv_write_mask=self._gathered_kv_write_mask,
            global_block_table_num_blocks=global_block_table_num_blocks,
        )
