# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
from copy import copy
from typing import Any

import numpy as np
import torch
from vllm.config import VllmConfig, replace
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.attn_utils import build_slot_mappings_by_layer
from vllm.v1.worker.gpu.dp_utils import dispatch_cg_and_sync_dp
from vllm.v1.worker.gpu.spec_decode.utils import get_parallel_drafting_token_id
from vllm.v1.worker.utils import get_uniform_decode_token_count

from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.utils import vllm_version_is
from vllm_ascend.worker.v2.attn_utils import build_attn_metadata, build_attn_metadata_wrapper
from vllm_ascend.worker.v2.input_batch import AscendInputBuffers
from vllm_ascend.worker.v2.spec_decode.autoregressive.aclgraph import AutoRegressiveAclGraphManager
from vllm_ascend.worker.v2.spec_decode.eagle.speculator import AscendEagleSpeculator
from vllm_ascend.worker.v2.spec_decode.pcp_utils import disable_target_pcp_for_replicated_draft


class AscendParallelEagleSpeculator(AscendEagleSpeculator):
    """P-EAGLE: expand masked queries and sample K positions in one V2 forward."""

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        super().__init__(vllm_config, device)
        self.extra_query_tokens = self.num_speculative_steps - 1
        self.max_num_tokens += self.max_num_reqs * self.extra_query_tokens
        self.input_buffers = AscendInputBuffers(self.max_num_reqs, self.max_num_tokens, device)
        self.hidden_states = torch.zeros(self.max_num_tokens, self.hidden_size, dtype=self.dtype, device=device)
        self.parallel_token_id = get_parallel_drafting_token_id(self.draft_model_config.hf_config)
        self.parallel_sample_indices = torch.zeros(
            self.max_num_reqs * self.num_speculative_steps, dtype=torch.long, device=device
        )
        self.parallel_sample_steps = torch.arange(self.num_speculative_steps, device=device).repeat(self.max_num_reqs)
        self.parallel_metadata: dict[str, Any] = {}

    def load_draft_model(self, target_model, target_attn_layer_names):
        model = super().load_draft_model(target_model, target_attn_layer_names)
        # P-EAGLE's learned mask is in the target auxiliary-hidden space, just
        # like real target activations. Project both through the same FC layer.
        self.parallel_hidden = model.combine_hidden_states(model.mask_hidden.view(-1))
        return model

    def set_attn(self, model_state, kv_cache_config, block_tables, target_input_buffers, target_attn_groups):
        super().set_attn(model_state, kv_cache_config, block_tables, target_input_buffers, target_attn_groups)
        # Share the request-to-block registry, but own the expanded draft's
        # slot buffer: target-sized storage cannot hold K-1 extra queries/row.
        self.block_tables = copy(block_tables)
        self.block_tables.max_num_batched_tokens = self.max_num_tokens
        self.block_tables.slot_mappings = torch.full(
            (len(kv_cache_config.kv_cache_groups), self.max_num_tokens),
            -1,
            dtype=torch.int32,
            device=self.device,
        )

    def init_cudagraph_manager(self, cudagraph_mode: CUDAGraphMode) -> None:
        if self.speculative_config.enforce_eager:
            cudagraph_mode = CUDAGraphMode.NONE
        graph_config = replace(
            self.vllm_config,
            scheduler_config=replace(self.scheduler_config, max_num_batched_tokens=self.max_num_tokens),
        )
        # Target decode has K+1 tokens; adding K-1 masked queries gives 2K.
        # The single parallel forward uses the draft-prefill graph pool.
        self.prefill_cudagraph_manager = AutoRegressiveAclGraphManager(
            graph_config, self.device, cudagraph_mode, decode_query_len=2 * self.num_speculative_steps
        )
        self.prefill_cudagraph_manager.speculator = self
        self.prefill_cudagraph_manager.update_stream = self.update_stream

    def capture(self) -> None:
        manager = self.prefill_cudagraph_manager
        assert manager is not None
        self.parallel_sample_indices.zero_()
        self.idx_mapping.zero_()
        if manager.use_breakable_cg:
            manager.init_breakable_cg_runner(self.model)
        with disable_target_pcp_for_replicated_draft(self), build_attn_metadata_wrapper():
            manager.capture(
                self._parallel_forward,
                self.model_state,
                self.input_buffers,
                self.block_tables,
                self.attn_groups,
                self.kv_cache_config,
                progress_bar_desc="Capturing parallel EAGLE CUDA graphs",
            )

    def _parallel_forward(
        self,
        num_reqs,
        num_tokens,
        attn_metadata,
        slot_mappings,
        num_tokens_across_dp,
        cudagraph_runtime_mode=CUDAGraphMode.NONE,
        mm_inputs=None,
    ) -> None:
        hidden_states, _ = self._run_model(
            num_tokens, attn_metadata, slot_mappings, num_tokens_across_dp, cudagraph_runtime_mode, mm_inputs
        )
        num_samples = num_reqs * self.num_speculative_steps
        indices = self.parallel_sample_indices[:num_samples]
        tokens = self.sample_draft(
            hidden_states[indices],
            self.input_buffers.positions[indices] + 1,
            self.idx_mapping[:num_reqs].repeat_interleave(self.num_speculative_steps),
            self.temperature,
            self.seeds,
            self.parallel_sample_steps[:num_samples],
            self.draft_logits,
        )
        self.draft_tokens[:num_reqs].copy_(tokens.view(num_reqs, self.num_speculative_steps))

    def _prepare_parallel_inputs(self, input_batch, hidden_states, next_tokens, num_rejected):
        num_reqs = input_batch.num_reqs
        num_tokens = input_batch.num_tokens
        expanded_tokens = num_tokens + num_reqs * self.extra_query_tokens
        ids, positions, rejected, masked, indices, hidden_mapping = (
            torch.ops._C_ascend.npu_copy_and_expand_eagle_inputs(
                input_batch.input_ids[:num_tokens],
                input_batch.positions[:num_tokens].to(torch.int32),
                next_tokens.to(torch.int32),
                input_batch.query_start_loc[: num_reqs + 1],
                input_batch.query_start_loc[1 : num_reqs + 1] - 1 - num_rejected,
                0,
                self.parallel_token_id,
                self.num_speculative_steps,
                True,
                expanded_tokens,
            )
        )
        self.input_buffers.input_ids[:expanded_tokens].copy_(ids)
        self.input_buffers.positions[:expanded_tokens].copy_(positions.clamp(0, self.max_model_len - 1))
        self.hidden_states[hidden_mapping] = hidden_states[:num_tokens]
        torch.where(
            masked.bool().unsqueeze(1),
            self.parallel_hidden,
            self.hidden_states[:expanded_tokens],
            out=self.hidden_states[:expanded_tokens],
        )
        num_samples = num_reqs * self.num_speculative_steps
        self.parallel_sample_indices[:num_samples].copy_(indices)
        self.parallel_sample_indices[num_samples:].zero_()
        # Rejected slots remain at the end of each row, after its mask queries.
        # Keep their space for asynchronous batches but never write them to KV.
        rejected = rejected.bool() | (positions >= self.max_model_len) | (positions < 0)
        return expanded_tokens, rejected

    def propose(
        self,
        input_batch,
        attn_metadata,
        slot_mappings,
        last_hidden_states,
        aux_hidden_states,
        num_sampled,
        num_rejected,
        last_sampled,
        next_prefill_tokens,
        temperature,
        seeds,
        num_tokens_across_dp=None,
        dummy_run=False,
        skip_attn_for_dummy_run=False,
        mm_inputs=None,
        is_profile=False,
        dp_sync=None,
    ):
        self.input_batch = input_batch
        num_reqs = input_batch.num_reqs
        hidden_states = (
            self.model.combine_hidden_states(torch.cat(aux_hidden_states, dim=-1))
            if aux_hidden_states
            else last_hidden_states
        )
        self._copy_request_inputs(num_reqs, input_batch.idx_mapping, temperature, seeds)
        next_tokens = torch.where(
            num_sampled[:num_reqs] > 0,
            last_sampled[input_batch.idx_mapping],
            next_prefill_tokens.reshape(-1)[input_batch.idx_mapping],
        )
        expanded_tokens, rejected = self._prepare_parallel_inputs(
            input_batch, hidden_states, next_tokens, num_rejected[:num_reqs]
        )
        query_np = input_batch.query_start_loc_np[: num_reqs + 1] + (
            np.arange(num_reqs + 1, dtype=np.int32) * self.extra_query_tokens
        )
        query_cpu = torch.from_numpy(query_np)
        self.input_buffers.query_start_loc[: num_reqs + 1].copy_(query_cpu)
        self.input_buffers.query_start_loc[num_reqs + 1 :].fill_(expanded_tokens)
        self.input_buffers.seq_lens[:num_reqs].copy_(
            (input_batch.seq_lens[:num_reqs] + self.extra_query_tokens).clamp(max=self.max_model_len)
        )
        self.input_buffers.seq_lens[num_reqs:].zero_()
        uniform = get_uniform_decode_token_count(
            num_reqs, expanded_tokens, int(np.diff(query_np).max()), input_batch.has_prefill
        )
        # Expansion changes the token count, so take the draft's own DP sync
        # instead of reusing the target graph's unexpanded padding decision.
        desc, sync = dispatch_cg_and_sync_dp(
            self.prefill_cudagraph_manager,
            num_reqs,
            expanded_tokens,
            uniform,
            dp_size=self.dp_size,
            dp_rank=self.dp_rank,
            need_eager=is_profile,
        )
        num_reqs_padded = desc.num_reqs or num_reqs
        # vLLM 0.28 returns the token-count tensor directly; main returns
        # DPSyncState. Both decisions were computed for the expanded draft.
        draft_dp_tokens = (
            sync if vllm_version_is("0.28.0") else (sync.num_tokens_across_dp if sync is not None else None)
        )
        self.parallel_metadata = {}
        draft_slots = None
        if not (dummy_run and skip_attn_for_dummy_run):
            self.block_tables.gather_block_tables(input_batch.idx_mapping, num_reqs_padded=num_reqs_padded)
            slots = self.block_tables.compute_slot_mappings(
                input_batch.idx_mapping,
                self.input_buffers.query_start_loc,
                self.input_buffers.positions,
                num_tokens_padded=desc.num_tokens,
            )
            slots[:, :expanded_tokens].masked_fill_(rejected.unsqueeze(0), -1)
            query_padded = torch.full((num_reqs_padded + 1,), expanded_tokens, dtype=torch.int32)
            query_padded[: num_reqs + 1].copy_(query_cpu)
            if desc.cg_mode == CUDAGraphMode.FULL:
                query_padded = torch.arange(num_reqs_padded + 1, dtype=torch.int32) * (2 * self.num_speculative_steps)
                # Slot mapping uses only real requests. Attention must see the
                # same padded query boundaries on both host and device.
                self.input_buffers.query_start_loc[: num_reqs_padded + 1].copy_(query_padded)
            seq_np = np.zeros(num_reqs_padded, dtype=np.int32)
            seq_np[:num_reqs] = np.minimum(
                input_batch.seq_lens_np[:num_reqs] + self.extra_query_tokens, self.max_model_len
            )
            self.parallel_metadata = build_attn_metadata(
                attn_groups=self.attn_groups,
                num_reqs=num_reqs_padded,
                num_actual_reqs=num_reqs,
                num_tokens=desc.num_tokens,
                num_actual_tokens=expanded_tokens,
                query_start_loc_gpu=self.input_buffers.query_start_loc[: num_reqs_padded + 1],
                query_start_loc_cpu=query_padded,
                max_query_len=int(np.diff(query_np).max()),
                seq_lens=self.input_buffers.seq_lens[:num_reqs_padded],
                seq_lens_np=seq_np,
                max_seq_len=int(seq_np.max()),
                block_tables=[table[:num_reqs_padded] for table in self.block_tables.input_block_tables],
                slot_mappings=slots,
                kv_cache_config=self.kv_cache_config,
                positions=self.input_buffers.positions[: desc.num_tokens],
                is_prefilling=torch.from_numpy(input_batch.is_prefilling_np),
                attn_state=(
                    AscendAttentionState.ChunkedPrefill
                    if input_batch.has_prefill
                    else AscendAttentionState.SpecDecoding
                ),
            )
            draft_slots = build_slot_mappings_by_layer(slots, self.kv_cache_config)
        self._prepare_eplb_forward(expanded_tokens)
        if desc.cg_mode == CUDAGraphMode.FULL:
            assert self.prefill_cudagraph_manager is not None
            self.prefill_cudagraph_manager.run_fullgraph(desc)
        else:
            self._parallel_forward(
                num_reqs,
                desc.num_tokens,
                self.parallel_metadata or None,
                draft_slots,
                draft_dp_tokens,
                desc.cg_mode,
                mm_inputs,
            )
        return self.draft_tokens[:num_reqs]

    def build_draft_attn_metadatas(self, num_reqs_padded, num_tokens_padded, is_draft_model_prefill):
        return [self.parallel_metadata]

    def build_fia_params(self, num_reqs_padded, is_draft_model_prefill):
        metadata = next(iter(self.parallel_metadata.values()))
        return [
            {
                "actual_seq_lengths": metadata.actual_seq_lengths_q,
                "actual_seq_lengths_kv": metadata.seq_lens_list,
                "block_table": metadata.block_tables,
            }
        ]
