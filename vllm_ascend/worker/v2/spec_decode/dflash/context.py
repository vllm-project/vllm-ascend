# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Context-KV capture backport for matched vLLM snapshots before PR #57632.

The native upstream implementation takes precedence when available. Remove
this mixin once all supported snapshots include the context-prefix callback.
The backport is restricted to the standard single-card FULL DFlash path.
"""

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import torch
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.spec_decode.dflash import speculator as dflash_speculator
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator as _DFlashContextGraphBase
else:
    _DFlashContextGraphBase = object


class DFlashContextGraphMixin(_DFlashContextGraphBase):
    # The hosting Ascend speculator supplies these capability guards. The
    # upstream base is used for type checking only; runtime MRO stays unchanged.
    _reuse_draft_layout: bool
    _supports_context_graph_backport: Callable[[], bool]

    def capture(self) -> None:
        # Older vLLM lacks the context-prefix callback. Match the newer upstream
        # FULL-graph layout only for the standard single-card DFlash path.
        manager = self.query_cudagraph_manager
        self._compat_context_graph = (
            self._supports_context_graph_backport()
            and not hasattr(DFlashSpeculator, "_num_graph_context_tokens")
            and self._reuse_draft_layout
            and self.vllm_config.parallel_config.tensor_parallel_size == 1
            and self.dp_size == 1
            and manager is not None
            and any(manager._capture_descs.values())
            and all(mode == CUDAGraphMode.FULL for mode, descs in manager._capture_descs.items() if descs)
        )
        if not self._compat_context_graph:
            return super().capture()
        self.sample_indices.zero_()
        self.sample_pos.zero_()
        self.sample_idx_mapping.fill_(-1)
        self._context_slot_mappings.fill_(dflash_speculator.PAD_SLOT_ID)
        manager.capture(
            self._generate_draft,
            self.input_buffers,
            self.block_tables,
            self.attn_groups,
            self.kv_cache_config,
            self.max_model_len,
            causal=self._group_causal,
            precompute_context_kv=lambda num_reqs: self._compat_precompute_context_kv(
                0, self._compat_num_graph_context_tokens(num_reqs)
            ),
            progress_bar_desc="Capturing DFlash context and query ACL graphs",
        )

    def _compat_num_graph_context_tokens(self, num_reqs: int) -> int:
        # Context rows a captured draft step stores: one full verify per request.
        return min(num_reqs * (self.num_speculative_steps + 1), self.max_num_tokens)

    def _compat_precompute_context_kv(self, start: int, end: int, dummy_run: bool = False) -> None:
        if dummy_run:
            context_slots: torch.Tensor | list[torch.Tensor | None] | None = None
        elif self._layer_group_idx is not None:
            context_slots = [self._context_slot_mappings[gidx][start:end] for gidx in self._layer_group_idx]
        else:
            context_slots = self._context_slot_mappings[0][start:end]
        self.model.precompute_and_store_context_kv(
            self.hidden_states[start:end],
            self.context_positions[start:end],
            context_slots,
        )

    def _prepare_context_kv_for_graph(self, batch_desc, num_target_tokens, dummy_run=False):
        if batch_desc.cg_mode == CUDAGraphMode.FULL:
            # The graph stores the first num_context context rows.
            assert batch_desc.num_reqs is not None
            num_context = self._compat_num_graph_context_tokens(batch_desc.num_reqs)
            if dummy_run:
                # Dummy block tables are placeholders: write no context K/V.
                self._context_slot_mappings[:, :num_context].fill_(dflash_speculator.PAD_SLOT_ID)
            elif num_target_tokens <= num_context:
                # Rows past the batch keep stale positions but write no K/V.
                self._context_slot_mappings[:, num_target_tokens:num_context].fill_(dflash_speculator.PAD_SLOT_ID)
            else:
                # Prefill context beyond the graph's rows is stored before replay.
                self._compat_precompute_context_kv(num_context, num_target_tokens)
        else:
            self._compat_precompute_context_kv(0, num_target_tokens, dummy_run)

    @torch.inference_mode()
    def _propose_with_context_graph(
        self,
        input_batch: InputBatch,
        attn_metadata: dict[str, Any],
        slot_mappings: dict[str, torch.Tensor],
        # [num_tokens, hidden_size]
        last_hidden_states: torch.Tensor,
        # num_layers x [num_tokens, hidden_size]
        aux_hidden_states: list[torch.Tensor] | None,
        # [num_reqs]
        num_sampled: torch.Tensor,
        # [num_reqs]
        num_rejected: torch.Tensor,
        # [max_num_reqs]
        last_sampled: torch.Tensor,
        # [max_num_reqs]
        next_prefill_tokens: torch.Tensor,
        # [max_num_reqs]
        temperature: torch.Tensor,
        # [max_num_reqs]
        seeds: torch.Tensor,
        dp_sync: Any = None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
        mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
        is_profile: bool = False,
    ) -> torch.Tensor:
        num_reqs = input_batch.num_reqs
        num_target_tokens = input_batch.num_tokens
        num_query_tokens = num_reqs * self.num_query_per_req
        max_seq_len = input_batch.seq_lens_cpu_upper_bound[:num_reqs].max().item()
        self.draft_max_seq_len = min(max_seq_len + self.num_query_per_req, self.max_model_len)

        # NOTE: To avoid CPU-GPU synchronization without CPU knowing the
        # number of rejected tokens, we maintain the size of input_ids and
        # hidden_states the same as the target model's. This means, we pad each
        # request's query length to include any rejected positions.
        if aux_hidden_states:
            hidden_states = self.model.combine_hidden_states(torch.cat(aux_hidden_states, dim=-1))
        else:
            hidden_states = last_hidden_states
        self.hidden_states[:num_target_tokens].copy_(hidden_states[:num_target_tokens])

        if dummy_run and skip_attn_for_dummy_run:
            # Memory profiling path: block_tables / kv_cache_config are not initialized.
            # Since DFlash needs to build its own attention metadata, we must skip the
            # preparation in this path and run a minimal forward pass.
            self.model.precompute_and_store_context_kv(
                self.hidden_states[:num_target_tokens],
                self.context_positions[:num_target_tokens],
            )
            # DFlash processes all speculative tokens in one forward pass,
            # so the real token count is num_query_tokens.
            self._prepare_eplb_forward(num_query_tokens)
            self._generate_draft(
                num_reqs,
                num_query_tokens,
                attn_metadata=None,
                slot_mappings=None,
                num_tokens_across_dp=None,
                cudagraph_runtime_mode=CUDAGraphMode.NONE,
            )
            return self.draft_tokens[:num_reqs]

        if self.pcp_manager is not None and not dummy_run:
            self.block_tables.gather_block_tables(input_batch.idx_mapping, num_reqs_padded=num_reqs)

        # The query slot mapping is written into the shared BlockTables slot_mappings.
        # That buffer's address is what the captured CUDA graph reads from at replay.
        assert self.draft_kv_cache_group_id >= 0
        # Support multiple draft KV cache groups by preparing inputs once for each
        for i, gid in enumerate(self.draft_kv_cache_group_ids):
            dflash_speculator.prepare_dflash_inputs(
                self.input_buffers,
                self.block_tables.slot_mappings[gid],
                self.context_positions,
                self._context_slot_mappings[i],
                self.sample_indices,
                self.sample_pos,
                self.sample_idx_mapping,
                self.temperature,
                self.seeds,
                input_batch,
                num_sampled,
                num_rejected,
                last_sampled,
                next_prefill_tokens,
                temperature,
                seeds,
                self.block_tables.input_block_tables[gid],
                self.block_tables.kernel_block_sizes[gid],
                self.block_tables.cp_rank,
                self.block_tables.cp_size,
                self.block_tables.cp_interleave,
                self.parallel_drafting_token_id,
                self.num_query_per_req,
                self.num_speculative_steps,
                self.max_num_reqs,
                self.max_num_tokens,
                self.max_model_len,
                self.sample_from_anchor,
            )

        batch_sync, num_batch_tokens = (
            self._build_uniform_batch_dp_sync(dp_sync, num_reqs, self.num_query_per_req)
            if dp_sync is not None
            else (None, num_query_tokens)
        )
        # Every DFlash step has exactly num_query_per_req tokens, so we can use FULL CGs
        batch_desc, batch_sync = dflash_speculator.dispatch_cg_and_sync_dp(
            self.query_cudagraph_manager,
            num_reqs,
            num_batch_tokens,
            uniform_token_count=self.num_query_per_req,
            dp_size=self.dp_size,
            dp_rank=self.dp_rank,
            need_eager=is_profile,
            dp_sync=batch_sync,
        )
        num_tokens_padded = batch_desc.num_tokens
        num_tokens_across_dp = batch_sync.num_tokens_across_dp if batch_sync is not None else None

        self._prepare_context_kv_for_graph(batch_desc, num_target_tokens, dummy_run)

        # Rebuild the draft attention metadata even when replaying the FULL
        # graph so that any attention metadata builder state is updated.
        draft_attn_metadata = self._build_uniform_attn_metadata(
            num_reqs=num_reqs,
            batch_desc=batch_desc,
            num_query_per_req=self.num_query_per_req,
            seq_lens_cpu_upper_bound=input_batch.seq_lens_cpu_upper_bound,
            step=self.num_query_per_req,
            causal=self._group_causal,
        )
        draft_slot_mappings_by_layer = dflash_speculator.build_slot_mappings_by_layer(
            self.block_tables.slot_mappings[:, :num_tokens_padded],
            self.kv_cache_config,
        )

        # DFlash processes all speculative tokens in one forward pass,
        # so the real token count is num_query_tokens.
        self._prepare_eplb_forward(num_query_tokens)

        if batch_desc.cg_mode == CUDAGraphMode.FULL:
            assert self.query_cudagraph_manager is not None
            self.query_cudagraph_manager.run_fullgraph(batch_desc)
        else:
            self._generate_draft(
                num_reqs,
                num_tokens_padded,
                draft_attn_metadata,
                draft_slot_mappings_by_layer,
                num_tokens_across_dp=num_tokens_across_dp,
                cudagraph_runtime_mode=batch_desc.cg_mode,
            )

        return self.draft_tokens[:num_reqs]
