# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import torch
from vllm.config import CUDAGraphMode, VllmConfig, get_layers_from_vllm_config
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.v1.attention.backends.utils import CommonAttentionMetadata
from vllm.v1.kv_cache_interface import UniformTypeKVCacheSpecs
from vllm.v1.worker.utils import AttentionGroup

from vllm_ascend.ascend_forward_context import _EXTRA_CTX, set_ascend_forward_context
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.utils import AscendCommonAttentionMetadata
from vllm_ascend.spec_decode.dflash_proposer import AscendDflashProposer
from vllm_ascend.ops.triton.spec_decode.utils import (
    copy_and_expand_dflash_inputs_py,
)


class AscendDSparkProposer(AscendDflashProposer):
    """DSpark block proposer.

    DSpark uses vLLM's ``mtp`` method in user config, but its execution shape is
    closer to DFlash: target hidden states prepopulate draft K/V, then one
    anchor-first query block emits all speculative tokens.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        runner=None,
    ):
        super().__init__(vllm_config, device, runner=runner)
        assert vllm_config.speculative_config is not None
        if vllm_config.speculative_config.draft_sample_method == "probabilistic":
            raise ValueError(
                "DSpark probabilistic draft sampling is not supported on the v1 "
                "model runner; use greedy (the default) instead."
            )
        self._dspark_seed_buffer = torch.zeros(self.max_batch_size, dtype=torch.int64, device=device)
        # DSpark is not supported in vllm v1, so related property needs to be reset here.
        self.hidden_size = vllm_config.speculative_config.draft_model_config.get_hidden_size()
        self.hidden_states = torch.zeros(
            (self.max_num_tokens, self.hidden_size),
            dtype=self.dtype,
            device=self.device,
        )
        self._dflash_hidden_states = torch.zeros(
            (self.max_num_tokens, self.hidden_size),
            dtype=self.dtype,
            device=self.device,
        )
        # DSpark runs eager only (Ascend cudagraph unsupported on this path).
        self.use_cuda_graph = False
        # Max query tokens = max_batch_size * num_speculative_tokens
        # (anchor-first: N query tokens per request, no bonus token, unlike
        # DFlash's 1+N). Overrides dflash:28; v2 derives via num_query_per_req.
        self.max_query_tokens = self.max_batch_size * self.num_speculative_tokens
        # max_num_tokens + max_query_tokens. Overrides dflash:29; v2 drops it.
        self.max_positions = self.max_num_tokens + self.max_query_tokens
        # Position ids for the draft query block [max_query_tokens].
        # Overrides dflash:49; v2 uses input_buffers.positions.
        self.positions = torch.zeros(
            self.max_query_tokens,
            dtype=torch.int32,
            device=device,
        )
        # Primary-group query slot mapping buffer [max_query_tokens].
        # Overrides dflash:37; v2 uses BlockTables.slot_mappings. Per-non-
        # primary-gid buffers live in _per_group_query_slot_mapping_buffers.
        self._slot_mapping_buffer = torch.zeros(
            self.max_query_tokens,
            dtype=torch.int32,
            device=device,
        )
        # Markov block drafting: ``_markov_anchor_tokens`` is the target model's
        # next token (the seed that starts the Markov chain); ``_markov_draft_tokens``
        # holds each step's sampled token and feeds it back as the next step's
        # input. DSpark-only -- DFlash/Eagle sample in parallel without Markov.
        self._markov_anchor_tokens = torch.zeros(
            self.max_batch_size,
            dtype=torch.int64,
            device=device,
        )
        # Per-token -> request index map consumed by the SAS attention op. Sliced
        # to num_query_total for real query tokens; padding slots in
        # [num_actual_tokens, num_input_tokens) are filled with -1.
        self.token_to_req_indices_buffer = torch.zeros(
            self.max_query_tokens,
            dtype=torch.int32,
            device=device,
        )
        # Cached slices of the runner's common_attn_metadata (cad): in
        # set_inputs_first_pass they alias cad.query_start_loc_cpu / cad.seq_lens;
        # in dummy_run/profile_run (no cad available) they are synthesized locally.
        # Kept under the _dspark_ prefix because llm_base already defines
        # self.query_start_loc for Eagle.
        self._dspark_query_start_loc: torch.Tensor | None = None
        self._dspark_seq_lens: torch.Tensor | None = None
        self._markov_draft_tokens = torch.zeros(
            (self.max_batch_size, self.num_speculative_tokens),
            dtype=torch.int64,
            device=device,
        )

        # TODO simplify these comments
        # block_table / slot_mapping bookkeeping (10 dicts below). v1 self-
        # manages per kv_cache_group_id / per layer because it lacks v2's
        # BlockTables scaffold; v2 injects a single self.block_tables
        # (BlockTables, with .slot_mappings) + build_slot_mappings_by_layer,
        # so the speculator holds none of these. P2 refactor target (move to
        # runner).

        # per-gid block_table from runner (just read)
        self._per_group_block_tables: dict[int, torch.Tensor] = {}
        # per-gid slot_mapping from runner (just read)
        self._per_group_slot_mappings: dict[int, torch.Tensor] = {}

        # per-gid block_table (use in proposer)
        self._per_group_block_table_buffers: dict[int, torch.Tensor] = {}
        # per-gid query slot_mapping buffer
        self._per_group_query_slot_mapping_buffers: dict[int, torch.Tensor] = {}
        # per-gid context slot_mapping buffer
        self._per_group_context_slot_mapping_buffers: dict[int, torch.Tensor] = {}

        # per-layer context slot mappings as a flat list
        self._context_slots: list[torch.Tensor | None] = []


    def initialize_attn_backend(self, kv_cache_config, kernel_block_sizes=None) -> None:
        # Find draft layers (attention layers added by draft model)
        all_attn_layers = get_layers_from_vllm_config(
            self.vllm_config,
            AttentionLayerBase,  # type: ignore[type-abstract]
        )

        attention_groups_list: list[dict[tuple[str, str], AttentionGroup]] = []
        # the draft layers have multiple kv_cahce_groups
        if not hasattr(self.model, "get_draft_kv_cache_layer_names"):
            raise RuntimeError(
                "DSpark standard-cache path requires the draft model to expose "
                "get_draft_kv_cache_layer_names"
            )

        self._draft_attn_layer_names = sorted(self.model.get_draft_kv_cache_layer_names())

        # there are many kv groups other than one
        for kv_cache_gid, kv_cache_group_spec in enumerate(kv_cache_config.kv_cache_groups):
            draft_layer_names_in_group = set(kv_cache_group_spec.layer_names) & set(self._draft_attn_layer_names)
            if not draft_layer_names_in_group:
                continue

            attention_groups: dict[tuple[str, Any], AttentionGroup] = {}
            # iterate in a way like vllm's llm_base_proposer
            for layer_name in draft_layer_names_in_group:
                attn_backend = all_attn_layers[layer_name].get_attn_backend()
                layer_kv_cache_spec = kv_cache_group_spec.kv_cache_spec
                if isinstance(layer_kv_cache_spec, UniformTypeKVCacheSpecs):
                    layer_kv_cache_spec = layer_kv_cache_spec.kv_cache_specs[layer_name]
                key = (attn_backend.full_cls_name(), layer_kv_cache_spec)
                
                if key not in attention_groups:
                    attn_group = AttentionGroup(
                        attn_backend,
                        [layer_name],
                        layer_kv_cache_spec,
                        kv_cache_gid,
                    )
                    attn_group.create_metadata_builders(self.vllm_config, self.device)
                    attention_groups[key] = attn_group
                else:
                    attention_groups[key].layer_names.append(layer_name)

            attention_groups_list.append(attention_groups)

        self.draft_attn_groups = [attention_group for attention_groups in attention_groups_list for attention_group in attention_groups.values()]
        self.kv_cache_gid = 0
        if not self.draft_attn_groups:
            raise RuntimeError(
                "DSpark standard-cache path requires registered draft attention "
                f"groups. Missing layers: {self._draft_attn_layer_names}"
            )

        self.kv_cache_gid = self.draft_attn_groups[0].kv_cache_group_id
        self.kernel_block_size = int(self.draft_attn_groups[0].kv_cache_spec.block_size)

        name_to_gid = {
            ln: gid
            for gid, group in enumerate(kv_cache_config.kv_cache_groups)
            for ln in group.layer_names if ln in self._draft_attn_layer_names
        }
        self._layer_group_idx = [name_to_gid[name] for name in self._draft_attn_layer_names]

        # some buffers need information of groups
        self._per_group_query_slot_mapping_buffers = {attn_group.kv_cache_group_id: torch.zeros(self.max_query_tokens, dtype=torch.int32, device=self.device) for attn_group in self.draft_attn_groups}
        self._per_group_context_slot_mapping_buffers = {attn_group.kv_cache_group_id: torch.zeros(self.max_num_tokens, dtype=torch.int32, device=self.device) for attn_group in self.draft_attn_groups}


    def set_per_group_attn_metadata(
        self,
        gid: int,
        block_table: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        self._per_group_block_tables[gid] = block_table
        self._per_group_slot_mappings[gid] = slot_mapping

    def _slot_mapping_buffer_for_gid(self, gid: int, *, context: bool) -> torch.Tensor:
        if gid == getattr(self, "kv_cache_gid", 0):
            return self._context_slot_mapping_buffer if context else self._slot_mapping_buffer
        buffers = self._per_group_context_slot_mapping_buffers if context else self._per_group_query_slot_mapping_buffers
        buf = buffers.get(gid)
        return buf

    def _get_draft_block_table_for_gid(
        self,
        cad: CommonAttentionMetadata,
        batch_size: int,
        gid: int,
    ) -> torch.Tensor | None:
        block_table = getattr(self, "_per_group_block_tables", {}).get(gid)
        input_batch = getattr(getattr(self, "runner", None), "input_batch", None)
        block_tables = getattr(input_batch, "block_table", None)
        if block_table is None and block_tables is not None:
            try:
                draft_block_table = block_tables[gid]
            except (IndexError, KeyError, TypeError):
                draft_block_table = None
            if draft_block_table is not None:
                block_table = block_table.get_device_tensor(batch_size)
        if block_table is None and gid == getattr(self, "kv_cache_gid", 0):
            block_table = getattr(cad, "block_table_tensor", None)
        if block_table is None:
            return None
        block_table = block_table[:batch_size]
        # Ascend block-table tensors are reused by the runner; DSpark consumes
        # them after query slot mappings have been built.
        block_table = block_table.clone()
        return block_table

    def _get_draft_block_tables(
        self,
        cad: CommonAttentionMetadata,
        batch_size: int,
    ) -> dict[int, torch.Tensor]:
        by_gid: dict[int, torch.Tensor] = {}
        for attn_group in self.draft_attn_groups:
            gid = attn_group.kv_cache_group_id
            if gid in by_gid:
                continue
            block_table = self._get_draft_block_table_for_gid(cad, batch_size, gid)
            if block_table is not None:
                by_gid[gid] = block_table
        return by_gid

    def set_inputs_first_pass(
        self,
        target_token_ids: torch.Tensor,
        next_token_ids: torch.Tensor,
        target_positions: torch.Tensor,
        target_hidden_states: torch.Tensor,
        token_indices_to_sample: torch.Tensor | None,
        cad: CommonAttentionMetadata,
        num_rejected_tokens_gpu: torch.Tensor | None,
        req_scheduled_tokens=None,
        long_seq_metadata=None,
        num_prefill_reqs=0,
        num_decode_reqs=0,
    ) -> tuple[int, torch.Tensor, CommonAttentionMetadata, tuple[Any, Any] | None]:
        # The initial input token of markovHead is the next token
        n = next_token_ids.shape[0]
        self._dspark_seed_buffer[:n].copy_(next_token_ids)
        if n < self._dspark_seed_buffer.shape[0]:
            self._dspark_seed_buffer[n:].fill_(0)
        batch_size = cad.num_reqs
        block_size = self.num_speculative_tokens
        num_query_total = batch_size * block_size
        has_num_rejected = num_rejected_tokens_gpu is not None
        primary_gid = getattr(self, "kv_cache_gid", 0)
        block_tables_by_gid = self._get_draft_block_tables(cad, batch_size)
        self._per_group_block_table_buffers = block_tables_by_gid
        self._context_slots = None
        self._markov_anchor_tokens[:batch_size].copy_(next_token_ids)
        if batch_size < self._markov_anchor_tokens.shape[0]:
            self._markov_anchor_tokens[batch_size:].fill_(0)

        self._dflash_num_context = cad.query_start_loc_cpu[batch_size]
        self._dflash_hidden_states[:self._dflash_num_context] = target_hidden_states[:self._dflash_num_context]

        # token_indices_to_sample is filled by copy_and_expand_dflash_inputs_py
        # below (SAMPLE_FROM_ANCHOR=True, anchor included) -- not arange here.
        token_indices_to_sample = torch.empty(
            num_query_total,
            dtype=torch.int32,
            device=self.device,
        )

        # Query block: reuse the DFlash inputs kernel logic (host-side ref)
        # per kv-cache-group to fill positions / input_ids / query slot_mapping
        # / token_indices (SAMPLE_FROM_ANCHOR: anchor at q_idx=0 is sampled too).
        draft_attn_groups = getattr(self, "draft_attn_groups", [])
        if block_tables_by_gid and draft_attn_groups:
            for attn_group in draft_attn_groups:
                gid = attn_group.kv_cache_group_id
                gid_block_table = block_tables_by_gid.get(gid)
                if gid_block_table is None:
                    continue
                kv_block_size = int(attn_group.kv_cache_spec.block_size)
                copy_and_expand_dflash_inputs_py(
                    # Inputs
                    next_token_ids=next_token_ids,
                    target_positions=target_positions,
                    context_slot_mapping=self._per_group_slot_mappings[gid],
                    # Outputs
                    out_input_ids=self.input_ids,
                    out_context_positions=self._context_positions_buffer,
                    out_query_positions=self.positions,
                    out_context_slot_mapping=self._slot_mapping_buffer_for_gid(gid, context=True),
                    out_query_slot_mapping=self._slot_mapping_buffer_for_gid(gid, context=False),
                    out_token_indices=token_indices_to_sample,
                    # Block table
                    block_table=gid_block_table,
                    block_table_stride=gid_block_table.stride(0),
                    # Metadata
                    query_start_loc=cad.query_start_loc,
                    seq_lens=cad.seq_lens,
                    num_rejected_tokens=num_rejected_tokens_gpu,
                    # Scalars
                    parallel_drafting_token_id=self.parallel_drafting_token_id,
                    block_size=kv_block_size,
                    num_query_per_req=block_size,
                    num_speculative_tokens=block_size,
                    total_input_tokens=self._dflash_num_context,
                    batch_size=batch_size,
                    HAS_NUM_REJECTED=has_num_rejected,
                    SAMPLE_FROM_ANCHOR=True,
                )
        else:
            # No draft attn groups (profile/dummy without standard path):
            # fill positions/input_ids only.
            for req_idx in range(batch_size):
                ctx_end = int(cad.query_start_loc[req_idx + 1].item())
                valid_ctx_end = ctx_end
                if has_num_rejected:
                    assert num_rejected_tokens_gpu is not None
                    valid_ctx_end -= int(num_rejected_tokens_gpu[req_idx].item())
                last_pos = target_positions[valid_ctx_end - 1]
                out_start = req_idx * block_size
                out_end = out_start + block_size
                self.positions[out_start:out_end] = last_pos + 1 + self.arange_dflash[:block_size]
                self.input_ids[out_start] = next_token_ids[req_idx]
                if block_size > 1:
                    self.input_ids[out_start + 1 : out_end] = self.parallel_drafting_token_id

        # to compute self._context_slots
        if block_tables_by_gid:
            self._dspark_context_slot_mappings_by_gid = {
                gid: self._slot_mapping_buffer_for_gid(gid, context=True)[:self._dflash_num_context]
                for gid in block_tables_by_gid
            }
            self._context_slots = [self._dspark_context_slot_mappings_by_gid[gidx] for gidx in self._layer_group_idx]

        # token_to_req: per-token request index (vectorized; equivalent to
        # token_to_req_indices[req*block:(req+1)*block] = req per req).
        self.token_to_req_indices_buffer[:num_query_total] = (
            torch.arange(num_query_total, device=self.device, dtype=torch.int32) // block_size
        )
        self.token_to_req_indices_buffer[num_query_total:].fill_(-1)

        effective_seq_lens = cad.seq_lens
        if has_num_rejected:
            effective_seq_lens = effective_seq_lens - num_rejected_tokens_gpu

        cad.query_start_loc = self.arange_dflash[: batch_size + 1] * block_size
        cad.seq_lens = effective_seq_lens + block_size
        cad.query_start_loc_cpu = (torch.from_numpy(self.token_arange_np[: batch_size + 1]).clone() * block_size).to(
            torch.int32
        )

        if hasattr(cad, "actual_seq_lengths_q"):
            cad.actual_seq_lengths_q = [block_size] * batch_size
        if hasattr(cad, "decode_token_per_req"):
            cad.decode_token_per_req = block_size

        cad.num_actual_tokens = num_query_total
        cad.num_input_tokens = num_query_total
        cad.max_query_len = block_size
        cad.max_seq_len = cad.max_seq_len + block_size
        cad.slot_mapping = self._slot_mapping_buffer[:num_query_total]
        if block_tables_by_gid:
            self._per_group_query_slot_mapping_buffers = {
                gid: self._slot_mapping_buffer_for_gid(gid, context=False) for gid in block_tables_by_gid
            }
            if primary_gid in self._per_group_query_slot_mapping_buffers:
                cad.slot_mapping = self._per_group_query_slot_mapping_buffers[primary_gid][:num_query_total]
        cad.positions = self.positions[:num_query_total]
        cad.causal = False
        cad.attn_mask = None
        cad.attn_state = AscendAttentionState.ChunkedPrefill
        cad.token_to_req_indices = self.token_to_req_indices_buffer[:num_query_total]
        self._dspark_query_start_loc = cad.query_start_loc_cpu[: batch_size + 1]
        self._dspark_seq_lens = cad.seq_lens[:batch_size]

        return num_query_total, token_indices_to_sample, cad, None

    def _prepare_dspark_dummy_standard_inputs(
        self,
        num_reqs: int,
        num_input_tokens: int,
        model_num_query_tokens: int,
    ) -> None:
        """Build dummy paged SWA inputs so dummy_run/profile_run exercises the
        standard-DSA path (which needs block_table/slot_mapping/indices, unlike
        the private ring-buffer path). All-zero block tables / slot mappings
        are fine: profile_run only needs correct shapes, not correct values.
        """
        batch_size = max(num_reqs, 1)
        block_size = self.num_speculative_tokens
        cache_block_size = int(self.draft_attn_groups[0].kv_cache_spec.block_size)
        # The dummy SWA block table is captured into the target model's
        # FULL_DECODE_ONLY aclgraph and reused (contents overwritten in place) at
        # replay time, so its width must cover the full max_model_len -- not just
        # the per-step max_positions. A too-small table makes the graph-captured
        # draft attention gather block_table[req_idx] with absolute
        # ``position // cache_block_size`` indices that overflow for long
        # sequences (GatherV3 "Index out of range").
        num_blocks = (int(self.max_model_len) + cache_block_size - 1) // cache_block_size
        block_tables_by_gid: dict[int, torch.Tensor] = {}
        context_slot_mappings_by_gid: dict[int, torch.Tensor] = {}
        for attn_group in self.draft_attn_groups:
            gid = attn_group.kv_cache_group_id
            block_tables_by_gid[gid] = torch.zeros(
                (batch_size, num_blocks), dtype=torch.int32, device=self.device
            )
            self._per_group_query_slot_mapping_buffers[gid].fill_(-1)
            context_slot_mappings_by_gid[gid] = torch.zeros(
                num_input_tokens, dtype=torch.int32, device=self.device
            )
        self._per_group_block_table_buffers = block_tables_by_gid
        self._dspark_context_slot_mappings_by_gid = context_slot_mappings_by_gid
        self._context_slots = [self._dspark_context_slot_mappings_by_gid[gidx] for gidx in self._layer_group_idx]
        self._dspark_query_start_loc = (
            self.arange_dflash[: batch_size + 1] * block_size
        ).to(torch.int32)
        self._dspark_seq_lens = torch.full(
            (batch_size,), block_size, dtype=torch.int32, device=self.device
        )
        req_ids = (
            torch.arange(model_num_query_tokens, device=self.device, dtype=torch.int32) // block_size % batch_size
        )
        self.token_to_req_indices_buffer[:model_num_query_tokens].copy_(req_ids)
        self.token_to_req_indices_buffer[model_num_query_tokens:].fill_(-1)

    @torch.inference_mode()
    def dummy_run(
        self,
        num_tokens: int,
        num_reqs: int = 0,
        num_tokens_across_dp: torch.Tensor | None = None,
        aclgraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
        batch_descriptor=None,
        dummy_compute_logits=lambda hidden_states: None,
        is_profile=False,
        **kwargs,
    ) -> None:
        del dummy_compute_logits, kwargs
        block_size = self.num_speculative_tokens
        num_query_tokens = min(num_tokens, self.max_query_tokens)

        (
            num_input_tokens,
            num_tokens_across_dp,
            _,
        ) = self.runner._sync_metadata_across_dp(
            num_query_tokens,
            is_draft_model=True,
        )
        if not self.use_cuda_graph:
            aclgraph_runtime_mode = CUDAGraphMode.NONE
        num_query_total = min(num_reqs * block_size, num_query_tokens)
        model_num_query_tokens = num_input_tokens
        self._pad_draft_query_buffers(num_query_total, num_input_tokens)

        standard_ready = (
            bool(getattr(self, "draft_attn_groups", []))
            and model_num_query_tokens > 0
        )
        if standard_ready:
            self._prepare_dspark_dummy_standard_inputs(
                num_reqs, num_input_tokens, model_num_query_tokens
            )

        multi_steps_attn_metadata = []
        if standard_ready:
            dummy_cad = AscendCommonAttentionMetadata(
                query_start_loc=self._dspark_query_start_loc,
                query_start_loc_cpu=(
                    torch.from_numpy(self.token_arange_np[: num_reqs + 1]).clone() * block_size
                ).to(torch.int32),
                seq_lens=self._dspark_seq_lens,
                seq_lens_cpu=torch.full((num_reqs,), block_size, dtype=torch.int32),
                num_reqs=num_reqs,
                num_actual_tokens=num_query_total,
                max_query_len=block_size,
                max_seq_len=0,
                slot_mapping=self._slot_mapping_buffer[:num_input_tokens],
                attn_state=AscendAttentionState.ChunkedPrefill,
                causal=False,
                block_table_tensor=self._per_group_block_table_buffers[self.kv_cache_gid][:num_reqs],
            )
            multi_steps_attn_metadata, _ = self.build_draft_attn_metadata(
                dummy_cad, num_input_tokens, num_query_total
            )

        with set_ascend_forward_context(
            multi_steps_attn_metadata[0] if multi_steps_attn_metadata else None,
            self.vllm_config,
            num_tokens=num_input_tokens,
            num_tokens_across_dp=num_tokens_across_dp,
            num_actual_tokens=num_input_tokens,
            in_profile_run=is_profile,
            batch_descriptor=batch_descriptor,
            aclgraph_runtime_mode=aclgraph_runtime_mode,
            is_draft_model=True,
            draft_attn_metadatas=multi_steps_attn_metadata,
        ):
            self._dflash_num_context = num_input_tokens
            self.model.precompute_and_store_context_kv(
                self.hidden_states[:num_input_tokens],
                self._context_positions_buffer[:num_input_tokens],
                self._context_slots if standard_ready else None,
            )
            if model_num_query_tokens:
                self.model(
                    input_ids=self.input_ids[:model_num_query_tokens],
                    positions=self.positions[:model_num_query_tokens],
                )
            forward_context = get_forward_context()
            if (
                forward_context.cudagraph_runtime_mode == CUDAGraphMode.FULL
                and not _EXTRA_CTX.capturing
                and self.draft_attn_groups
            ):
                self._update_full_graph_params(forward_context, num_tokens, [])
