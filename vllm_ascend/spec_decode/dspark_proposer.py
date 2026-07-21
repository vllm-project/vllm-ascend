# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from typing import Any

import torch
from vllm.config import CUDAGraphMode, VllmConfig, get_layers_from_vllm_config
from vllm.forward_context import BatchDescriptor, ForwardContext
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.v1.attention.backends.utils import CommonAttentionMetadata
from vllm.v1.kv_cache_interface import UniformTypeKVCacheSpecs
from vllm.v1.worker.utils import AttentionGroup

from vllm_ascend.ascend_forward_context import set_ascend_forward_context
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.compilation.acl_graph import ACLGraphWrapper
from vllm_ascend.ops.triton.spec_decode.utils import copy_and_expand_dflash_and_dspark_inputs_kernel_single_grid
from vllm_ascend.spec_decode.dflash_proposer import AscendDflashProposer


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
        blk = 1 + self.num_speculative_tokens
        checkpoint_block_size = getattr(
            self.speculative_config.draft_model_config.hf_config,
            "block_size",
            None,
        )
        if checkpoint_block_size is None:
            checkpoint_block_size = self.num_speculative_tokens
        if checkpoint_block_size not in (self.num_speculative_tokens, blk):
            raise ValueError(
                "DSpark checkpoint block_size must equal num_speculative_tokens "
                "or one anchor plus all speculative tokens: "
                f"checkpoint block_size={checkpoint_block_size}, "
                f"num_speculative_tokens={self.num_speculative_tokens}"
            )
        self._dspark_query_tokens_per_req = int(checkpoint_block_size)
        self._dspark_sample_from_anchor = (
            self._dspark_query_tokens_per_req == self.num_speculative_tokens
        )
        self._dspark_draft_buffer = torch.zeros((self.max_batch_size, blk), dtype=torch.int64, device=device)
        self._dspark_seed_buffer = torch.zeros(self.max_batch_size, dtype=torch.int64, device=device)
        # DSpark is not supported in vllm v1, so related property needs to be reset here.
        del self.hidden_size, self.hidden_states, self._dflash_hidden_states  # type: ignore[has-type]
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
        # The v1 graph path is validated only for the GLM/Qwen checkpoint
        # contract that carries one anchor plus N supervised query positions.
        self.use_cuda_graph = self.use_cuda_graph and not self._dspark_sample_from_anchor
        self.max_query_tokens = self.max_batch_size * self._dspark_query_tokens_per_req
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
        self._context_slot_mapping_buffers: list[torch.Tensor | None] | None = None
        self._dspark_context_kv_graph: ACLGraphWrapper | None = None
        self._dspark_context_kv_precomputed = False

    def _primary_context_slot_mapping(self) -> torch.Tensor | None:
        context_slots = self._context_slot_mapping_buffers
        if isinstance(context_slots, list):
            return next((slots for slots in context_slots if slots is not None), None)
        return context_slots

    def _use_dspark_context_kv_bucket_graph(
        self,
        forward_context: ForwardContext,
    ) -> bool:
        return (
            self.use_cuda_graph
            and forward_context.cudagraph_runtime_mode == CUDAGraphMode.FULL
            and os.getenv(
                "VLLM_ASCEND_ENABLE_GLM_DSPARK_CONTEXT_KV_BUCKET_GRAPH",
                "0",
            ).lower()
            in ("1", "true", "yes", "on")
        )

    def _get_dspark_context_kv_bucket(self, num_context: int) -> int | None:
        if num_context <= 0 or num_context > self.max_query_tokens:
            return None
        width = self._dspark_query_tokens_per_req
        return min(
            self.max_query_tokens,
            ((num_context + width - 1) // width) * width,
        )

    def _precompute_padded_context_kv(
        self,
        context_states: torch.Tensor,
        context_positions: torch.Tensor,
        context_slot_mapping: torch.Tensor,
    ) -> torch.Tensor:
        self.model.precompute_and_store_context_kv(
            context_states,
            context_positions,
            context_slot_mapping,
        )
        return context_states

    def _precompute_context_kv_bucket_graph(
        self,
        forward_context: ForwardContext,
    ) -> bool:
        bucket = self._get_dspark_context_kv_bucket(self._dflash_num_context)
        context_slots = self._primary_context_slot_mapping()
        if bucket is None or context_slots is None:
            return False

        if self._dspark_context_kv_graph is None:
            self._dspark_context_kv_graph = ACLGraphWrapper(
                self._precompute_padded_context_kv,
                self.vllm_config,
                runtime_mode=CUDAGraphMode.FULL,
                use_eagle=self.use_eagle,
                enable_enpu=self.enable_enpu,
            )

        query_batch_descriptor = forward_context.batch_descriptor
        forward_context.batch_descriptor = BatchDescriptor(num_tokens=bucket)
        try:
            self._dspark_context_kv_graph(
                self._dflash_hidden_states[:bucket],
                self._context_positions_buffer[:bucket],
                context_slots[:bucket],
            )
        finally:
            forward_context.batch_descriptor = query_batch_descriptor

        self._dspark_context_kv_precomputed = True
        return True

    def _precompute_live_context_kv(self) -> None:
        num_context = self._dflash_num_context
        context_slots = self._context_slot_mapping_buffers
        primary_slots = self._primary_context_slot_mapping()
        if primary_slots is None:
            self.model.precompute_and_store_context_kv(
                self._dflash_hidden_states[:num_context],
                self._context_positions_buffer[:num_context],
                None,
            )
            return

        valid_context = primary_slots[:num_context] >= 0
        filtered_slots: torch.Tensor | list[torch.Tensor | None]
        if isinstance(context_slots, list):
            filtered_slots = [
                (
                    slots[:num_context][valid_context]
                    .to(torch.int32)
                    .contiguous()
                    if slots is not None
                    else None
                )
                for slots in context_slots
            ]
        else:
            filtered_slots = (
                primary_slots[:num_context][valid_context]
                .to(torch.int32)
                .contiguous()
            )

        self.model.precompute_and_store_context_kv(
            self._dflash_hidden_states[:num_context][valid_context].contiguous(),
            self._context_positions_buffer[:num_context][valid_context].contiguous(),
            filtered_slots,
        )

    def _initialize_graph_padding(
        self,
        num_context: int,
        num_query_total: int,
    ) -> None:
        context_bucket = (
            self._get_dspark_context_kv_bucket(num_context)
            if os.getenv(
                "VLLM_ASCEND_ENABLE_GLM_DSPARK_CONTEXT_KV_BUCKET_GRAPH",
                "0",
            ).lower()
            in ("1", "true", "yes", "on")
            else None
        ) or num_context

        if context_bucket > num_context:
            self._dflash_hidden_states[num_context:context_bucket].zero_()
            self._context_positions_buffer[num_context:context_bucket].zero_()
            for slots in self._per_group_context_slot_mapping_buffers.values():
                slots[num_context:context_bucket].fill_(-1)

        if self.max_query_tokens > num_query_total:
            self.input_ids[num_query_total : self.max_query_tokens].fill_(
                self.parallel_drafting_token_id
            )
            self.positions[num_query_total : self.max_query_tokens].zero_()
            for slots in self._per_group_query_slot_mapping_buffers.values():
                slots[num_query_total : self.max_query_tokens].fill_(-1)

    def prepare_dspark_context_kv_for_graph(
        self,
        forward_context: ForwardContext,
    ) -> bool:
        if (
            not self.use_cuda_graph
            or forward_context.cudagraph_runtime_mode != CUDAGraphMode.FULL
        ):
            return False
        if self._use_dspark_context_kv_bucket_graph(forward_context):
            if self._precompute_context_kv_bucket_graph(forward_context):
                return True
        self._precompute_live_context_kv()
        self._dspark_context_kv_precomputed = True
        return True

    def _stabilize_padded_graph_metadata(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        actual_num_reqs: int,
        padded_num_reqs: int,
    ) -> None:
        if padded_num_reqs <= actual_num_reqs:
            return
        common_attn_metadata.seq_lens[actual_num_reqs:padded_num_reqs].fill_(1)
        for attr_name in ("_seq_lens_cpu", "seq_lens_cpu"):
            seq_lens_cpu = getattr(common_attn_metadata, attr_name, None)
            if seq_lens_cpu is None:
                continue
            seq_lens_cpu = self._adjust_tensor(seq_lens_cpu, padded_num_reqs)
            seq_lens_cpu[actual_num_reqs:padded_num_reqs].fill_(1)
            setattr(common_attn_metadata, attr_name, seq_lens_cpu)
        if hasattr(common_attn_metadata, "actual_seq_lengths_q"):
            common_attn_metadata.actual_seq_lengths_q = [
                self._dspark_query_tokens_per_req
            ] * padded_num_reqs

    def build_model_inputs_first_pass(
        self,
        num_input_tokens: int,
        context_slots: torch.Tensor | list[torch.Tensor | None] | None,
    ) -> None:
        if not self._dspark_context_kv_precomputed:
            self._precompute_live_context_kv()

    def initialize_attn_backend(self, kv_cache_config, kernel_block_sizes=None) -> None:
        # Find draft layers (attention layers added by draft model)
        all_attn_layers = get_layers_from_vllm_config(
            self.vllm_config,
            AttentionLayerBase,  # type: ignore[type-abstract]
        )

        attention_groups_list: list[dict[tuple[str, str], AttentionGroup]] = []
        # the draft layers have multiple kv_cache_groups
        if not hasattr(self.model, "get_draft_kv_cache_layer_names"):
            raise RuntimeError(
                "DSpark standard-cache path requires the draft model to expose get_draft_kv_cache_layer_names"
            )

        self._draft_attn_layer_names = set(self.model.get_draft_kv_cache_layer_names())
        self.attn_layer_names = list(sorted(self._draft_attn_layer_names))

        # there are many kv groups other than one
        for kv_cache_gid, kv_cache_group_spec in enumerate(kv_cache_config.kv_cache_groups):
            draft_layer_names_in_group = set(kv_cache_group_spec.layer_names) & self._draft_attn_layer_names
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

        self.draft_attn_groups = [
            attention_group
            for attention_groups in attention_groups_list
            for attention_group in attention_groups.values()
        ]
        self.kv_cache_gid = 0
        if not self.draft_attn_groups:
            raise RuntimeError(
                "DSpark standard-cache path requires registered draft attention "
                f"groups. Missing layers: {self.attn_layer_names}"
            )

        self.kv_cache_gid = self.draft_attn_groups[0].kv_cache_group_id
        self.kernel_block_size = int(self.draft_attn_groups[0].kv_cache_spec.block_size)

        name_to_gid = {
            ln: gid
            for gid, group in enumerate(kv_cache_config.kv_cache_groups)
            for ln in group.layer_names
            if ln in self.attn_layer_names
        }
        self._layer_group_idx = [name_to_gid[name] for name in self.attn_layer_names]

        # some buffers need information of groups
        self._per_group_query_slot_mapping_buffers = {
            attn_group.kv_cache_group_id: torch.zeros(self.max_query_tokens, dtype=torch.int32, device=self.device)
            for attn_group in self.draft_attn_groups
        }
        self._per_group_context_slot_mapping_buffers = {
            attn_group.kv_cache_group_id: torch.zeros(self.max_num_tokens, dtype=torch.int32, device=self.device)
            for attn_group in self.draft_attn_groups
        }
        self._slot_mapping_buffer = self._per_group_query_slot_mapping_buffers[
            self.kv_cache_gid
        ]

    def set_per_group_attn_metadata(
        self,
        gid: int,
        block_table: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        self._per_group_block_tables[gid] = block_table
        self._per_group_slot_mappings[gid] = slot_mapping

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
        self._dspark_seed_buffer[n:].fill_(0)
        batch_size = cad.num_reqs
        num_query_per_req = self._dspark_query_tokens_per_req
        num_query_total = batch_size * num_query_per_req
        has_num_rejected = num_rejected_tokens_gpu is not None
        primary_gid = getattr(self, "kv_cache_gid", 0)
        old_query_start_loc = cad.query_start_loc
        self._per_group_block_table_buffers = {
            attn_group.kv_cache_group_id: self._per_group_block_tables[attn_group.kv_cache_group_id]
            for attn_group in self.draft_attn_groups
        }
        self._context_slot_mapping_buffers = None
        self._dflash_num_context = int(cad.query_start_loc_cpu[batch_size])
        self._dflash_hidden_states[: self._dflash_num_context] = target_hidden_states[: self._dflash_num_context]

        # below (SAMPLE_FROM_ANCHOR=True, anchor included) -- not arange here.
        token_indices_to_sample = torch.empty(
            batch_size * self.num_speculative_tokens,
            dtype=torch.int32,
            device=self.device,
        )

        # Fill each KV group's context/query slots while sharing query tokens,
        # positions, and sample indices across groups.
        draft_attn_groups = getattr(self, "draft_attn_groups", [])
        for attn_group in draft_attn_groups:
            gid = attn_group.kv_cache_group_id
            gid_block_table = self._per_group_block_table_buffers.get(gid)
            if gid_block_table is None:
                continue
            kv_block_size = int(attn_group.kv_cache_spec.block_size)
            copy_and_expand_dflash_and_dspark_inputs_kernel_single_grid[1,](
                # Inputs
                next_token_ids_ptr=next_token_ids,
                target_positions_ptr=target_positions,
                context_slot_mapping_ptr=self._per_group_slot_mappings[gid],
                # Outputs
                out_input_ids_ptr=self.input_ids,
                out_context_positions_ptr=self._context_positions_buffer,
                out_query_positions_ptr=self.positions,
                out_context_slot_mapping_ptr=self._per_group_context_slot_mapping_buffers[gid],
                out_query_slot_mapping_ptr=self._per_group_query_slot_mapping_buffers[gid],
                out_token_indices_ptr=token_indices_to_sample,
                # Block table
                block_table_ptr=gid_block_table,
                block_table_stride=gid_block_table.stride(0),
                # Metadata
                query_start_loc_ptr=old_query_start_loc,
                seq_lens_ptr=cad.seq_lens,
                num_rejected_tokens_ptr=num_rejected_tokens_gpu,
                # Scalars
                parallel_drafting_token_id=self.parallel_drafting_token_id,
                block_size=kv_block_size,
                num_query_per_req=num_query_per_req,
                num_speculative_tokens=self.num_speculative_tokens,
                total_input_tokens=self._dflash_num_context,
                batch_size=batch_size,
                HAS_NUM_REJECTED=has_num_rejected,
                SAMPLE_FROM_ANCHOR=self._dspark_sample_from_anchor,
                RECOMPUTE_CONTEXT_SLOTS=not self._dspark_sample_from_anchor,
            )
        # to compute self._context_slot_mapping_buffers from dict to list
        self._context_slot_mapping_buffers = [
            self._per_group_context_slot_mapping_buffers[gidx] for gidx in self._layer_group_idx
        ]
        if self.use_cuda_graph:
            self._initialize_graph_padding(
                self._dflash_num_context,
                num_query_total,
            )

        if self._dspark_sample_from_anchor:
            new_seq_lens = cad.seq_lens
            if has_num_rejected:
                new_seq_lens = new_seq_lens - num_rejected_tokens_gpu
            new_seq_lens = new_seq_lens + num_query_per_req
        else:
            last_position_indices = old_query_start_loc[1:] - 1
            if has_num_rejected:
                last_position_indices = (
                    last_position_indices - num_rejected_tokens_gpu
                )
            last_position_indices = torch.maximum(
                last_position_indices,
                old_query_start_loc[:-1],
            )
            last_valid_positions = target_positions.index_select(
                0,
                last_position_indices.long(),
            ).to(torch.int32)
            new_seq_lens = last_valid_positions + num_query_per_req + 1

        cad.query_start_loc = (
            self.arange_dflash[: batch_size + 1] * num_query_per_req
        )
        cad.seq_lens = new_seq_lens
        cad.query_start_loc_cpu = (
            torch.from_numpy(self.token_arange_np[: batch_size + 1]).clone()
            * num_query_per_req
        ).to(torch.int32)
        if getattr(cad, "_seq_lens_cpu", None) is not None:
            cad._seq_lens_cpu = new_seq_lens.detach().cpu()
        if getattr(cad, "seq_lens_cpu", None) is not None:
            cad.seq_lens_cpu = new_seq_lens.detach().cpu()

        if hasattr(cad, "actual_seq_lengths_q"):
            cad.actual_seq_lengths_q = [num_query_per_req] * batch_size
        if hasattr(cad, "decode_token_per_req"):
            cad.decode_token_per_req = num_query_per_req

        cad.num_actual_tokens = num_query_total
        cad.num_input_tokens = num_query_total
        cad.max_query_len = num_query_per_req
        cad.max_seq_len = cad.max_seq_len + num_query_per_req
        cad.slot_mapping = self._per_group_query_slot_mapping_buffers[primary_gid][:num_query_total]
        cad.positions = self.positions[:num_query_total]
        if getattr(cad, "positions_cpu", None) is not None:
            cad.positions_cpu = cad.positions.detach().cpu()
        cad.causal = False
        cad.attn_mask = None
        cad.attn_state = AscendAttentionState.ChunkedPrefill

        return num_query_total, token_indices_to_sample, cad, None

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
        if not self._dspark_sample_from_anchor:
            return AscendDflashProposer.dummy_run(
                self,
                num_tokens=num_tokens,
                num_reqs=num_reqs,
                num_tokens_across_dp=num_tokens_across_dp,
                aclgraph_runtime_mode=aclgraph_runtime_mode,
                batch_descriptor=batch_descriptor,
                dummy_compute_logits=dummy_compute_logits,
                is_profile=is_profile,
                **kwargs,
            )

        # The DSv4 contract samples the anchor query itself and therefore uses
        # exactly num_speculative_tokens query rows per request.
        # Ensure that the maximum batch token is within the limit of self.max_query_tokens.
        num_query_per_req = self._dspark_query_tokens_per_req
        num_query_total = num_reqs * num_query_per_req
        num_query_tokens = min(num_query_total if num_reqs > 0 else num_tokens, self.max_query_tokens)

        (
            num_input_tokens,
            num_tokens_across_dp,
            _,
        ) = self.runner._sync_metadata_across_dp(num_query_tokens, is_draft_model=True)

        if not self.use_cuda_graph:
            aclgraph_runtime_mode = CUDAGraphMode.NONE

        context_positions = self._context_positions_buffer[:num_input_tokens]
        context_states = self.hidden_states[:num_input_tokens]

        self.token_indices_to_sample.fill_(0)
        self._pad_draft_buffers(num_query_total, num_input_tokens)

        with set_ascend_forward_context(
            None,
            self.vllm_config,
            num_tokens=num_input_tokens,
            num_tokens_across_dp=num_tokens_across_dp,
            num_actual_tokens=num_input_tokens,
            in_profile_run=is_profile,
            batch_descriptor=batch_descriptor,
            aclgraph_runtime_mode=aclgraph_runtime_mode,
            is_draft_model=True,
            draft_attn_metadatas=[],
        ):
            if is_profile:
                self.model.precompute_and_store_context_kv(context_states, context_positions)
                self.model(
                    input_ids=self.input_ids[:num_query_total],
                    positions=self._get_positions(num_query_total),
                    inputs_embeds=None,
                )

            else:
                self._dflash_num_context = num_input_tokens
                self._runnable(
                    num_input_tokens=num_input_tokens,
                    batch_size=num_reqs,
                    token_indices_to_sample=self.token_indices_to_sample[: num_reqs * self.num_speculative_tokens],
                    target_positions=self._get_positions(num_input_tokens),
                    inputs_embeds=None,
                    multi_steps_attn_metadata=[],
                    num_tokens=num_input_tokens,
                )
