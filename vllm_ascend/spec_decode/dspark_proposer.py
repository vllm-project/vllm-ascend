# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import torch
from vllm.config import CUDAGraphMode, VllmConfig, get_layers_from_vllm_config
from vllm.forward_context import BatchDescriptor, get_forward_context
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.v1.attention.backends.utils import CommonAttentionMetadata
from vllm.v1.kv_cache_interface import UniformTypeKVCacheSpecs
from vllm.v1.worker.utils import AttentionGroup

from vllm_ascend.ascend_forward_context import _EXTRA_CTX, set_ascend_forward_context
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.utils import AscendCommonAttentionMetadata
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
        self.sample_from_anchor = not getattr(self.draft_model_config.hf_config, "dspark_bonus_anchor", False)
        if self.sample_from_anchor:
            self.num_query_per_req = self.num_speculative_tokens
        else:
            self.num_query_per_req = 1 + self.num_speculative_tokens

        blk = 1 + self.num_speculative_tokens
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
        # DSpark ACLGraph is now supported; use_cuda_graph is inherited from
        # AscendSpecDecodeBaseProposer (which checks runner._use_aclgraph() and
        # speculative_config.enforce_eager).  Do NOT override it here.
        # Keep enough capacity for both native DSpark query graphs and the
        # target model's larger 1+N context/capture descriptors. Bonus-anchor
        # DSpark also uses the full 1+N query width.
        self.max_query_tokens = self.max_batch_size * (1 + self.num_speculative_tokens)
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

        # Populated by initialize_attn_backend after memory profiling and KV
        # cache configuration have completed.
        self._layer_group_idx: list[int] = []

        # per-layer context slot mappings as a flat list
        self._context_slot_mapping_buffers: list[torch.Tensor | None] | None = None

    def get_graph_num_input_tokens(self, batch_descriptor: BatchDescriptor) -> int:
        """Use DSpark's native query width for uniform decode graphs.

        The target verifies ``1 + N`` tokens per request, while anchor-first
        DSpark only evaluates ``N`` query tokens. The target descriptor remains
        the ACLGraph cache key, but the draft model and FIA graph are captured
        with their own token count.
        """
        if batch_descriptor.uniform and batch_descriptor.num_reqs is not None:
            return batch_descriptor.num_reqs * self.num_query_per_req
        return batch_descriptor.num_tokens

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
        # Bind the per-layer view as soon as the persistent per-group buffers
        # exist. ACLGraph capture runs before the first real request; delaying
        # this binding until set_inputs_first_pass makes the Qwen context-KV
        # precompute return early and permanently omits all cache-update ops
        # from the captured graph.
        self._bind_context_slot_mapping_buffers()

    def _bind_context_slot_mapping_buffers(self) -> None:
        """Bind each draft layer to its persistent context slot buffer.

        The list itself is only model-facing bookkeeping. Its tensors are the
        persistent per-group buffers populated in-place for every request, so
        their addresses stay stable across ACLGraph capture and replay.
        """
        self._context_slot_mapping_buffers = [
            self._per_group_context_slot_mapping_buffers[group_idx]
            for group_idx in self._layer_group_idx
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
        num_query_total = batch_size * self.num_query_per_req
        num_sample_total = batch_size * self.num_speculative_tokens
        has_num_rejected = num_rejected_tokens_gpu is not None
        primary_gid = getattr(self, "kv_cache_gid", 0)
        num_context = int(cad.query_start_loc_cpu[batch_size])
        self._per_group_block_table_buffers = {
            attn_group.kv_cache_group_id: self._per_group_block_tables[attn_group.kv_cache_group_id]
            for attn_group in self.draft_attn_groups
        }
        self._dflash_num_context = num_context
        self._dflash_hidden_states[: self._dflash_num_context] = target_hidden_states[: self._dflash_num_context]

        token_indices_to_sample = torch.empty(
            num_sample_total,
            dtype=torch.int32,
            device=self.device,
        )

        # Query block: reuse the DFlash inputs kernel logic (host-side ref)
        # per kv-cache-group to fill positions / input_ids / query slot_mapping
        # / token_indices.
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
                query_start_loc_ptr=cad.query_start_loc,
                seq_lens_ptr=cad.seq_lens,
                num_rejected_tokens_ptr=num_rejected_tokens_gpu,
                # Scalars
                parallel_drafting_token_id=self.parallel_drafting_token_id,
                block_size=kv_block_size,
                num_query_per_req=self.num_query_per_req,
                num_speculative_tokens=self.num_speculative_tokens,
                total_input_tokens=self._dflash_num_context,
                batch_size=batch_size,
                HAS_NUM_REJECTED=has_num_rejected,
                SAMPLE_FROM_ANCHOR=self.sample_from_anchor,
            )
        # Rebind defensively in case attention groups were rebuilt. The tensor
        # objects remain the persistent buffers filled by the kernel above.
        self._bind_context_slot_mapping_buffers()

        effective_seq_lens = cad.seq_lens
        if has_num_rejected:
            effective_seq_lens = effective_seq_lens - num_rejected_tokens_gpu

        cad.query_start_loc = self.arange_dflash[: batch_size + 1] * self.num_query_per_req
        cad.seq_lens = effective_seq_lens + self.num_query_per_req
        cad.query_start_loc_cpu = (
            torch.from_numpy(self.token_arange_np[: batch_size + 1]).clone() * self.num_query_per_req
        ).to(torch.int32)

        if hasattr(cad, "actual_seq_lengths_q"):
            cad.actual_seq_lengths_q = [self.num_query_per_req] * batch_size
        if hasattr(cad, "decode_token_per_req"):
            cad.decode_token_per_req = self.num_query_per_req

        cad.num_actual_tokens = num_query_total
        cad.num_input_tokens = num_query_total
        cad.max_query_len = self.num_query_per_req
        cad.max_seq_len = cad.max_seq_len + self.num_query_per_req
        cad.slot_mapping = self._per_group_query_slot_mapping_buffers[primary_gid][:num_query_total]
        cad.positions = self.positions  # this would be sliced in attention backend
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
        num_query_total = num_reqs * self.num_query_per_req
        # The target descriptor is based on the verification width (1 + N),
        # but anchor-first DSpark's query backbone only consumes N tokens per
        # request. Keep the descriptor for ACLGraph dispatch and capture the
        # draft computation at its native width.
        if aclgraph_runtime_mode == CUDAGraphMode.FULL and batch_descriptor is not None:
            graph_query_tokens = self.get_graph_num_input_tokens(batch_descriptor)
        else:
            graph_query_tokens = num_query_total if num_reqs > 0 else num_tokens
        num_query_tokens = min(graph_query_tokens, self.max_query_tokens)

        # Context K/V precomputation consumes the target model's verification
        # hidden states and therefore retains the original 1 + N token width.
        # This is intentionally independent from num_query_tokens.
        num_context_tokens = min(num_tokens, self.max_num_tokens)

        # Memory profiling also enters dummy_run, but it runs before KV-cache
        # initialization and therefore has no layer-to-group mapping yet. Bind
        # only after initialize_attn_backend has created that mapping. The
        # subsequent ACLGraph capture then records the context cache updates,
        # while the pre-KV profile keeps the original no-cache behavior.
        if getattr(self, "_layer_group_idx", None):
            self._bind_context_slot_mapping_buffers()

        (
            num_input_tokens,
            num_tokens_across_dp,
            _,
        ) = self.runner._sync_metadata_across_dp(num_query_tokens, is_draft_model=True)

        if not self.use_cuda_graph:
            aclgraph_runtime_mode = CUDAGraphMode.NONE

        context_positions = self._context_positions_buffer[:num_context_tokens]
        context_states = self.hidden_states[:num_context_tokens]

        # Build capture metadata for ACLGraph FULL mode, mirroring dFlash but
        # with DSpark-specific query geometry and per-group block table / slot
        # mapping.
        multi_steps_attn_metadata: list[dict[str, Any]] = []
        if aclgraph_runtime_mode == CUDAGraphMode.FULL and len(self.draft_attn_groups) > 0:
            # The native DSpark graph normally has exactly N query tokens per
            # captured request. Retain a fallback tail request only for
            # non-uniform or externally padded descriptors.
            qsl_cpu = (torch.from_numpy(self.token_arange_np[: num_reqs + 1]).clone() * self.num_query_per_req).to(
                torch.int32
            )
            self.query_start_loc.cpu[: num_reqs + 1].copy_(qsl_cpu)
            # Add virtual-request padding
            num_reqs_padded = num_reqs
            if self.query_start_loc.np[num_reqs] < num_input_tokens:
                self.query_start_loc.np[num_reqs + 1] = num_input_tokens
                num_reqs_padded = num_reqs + 1
            self.query_start_loc.copy_to_gpu()

            per_layer_attn_metadata: dict[str, Any] = {}
            for attn_group in self.draft_attn_groups:
                gid = attn_group.kv_cache_group_id
                builder = attn_group.get_metadata_builder()
                block_table = self._per_group_block_table_buffers.get(gid)
                if block_table is None:
                    # Fallback: read from runner's input_batch (populated during
                    # _dummy_run per-kv-group loop).
                    block_table = self.runner.input_batch.block_table[gid].get_device_tensor()[:num_reqs_padded]

                # Pad block_table for the fallback virtual request above
                # (mirrors the builder's own replay-path padding).
                if block_table.shape[0] < num_reqs_padded:
                    block_table = torch.cat(
                        [
                            block_table,
                            block_table.new_zeros((num_reqs_padded - block_table.shape[0], block_table.shape[1])),
                        ],
                        dim=0,
                    )

                # Pad seq_lens for the dummy request; its KV length is
                # irrelevant because the FIA output for padding tokens is
                # ignored, but the list/tensor length must match
                # num_reqs_padded.
                seq_lens = self.runner.seq_lens[:num_reqs]
                if seq_lens.shape[0] < num_reqs_padded:
                    seq_lens = torch.cat([seq_lens, seq_lens.new_ones(num_reqs_padded - seq_lens.shape[0])])

                common_attn_metadata = AscendCommonAttentionMetadata(
                    query_start_loc=self.query_start_loc.gpu[: num_reqs_padded + 1],
                    query_start_loc_cpu=self.query_start_loc.cpu[: num_reqs_padded + 1],
                    seq_lens_cpu=self.runner.optimistic_seq_lens_cpu,
                    seq_lens_cpu_upper_bound=self.runner.optimistic_seq_lens_cpu,
                    seq_lens=seq_lens,
                    num_reqs=num_reqs_padded,
                    num_actual_tokens=num_input_tokens,
                    num_input_tokens=num_input_tokens,
                    max_query_len=self.num_query_per_req,
                    max_seq_len=0,
                    slot_mapping=self._per_group_query_slot_mapping_buffers[gid][:num_input_tokens],
                    positions=self.positions,
                    attn_state=AscendAttentionState.ChunkedPrefill,
                    causal=False,
                    is_prefilling=torch.zeros(num_reqs_padded, dtype=torch.bool),
                    block_table_tensor=block_table,
                )

                attn_metadata = builder.build_for_graph_capture(
                    common_attn_metadata,
                    AscendAttentionState.ChunkedPrefill,
                )
                attn_metadata.attn_mask = None
                attn_metadata.attn_state = AscendAttentionState.ChunkedPrefill

                for layer_name in attn_group.layer_names:
                    per_layer_attn_metadata[layer_name] = attn_metadata

            multi_steps_attn_metadata.append(per_layer_attn_metadata)

        self.token_indices_to_sample.fill_(0)
        self._pad_draft_buffers(num_query_total, num_input_tokens)

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
            if is_profile:
                self.model.precompute_and_store_context_kv(context_states, context_positions)
                self.model(
                    input_ids=self.input_ids[:num_query_total],
                    positions=self._get_positions(num_query_total),
                    inputs_embeds=None,
                )

            else:
                self._dflash_num_context = num_context_tokens
                self._runnable(
                    num_input_tokens=num_input_tokens,
                    batch_size=num_reqs,
                    token_indices_to_sample=self.token_indices_to_sample[: num_reqs * self.num_speculative_tokens],
                    target_positions=self._get_positions(num_input_tokens),
                    inputs_embeds=None,
                    multi_steps_attn_metadata=multi_steps_attn_metadata,
                    num_tokens=num_input_tokens,
                )

            forward_context = get_forward_context()
            if forward_context.cudagraph_runtime_mode == CUDAGraphMode.FULL and not _EXTRA_CTX.capturing:
                self._update_full_graph_params(forward_context, num_input_tokens, multi_steps_attn_metadata)
