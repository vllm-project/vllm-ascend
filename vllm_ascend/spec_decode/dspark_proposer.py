# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import copy
import os
from typing import Any

import torch
from vllm.config import CUDAGraphMode, VllmConfig, get_layers_from_vllm_config
from vllm.forward_context import get_forward_context
from vllm.logger import logger
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.v1.attention.backends.utils import CommonAttentionMetadata
from vllm.v1.kv_cache_interface import UniformTypeKVCacheSpecs
from vllm.v1.worker.utils import AttentionGroup

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.ascend_forward_context import set_ascend_forward_context
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.dsa_v1 import AscendDSAMetadataBuilder
from vllm_ascend.attention.utils import AscendCommonAttentionMetadata, enable_pcp
from vllm_ascend.ops.triton.spec_decode.utils import copy_and_expand_dflash_and_dspark_inputs_kernel
from vllm_ascend.spec_decode.dflash_proposer import AscendDflashProposer, _compute_num_programs
from vllm_ascend.spec_decode.utils import DynamicSpecScheduler

# ACL-graph capture for the DSpark draft model (DeepSeek V4.x MTP heads).
#
# Upstream runs the DSpark draft eager-only (``use_cuda_graph = False``), which
# costs ~20% of end-to-end decode throughput.  Running the draft under ACL
# graphs requires the capture to respect three invariants that the generic
# dummy-run path violates; each was isolated with bidirectional A/B on a
# single chip (same inputs, same metadata, identical kernel histograms):
#
# 1. Real capture metadata.  Without draft attention metadata the DSA
#    attention falls into its ``no metadata`` branch during capture and the
#    replayed graph contains no attention at all (acceptance length collapses
#    to ~1.0 with no error raised).
#
# 2. Value-domain baking ("what is captured is what replays").  Python scalars
#    and tensor *values* read during capture are baked into the graph; refresh
#    buffer contents at replay does not fix them.  Concretely:
#    ``seq_lens``-derived lengths must be representative at capture time
#    (dummy values such as 0/5 bake a wrong value domain), and the context-KV
#    write (``precompute_and_store_context_kv``) must be captured *inside* the
#    graph against resident slot buffers so the replay writes to the right
#    locations.
#
# 3. Bucket alignment.  The capture-side ``num_context`` must equal the
#    replay-side value ``num_reqs * (1 + num_speculative_tokens)``; the
#    bucket-aligned token count diverges from it for more than one request
#    and bakes a wrong slice length into the graph.
#
# The feature is opt-in and disabled by default:
#     VLLM_ASCEND_ENABLE_DSPARK_DRAFT_GRAPH=1
_DSPARK_DRAFT_GRAPH_ENABLED = os.environ.get("VLLM_ASCEND_ENABLE_DSPARK_DRAFT_GRAPH", "0") == "1"


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
        hf_config = self.draft_model_config.hf_config
        hf_config = getattr(hf_config, "text_config", hf_config)
        self.sample_from_anchor = getattr(hf_config, "sample_from_anchor", True)
        if self.sample_from_anchor:
            self.num_query_per_req = self.num_speculative_tokens
        else:
            self.num_query_per_req = 1 + self.num_speculative_tokens

        blk = 1 + self.num_speculative_tokens
        self._dspark_draft_buffer = torch.zeros((self.max_batch_size, blk), dtype=torch.int64, device=device)
        self._dspark_seed_buffer = torch.zeros(self.max_batch_size, dtype=torch.int64, device=device)
        # Replace the target-sized DFlash buffers with the draft model's hidden
        # size. Assignment releases the old tensors without an explicit del.
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
        dynamic_spec_config = get_ascend_config().dynamic_spec_config
        self.dynamic_spec = None

        if dynamic_spec_config.method == "dspark":
            self.dynamic_spec = DynamicSpecScheduler(
                method="dspark",
                method_params=dynamic_spec_config.method_params,
                max_batch_size=self.max_batch_size,
                num_speculative_tokens=self.num_speculative_tokens,
                device=device,
            )
        # Upstream runs the DSpark draft eager-only.  With the three capture
        # invariants documented at the top of this file handled, the draft can
        # run under ACL graphs like the other drafters; keep it strictly
        # opt-in so the default is bit-identical to upstream.
        if _DSPARK_DRAFT_GRAPH_ENABLED:
            _runner_use_aclgraph = getattr(runner, "_use_aclgraph", None)
            self.use_cuda_graph = bool(
                callable(_runner_use_aclgraph)
                and _runner_use_aclgraph()
                and not vllm_config.speculative_config.enforce_eager
            )
        else:
            self.use_cuda_graph = False
        # Max query tokens depend on whether sampling from anchor or not.
        #
        # [DRAFT-GRAPH-NUMINPUT-FIX] This attribute is a *capacity* for the
        # per-query buffers (positions / query slot mappings), not the shape the
        # draft graph computes: DSpark's real query block is
        # `max_batch_size * num_query_per_req` (5 per request here, one row per
        # speculative token — `copy_and_expand_dflash_and_dspark_inputs_kernel`
        # loops `num_query_total = batch_size * num_query_per_req` and its
        # SAMPLE_FROM_ANCHOR branch maps all of a request's rows onto that
        # request's `num_speculative_tokens` samples), and `dummy_run` clamps the
        # *capture* count to this capacity. Under ACLGraph, `_propose` however
        # pads the draft to the runner's capture bucket
        # (`cudagraph_dispatcher.dispatch(...).num_tokens`, always a multiple of
        # `1 + num_speculative_tokens` — see
        # `adjust_cudagraph_sizes_for_spec_decode`) and then indexes the per-query
        # buffers with that padded count (`_pad_draft_buffers`,
        # `build_draft_attn_metadata` -> `dsa_v1.build_for_drafting` ->
        # `spec_slot_mapping[draft_index - 1][:num_input_tokens]`).
        # A capacity below the bucket makes `_pad_draft_buffers` a silent no-op
        # (its `buf[num_tokens:num_input_tokens].fill_(-1)` then covers nothing)
        # while the padded slice still stays shorter than the assignment target —
        # that is the `[6, 2] = [5, 2]` crash. Size it like DFlash, i.e. for the
        # largest bucket the runner can dispatch a draft step to; the extra rows
        # stay inert (slot -1 / position 0 / parallel-drafting token id) and are
        # never read by the captured graph, which is captured with the *unpadded*
        # count, so FIA never computes a junk row.
        self.max_query_tokens = max(
            self.max_batch_size * self.num_query_per_req,
            self.max_batch_size * (1 + self.num_speculative_tokens),
        )
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

        # The v1 runner owns block tables and slot mappings. Keep per-group
        # references here because K3 draft layers can span multiple cache
        # groups with different logical block sizes.
        self._per_group_block_tables: dict[int, torch.Tensor] = {}
        self._per_group_slot_mappings: dict[int, torch.Tensor] = {}
        # Per-gid logical block size used to expand slot mappings. The KV
        # manager's physical page can be larger when hybrid cache groups share
        # one allocation, so kv_cache_spec.block_size is not interchangeable
        # with the attention kernel's block size.
        self._per_group_kernel_block_sizes: dict[int, int] = {}

        self._per_group_block_table_buffers: dict[int, torch.Tensor] = {}
        self._per_group_query_slot_mapping_buffers: dict[int, torch.Tensor] = {}
        self._per_group_context_slot_mapping_buffers: dict[int, torch.Tensor] = {}
        self._context_slot_mapping_buffers: list[torch.Tensor | None] | None = None

    def _compute_confidence(
        self,
        last_hidden_states: torch.Tensor,
        draft_token_ids: torch.Tensor,
        num_reqs: int,
    ) -> torch.Tensor:
        num_tokens = num_reqs * self.num_speculative_tokens
        flat_hidden = last_hidden_states.reshape(num_tokens, last_hidden_states.shape[-1])
        # Markov embeddings of the draft input tokens (cheap lookup, so they
        # are recomputed here instead of being captured in the drafting loop).
        markov_embs = self.model.markov_embed(draft_token_ids[:, : self.num_speculative_tokens])
        # The confidence head concatenates both inputs, so their dtypes must
        # match; it upcasts to float32 internally.
        flat_markov = markov_embs.reshape(num_tokens, markov_embs.shape[-1]).to(flat_hidden.dtype)
        conf_raw = self.model.compute_confidence(flat_hidden, flat_markov)
        confidence = self._dspark_confidence_logits_buffer[:num_reqs]
        confidence.copy_(conf_raw.reshape(num_reqs, self.num_speculative_tokens))
        return confidence

    def initialize_attn_backend(
        self,
        kv_cache_config,
        kernel_block_sizes: list[int] | None = None,
    ) -> None:
        # Find draft layers (attention layers added by draft model)
        all_attn_layers = get_layers_from_vllm_config(
            self.vllm_config,
            AttentionLayerBase,  # type: ignore[type-abstract]
        )

        self._draft_attn_layer_names = set(self.model.get_draft_kv_cache_layer_names())
        self.attn_layer_names = list(sorted(self._draft_attn_layer_names))
        self._per_group_kernel_block_sizes = {}
        self.draft_attn_groups: list[AttentionGroup] = []

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
                    kernel_block_size = int(
                        kernel_block_sizes[kv_cache_gid]
                        if kernel_block_sizes is not None and kv_cache_gid < len(kernel_block_sizes)
                        else layer_kv_cache_spec.block_size
                    )
                    attn_group = AttentionGroup(
                        attn_backend,
                        [layer_name],
                        layer_kv_cache_spec,
                        kv_cache_gid,
                    )
                    attn_group.create_metadata_builders(
                        self.vllm_config,
                        self.device,
                        kernel_block_size=kernel_block_size,
                    )
                    self._per_group_kernel_block_sizes[kv_cache_gid] = kernel_block_size
                    attention_groups[key] = attn_group
                else:
                    attention_groups[key].layer_names.append(layer_name)

            self.draft_attn_groups.extend(attention_groups.values())

        if (
            getattr(self.runner, "device_metadata_executor", None) is not None
            and self.dcp_size == 1
            and not enable_pcp()
        ):
            for attn_group in self.draft_attn_groups:
                builder = attn_group.get_metadata_builder()
                if isinstance(builder, AscendDSAMetadataBuilder):
                    # Register the exact per-request query count first so the
                    # buffer quota and the triton capacity grid agree: anchor
                    # sampling drafts spec queries per request, the non-anchor
                    # path drafts spec+1.
                    builder.set_dspark_num_query_per_req(self.num_query_per_req)
                    builder.enable_dspark_device_metadata(self.max_query_tokens)
                else:
                    from vllm_ascend.attention.dsa_v41 import AscendDSAV41MetadataBuilder

                    if isinstance(builder, AscendDSAV41MetadataBuilder):
                        builder.enable_device_metadata()

        self.kv_cache_gid = self.draft_attn_groups[0].kv_cache_group_id
        self.kernel_block_size = self._per_group_kernel_block_sizes[self.kv_cache_gid]

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
        dcp_size = getattr(self, "dcp_size", 1)
        dcp_rank = getattr(self, "dcp_rank", 0)
        cp_interleave_size = self.vllm_config.parallel_config.cp_kv_cache_interleave_size if dcp_size > 1 else 1
        long_seq_args = None
        primary_gid = getattr(self, "kv_cache_gid", 0)
        self._per_group_block_table_buffers = {
            attn_group.kv_cache_group_id: self._per_group_block_tables[attn_group.kv_cache_group_id]
            for attn_group in self.draft_attn_groups
        }
        self._context_slot_mapping_buffers = None
        self._dflash_num_context = int(cad.query_start_loc_cpu[batch_size])
        self._dflash_hidden_states[: self._dflash_num_context] = target_hidden_states[: self._dflash_num_context]

        token_indices_to_sample = torch.empty(
            num_sample_total,
            dtype=torch.int32,
            device=self.device,
        )

        # Query block: reuse the DFlash inputs kernel logic (host-side ref)
        # per kv-cache-group to fill positions / input_ids / query slot_mapping
        # / token_indices.
        for attn_group in self.draft_attn_groups:
            gid = attn_group.kv_cache_group_id
            gid_block_table = self._per_group_block_table_buffers[gid]
            kernel_block_size = self._per_group_kernel_block_sizes[gid]
            copy_and_expand_dflash_and_dspark_inputs_kernel[
                (_compute_num_programs(self._dflash_num_context, num_query_total),)
            ](
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
                block_size=kernel_block_size,
                num_query_per_req=self.num_query_per_req,
                num_speculative_tokens=self.num_speculative_tokens,
                total_input_tokens=self._dflash_num_context,
                batch_size=batch_size,
                HAS_NUM_REJECTED=has_num_rejected,
                SAMPLE_FROM_ANCHOR=self.sample_from_anchor,
                DCP_SIZE=dcp_size,
                DCP_RANK=dcp_rank,
                CP_INTERLEAVE_SIZE=cp_interleave_size,
            )
        # to compute self._context_slot_mapping_buffers from dict to list
        self._context_slot_mapping_buffers = [
            self._per_group_context_slot_mapping_buffers[gidx] for gidx in self._layer_group_idx
        ]

        effective_seq_lens = cad.seq_lens
        if has_num_rejected:
            effective_seq_lens = effective_seq_lens - num_rejected_tokens_gpu

        cad.query_start_loc = self.arange_dflash[: batch_size + 1] * self.num_query_per_req
        cad.seq_lens = effective_seq_lens + self.num_query_per_req
        # The model runner has already corrected this canonical host mirror
        # with the accepted-token count. Extend it on CPU alongside the device
        # lengths, without another reject D2H copy or attention-side wait.
        if cad._seq_lens_cpu is not None:
            draft_seq_lens_cpu = cad._seq_lens_cpu.clone()
            draft_seq_lens_cpu[:batch_size].add_(self.num_query_per_req)
            cad._seq_lens_cpu = draft_seq_lens_cpu
            if getattr(cad, "seq_lens_cpu", None) is not None:
                cad.seq_lens_cpu = draft_seq_lens_cpu
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
        if hasattr(self.model, "get_draft_attn_causal"):
            # Currently, attention causality across draft layers are uniform.
            cad.causal = self.model.get_draft_attn_causal()[0]
        else:
            cad.causal = False
        cad.attn_mask = None
        cad.attn_state = AscendAttentionState.ChunkedPrefill

        if dcp_size > 1:
            assert self.runner is not None
            dcp_manager = getattr(self.runner, "dcp_manager", None)
            assert dcp_manager is not None
            dcp_manager.prepare_parallel_draft_metadata(cad, self.draft_attn_groups)

        return num_query_total, token_indices_to_sample, cad, long_seq_args

    def _clear_dummy_slot_mappings(self) -> None:
        for buf in self._per_group_query_slot_mapping_buffers.values():
            buf.fill_(-1)
        for buf in self._per_group_context_slot_mapping_buffers.values():
            buf.fill_(-1)

    def _build_capture_draft_attn_metadata(
        self,
        num_reqs: int,
        num_input_tokens: int,
    ) -> tuple[list[dict[str, Any]], list[Any]]:
        """Build the draft attention metadata that FULL-graph capture bakes in.

        Only used from ``dummy_run`` while capturing. Every tensor referenced by
        the returned metadata must be one of the persistent buffers that the
        replay path (``set_inputs_first_pass`` + ``build_draft_attn_metadata``)
        refreshes in place on every step: ``spec_slot_mapping``,
        ``spec_sas_metadata``, ``dspark_swa_indices_buffer``, ``seq_lens_group``,
        ``query_start_loc_group``, the per-group block tables / query slot
        mappings and ``positions``. The construction therefore mirrors
        ``build_draft_attn_metadata`` so capture and replay resolve to the exact
        same device addresses.
        """
        batch_size = num_reqs
        num_query_total = batch_size * self.num_query_per_req
        num_actual_tokens = min(num_query_total, num_input_tokens)
        primary_gid = getattr(self, "kv_cache_gid", 0)

        if not self._per_group_block_table_buffers:
            for attn_group in self.draft_attn_groups:
                gid = attn_group.kv_cache_group_id
                block_table = self._per_group_block_tables.get(gid)
                if block_table is not None:
                    self._per_group_block_table_buffers[gid] = block_table
        if primary_gid not in self._per_group_block_table_buffers:
            for gid, block_table in self._per_group_block_table_buffers.items():
                primary_gid = gid
                break
        if primary_gid not in self._per_group_query_slot_mapping_buffers:
            query_slot_mapping = self._slot_mapping_buffer[:num_input_tokens]
        else:
            query_slot_mapping = self._per_group_query_slot_mapping_buffers[primary_gid][:num_input_tokens]

        query_start_loc = self.query_start_loc_group[0]
        query_start_loc[: batch_size + 1].copy_(self.arange_dflash[: batch_size + 1] * self.num_query_per_req)
        query_start_loc[batch_size + 1 :].fill_(0)
        seq_lens = self.seq_lens_group[0]
        seq_lens[:batch_size].copy_(self.runner.seq_lens[:batch_size] + self.num_query_per_req)
        seq_lens[batch_size:].fill_(0)

        cad = AscendCommonAttentionMetadata(
            query_start_loc=query_start_loc[: batch_size + 1],
            query_start_loc_cpu=(
                torch.from_numpy(self.token_arange_np[: batch_size + 1]).clone() * self.num_query_per_req
            ).to(torch.int32),
            seq_lens=seq_lens[:batch_size],
            seq_lens_cpu=self.runner.optimistic_seq_lens_cpu[:batch_size],
            num_reqs=batch_size,
            num_actual_tokens=num_actual_tokens,
            num_input_tokens=num_input_tokens,
            max_query_len=self.num_query_per_req,
            max_seq_len=0,
            decode_token_per_req=self.num_query_per_req,
            slot_mapping=query_slot_mapping,
            block_table_tensor=self._per_group_block_table_buffers[primary_gid][:batch_size],
            positions=self.positions,
            attn_state=AscendAttentionState.ChunkedPrefill,
            causal=False,
            is_prefilling=torch.zeros(batch_size, dtype=torch.bool),
        )

        shared_cache: dict[str, Any] = dict(common_ratio_to_sas_metadata=dict()) if self.use_compress else {}
        per_layer_attn_metadata: dict[str, Any] = {}
        captured_tasks: list[Any] = []
        for attn_group in self.draft_attn_groups:
            gid = attn_group.kv_cache_group_id
            builder = attn_group.get_metadata_builder()
            group_cad = copy.copy(cad)
            block_table = self._per_group_block_table_buffers.get(gid)
            if block_table is not None:
                group_cad.block_table_tensor = block_table[:batch_size]
            slot_mapping = self._per_group_query_slot_mapping_buffers.get(gid)
            if slot_mapping is not None:
                group_cad.slot_mapping = slot_mapping[:num_input_tokens]
            if self.sliding_window is not None:
                self.sliding_window.apply(group_cad)
            attn_metadata = builder.build_for_drafting(group_cad, draft_index=1, **dict(shared_cache))
            take_tasks = getattr(builder, "take_device_metadata_tasks", None)
            if callable(take_tasks):
                captured_tasks.extend(take_tasks())
            for layer_name in attn_group.layer_names:
                per_layer_attn_metadata[layer_name] = attn_metadata
        logger.info(
            "[dspark-graph-capture] built capture metadata: %d layers, "
            "num_reqs=%d num_input_tokens=%d num_query_total=%d",
            len(per_layer_attn_metadata),
            num_reqs,
            num_input_tokens,
            num_query_total,
        )
        return [per_layer_attn_metadata], captured_tasks

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
        self._clear_dummy_slot_mappings()

        multi_steps_attn_metadata: list[dict[str, Any]] = []
        if (
            _DSPARK_DRAFT_GRAPH_ENABLED
            and self.use_cuda_graph
            and aclgraph_runtime_mode == CUDAGraphMode.FULL
            and not is_profile
            and num_reqs > 0
        ):
            # Invariant 2 (see module docstring) must be applied *before* the
            # capture metadata is built: the builder reads ``runner.seq_lens``
            # and ``optimistic_seq_lens_cpu``, and building the metadata with
            # dummy values bakes a wrong value domain into the graph.  Running
            # this block after the builder was verified to make the capture
            # completely ineffective (same config, only the order differs).
            # (a) Representative value domain: ``max_num_tokens`` is the upper
            #     bound of ``num_context``; verified to replay correctly for
            #     sequence lengths from 6 to 262144 (small values such as 6
            #     bake a wrong domain and break replay).
            # (b) Restore the in-graph context-KV write: point
            #     ``_context_slot_mapping_buffers`` at the resident per-group
            #     buffers so ``precompute_and_store_context_kv`` is captured
            #     against stable addresses that ``set_inputs_first_pass``
            #     refreshes in place on every replayed step.
            self._context_slot_mapping_buffers = [
                self._per_group_context_slot_mapping_buffers[gidx] for gidx in self._layer_group_idx
            ]
            try:
                _cap_len = int(getattr(self, "max_num_tokens", 0) or 0) or 8192
                self.runner.seq_lens[:num_reqs].fill_(_cap_len)
                self.runner.optimistic_seq_lens_cpu[:num_reqs].fill_(_cap_len)
            except Exception as e:  # pragma: no cover - defensive
                logger.warning("[dspark-graph-capture] failed to set representative seq_lens: %r", e)
            multi_steps_attn_metadata, _capture_tasks = self._build_capture_draft_attn_metadata(
                num_reqs, num_input_tokens
            )
            # The deferred draft metadata tasks are intentionally NOT run here:
            # the per-step path (llm_base_proposer.build_draft_attn_metadata)
            # executes them synchronously before every replay, which keeps the
            # graph free of device-metadata waits.

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
            model_instance=self.model,
            draft_attn_metadatas=multi_steps_attn_metadata,
            eplb_heat_collection_status=(
                self.runner.eplb_heat_collection_status if self.runner.dynamic_eplb else False
            ),
        ):
            if is_profile:
                self.model.precompute_and_store_context_kv(context_states, context_positions)
                self.model(
                    input_ids=self.input_ids[:num_query_total],
                    positions=self._get_positions(num_query_total),
                    inputs_embeds=None,
                )

            else:
                # Invariant 3 (bucket alignment): the capture-side
                # ``num_context`` must equal the replay-side value
                # ``num_reqs * (1 + num_speculative_tokens)`` computed by
                # ``set_inputs_first_pass``.  The bucket-aligned token count
                # diverges from it for more than one request and would bake a
                # wrong slice length (too few context-KV rows) into the graph.
                self._dflash_num_context = int(num_reqs) * (1 + int(self.num_speculative_tokens))
                self._runnable(
                    num_input_tokens=num_input_tokens,
                    batch_size=num_reqs,
                    token_indices_to_sample=self.token_indices_to_sample[: num_reqs * self.num_speculative_tokens],
                    target_positions=self._get_positions(num_input_tokens),
                    inputs_embeds=None,
                    multi_steps_attn_metadata=multi_steps_attn_metadata,
                    num_tokens=num_input_tokens,
                )
            if multi_steps_attn_metadata:
                forward_context = get_forward_context()
                if forward_context.cudagraph_runtime_mode == CUDAGraphMode.FULL and not getattr(
                    forward_context, "capturing", False
                ):
                    # Mirror DFlash: refresh per-step graph params after a real replay
                    # (a no-op for DSA, kept for parity with the other draft backends).
                    self._update_full_graph_params(forward_context, num_input_tokens, multi_steps_attn_metadata)
