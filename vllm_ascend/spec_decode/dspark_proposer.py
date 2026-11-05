# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import torch
from vllm.config import CUDAGraphMode, VllmConfig, get_layers_from_vllm_config
from vllm.logger import logger
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.v1.attention.backends.utils import CommonAttentionMetadata
from vllm.v1.kv_cache_interface import UniformTypeKVCacheSpecs
from vllm.v1.worker.utils import AttentionGroup

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.ascend_forward_context import set_ascend_forward_context
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.dsa_v1 import AscendDSAMetadataBuilder
from vllm_ascend.attention.utils import enable_pcp
from vllm_ascend.ops.triton.spec_decode.utils import copy_and_expand_dflash_and_dspark_inputs_kernel
from vllm_ascend.spec_decode.dflash_proposer import AscendDflashProposer, _compute_num_programs
from vllm_ascend.spec_decode.utils import DynamicSpecScheduler


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
        self.sample_from_anchor = getattr(self.draft_model_config.hf_config, "sample_from_anchor", True)
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
        # Remove the hard-coded False: use the same expression as the base class.
        # Real capture metadata comes from _build_capture_cad in dummy_run.
        try:
            self.use_cuda_graph = bool(self.runner._use_aclgraph() and not self.speculative_config.enforce_eager)
        except Exception:
            self.use_cuda_graph = False
        # Draft graph replay depends on the batch contract and on metadata
        # producers being ordered on the consumer stream; both are always on.
        self._draft_graph_contract = True
        self._draft_metadata_current_stream = True
        # Target descriptor tokens are a dispatch key, NOT draft query rows.
        # For SP=5, descriptor (192, 32) executes 32*5=160 draft queries.
        self.max_query_tokens = self.max_batch_size * self.num_query_per_req
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
        if not getattr(self, "use_cuda_graph", False):
            self._per_group_block_tables[gid] = block_table
            self._per_group_slot_mappings[gid] = slot_mapping
            return
        # Keep an enlarged persistent block-table buffer.
        # The runner passes a view sized by the target-side num_reqs_padded; the draft
        # mixed-batch branch of _pad_query_start_loc_for_fia adds one dummy request
        # (num_reqs_padded + 1), so the two sides disagree on the row count;
        #   AssertionError: assert num_reqs <= num_reqs_padded
        # allocate max_num_reqs + 2 rows and zero the extra rows.
        _rows = int(block_table.shape[0])
        _cols = int(block_table.shape[1])
        _cap = int(getattr(self.runner, "max_num_reqs", _rows)) + 2
        _buffers = self._block_table_buffer_map()
        _buf = _buffers.get(gid)
        if _buf is None or int(_buf.shape[0]) < max(_rows, _cap) or int(_buf.shape[1]) != _cols:
            _buf = torch.zeros(
                (max(_rows, _cap), _cols),
                dtype=block_table.dtype,
                device=block_table.device,
            )
            _buffers[gid] = _buf
        _buf[:_rows].copy_(block_table)
        if int(_buf.shape[0]) > _rows:
            _buf[_rows:].fill_(0)
        self._per_group_block_tables[gid] = _buf
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

    def _block_table_buffer_map(self) -> dict[int, torch.Tensor]:
        """Return the per-group block-table buffer map, creating it on demand.

        Some callers build the proposer without running ``__init__`` (unit tests,
        warm-up helpers), so the map must not be assumed to exist.
        """
        buffers = getattr(self, "_per_group_block_table_buffers", None)
        if buffers is None:
            buffers = {}
            self._per_group_block_table_buffers = buffers
        return buffers

    def _pad_uniform_draft(self, cad, padded_reqs: int) -> int:
        """Pad draft requests using q, never the target's q+1 layout."""
        actual_reqs = cad.num_reqs
        query_tokens = padded_reqs * self.num_query_per_req
        assert actual_reqs <= padded_reqs <= self.max_batch_size
        assert query_tokens <= self.max_query_tokens
        self.query_start_loc.cpu[: actual_reqs + 1].copy_(cad.query_start_loc_cpu)
        if actual_reqs < padded_reqs:
            self.query_start_loc.np[actual_reqs + 1 : padded_reqs + 1] = (
                self.token_arange_np[actual_reqs + 1 : padded_reqs + 1] * self.num_query_per_req
            )
        self.query_start_loc.copy_to_gpu()
        cad.num_reqs = padded_reqs
        cad.query_start_loc = self.query_start_loc.gpu[: padded_reqs + 1]
        cad.query_start_loc_cpu = self.query_start_loc.cpu[: padded_reqs + 1]
        cad.seq_lens = self._adjust_tensor(cad.seq_lens, padded_reqs)
        if cad.seq_lens_cpu is not None:
            cad.seq_lens_cpu = self._adjust_tensor(cad.seq_lens_cpu, padded_reqs)
        if cad._seq_lens_cpu is not None:
            cad._seq_lens_cpu = self._adjust_tensor(cad._seq_lens_cpu, padded_reqs)
        if cad.num_computed_tokens_cpu is not None:
            cad.num_computed_tokens_cpu = self._adjust_tensor(cad.num_computed_tokens_cpu, padded_reqs)
        cad.block_table_tensor = self._adjust_tensor(cad.block_table_tensor, padded_reqs)
        for table in self._block_table_buffer_map().values():
            table[actual_reqs:padded_reqs].zero_()
        # Graph precompute captures padded_reqs*(SP+1) context rows. Real
        # rows can be fewer after rejection; inactive rows must never write KV.
        context_capacity = padded_reqs * (1 + self.num_speculative_tokens)
        actual_context = self._dflash_num_context
        assert actual_context <= context_capacity <= self.max_num_tokens
        self._dflash_hidden_states[actual_context:context_capacity].zero_()
        self._context_positions_buffer[actual_context:context_capacity].zero_()
        for slots in self._per_group_context_slot_mapping_buffers.values():
            slots[actual_context:].fill_(-1)
        return padded_reqs

    def _clear_dummy_slot_mappings(self) -> None:
        for buf in self._per_group_query_slot_mapping_buffers.values():
            buf.fill_(-1)
        for buf in self._per_group_context_slot_mapping_buffers.values():
            buf.fill_(-1)

    @torch.inference_mode()
    def _build_capture_cad(self, num_reqs: int, num_input_tokens: int):
        """Build a representative CAD for capture.

        The goal is for `build_draft_attn_metadata` to emit metadata that references
        persistent buffers during capture, so the graph bakes those addresses in;
        `set_inputs_first_pass` refreshes the same buffers in place for replay.

        This is required for graph capture: every tensor comes from a persistent buffer.
        """
        batch_size = num_reqs
        num_query_total = batch_size * self.num_query_per_req
        num_actual_tokens = min(num_query_total, num_input_tokens)
        primary_gid = getattr(self, "kv_cache_gid", 0)

        buffers = self._block_table_buffer_map()
        if not buffers:
            for attn_group in self.draft_attn_groups:
                gid = attn_group.kv_cache_group_id
                bt = self._per_group_block_tables.get(gid)
                if bt is not None:
                    buffers[gid] = bt
        if primary_gid not in buffers:
            for gid in buffers:
                primary_gid = gid
                break

        if primary_gid not in self._per_group_query_slot_mapping_buffers:
            query_slot_mapping = self._slot_mapping_buffer[:num_input_tokens]
        else:
            query_slot_mapping = self._per_group_query_slot_mapping_buffers[primary_gid][:num_input_tokens]

        # Capture needs a representative seq_len: dummy or small values leave replay
        # ineffective at real sequence lengths. Use max_num_tokens (8192 in production).
        # Restore context KV writes: attach the per-group context slot-mapping buffers
        # during capture so attention writes context KV; otherwise the draft sees no context.
        try:
            if getattr(self, "_layer_group_idx", None) and getattr(
                self, "_per_group_context_slot_mapping_buffers", None
            ):
                self._context_slot_mapping_buffers = [
                    self._per_group_context_slot_mapping_buffers[gidx] for gidx in self._layer_group_idx
                ]
                logger.warning(
                    "[draft-graph] capture context KV writes restored for %d groups",
                    len(self._context_slot_mapping_buffers),
                )
        except Exception as _ce:
            logger.warning("[draft-graph] failed to restore context slot mapping: %r", _ce)

        _cap_len = int(getattr(self, "max_num_tokens", 0) or 0) or 8192
        try:
            self.runner.seq_lens[:batch_size].fill_(_cap_len)
            self.runner.optimistic_seq_lens_cpu[:batch_size].fill_(_cap_len)
            logger.warning("[draft-graph] capture representative seq_len=%d (batch=%d)", _cap_len, batch_size)
        except Exception as _ce:
            logger.warning("[draft-graph] failed to set representative seq_len: %r", _ce)

        query_start_loc = self.query_start_loc_group[0]
        query_start_loc[: batch_size + 1].copy_(self.arange_dflash[: batch_size + 1] * self.num_query_per_req)
        query_start_loc[batch_size + 1 :].fill_(0)
        seq_lens = self.seq_lens_group[0]
        seq_lens[:batch_size].copy_(self.runner.seq_lens[:batch_size] + self.num_query_per_req)
        seq_lens[batch_size:].fill_(0)

        # Capture slot_mapping values are dummies: set them to -1 (pad, no KV write).
        _slot = self._per_group_query_slot_mapping_buffers.get(primary_gid)
        if _slot is None:
            _slot = self._slot_mapping_buffer
        _slot[:num_input_tokens].fill_(-1)

        # AscendAttentionState is imported at module scope; only the CAD class is added here.
        from vllm_ascend.attention.utils import AscendCommonAttentionMetadata

        return AscendCommonAttentionMetadata(
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
        )

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

        # Bake real draft attention metadata into the graph (the original passed an empty list).
        multi_steps_attn_metadata = []
        if self.use_cuda_graph and aclgraph_runtime_mode == CUDAGraphMode.FULL and not is_profile and num_reqs > 0:
            try:
                _cad = self._build_capture_cad(num_reqs, num_input_tokens)
                # Returns a pair: (multi_steps_attn_metadata, attn_metadata_i)
                _md_list, _md_first = self.build_draft_attn_metadata(_cad, num_input_tokens, num_input_tokens)
                _md = _md_list
                if self._draft_graph_contract and (not _md or not _md[0]):
                    raise RuntimeError("Draft graph capture requires nonempty attention metadata")
                self._draft_capture_first_md = _md_first
                # Capture must release the device-side metadata submission, otherwise the next
                # submission fails with "The previous device metadata submission has not been released".
                _ex = getattr(self.runner, "device_metadata_executor", None)
                if _ex is not None and (not self._draft_graph_contract or _ex.submission_in_flight):
                    try:
                        _ex.release()
                    except Exception as _re:
                        logger.warning("[draft-graph] executor.release() failed: %r", _re)
                if _md:
                    # _md is list[dict[layer_name -> attn_metadata]], one entry per step
                    multi_steps_attn_metadata = list(_md)
                    logger.warning(
                        "[draft-graph] capture: built draft attn metadata "
                        "(steps=%d layers=%d num_query_total=%d num_input_tokens=%d)",
                        len(multi_steps_attn_metadata),
                        len(multi_steps_attn_metadata[0]) if multi_steps_attn_metadata else 0,
                        num_query_total,
                        num_input_tokens,
                    )
            except Exception as _e:
                if self._draft_graph_contract:
                    raise RuntimeError("Draft capture metadata contract failed") from _e
                logger.warning("[draft-graph] capture metadata build FAILED: %r", _e)

        with set_ascend_forward_context(
            # The model forward looks up draft metadata by key from its first positional
            # attn_metadata argument; passing None left the draft without attention. Pass the dict.
            (multi_steps_attn_metadata[0] if multi_steps_attn_metadata else None),
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
                # Capture must use the same context row count that replay computes,
                # otherwise the graph writes context KV for only num_input_tokens rows.
                self._dflash_num_context = num_input_tokens
                self._runnable(
                    num_input_tokens=num_input_tokens,
                    batch_size=num_reqs,
                    token_indices_to_sample=self.token_indices_to_sample[: num_reqs * self.num_speculative_tokens],
                    target_positions=self._get_positions(num_input_tokens),
                    inputs_embeds=None,
                    multi_steps_attn_metadata=multi_steps_attn_metadata,
                    num_tokens=num_input_tokens,
                )
