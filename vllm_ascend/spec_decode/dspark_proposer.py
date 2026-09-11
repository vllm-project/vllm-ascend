# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import torch
from vllm.config import CUDAGraphMode, VllmConfig, get_layers_from_vllm_config
from vllm.forward_context import BatchDescriptor, get_forward_context
from vllm.logger import logger
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.v1.attention.backends.utils import CommonAttentionMetadata
from vllm.v1.kv_cache_interface import UniformTypeKVCacheSpecs
from vllm.v1.worker.utils import AttentionGroup

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.ascend_forward_context import _EXTRA_CTX, set_ascend_forward_context
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.dsa_v1 import AscendDSAMetadataBuilder
from vllm_ascend.attention.utils import AscendCommonAttentionMetadata, enable_pcp
from vllm_ascend.ops.triton.spec_decode.utils import copy_and_expand_dflash_and_dspark_inputs_kernel
from vllm_ascend.spec_decode.dflash_proposer import AscendDflashProposer, _compute_num_programs
from vllm_ascend.spec_decode.utils import DynamicSpecScheduler

_DSPARK_ACLGRAPH_CAPABILITY = "supports_dspark_aclgraph"


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
        # Reserve fixed-address space for both the anchor-first N-token query
        # and the target verifier's (N + 1)-token capture descriptor.
        graph_query_width = max(
            self.num_query_per_req,
            1 + self.num_speculative_tokens,
        )
        self.max_query_tokens = self.max_batch_size * graph_query_width
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

    def _model_supports_dspark_aclgraph(self) -> bool:
        """Whether the loaded DSpark adapter opted in to ACLGraph."""
        return bool(
            getattr(
                self.get_model(),
                _DSPARK_ACLGRAPH_CAPABILITY,
                False,
            )
        )

    def _maybe_share_lm_head(self, model) -> None:
        # This hook runs after loading the draft model and before the base
        # proposer creates ACLGraphWrapper, so it is the last safe point to
        # reject unsupported adapters without disabling the target graph.
        if self.use_cuda_graph:
            if self.dynamic_spec is not None:
                logger.warning_once(
                    "Dynamic DSpark verify length is not compatible with ACLGraph yet; falling back to eager drafting."
                )
                self.use_cuda_graph = False
            elif not self._model_supports_dspark_aclgraph():
                logger.warning_once(
                    "The loaded DSpark draft model does not declare ACLGraph "
                    "support; falling back to eager drafting while keeping "
                    "the target graph enabled."
                )
                self.use_cuda_graph = False
            elif self.vllm_config.compilation_config.cudagraph_mode != CUDAGraphMode.FULL_DECODE_ONLY:
                logger.warning_once(
                    "DSpark ACLGraph currently supports only FULL_DECODE_ONLY; falling back to eager drafting."
                )
                self.use_cuda_graph = False
        super()._maybe_share_lm_head(model)

    def uses_target_batch_descriptor_for_graph(self) -> bool:
        """Cache DSpark graphs with the target verifier descriptor."""
        return True

    def get_graph_num_input_tokens(
        self,
        batch_descriptor: BatchDescriptor,
    ) -> int:
        """Map target (N + 1) capture geometry to DSpark's query width."""
        if batch_descriptor.uniform and batch_descriptor.num_reqs is not None:
            return batch_descriptor.num_reqs * self.num_query_per_req
        return batch_descriptor.num_tokens

    def _pad_request_tensor(
        self,
        tensor: torch.Tensor | None,
        num_actual_reqs: int,
        num_reqs_padded: int,
        pad_value: int = 0,
    ) -> torch.Tensor | None:
        if tensor is None:
            return None
        padded = self._adjust_tensor(tensor[:num_actual_reqs], num_reqs_padded)
        if num_reqs_padded > num_actual_reqs and pad_value != 0:
            padded[num_actual_reqs:].fill_(pad_value)
        return padded

    def prepare_target_batch_descriptor_for_graph(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        batch_descriptor: BatchDescriptor,
        num_actual_tokens: int,
    ) -> int:
        """Translate verifier padding to DSpark's fixed-width draft batch."""
        if batch_descriptor.num_reqs is None:
            raise ValueError("A uniform DSpark graph descriptor must include num_reqs")

        num_actual_reqs = common_attn_metadata.num_reqs
        num_reqs_padded = batch_descriptor.num_reqs
        if num_reqs_padded < num_actual_reqs:
            raise ValueError(
                "DSpark graph request capacity is smaller than the runtime batch: "
                f"{num_reqs_padded} < {num_actual_reqs}"
            )

        num_input_tokens = num_reqs_padded * self.num_query_per_req
        if num_input_tokens < num_actual_tokens:
            raise ValueError(
                "DSpark graph token capacity is smaller than the runtime draft: "
                f"{num_input_tokens} < {num_actual_tokens}"
            )

        # The target and draft share a graph cache key, but their uniform query
        # widths differ: the verifier has K + 1 tokens/request and anchor-first
        # DSpark has K. Materialize the latter explicitly for all padded rows.
        common_attn_metadata.query_start_loc = self.arange_dflash[: num_reqs_padded + 1] * self.num_query_per_req
        common_attn_metadata.query_start_loc_cpu = (
            torch.from_numpy(self.token_arange_np[: num_reqs_padded + 1]).clone() * self.num_query_per_req
        ).to(torch.int32)

        common_attn_metadata.seq_lens = self._pad_request_tensor(
            common_attn_metadata.seq_lens,
            num_actual_reqs,
            num_reqs_padded,
            self.num_query_per_req,
        )
        seq_lens_cpu = self._pad_request_tensor(
            common_attn_metadata._seq_lens_cpu,
            num_actual_reqs,
            num_reqs_padded,
            self.num_query_per_req,
        )
        if seq_lens_cpu is None:
            seq_lens_cpu = self._pad_request_tensor(
                common_attn_metadata.seq_lens_cpu,
                num_actual_reqs,
                num_reqs_padded,
                self.num_query_per_req,
            )
        common_attn_metadata._seq_lens_cpu = seq_lens_cpu
        common_attn_metadata.seq_lens_cpu = seq_lens_cpu.clone() if seq_lens_cpu is not None else None
        common_attn_metadata.seq_lens_cpu_upper_bound = seq_lens_cpu.clone() if seq_lens_cpu is not None else None
        common_attn_metadata.num_computed_tokens_cpu = self._pad_request_tensor(
            common_attn_metadata.num_computed_tokens_cpu,
            num_actual_reqs,
            num_reqs_padded,
        )
        if getattr(common_attn_metadata, "_num_computed_tokens_cpu", None) is not None:
            common_attn_metadata._num_computed_tokens_cpu = self._pad_request_tensor(
                common_attn_metadata._num_computed_tokens_cpu,
                num_actual_reqs,
                num_reqs_padded,
            )
        if common_attn_metadata.is_prefilling is not None:
            common_attn_metadata.is_prefilling = self._pad_request_tensor(
                common_attn_metadata.is_prefilling,
                num_actual_reqs,
                num_reqs_padded,
            )

        primary_gid = self.draft_attn_groups[0].kv_cache_group_id
        common_attn_metadata.block_table_tensor = self._per_group_block_table_buffers[primary_gid][:num_reqs_padded]
        common_attn_metadata.slot_mapping = self._per_group_query_slot_mapping_buffers[primary_gid][:num_input_tokens]
        common_attn_metadata.actual_seq_lengths_q = [self.num_query_per_req] * num_reqs_padded
        common_attn_metadata.num_reqs = num_reqs_padded
        common_attn_metadata.num_actual_tokens = num_input_tokens
        common_attn_metadata.num_input_tokens = num_input_tokens

        # Context KV still comes from the verifier input and therefore uses the
        # target K + 1 width. Clear padded rows so replay cannot consume stale
        # hidden states, positions, or cache slots from an earlier larger batch.
        num_actual_context = self._dflash_num_context
        num_graph_context = batch_descriptor.num_tokens
        if num_graph_context < num_actual_context:
            raise ValueError(
                "DSpark graph context capacity is smaller than the verifier input: "
                f"{num_graph_context} < {num_actual_context}"
            )
        if num_graph_context > num_actual_context:
            self._dflash_hidden_states[num_actual_context:num_graph_context].zero_()
            self._context_positions_buffer[num_actual_context:num_graph_context].zero_()
            for context_slots in self._per_group_context_slot_mapping_buffers.values():
                context_slots[num_actual_context:num_graph_context].fill_(-1)
        self._dflash_num_context = num_graph_context
        return num_reqs_padded

    def _bind_context_slot_mapping_buffers(self) -> None:
        """Bind persistent per-layer mappings before capture and replay."""
        self._context_slot_mapping_buffers = [
            self._per_group_context_slot_mapping_buffers[group_idx] for group_idx in self._layer_group_idx
        ]

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
            )
        self._bind_context_slot_mapping_buffers()

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

        return num_query_total, token_indices_to_sample, cad, None

    def _build_graph_capture_attn_metadata(
        self,
        num_reqs: int,
        num_input_tokens: int,
        batch_descriptor: BatchDescriptor,
    ) -> list[dict[str, Any]]:
        """Build DSpark metadata with graph-stable MRV1 buffers.

        Reuse the normal draft metadata path so sparse indices, SAS metadata,
        DSA-CP state, and external-event synchronization have the same
        lifecycle during capture and replay.
        """
        if not self.draft_attn_groups:
            return []

        num_query_total = num_reqs * self.num_query_per_req
        self._per_group_block_table_buffers = {
            group.kv_cache_group_id: self._per_group_block_tables[group.kv_cache_group_id]
            for group in self.draft_attn_groups
        }
        self._bind_context_slot_mapping_buffers()

        query_start_loc = self.query_start_loc_group[0][: num_reqs + 1]
        query_start_loc.copy_(self.arange_dflash[: num_reqs + 1] * self.num_query_per_req)
        query_start_loc_cpu = (
            torch.from_numpy(self.token_arange_np[: num_reqs + 1]).clone() * self.num_query_per_req
        ).to(torch.int32)

        seq_lens = self.seq_lens_group[0][:num_reqs]
        seq_lens.copy_(self.runner.seq_lens[:num_reqs])
        target_query_width = 1 + self.num_speculative_tokens
        seq_lens.sub_(target_query_width).clamp_(min=0).add_(self.num_query_per_req)
        seq_lens_cpu = (self.runner.optimistic_seq_lens_cpu[:num_reqs] - target_query_width).clamp(
            min=0
        ) + self.num_query_per_req

        primary_gid = self.draft_attn_groups[0].kv_cache_group_id
        common_attn_metadata = AscendCommonAttentionMetadata(
            query_start_loc=query_start_loc,
            query_start_loc_cpu=query_start_loc_cpu,
            seq_lens_cpu=seq_lens_cpu,
            _seq_lens_cpu=seq_lens_cpu,
            seq_lens_cpu_upper_bound=seq_lens_cpu,
            seq_lens=seq_lens,
            num_reqs=num_reqs,
            num_actual_tokens=num_query_total,
            num_input_tokens=num_input_tokens,
            max_query_len=self.num_query_per_req,
            max_seq_len=0,
            actual_seq_lengths_q=[self.num_query_per_req] * num_reqs,
            decode_token_per_req=self.num_query_per_req,
            block_table_tensor=self._per_group_block_table_buffers[primary_gid][:num_reqs],
            slot_mapping=self._per_group_query_slot_mapping_buffers[primary_gid],
            positions=self.positions,
            attn_state=AscendAttentionState.ChunkedPrefill,
            causal=False,
            is_prefilling=torch.zeros(num_reqs, dtype=torch.bool),
        )
        multi_steps_attn_metadata, _ = self.build_draft_attn_metadata(
            common_attn_metadata,
            num_input_tokens,
            num_query_total,
            batch_descriptor=batch_descriptor,
        )
        return multi_steps_attn_metadata

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
        if aclgraph_runtime_mode == CUDAGraphMode.FULL and batch_descriptor is not None:
            graph_query_tokens = self.get_graph_num_input_tokens(batch_descriptor)
        else:
            graph_query_tokens = num_query_total if num_reqs > 0 else num_tokens
        num_query_tokens = min(graph_query_tokens, self.max_query_tokens)

        (
            num_input_tokens,
            num_tokens_across_dp,
            _,
        ) = self.runner._sync_metadata_across_dp(num_query_tokens, is_draft_model=True)
        if num_tokens_across_dp is not None:
            num_input_tokens = int(num_tokens_across_dp[self.dp_rank].item())

        if not self.use_cuda_graph:
            aclgraph_runtime_mode = CUDAGraphMode.NONE

        graph_context_tokens = (
            batch_descriptor.num_tokens
            if aclgraph_runtime_mode == CUDAGraphMode.FULL and batch_descriptor is not None
            else num_input_tokens
        )
        context_positions = self._context_positions_buffer[:graph_context_tokens]
        context_states = self.hidden_states[:graph_context_tokens]

        self.token_indices_to_sample.fill_(0)
        self._pad_draft_buffers(num_query_total, num_input_tokens)

        multi_steps_attn_metadata = []
        if aclgraph_runtime_mode == CUDAGraphMode.FULL:
            if batch_descriptor is None:
                raise ValueError("FULL DSpark graph capture requires a batch descriptor")
            self._dflash_hidden_states[:graph_context_tokens].zero_()
            self._context_positions_buffer[:graph_context_tokens].zero_()
            for query_slots in self._per_group_query_slot_mapping_buffers.values():
                query_slots[:num_input_tokens].fill_(-1)
            for context_slots in self._per_group_context_slot_mapping_buffers.values():
                context_slots[:graph_context_tokens].fill_(-1)
            multi_steps_attn_metadata = self._build_graph_capture_attn_metadata(
                num_reqs,
                num_input_tokens,
                batch_descriptor,
            )

        active_device_metadata_executor = (
            getattr(self.runner, "device_metadata_executor", None) if multi_steps_attn_metadata else None
        )
        if active_device_metadata_executor is not None and not active_device_metadata_executor.submission_in_flight:
            active_device_metadata_executor = None

        with set_ascend_forward_context(
            multi_steps_attn_metadata[0] if multi_steps_attn_metadata else None,
            self.vllm_config,
            num_tokens=num_input_tokens,
            num_tokens_across_dp=num_tokens_across_dp,
            num_actual_tokens=num_query_total,
            in_profile_run=is_profile,
            batch_descriptor=batch_descriptor,
            aclgraph_runtime_mode=aclgraph_runtime_mode,
            is_draft_model=True,
            draft_attn_metadatas=multi_steps_attn_metadata,
            device_metadata_executor=active_device_metadata_executor,
            eplb_heat_collection_status=(
                self.runner.eplb_heat_collection_status if self.runner.dynamic_eplb else False
            ),
        ):
            if is_profile:
                self.model.precompute_and_store_context_kv(
                    context_states,
                    context_positions,
                    self._context_slot_mapping_buffers,
                )
                self.model(
                    input_ids=self.input_ids[:num_query_total],
                    positions=self._get_positions(num_query_total),
                    inputs_embeds=None,
                )

            else:
                self._dflash_num_context = graph_context_tokens
                self._runnable(
                    num_input_tokens=num_input_tokens,
                    batch_size=num_reqs,
                    token_indices_to_sample=self.token_indices_to_sample[: num_reqs * self.num_speculative_tokens],
                    target_positions=self._get_positions(num_input_tokens),
                    inputs_embeds=None,
                    multi_steps_attn_metadata=multi_steps_attn_metadata,
                    num_tokens=num_query_total,
                )

            forward_context = get_forward_context()
            if forward_context.cudagraph_runtime_mode == CUDAGraphMode.FULL and not _EXTRA_CTX.capturing:
                self._update_full_graph_params(
                    forward_context,
                    num_input_tokens,
                    multi_steps_attn_metadata,
                )

        if active_device_metadata_executor is not None:
            active_device_metadata_executor.release()
