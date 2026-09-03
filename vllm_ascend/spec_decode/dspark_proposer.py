# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import copy
from dataclasses import replace
from typing import Any

import torch
from vllm.config import (
    CUDAGraphMode,
    VllmConfig,
    get_layers_from_vllm_config,
    set_current_vllm_config,
)
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.v1.attention.backends.utils import CommonAttentionMetadata
from vllm.v1.kv_cache_interface import UniformTypeKVCacheSpecs
from vllm.v1.worker.utils import AttentionGroup

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.ascend_forward_context import set_ascend_forward_context
from vllm_ascend.attention.attention_v1 import (
    AscendAttentionMetadataBuilder,
    AscendAttentionState,
)
from vllm_ascend.attention.dsa_v1 import AscendDSAMetadataBuilder
from vllm_ascend.attention.utils import enable_dcp, enable_pcp
from vllm_ascend.core.kv_cache_interface import (
    AscendDCPReplicatedDraftAttentionSpec,
)
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
        self.replicated_draft_kv = self._uses_dcp_replicated_draft_kv()
        if self.replicated_draft_kv:
            # Target attention remains DCP-aware. The GQA draft attention is
            # deliberately built with a DCP=1 config and a replicated cache.
            self.dcp_size = 1
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
        # DSpark runs eager only (Ascend cudagraph unsupported on this path).
        self.use_cuda_graph = False
        # Max query tokens depend on whether sampling from anchor or not.
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
        self._per_group_manager_block_sizes: dict[int, int] = {}
        self._per_group_replication_sizes: dict[int, int] = {}

        self._per_group_block_table_buffers: dict[int, torch.Tensor] = {}
        self._per_group_query_slot_mapping_buffers: dict[int, torch.Tensor] = {}
        self._per_group_context_slot_mapping_buffers: dict[int, torch.Tensor] = {}
        self._context_slot_mapping_buffers: list[torch.Tensor | None] | None = None
        self._replicated_block_table_storage: dict[int, torch.Tensor] = {}
        self._replicated_block_table_arange: dict[int, torch.Tensor] = {}

    def _uses_dcp_replicated_draft_kv(self) -> bool:
        config = getattr(self, "vllm_config", None)
        if config is None:
            return False
        spec_config = config.speculative_config
        runner = getattr(self, "runner", None)
        if spec_config is None or runner is None:
            return False
        target_model_config = config.model_config
        target_architectures = {
            *(getattr(target_model_config, "architectures", ()) or ()),
            *(getattr(target_model_config.hf_config, "architectures", ()) or ()),
        }
        target_architecture = getattr(target_model_config, "architecture", None)
        if target_architecture:
            target_architectures.add(target_architecture)
        draft_hf_config = spec_config.draft_model_config.hf_config
        draft_architectures = {
            *(getattr(spec_config.draft_model_config, "architectures", ()) or ()),
            *(getattr(draft_hf_config, "architectures", ()) or ()),
        }
        return (
            (
                getattr(target_model_config.hf_config, "model_type", None) == "kimi_k3"
                or any("KimiK3" in architecture for architecture in target_architectures)
            )
            and getattr(draft_hf_config, "model_type", None) == "qwen3"
            and any(architecture in {"DSparkDraftModel", "Qwen3DSparkModel"} for architecture in draft_architectures)
        )

    def _get_model(self):
        if not self._uses_dcp_replicated_draft_kv():
            return super()._get_model()

        # enable_dcp() is cached process-wide. Refresh it while the parent
        # loader installs the draft's DCP=1 config so the draft Attention
        # layers construct the ordinary GQA implementation, then restore the
        # target DCP setting for the rest of worker initialization.
        enable_dcp.cache_clear()
        try:
            return super()._get_model()
        finally:
            enable_dcp.cache_clear()
            with set_current_vllm_config(self.vllm_config):
                enable_dcp()

    def _create_draft_vllm_config(self) -> VllmConfig:
        base = super()._create_draft_vllm_config()
        if not self._uses_dcp_replicated_draft_kv():
            return base
        spec_config = self.speculative_config
        draft_parallel_config = copy.copy(spec_config.draft_parallel_config)
        draft_parallel_config.rank = self.vllm_config.parallel_config.rank
        draft_parallel_config.decode_context_parallel_size = 1
        return replace(
            base,
            model_config=spec_config.draft_model_config,
            parallel_config=draft_parallel_config,
        )

    def _build_replicated_block_table(
        self,
        gid: int,
        dcp_block_table: torch.Tensor,
        seq_lens: torch.Tensor,
    ) -> torch.Tensor:
        replication_size = self._per_group_replication_sizes[gid]
        manager_block_size = self._per_group_manager_block_sizes[gid]
        kernel_block_size = self._per_group_kernel_block_sizes[gid]
        if manager_block_size % kernel_block_size != 0:
            raise RuntimeError(
                "Replicated DSpark KV requires manager block size "
                f"{manager_block_size} to be divisible by kernel block size "
                f"{kernel_block_size}."
            )
        blocks_per_phys_block = manager_block_size // kernel_block_size
        max_model_len = self.vllm_config.model_config.max_model_len
        max_local_cols = (
            (max_model_len + manager_block_size * replication_size - 1)
            // (manager_block_size * replication_size)
            * blocks_per_phys_block
        )
        local_cols = min(dcp_block_table.shape[1], max_local_cols)
        replicated_cols = local_cols * replication_size
        required_shape = (dcp_block_table.shape[0], replicated_cols)
        storage = self._replicated_block_table_storage.get(gid)
        if storage is None or any(have < need for have, need in zip(storage.shape, required_shape)):
            storage = torch.empty(
                required_shape,
                dtype=torch.int32,
                device=self.device,
            )
            self._replicated_block_table_storage[gid] = storage
        col_indices = self._replicated_block_table_arange.get(gid)
        if col_indices is None or col_indices.numel() < replicated_cols:
            col_indices = torch.arange(
                replicated_cols,
                dtype=torch.int32,
                device=self.device,
            )
            self._replicated_block_table_arange[gid] = col_indices
        col_indices = col_indices[:replicated_cols]
        local_col_indices = (
            col_indices // (replication_size * blocks_per_phys_block) * blocks_per_phys_block
            + col_indices % blocks_per_phys_block
        )
        lanes = (col_indices // blocks_per_phys_block) % replication_size
        local_blocks = torch.index_select(
            dcp_block_table[:, :local_cols],
            1,
            local_col_indices.to(torch.int64),
        )
        if blocks_per_phys_block == 1:
            replicated_blocks = local_blocks * replication_size + lanes
        else:
            local_sub_blocks = local_blocks % blocks_per_phys_block
            local_phys_blocks = local_blocks // blocks_per_phys_block
            replicated_blocks = (
                local_phys_blocks * replication_size + lanes
            ) * blocks_per_phys_block + local_sub_blocks
        valid_rows = (seq_lens[: dcp_block_table.shape[0]] > 0).view(-1, 1)
        result = storage[: required_shape[0], : required_shape[1]]
        result.copy_(torch.where(valid_rows, replicated_blocks, 0))
        return result

    def _build_replicated_context_slot_mapping(
        self,
        gid: int,
        block_table: torch.Tensor,
        positions: torch.Tensor,
        query_start_loc: torch.Tensor,
        num_reqs: int,
        num_tokens: int,
    ) -> torch.Tensor:
        result = self._per_group_context_slot_mapping_buffers[gid]
        result.fill_(-1)
        if num_tokens == 0:
            return result
        query_lens = query_start_loc[1 : num_reqs + 1] - query_start_loc[:num_reqs]
        req_indices = torch.repeat_interleave(
            torch.arange(num_reqs, dtype=torch.int32, device=self.device),
            query_lens.to(device=self.device),
            output_size=num_tokens,
        )
        kernel_block_size = self._per_group_kernel_block_sizes[gid]
        token_positions = positions[:num_tokens].to(torch.int32)
        logical_block_indices = token_positions // kernel_block_size
        flat_indices = (req_indices * block_table.shape[1] + logical_block_indices).to(torch.int64)
        block_numbers = block_table.flatten()[flat_indices]
        result[:num_tokens] = block_numbers * kernel_block_size + token_positions % kernel_block_size
        return result

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
        self._per_group_manager_block_sizes = {}
        self._per_group_replication_sizes = {}
        self.draft_attn_groups: list[AttentionGroup] = []
        draft_vllm_config = (
            self._create_draft_vllm_config() if getattr(self, "replicated_draft_kv", False) else self.vllm_config
        )

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
                    if getattr(self, "replicated_draft_kv", False):
                        builder_spec = layer_kv_cache_spec.copy_with_new_block_size(
                            kernel_block_size
                        )
                        attn_group.metadata_builders = [
                            AscendAttentionMetadataBuilder(
                                builder_spec,
                                attn_group.layer_names,
                                draft_vllm_config,
                                self.device,
                            )
                        ]
                    else:
                        attn_group.create_metadata_builders(
                            draft_vllm_config,
                            self.device,
                            kernel_block_size=kernel_block_size,
                        )
                    self._per_group_kernel_block_sizes[kv_cache_gid] = kernel_block_size
                    self._per_group_manager_block_sizes[kv_cache_gid] = layer_kv_cache_spec.block_size
                    if isinstance(
                        layer_kv_cache_spec,
                        AscendDCPReplicatedDraftAttentionSpec,
                    ):
                        self._per_group_replication_sizes[kv_cache_gid] = layer_kv_cache_spec.dcp_replication_size
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
        self._per_group_block_table_buffers = {}
        for attn_group in self.draft_attn_groups:
            gid = attn_group.kv_cache_group_id
            block_table = self._per_group_block_tables[gid]
            if gid in self._per_group_replication_sizes:
                block_table = self._build_replicated_block_table(
                    gid,
                    block_table,
                    cad.seq_lens,
                )
            self._per_group_block_table_buffers[gid] = block_table
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
            if gid in self._per_group_replication_sizes:
                self._build_replicated_context_slot_mapping(
                    gid,
                    gid_block_table,
                    target_positions,
                    cad.query_start_loc,
                    batch_size,
                    self._dflash_num_context,
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
        if getattr(self, "replicated_draft_kv", False):
            cad.context_parallel_metadata = None
            cad.dcp_local_seq_lens = None

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
