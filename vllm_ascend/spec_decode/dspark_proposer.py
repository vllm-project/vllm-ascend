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
from vllm_ascend.attention.dsa_v1 import AscendDSABackend
from vllm_ascend.attention.mla_v1 import AscendMLABackend, AscendMLAMetadataBuilder
from vllm_ascend.attention.utils import AscendCommonAttentionMetadata
from vllm_ascend.ops.triton.spec_decode.utils import copy_and_expand_dflash_and_dspark_inputs_kernel_single_grid
from vllm_ascend.spec_decode.dflash_proposer import AscendDflashProposer
from vllm_ascend.transformers_utils.configs.kimi_k3 import (
    K3_DSPARK_USE_MLA_ROPE,
    K3DSparkConfig,
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
        assert vllm_config.speculative_config is not None
        additional_config = getattr(vllm_config, "additional_config", None) or {}
        if additional_config.get("enable_reduce_sample", False):
            raise ValueError(
                "DSpark on the v1 model runner does not support "
                "enable_reduce_sample: the reduced sampling path bypasses the "
                "DSpark Markov-head correction. Set "
                "additional_config.enable_reduce_sample=false."
            )
        if vllm_config.speculative_config.draft_sample_method == "probabilistic":
            raise ValueError(
                "DSpark probabilistic draft sampling is not supported on the v1 "
                "model runner; use greedy (the default) instead."
            )
        super().__init__(vllm_config, device, runner=runner)
        self.sample_from_anchor = getattr(self.draft_model_config.hf_config, "sample_from_anchor", True)
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
        # The inherited DFlash capacity uses the target verification width
        # (1 + N). Anchor-first DSpark has N real query tokens per request, and
        # uses the extra capacity for graph padding.
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
        # Per-gid logical block size used by the expanded block table and
        # attention kernel. This may be smaller than kv_cache_spec.block_size,
        # which remains the KV manager's physical page size.
        self._per_group_kernel_block_sizes: dict[int, int] = {}

        # per-gid block_table (use in proposer)
        self._per_group_block_table_buffers: dict[int, torch.Tensor] = {}
        # per-gid query slot_mapping buffer
        self._per_group_query_slot_mapping_buffers: dict[int, torch.Tensor] = {}
        # per-gid context slot_mapping buffer
        self._per_group_context_slot_mapping_buffers: dict[int, torch.Tensor] = {}

        # per-layer context slot mappings as a flat list
        self._context_slot_mapping_buffers: list[torch.Tensor | None] | None = None

        # Populated by initialize_attn_backend. MLA builds the draft KV length
        # through parallel_draft_seq_lens_cpu, while DSA consumes the canonical
        # host mirror for max_seqlen_kv; their host-length contracts differ.
        self._draft_uses_mla_backend = False
        self._draft_uses_dsa_backend = False

    @staticmethod
    def _resolve_kernel_block_size(
        gid: int,
        kv_cache_spec,
        kernel_block_sizes: list[int] | None,
    ) -> int:
        """Return the logical block size used by the expanded block table."""
        if kernel_block_sizes is not None and gid < len(kernel_block_sizes):
            return int(kernel_block_sizes[gid])
        return int(kv_cache_spec.block_size)

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

        attention_groups_list: list[dict[tuple[str, str], AttentionGroup]] = []
        # the draft layers have multiple kv_cache_groups
        if not hasattr(self.model, "get_draft_kv_cache_layer_names"):
            raise RuntimeError(
                "DSpark standard-cache path requires the draft model to expose get_draft_kv_cache_layer_names"
            )

        self._draft_attn_layer_names = set(self.model.get_draft_kv_cache_layer_names())
        self.attn_layer_names = list(sorted(self._draft_attn_layer_names))
        self._per_group_kernel_block_sizes = {}

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
                    kernel_block_size = self._resolve_kernel_block_size(
                        kv_cache_gid,
                        layer_kv_cache_spec,
                        kernel_block_sizes,
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

        self._draft_uses_mla_backend = any(
            isinstance(attn_group.backend, type)
            and issubclass(attn_group.backend, AscendMLABackend)
            for attn_group in self.draft_attn_groups
        )
        self._draft_uses_dsa_backend = any(
            isinstance(attn_group.backend, type)
            and issubclass(attn_group.backend, AscendDSABackend)
            for attn_group in self.draft_attn_groups
        )

        self.kv_cache_gid = self.draft_attn_groups[0].kv_cache_group_id
        self.kernel_block_size = self._per_group_kernel_block_sizes[self.kv_cache_gid]

        # Kimi-K3 MLA dspark: the MLA metadata builder derives use_mla_rope
        # from the TARGET's hf_text_config (K3 target is NoPE), so the draft
        # groups' builders would emit identity cos/sin while the draft's
        # context KV is written with real YaRN rotations -- silently breaking
        # the draft's positional alignment. The builders created above serve
        # draft layers only, so flip them to the draft's own RoPE setting.
        draft_hf_config = self.draft_model_config.hf_config
        if isinstance(draft_hf_config, K3DSparkConfig):
            for attn_group in self.draft_attn_groups:
                for builder in attn_group.metadata_builders:
                    if not isinstance(builder, AscendMLAMetadataBuilder):
                        raise TypeError(
                            f"K3 DSpark requires Ascend MLA metadata builders, got {type(builder).__name__}."
                        )
                    builder.use_mla_rope = K3_DSPARK_USE_MLA_ROPE

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
        num_rows = block_table.shape[0]
        self._per_group_block_tables[gid] = block_table
        self._per_group_slot_mappings[gid] = slot_mapping
        buffer = self._per_group_block_table_buffers.get(gid)
        # Anchor-first graph padding can add one virtual request whose tokens
        # are discarded. Keep one zero block-table row for that metadata tail.
        buffer_shape = (self.max_batch_size + 1, *block_table.shape[1:])
        if buffer is None or buffer.shape != buffer_shape:
            buffer = block_table.new_zeros(buffer_shape)
            self._per_group_block_table_buffers[gid] = buffer
        buffer[:num_rows].copy_(block_table)
        buffer[num_rows:].zero_()

    def _prepare_parallel_draft_seq_lens_cpu(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        batch_size: int,
        num_draft_tokens_cpu: list[int] | None,
    ) -> None:
        # Only MLA consumes parallel_draft_seq_lens_cpu. DSA gets its max-KV
        # upper bound from the canonical host mirror, while ordinary attention
        # consumes the exact device seq_lens for parallel drafting.
        if not self._draft_uses_mla_backend:
            return

        super()._prepare_parallel_draft_seq_lens_cpu(
            common_attn_metadata,
            batch_size,
            num_draft_tokens_cpu,
        )

        # In padded synchronous scheduling, the reject count is available only
        # in the device-side seq_lens that set_inputs_first_pass has already
        # corrected to L_t + N. There is no pending-reject event, and the MLA
        # builder otherwise falls back to the stale host mirror L_t + rejected_t.
        # Publish an explicit CPU copy for MLA. The blocking D2H is confined to
        # synchronous scheduling; async+padded keeps its overlapped reject-copy
        # path and non-padded input preparation already has an exact host value.
        reject_event = getattr(self.runner, "num_rejected_tokens_event", None)
        sync_padded = num_draft_tokens_cpu is not None and reject_event is None
        if sync_padded:
            common_attn_metadata.parallel_draft_seq_lens_cpu = (
                common_attn_metadata.seq_lens[: common_attn_metadata.num_reqs]
                .detach()
                .to("cpu", copy=True)
            )

    def _sample_dspark_tokens(
        self,
        raw_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the Markov correction and sample one DSpark block."""
        sampling_metadata = self.runner.input_batch.sampling_metadata
        logits = raw_logits.view(
            -1,
            self.num_speculative_tokens,
            raw_logits.shape[-1],
        )
        num_reqs = logits.shape[0]
        draft_token_ids = self._dspark_draft_buffer[:num_reqs]
        draft_token_ids[:, 0].copy_(self._dspark_seed_buffer[:num_reqs])

        self._last_draft_probs = None
        use_probabilistic = (
            self._dspark_draft_probs is not None
            and sampling_metadata is not None
            and not sampling_metadata.all_greedy
        )
        draft_probs = None
        if use_probabilistic:
            assert self._dspark_draft_probs is not None
            if logits.shape[-1] != self._dspark_draft_probs.shape[-1]:
                raise RuntimeError(
                    "DSpark draft/target vocabulary mismatch for probabilistic "
                    f"sampling: draft={logits.shape[-1]}, "
                    f"target={self._dspark_draft_probs.shape[-1]}"
                )
            draft_probs = self._dspark_draft_probs[:num_reqs]

        for idx in range(self.num_speculative_tokens):
            markov_emb = self.model.markov_embed(draft_token_ids[:, idx])
            logits[:, idx].add_(self.model.markov_bias(markov_emb))
            if draft_probs is None:
                next_token_ids = logits[:, idx].argmax(dim=-1)
            else:
                next_token_ids, probs = self._sample_from_logits(
                    logits[:, idx],
                    sampling_metadata,
                )
                assert probs is not None
                draft_probs[:, idx].copy_(probs)
            draft_token_ids[:, idx + 1].copy_(next_token_ids)

        self._last_draft_probs = draft_probs
        return draft_token_ids

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
        draft_attn_groups = getattr(self, "draft_attn_groups", [])
        for attn_group in draft_attn_groups:
            gid = attn_group.kv_cache_group_id
            gid_block_table = self._per_group_block_table_buffers.get(gid)
            if gid_block_table is None:
                continue
            kernel_block_size = self._per_group_kernel_block_sizes[gid]
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
                block_size=kernel_block_size,
                num_query_per_req=self.num_query_per_req,
                num_speculative_tokens=self.num_speculative_tokens,
                total_input_tokens=self._dflash_num_context,
                batch_size=batch_size,
                HAS_NUM_REJECTED=has_num_rejected,
                SAMPLE_FROM_ANCHOR=self.sample_from_anchor,
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
        # Backend-specific host-length contract:
        # - MLA must keep the canonical host mirror unchanged. Its parallel
        #   draft path adds N exactly once and later finalizes pending rejects.
        # - DSA ignores parallel_draft_seq_lens_cpu and derives max_seqlen_kv
        #   from the canonical host mirror, so include N there as a safe upper
        #   bound for its device seqused_kv=L_t+N.
        # - Ordinary attention consumes device seq_lens while parallel drafting.
        if self._draft_uses_dsa_backend:
            host_seq_lens = cad._seq_lens_cpu
            if host_seq_lens is None:
                host_seq_lens = getattr(cad, "seq_lens_cpu", None)
            if host_seq_lens is not None:
                draft_seq_lens_cpu = host_seq_lens.clone()
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

    def pad_query_start_loc_for_graph(
        self,
        query_start_loc,
        num_input_tokens: int,
        real_num_reqs: int,
        graph_num_reqs: int,
        num_query_tokens: int | None = None,
    ) -> int:
        """Match DSpark query metadata to the captured attention Query T.

        ``num_input_tokens`` is the target verification graph width.  For the
        K3 MLA draft, the captured FIA Query T is narrower and is supplied as
        ``num_query_tokens``.  Other DSpark backends keep using the target
        width because their captured attention query includes graph padding.
        """
        if num_query_tokens is None:
            num_query_tokens = num_input_tokens

        last_loc = int(query_start_loc.np[real_num_reqs])
        num_metadata_reqs = real_num_reqs

        for req_idx in range(real_num_reqs, graph_num_reqs):
            last_loc = min(last_loc + self.num_query_per_req, num_query_tokens)
            query_start_loc.np[req_idx + 1] = last_loc
            num_metadata_reqs += 1

        if last_loc < num_query_tokens:
            query_start_loc.np[num_metadata_reqs + 1] = num_query_tokens
            num_metadata_reqs += 1

        query_start_loc.copy_to_gpu()
        return num_metadata_reqs

    def get_graph_query_num_tokens(
        self,
        num_input_tokens: int,
        graph_num_reqs: int,
    ) -> int:
        """Return the Query T used to capture this DSpark attention graph."""
        if not isinstance(self.draft_model_config.hf_config, K3DSparkConfig):
            return num_input_tokens

        return graph_num_reqs * self.num_query_per_req

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
        num_query_tokens = num_query_total if num_reqs > 0 else num_tokens
        if aclgraph_runtime_mode == CUDAGraphMode.FULL and batch_descriptor is not None:
            num_query_tokens = batch_descriptor.num_tokens
        num_query_tokens = min(num_query_tokens, self.max_query_tokens)

        (
            num_input_tokens,
            num_tokens_across_dp,
            _,
        ) = self.runner._sync_metadata_across_dp(num_query_tokens, is_draft_model=True)

        graph_query_tokens = self.get_graph_query_num_tokens(
            num_input_tokens,
            num_reqs,
        )

        if not self.use_cuda_graph:
            aclgraph_runtime_mode = CUDAGraphMode.NONE

        context_positions = self._context_positions_buffer[:num_input_tokens]
        context_states = self.hidden_states[:num_input_tokens]

        self.token_indices_to_sample.fill_(0)
        self._pad_draft_buffers(num_query_total, num_input_tokens)

        multi_steps_attn_metadata = []
        if aclgraph_runtime_mode == CUDAGraphMode.FULL and self.draft_attn_groups:
            query_start_loc_cpu = (
                torch.from_numpy(self.token_arange_np[: num_reqs + 1]).clone() * self.num_query_per_req
            )
            query_start_loc_cpu.clamp_(max=graph_query_tokens)
            if int(query_start_loc_cpu[-1]) < graph_query_tokens:
                query_start_loc_cpu = torch.cat(
                    (query_start_loc_cpu, query_start_loc_cpu.new_tensor([graph_query_tokens]))
                )
            num_metadata_reqs = query_start_loc_cpu.shape[0] - 1
            query_start_loc = query_start_loc_cpu.to(self.device)
            seq_lens_cpu = self._adjust_tensor(self.runner.optimistic_seq_lens_cpu, num_metadata_reqs)
            seq_lens = self._adjust_tensor(self.runner.seq_lens, num_metadata_reqs)
            causal = self.model.get_draft_attn_causal()[0]
            per_layer_attn_metadata: dict[str, Any] = {}
            for attn_group in self.draft_attn_groups:
                gid = attn_group.kv_cache_group_id
                common_attn_metadata = AscendCommonAttentionMetadata(
                    query_start_loc=query_start_loc,
                    query_start_loc_cpu=query_start_loc_cpu,
                    seq_lens_cpu=seq_lens_cpu,
                    _seq_lens_cpu=seq_lens_cpu,
                    seq_lens_cpu_upper_bound=seq_lens_cpu,
                    seq_lens=seq_lens,
                    num_reqs=num_metadata_reqs,
                    # Keep cache writes limited to real draft queries. The
                    # graph still executes ``num_input_tokens`` rows; padding
                    # rows have -1 slot mappings and are discarded.
                    num_actual_tokens=num_query_total,
                    num_input_tokens=num_input_tokens,
                    max_query_len=self.num_query_per_req,
                    max_seq_len=0,
                    slot_mapping=self._per_group_query_slot_mapping_buffers[gid][:num_input_tokens],
                    positions=self.positions,
                    attn_state=AscendAttentionState.SpecDecoding,
                    causal=causal,
                    is_prefilling=torch.zeros(num_metadata_reqs, dtype=torch.bool),
                    block_table_tensor=self._per_group_block_table_buffers[gid][:num_metadata_reqs],
                )
                metadata = attn_group.get_metadata_builder().build_for_graph_capture(
                    common_attn_metadata,
                    AscendAttentionState.SpecDecoding,
                )
                if hasattr(metadata, "attn_mask") and not causal:
                    metadata.attn_mask = None
                metadata.attn_state = AscendAttentionState.SpecDecoding
                for layer_name in attn_group.layer_names:
                    per_layer_attn_metadata[layer_name] = metadata
            multi_steps_attn_metadata.append(per_layer_attn_metadata)

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
                    multi_steps_attn_metadata=multi_steps_attn_metadata,
                    num_tokens=num_input_tokens,
                )

            forward_context = get_forward_context()
            if forward_context.cudagraph_runtime_mode == CUDAGraphMode.FULL and not _EXTRA_CTX.capturing:
                self._update_full_graph_params(forward_context, graph_query_tokens, multi_steps_attn_metadata)
