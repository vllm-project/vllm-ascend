"""Gemma4 MTP proposer for Ascend NPUs.

Reuses upstream ``Gemma4Proposer`` for model-generic behavior. Following the
dspark/step3p5 convention, the draft-loop methods that differ for Gemma4 are
forked here instead of extending the shared ``AscendSpecDecodeBaseProposer``:
Gemma4 drafts from a constant target position (no per-step position / seq-len
advancement), and attention metadata is built per KV cache group (sliding /
full attention) with per-group block tables. Branches that cannot apply to
Gemma4 MTP are omitted: other drafter methods, compressor metadata, multimodal
draft inputs, DCP, M-RoPE, GDN/sliding-window, sparse KV offload, LM-head TP.
"""

import copy
from functools import partial

import torch
from vllm.config import CUDAGraphMode, get_layers_from_vllm_config
from vllm.forward_context import BatchDescriptor, get_forward_context
from vllm.logger import logger
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.spec_decode.gemma4 import Gemma4Proposer
from vllm.v1.spec_decode.utils import PADDING_SLOT_ID

from vllm_ascend.ascend_forward_context import _EXTRA_CTX, set_ascend_forward_context
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.utils import AscendCommonAttentionMetadata
from vllm_ascend.spec_decode.llm_base_proposer import AscendSpecDecodeBaseProposer, _split_draft_outputs


class AscendGemma4Proposer(Gemma4Proposer, AscendSpecDecodeBaseProposer):
    """Reuse vLLM's Gemma4 proposer with Ascend execution support."""

    def _setup_centroids_cuda_graphs(self) -> None:
        self._centroids_sizes: list[int] = []

    def _maybe_share_lm_head(self, target_language_model) -> None:
        AscendSpecDecodeBaseProposer._maybe_share_lm_head(self, target_language_model)

    def load_model(self, target_model) -> None:
        super().load_model(target_model)
        self.supports_mm_inputs = False
        self._sync_kv_sharing_target_to_impl()

    def _sync_kv_sharing_target_to_impl(self) -> None:
        """Sync upstream late-bound kv_sharing_target_layer_name from the
        Attention wrapper to the Ascend backend impl, which was constructed
        with None; without this the draft reads its own cache."""
        attn_layers = get_layers_from_vllm_config(self.vllm_config, AttentionLayerBase)  # type: ignore[type-abstract]
        synced = 0
        for layer_name in self._draft_attn_layer_names:
            attn = attn_layers[layer_name]
            target = getattr(attn, "kv_sharing_target_layer_name", None)
            impl = getattr(attn, "impl", None)
            if target is not None and impl is not None:
                impl.kv_sharing_target_layer_name = target
                synced += 1
        logger.info("Gemma4 MTP: KV-sharing synced to %d/%d layers.", synced, len(self._draft_attn_layer_names))

    def _get_group_common_attn_metadata(self, attn_group, common_attn_metadata):
        block_table = self._per_group_block_tables.get(attn_group.kv_cache_group_id)
        if block_table is None:
            return common_attn_metadata
        group_metadata = copy.copy(common_attn_metadata)
        group_metadata.block_table_tensor = block_table[: common_attn_metadata.num_reqs]
        return group_metadata

    def _per_group_layer_metadata(self, common_attn_metadata, build):
        per_layer_attn_metadata = {}
        for attn_group in self.draft_attn_groups:
            group_metadata = self._get_group_common_attn_metadata(attn_group, common_attn_metadata)
            attn_metadata = build(attn_group, group_metadata)
            for layer_name in attn_group.layer_names:
                per_layer_attn_metadata[layer_name] = attn_metadata
        return per_layer_attn_metadata

    def build_draft_attn_metadata(self, common_attn_metadata, num_input_tokens, num_actual_tokens):
        def build(attn_group, group_metadata):
            attn_metadata = attn_group.get_metadata_builder().build(0, group_metadata, self.runner.get_model())
            attn_metadata.attn_state = AscendAttentionState.SpecDecoding
            if hasattr(attn_metadata, "causal") and not attn_metadata.causal:
                attn_metadata.attn_mask = None
            return attn_metadata

        per_layer_attn_metadata = self._per_group_layer_metadata(common_attn_metadata, build)
        attn_metadata = per_layer_attn_metadata[self.draft_attn_groups[0].layer_names[0]]
        return [per_layer_attn_metadata], attn_metadata

    def _bind_step_buffers(self, common_attn_metadata, draft_index, slot_pad):
        num_reqs = common_attn_metadata.seq_lens.shape[0]
        slot_len = common_attn_metadata.slot_mapping.shape[0]
        slot = self.slot_mapping_group[draft_index]
        seq = self.seq_lens_group[draft_index]
        qsl = self.query_start_loc_group[draft_index]
        slot[:slot_len].copy_(common_attn_metadata.slot_mapping)
        slot[slot_len:].fill_(slot_pad)
        common_attn_metadata.slot_mapping = slot
        seq[:num_reqs].copy_(common_attn_metadata.seq_lens)
        seq[num_reqs:].fill_(0)
        common_attn_metadata.seq_lens = seq[:num_reqs]
        qsl_len = common_attn_metadata.query_start_loc.shape[0]
        qsl[:qsl_len].copy_(common_attn_metadata.query_start_loc)
        qsl[qsl_len:].fill_(0)
        common_attn_metadata.query_start_loc = qsl[:qsl_len]

    def _run_draft_forward(self, model_kwargs):
        ret_hidden_states = self.model(**model_kwargs)
        last_hidden_states, hidden_states = _split_draft_outputs(ret_hidden_states)
        return self.maybe_all_gather_and_unpad(last_hidden_states, model_kwargs["positions"], hidden_states)

    def _sync_dp_num_tokens(self, num_tokens):
        _, num_tokens_across_dp, _ = self.runner._sync_metadata_across_dp(num_tokens, is_draft_model=True)
        if num_tokens_across_dp is not None:
            num_tokens = int(num_tokens_across_dp[self.dp_rank].item())
        return num_tokens, num_tokens_across_dp

    def _propose(
        self,
        num_speculative_tokens: int,
        target_token_ids: torch.Tensor,
        target_positions: torch.Tensor,
        target_hidden_states: torch.Tensor,
        next_token_ids: torch.Tensor,
        token_indices_to_sample: torch.Tensor | None,
        common_attn_metadata: AscendCommonAttentionMetadata,
        target_model_batch_desc: BatchDescriptor,
        sampling_metadata: SamplingMetadata,
        mm_embed_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
        req_scheduled_tokens=None,
        long_seq_metadata=None,
        num_prefill_reqs=0,
        num_decode_reqs=0,
        scheduler_output=None,
        num_scheduled_tokens: int = 0,
        num_rejected_tokens_gpu: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size = common_attn_metadata.batch_size()
        self.num_speculative_tokens = num_speculative_tokens
        if self.num_speculative_tokens == 0:
            return torch.empty(batch_size, 0, device=target_token_ids.device, dtype=torch.int64)
        if token_indices_to_sample is None:
            token_indices_to_sample = common_attn_metadata.query_start_loc[1:] - 1
        num_tokens, token_indices_to_sample, common_attn_metadata, _ = self.set_inputs_first_pass(
            target_token_ids=target_token_ids,
            next_token_ids=next_token_ids,
            target_positions=target_positions,
            target_hidden_states=target_hidden_states,
            token_indices_to_sample=token_indices_to_sample,
            cad=common_attn_metadata,
            num_rejected_tokens_gpu=num_rejected_tokens_gpu,
            req_scheduled_tokens=req_scheduled_tokens,
            long_seq_metadata=long_seq_metadata,
            num_prefill_reqs=num_prefill_reqs,
            num_decode_reqs=num_decode_reqs,
        )
        assert self.runner is not None
        dispatch = self.runner.cudagraph_dispatcher.dispatch
        has_lora = len(self.runner.input_batch.lora_id_to_lora_request) > 0
        uniform_decode = target_model_batch_desc.uniform
        if self.use_cuda_graph:
            _, batch_descriptor = dispatch(num_tokens=num_tokens, uniform_decode=uniform_decode, has_lora=has_lora)
            num_input_tokens = batch_descriptor.num_tokens
        else:
            num_input_tokens = num_tokens
        num_input_tokens, num_tokens_across_dp = self._sync_dp_num_tokens(num_input_tokens)
        if self.use_cuda_graph:
            aclgraph_runtime_mode, batch_descriptor = dispatch(
                num_tokens=num_input_tokens, uniform_decode=uniform_decode, has_lora=has_lora
            )
            num_input_tokens = batch_descriptor.num_tokens
        else:
            aclgraph_runtime_mode = CUDAGraphMode.NONE
            batch_descriptor = None
        if aclgraph_runtime_mode == CUDAGraphMode.FULL:
            num_reqs = common_attn_metadata.query_start_loc.shape[0]
            self.query_start_loc.gpu[:num_reqs].copy_(common_attn_metadata.query_start_loc)
            self.query_start_loc.cpu[:num_reqs].copy_(common_attn_metadata.query_start_loc_cpu)
            num_reqs_padded = self.runner._pad_query_start_loc_for_fia(
                self.query_start_loc,
                num_input_tokens,
                batch_descriptor.num_reqs if batch_descriptor.num_reqs is not None else common_attn_metadata.num_reqs,
                common_attn_metadata.num_reqs,
                aclgraph_runtime_mode,
                batch_descriptor.num_reqs,
            )
            common_attn_metadata.num_reqs = num_reqs_padded
            common_attn_metadata.query_start_loc = self.query_start_loc.gpu[: num_reqs_padded + 1]
            common_attn_metadata.query_start_loc_cpu = self.query_start_loc.cpu[: num_reqs_padded + 1]
            common_attn_metadata.block_table_tensor = self._adjust_tensor(
                common_attn_metadata.block_table_tensor, num_reqs_padded
            )
            common_attn_metadata.seq_lens = self._adjust_tensor(self.runner.seq_lens, num_reqs_padded)
            common_attn_metadata.seq_lens_cpu = self._adjust_tensor(
                self.runner.optimistic_seq_lens_cpu, num_reqs_padded
            )
            if common_attn_metadata._seq_lens_cpu is not None:
                common_attn_metadata._seq_lens_cpu = common_attn_metadata.seq_lens_cpu.clone()
            if common_attn_metadata.num_computed_tokens_cpu is not None:
                common_attn_metadata.num_computed_tokens_cpu = self._adjust_tensor(
                    common_attn_metadata.num_computed_tokens_cpu, num_reqs_padded
                )
        else:
            num_reqs_padded = common_attn_metadata.num_reqs
            if not self.vllm_config.model_config.use_mla:
                common_attn_metadata.block_table_tensor = self._adjust_tensor(
                    common_attn_metadata.block_table_tensor, num_reqs_padded
                )
        self._bind_step_buffers(common_attn_metadata, 0, -1)
        common_attn_metadata.num_input_tokens = num_input_tokens
        self._pad_draft_buffers(num_tokens, num_input_tokens)
        multi_steps_attn_metadata, attn_metadata_i = self.build_draft_attn_metadata(
            common_attn_metadata, num_input_tokens, num_tokens
        )
        used_update_positions = self.positions[token_indices_to_sample]
        common_attn_metadata.block_table_tensor = common_attn_metadata.block_table_tensor.clone()
        is_prefill_batch = num_prefill_reqs > 0 or bool(getattr(attn_metadata_i, "num_prefills", 0))
        if not self.parallel_drafting:
            for draft_index in range(1, self.num_speculative_tokens):
                per_layer_attn_metadata = dict()
                for attn_group in self.draft_attn_groups:
                    common_attn_metadata, attn_metadata = self.attn_update_stack_num_spec_norm(
                        draft_index,
                        common_attn_metadata,
                        batch_size,
                        num_input_tokens,
                        used_update_positions,
                        aclgraph_runtime_mode,
                        attn_group=attn_group,
                    )
                    per_layer_attn_metadata.update({n: attn_metadata for n in attn_group.layer_names})
                multi_steps_attn_metadata.append(per_layer_attn_metadata)
        token_indices_to_sample_len = token_indices_to_sample.shape[0]
        self.token_indices_to_sample[:token_indices_to_sample_len].copy_(token_indices_to_sample)
        self.token_indices_to_sample[token_indices_to_sample_len:].fill_(0)
        with set_ascend_forward_context(
            multi_steps_attn_metadata[0],
            self.vllm_config,
            num_tokens=num_input_tokens,
            num_tokens_across_dp=num_tokens_across_dp,
            num_actual_tokens=num_tokens,
            batch_descriptor=batch_descriptor,
            aclgraph_runtime_mode=aclgraph_runtime_mode,
            is_draft_model=True,
            draft_attn_metadatas=multi_steps_attn_metadata,
            eplb_heat_collection_status=(
                self.runner.eplb_heat_collection_status if self.runner.dynamic_eplb else False
            ),
        ):
            forward_context = get_forward_context()
            if forward_context is not None:
                forward_context.moe_layer_index = 0
            model_inputs = {
                "num_input_tokens": num_input_tokens,
                "batch_size": batch_size,
                "token_indices_to_sample": self.token_indices_to_sample[:token_indices_to_sample_len],
                "target_positions": target_positions,
                "inputs_embeds": None,
                "multi_steps_attn_metadata": multi_steps_attn_metadata,
                "num_tokens": num_tokens,
                "is_prefill": is_prefill_batch,
                "sampling_metadata": sampling_metadata,
            }
            run_draft = partial(self._runnable, **model_inputs)
            if self.enable_enpu:
                self._update_full_graph_params_if_needed(forward_context, num_input_tokens, multi_steps_attn_metadata)
                draft_token_ids = run_draft()
            else:
                draft_token_ids = run_draft()
                self._update_full_graph_params_if_needed(forward_context, num_input_tokens, multi_steps_attn_metadata)
        return draft_token_ids

    @torch.inference_mode()
    def dummy_run(
        self,
        num_tokens: int,
        with_prefill: bool = False,
        in_graph_capturing: bool = False,
        num_reqs: int = 0,
        num_tokens_across_dp: torch.Tensor | None = None,
        aclgraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
        batch_descriptor=None,
        dummy_compute_logits=lambda hidden_states: None,
        is_profile=False,
    ):
        num_tokens, num_tokens_across_dp = self._sync_dp_num_tokens(num_tokens)
        multi_steps_attn_metadata = []
        if not self.use_cuda_graph:
            aclgraph_runtime_mode = CUDAGraphMode.NONE
        with self.runner.synchronize_input_prep():
            if aclgraph_runtime_mode == CUDAGraphMode.FULL and len(self.runner.attn_groups) > 0:
                num_computed_tokens_cpu = self.runner.input_batch.num_computed_tokens_cpu_tensor[:num_reqs]
                self.query_start_loc.cpu[: num_reqs + 1].copy_(self.runner.query_start_loc.cpu[: num_reqs + 1])
                self.query_start_loc.copy_to_gpu()
                common_attn_metadata = AscendCommonAttentionMetadata(
                    query_start_loc=self.query_start_loc.gpu[: num_reqs + 1],
                    query_start_loc_cpu=self.query_start_loc.cpu[: num_reqs + 1],
                    seq_lens_cpu=self.runner.optimistic_seq_lens_cpu,
                    _seq_lens_cpu=self.runner.optimistic_seq_lens_cpu,
                    seq_lens_cpu_upper_bound=self.runner.optimistic_seq_lens_cpu,
                    seq_lens=self.runner.seq_lens[:num_reqs],
                    num_reqs=num_reqs,
                    num_actual_tokens=num_tokens,
                    num_input_tokens=num_tokens,
                    max_query_len=self.num_speculative_tokens + 1,
                    num_computed_tokens_cpu=num_computed_tokens_cpu,
                    block_table_tensor=self.runner.input_batch.block_table[self.kv_cache_gid].get_device_tensor()[
                        :num_reqs
                    ],
                    slot_mapping=self.runner.input_batch.block_table[self.kv_cache_gid].slot_mapping.gpu,
                    positions=self.runner.positions,
                    attn_state=self.runner.attn_state,
                    decode_token_per_req=self.runner.decode_token_per_req,
                    is_prefilling=torch.zeros(num_reqs, dtype=torch.bool),
                    max_seq_len=0,
                )
                assert len(self.draft_attn_groups) > 0
                for draft_index in range(self.num_speculative_tokens):
                    common_attn_metadata = self.shallow_copy_metadata(common_attn_metadata)
                    self._bind_step_buffers(common_attn_metadata, draft_index, PADDING_SLOT_ID)

                    def build(attn_group, group_metadata):
                        return attn_group.get_metadata_builder().build_for_graph_capture(
                            group_metadata, AscendAttentionState.SpecDecoding
                        )

                    multi_steps_attn_metadata.append(self._per_group_layer_metadata(common_attn_metadata, build))
        model_positions = self._get_positions(num_tokens)
        batch_size = max(num_tokens // (self.num_speculative_tokens + 1), 1)
        if is_profile:
            batch_size = min(batch_size, self.runner.max_num_reqs)
        self.token_indices_to_sample.fill_(0)
        with set_ascend_forward_context(
            multi_steps_attn_metadata[0] if multi_steps_attn_metadata else None,
            self.vllm_config,
            num_tokens=num_tokens,
            num_tokens_across_dp=num_tokens_across_dp,
            num_actual_tokens=0,
            in_profile_run=is_profile,
            batch_descriptor=batch_descriptor,
            aclgraph_runtime_mode=aclgraph_runtime_mode,
            is_draft_model=True,
            draft_attn_metadatas=multi_steps_attn_metadata,
            eplb_heat_collection_status=(
                self.runner.eplb_heat_collection_status if self.runner.dynamic_eplb else False
            ),
        ):
            forward_context = get_forward_context()
            if forward_context is not None:
                forward_context.moe_layer_index = 0
            self._runnable(
                num_input_tokens=num_tokens,
                batch_size=batch_size,
                token_indices_to_sample=self.token_indices_to_sample[: batch_size * self.extra_slots_per_request],
                target_positions=model_positions,
                inputs_embeds=None,
                multi_steps_attn_metadata=multi_steps_attn_metadata,
                num_tokens=num_tokens,
            )
            forward_context = get_forward_context()
            if forward_context.cudagraph_runtime_mode == CUDAGraphMode.FULL and not _EXTRA_CTX.capturing:
                self._update_full_graph_params(forward_context, num_tokens, multi_steps_attn_metadata)

    def attn_update_stack_num_spec_norm(
        self,
        draft_index,
        old_common_metadata,
        batch_size,
        input_batch_size,
        used_update_positions,
        aclgraph_runtime_mode,
        ori_seq_len=None,
        ori_seq_len_cpu=None,
        slot_indices=None,
        mtp_slot_mapping=None,
        attn_group=None,
    ):
        assert draft_index > 0
        assert attn_group is not None, "vllm-ascend v0.17.0rc1 requires attn_group"
        old_common_metadata = self._get_group_common_attn_metadata(attn_group, old_common_metadata)
        common_attn_metadata = self.shallow_copy_metadata(old_common_metadata)
        if draft_index == 1:
            if aclgraph_runtime_mode == CUDAGraphMode.FULL:
                common_attn_metadata.num_reqs = input_batch_size
                padded = ("block_table_tensor", "seq_lens", "seq_lens_cpu")
                for field in (*padded, "_seq_lens_cpu", "num_computed_tokens_cpu"):
                    value = getattr(common_attn_metadata, field)
                    if value is not None or field in padded:
                        setattr(common_attn_metadata, field, self._adjust_tensor(value, input_batch_size))
            n = input_batch_size if aclgraph_runtime_mode == CUDAGraphMode.FULL else batch_size
            common_attn_metadata.query_start_loc = self.arange[: n + 1]
            common_attn_metadata.query_start_loc_cpu = torch.from_numpy(self.token_arange_np[: n + 1]).clone()
            common_attn_metadata.num_actual_tokens = batch_size
            common_attn_metadata.max_query_len = 1
            common_attn_metadata.decode_token_per_req = 1
            common_attn_metadata.attn_state = AscendAttentionState.SpecDecoding
            common_attn_metadata.graph_pad_size = -1
            common_attn_metadata.num_input_tokens = input_batch_size
        common_attn_metadata.seq_lens = common_attn_metadata.seq_lens.clone()
        for field in ("seq_lens_cpu", "_seq_lens_cpu", "num_computed_tokens_cpu"):
            value = getattr(common_attn_metadata, field)
            if value is not None:
                setattr(common_attn_metadata, field, value.clone())
        common_attn_metadata.positions = common_attn_metadata.positions.clone()
        exceeds_max_model_len = used_update_positions >= self.max_model_len
        clamped_positions = torch.where(exceeds_max_model_len, 0, used_update_positions)
        exceeds_mask = common_attn_metadata.seq_lens[:batch_size] > self.max_model_len
        common_attn_metadata.seq_lens[:batch_size].masked_fill_(exceeds_mask, 1)
        for field in ("seq_lens_cpu", "_seq_lens_cpu"):
            value = getattr(common_attn_metadata, field)
            if value is not None:
                value[:batch_size].masked_fill_(value[:batch_size] > self.max_model_len, 1)
        common_attn_metadata.positions[:batch_size].copy_(clamped_positions)
        block_numbers = clamped_positions // self.block_size
        block_ids = old_common_metadata.block_table_tensor.gather(dim=1, index=block_numbers.view(-1, 1)).view(-1)
        slot_mapping = block_ids * self.block_size + clamped_positions % self.block_size
        slot_mapping.masked_fill_(exceeds_max_model_len, PADDING_SLOT_ID)
        self.slot_mapping_group[draft_index][: slot_mapping.shape[0]].copy_(slot_mapping.to(torch.int32))
        self.slot_mapping_group[draft_index][slot_mapping.shape[0] :].fill_(PADDING_SLOT_ID)
        common_attn_metadata.slot_mapping = self.slot_mapping_group[draft_index]
        self._bind_step_buffers(common_attn_metadata, draft_index, PADDING_SLOT_ID)
        attn_metadata = attn_group.get_metadata_builder().build_for_drafting(common_attn_metadata, draft_index)
        return common_attn_metadata, attn_metadata

    def _run_merged_draft(
        self,
        num_input_tokens,
        batch_size,
        token_indices_to_sample,
        target_positions,
        inputs_embeds,
        multi_steps_attn_metadata,
        num_tokens,
        is_prefill=None,
        sampling_metadata: SamplingMetadata | None = None,
    ) -> torch.Tensor:
        self._last_draft_probs = None
        if sampling_metadata is None and self.runner is not None:
            sampling_metadata = self.runner.input_batch.sampling_metadata
        model_kwargs = {
            "input_ids": self.input_ids[:num_input_tokens],
            "positions": self._get_positions(num_input_tokens),
            "inputs_embeds": inputs_embeds,
        }
        if self.pass_hidden_states_to_model:
            model_hidden_states, model_positions = self.maybe_pad_and_reduce(
                self.hidden_states[:num_input_tokens], model_kwargs["positions"]
            )
            model_kwargs["hidden_states"] = model_hidden_states
            model_kwargs["positions"] = model_positions
        last_hidden_states, _, hidden_states = self._run_draft_forward(model_kwargs)
        sample_hidden_states = last_hidden_states[token_indices_to_sample]
        draft_token_ids, draft_probs_step0 = self.compute_draft_token_ids(sample_hidden_states, sampling_metadata)
        if self.num_speculative_tokens == 1 or self.parallel_drafting:
            if draft_probs_step0 is not None:
                self._last_draft_probs = draft_probs_step0.view(
                    -1, self.num_speculative_tokens, draft_probs_step0.shape[-1]
                ).contiguous()
            return draft_token_ids.view(-1, self.num_speculative_tokens)
        draft_token_ids_tensor = torch.zeros(
            (self.num_speculative_tokens, *draft_token_ids.shape), dtype=draft_token_ids.dtype, device=self.device
        )
        draft_token_ids_tensor[0] = draft_token_ids
        draft_probs_list = [draft_probs_step0] if draft_probs_step0 is not None else None
        positions = self.positions[token_indices_to_sample]
        hidden_states = hidden_states[token_indices_to_sample]
        token_indices_to_sample = self.arange[:batch_size]
        _EXTRA_CTX.num_tokens = num_input_tokens
        _EXTRA_CTX.num_accept_tokens = batch_size
        for draft_index in range(self.num_speculative_tokens - 1):
            forward_context = get_forward_context()
            if forward_context is not None:
                forward_context.moe_layer_index = 0
            self.input_ids[:batch_size] = draft_token_ids_tensor[draft_index]
            exceeds_max_model_len = positions >= self.vllm_config.model_config.max_model_len
            clamped_positions = torch.where(exceeds_max_model_len, 0, positions)
            self._set_positions(batch_size, clamped_positions)
            self.hidden_states[:batch_size] = hidden_states.view(batch_size, -1)
            model_kwargs = {
                "input_ids": self.input_ids[:num_input_tokens],
                "positions": self._get_positions(num_input_tokens),
                "inputs_embeds": None,
            }
            if self.pass_hidden_states_to_model:
                model_hidden_states, model_positions = self.maybe_pad_and_reduce(
                    self.hidden_states[:num_input_tokens], model_kwargs["positions"]
                )
                model_kwargs["hidden_states"] = model_hidden_states
            forward_context.attn_metadata = (
                multi_steps_attn_metadata[draft_index + 1] if multi_steps_attn_metadata else None
            )
            last_hidden_states, _, hidden_states = self._run_draft_forward(model_kwargs)
            sample_hidden_states = last_hidden_states[token_indices_to_sample]
            draft_token_ids, draft_probs_step = self.compute_draft_token_ids(sample_hidden_states, sampling_metadata)
            hidden_states = hidden_states[:batch_size]
            draft_token_ids_tensor[draft_index + 1] = draft_token_ids
            if draft_probs_list is not None and draft_probs_step is None:
                draft_probs_list = None
            elif draft_probs_list is not None:
                draft_probs_list.append(draft_probs_step)
        draft_token_ids = draft_token_ids_tensor.swapaxes(0, 1)
        if draft_probs_list is not None:
            self._last_draft_probs = torch.stack(draft_probs_list, dim=1).contiguous()
        return draft_token_ids
