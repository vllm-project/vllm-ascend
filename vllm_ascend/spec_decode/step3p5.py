# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable
from functools import partial
from typing import Any

import torch
from vllm.config import CUDAGraphMode, VllmConfig, get_layers_from_vllm_config, replace
from vllm.forward_context import BatchDescriptor, get_forward_context
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.models.utils import get_draft_quant_config
from vllm.v1.attention.backends.utils import CommonAttentionMetadata
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheSpec, UniformTypeKVCacheSpecs
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.spec_decode.llm_base_proposer import compute_probs_and_sample_next_token
from vllm.v1.spec_decode.utils import PADDING_SLOT_ID
from vllm.v1.worker.utils import AttentionGroup

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.ascend_forward_context import _EXTRA_CTX, set_ascend_forward_context
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.utils import AscendCommonAttentionMetadata
from vllm_ascend.distributed.parallel_state import get_lmhead_tp_group
from vllm_ascend.spec_decode.eagle_proposer import AscendEagleProposer
from vllm_ascend.utils import lmhead_tp_enable


class AscendStep3p5MTPProposer(AscendEagleProposer):
    """Step3.5 MTP proposer with per-MTP-layer independent KV cache groups.

    Each Step3.5 MTP layer owns a different attention/KV-cache group. The
    generic Ascend MTP proposer assumes one draft KV-cache group and builds one
    attention metadata object for all draft layers. Step3.5 therefore keeps the
    whole propose flow in this subclass: it builds per-group metadata for all MTP
    layers, reuses that step0 metadata across layer calls, and refreshes the
    metadata tensors after each in-graph input update.
    """

    _runnable: Callable

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        runner=None,
    ) -> None:
        super().__init__(vllm_config, device, runner=runner)
        # Per KV cache group block tables / slot mappings captured from the
        # model runner each step (see set_per_group_attn_metadata).
        self._per_group_block_tables: dict[int, torch.Tensor] = {}
        self._per_group_slot_mappings: dict[int, torch.Tensor] = {}
        # Slot-mapping buffers for additional KV cache groups. FULL graph
        # capture records tensor addresses for reshape/cache, so every group
        # that can appear in the shared step0 metadata needs a persistent
        # buffer.
        self._group_slot_buffers: dict[int, torch.Tensor] = {}

    def set_per_group_attn_metadata(
        self,
        gid: int,
        block_table: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        self._per_group_block_tables[gid] = block_table
        self._per_group_slot_mappings[gid] = slot_mapping

    def _seed_graph_capture_per_group_metadata(
        self,
        num_reqs: int,
        num_tokens: int,
    ) -> None:
        if self.runner is None or not self.draft_attn_groups:
            return

        block_tables = self.runner.input_batch.block_table
        for attn_group in self.draft_attn_groups:
            gid = attn_group.kv_cache_group_id
            try:
                block_table = block_tables[gid]
            except (IndexError, KeyError):
                continue
            block_table_tensor = block_table.get_device_tensor()
            if num_reqs > 0:
                block_table_tensor = block_table_tensor[:num_reqs]
            slot_mapping = block_table.slot_mapping.gpu[:num_tokens]
            self.set_per_group_attn_metadata(gid, block_table_tensor, slot_mapping)

    def _slot_mapping_buffer_for_group(
        self,
        attn_group: AttentionGroup,
    ) -> torch.Tensor:
        gid = attn_group.kv_cache_group_id
        if gid == self.kv_cache_gid:
            return self.slot_mapping_group[0]
        buf = self._group_slot_buffers.get(gid)
        if buf is None:
            buf = torch.zeros_like(self.slot_mapping_group[0])
            self._group_slot_buffers[gid] = buf
        return buf

    def _common_attn_metadata_for_group(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        attn_group: AttentionGroup,
    ) -> CommonAttentionMetadata:
        group_common_attn_metadata = self.shallow_copy_metadata(common_attn_metadata)
        gid = attn_group.kv_cache_group_id

        block_table = self._per_group_block_tables.get(gid)
        if block_table is not None:
            group_common_attn_metadata.block_table_tensor = block_table[: group_common_attn_metadata.num_reqs]

        slot_mapping = self._per_group_slot_mappings.get(gid)
        if slot_mapping is None:
            return group_common_attn_metadata
        slot_mapping_buffer = self._slot_mapping_buffer_for_group(attn_group)
        slot_mapping_len = slot_mapping.shape[0]
        if slot_mapping.data_ptr() != slot_mapping_buffer.data_ptr():
            slot_mapping_buffer[:slot_mapping_len].copy_(slot_mapping.to(torch.int32))
        slot_mapping_buffer[slot_mapping_len:].fill_(PADDING_SLOT_ID)
        slot_mapping_view = slot_mapping_buffer[: group_common_attn_metadata.num_actual_tokens]
        self._per_group_slot_mappings[gid] = slot_mapping_view
        group_common_attn_metadata.slot_mapping = slot_mapping_view

        return group_common_attn_metadata

    def _build_step_attn_metadatas(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        *,
        graph_capture: bool = False,
    ) -> tuple[list[Any], list[dict[str, Any]]]:
        """Build base-proposer-style per-step metadata for Step3.5 MTP.

        The full-window path still reuses the same logical window for every MTP
        layer, but graph-param update/replay expects draft metadata to be
        indexed by speculative step.  Each Step3.5 MTP attention group maps to
        one draft step/KV-cache group, so return:

        ``[{layer0: meta0}, {layer1: meta1}, {layer2: meta2}]``

        instead of one dict containing all MTP layers.
        """
        per_group_attn_metadata: list[Any] = []
        multi_steps_attn_metadata: list[dict[str, Any]] = []
        extra_attn_metadata_args: dict[str, Any] = {}
        if self.use_compress:
            extra_attn_metadata_args = dict(
                prefill_ratio_to_sas_metadata=dict(),
                decode_ratio_to_sas_metadata=dict(),
                common_ratio_to_sas_metadata=dict(),
                block_size=self.draft_attn_groups[0].kv_cache_spec.block_size,
            )

        for attn_group in self.draft_attn_groups:
            group_common_attn_metadata = self._common_attn_metadata_for_group(common_attn_metadata, attn_group)
            builder = attn_group.get_metadata_builder()
            if graph_capture:
                attn_metadata = builder.build_for_graph_capture(
                    group_common_attn_metadata,
                    AscendAttentionState.SpecDecoding,
                )
            else:
                attn_metadata = builder.build(
                    0,
                    group_common_attn_metadata,
                    self.runner.get_model(),
                    **extra_attn_metadata_args,
                )
                if hasattr(attn_metadata, "causal") and not attn_metadata.causal:
                    attn_metadata.attn_mask = None
            per_group_attn_metadata.append(attn_metadata)
            per_step_attn_metadata: dict[str, Any] = {}
            for layer_name in attn_group.layer_names:
                per_step_attn_metadata[layer_name] = attn_metadata
            multi_steps_attn_metadata.append(per_step_attn_metadata)
        return per_group_attn_metadata, multi_steps_attn_metadata

    def _sample_draft_tokens_for_step(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        spec_step_idx: int,
        num_indices: int,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """GPU Step3.5 sampling semantics with Ascend TP/reduce-sample paths."""
        logits: torch.Tensor | None = None
        if get_ascend_config().enable_reduce_sample and self.method == "mtp":
            if not hasattr(self.model.model, "compute_logits"):
                draft_token_ids = self.compute_draft_token_ids(hidden_states)
                if lmhead_tp_enable() and num_indices < draft_token_ids.shape[0]:
                    draft_token_ids = draft_token_ids[:num_indices]
                return draft_token_ids, None
            logits = self.model.compute_logits(hidden_states, spec_step_idx=spec_step_idx)
            if lmhead_tp_enable():
                logits = get_lmhead_tp_group().all_to_all(logits)
            else:
                logits = self.model.model.logits_processor._gather_logits(logits)
        else:
            logits = self.model.compute_logits(hidden_states, spec_step_idx=spec_step_idx)

        if lmhead_tp_enable() and num_indices < logits.shape[0]:
            logits = logits[:num_indices]
        if not self._enable_probabilistic_draft_probs or sampling_metadata.all_greedy:
            return logits.argmax(dim=-1), None
        return compute_probs_and_sample_next_token(logits, sampling_metadata)

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
        (
            num_tokens,
            num_tokens_across_dp,
            _,
        ) = self.runner._sync_metadata_across_dp(num_tokens, is_draft_model=True)

        multi_steps_attn_metadata: list[dict[str, Any]] = []
        if not self.use_cuda_graph:
            aclgraph_runtime_mode = CUDAGraphMode.NONE

        if (
            self.pcp_size * self.dcp_size > 1
            and self.use_cuda_graph
            and not is_profile
            and self.block_table_tensor_clone is None
        ):
            self.block_table_tensor_clone = torch.zeros(
                (
                    self.runner.max_num_tokens + 2 * self.pcp_size * self.runner.max_num_reqs,
                    self.runner.input_batch.block_table[0].get_device_tensor().shape[1],
                ),
                dtype=torch.int32,
                device=self.device,
                pin_memory=self.runner.pin_memory,
            )

        batch_size = max(num_tokens // (self.num_speculative_tokens + 1), 1)
        if is_profile:
            batch_size = min(batch_size, self.runner.max_num_reqs)

        if aclgraph_runtime_mode == CUDAGraphMode.FULL and len(self.runner.attn_groups) > 0:
            num_computed_tokens_cpu = self.runner.input_batch.num_computed_tokens_cpu_tensor[:num_reqs]

            self.query_start_loc.cpu[: num_reqs + 1].copy_(self.runner.query_start_loc.cpu[: num_reqs + 1])
            self.query_start_loc.copy_to_gpu()

            common_attn_metadata = AscendCommonAttentionMetadata(
                query_start_loc=self.query_start_loc.gpu[: num_reqs + 1],
                query_start_loc_cpu=self.query_start_loc.cpu[: num_reqs + 1],
                seq_lens_cpu=self.runner.optimistic_seq_lens_cpu,
                seq_lens_cpu_upper_bound=self.runner.optimistic_seq_lens_cpu,
                seq_lens=self.runner.seq_lens[:num_reqs],
                num_reqs=num_reqs,
                num_actual_tokens=num_tokens,
                num_input_tokens=num_tokens,
                max_query_len=self.num_speculative_tokens + 1,
                num_computed_tokens_cpu=num_computed_tokens_cpu,
                actual_seq_lengths_q=self.runner.actual_seq_lengths_q,
                block_table_tensor=self.runner.input_batch.block_table[0].get_device_tensor()[:num_reqs],
                slot_mapping=self.runner.input_batch.block_table[0].slot_mapping.gpu,
                slot_mapping_cpu=self.runner.input_batch.block_table[0].slot_mapping.cpu,
                positions=self.runner.positions,
                attn_state=self.runner.attn_state,
                decode_token_per_req=self.runner.decode_token_per_req,
                max_seq_len=0,
            )
            if self.pcp_size * self.dcp_size > 1:
                common_attn_metadata.prefill_context_parallel_metadata = self.runner.pcp_manager.long_seq_metadata

            common_attn_metadata = self.shallow_copy_metadata(common_attn_metadata)
            common_attn_metadata.slot_mapping = self.slot_mapping_group[0]
            common_attn_metadata.seq_lens = self.seq_lens_group[0][:num_reqs]
            common_attn_metadata.query_start_loc = self.query_start_loc_group[0][: num_reqs + 1]
            self._seed_graph_capture_per_group_metadata(num_reqs, num_tokens)
            _, multi_steps_attn_metadata = self._build_step_attn_metadatas(common_attn_metadata, graph_capture=True)

        model_positions = self._get_positions(num_tokens)

        if self.supports_mm_inputs:
            inputs_embeds = self.model.embed_input_ids(self.input_ids[:num_tokens])
            self.inputs_embeds[:num_tokens] = inputs_embeds
            inputs_embeds = self.inputs_embeds[:num_tokens]
        else:
            inputs_embeds = None

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
        ):
            forward_context = get_forward_context()
            if forward_context is not None:
                forward_context.moe_layer_index = 0

            self._runnable(
                num_input_tokens=num_tokens,
                batch_size=batch_size,
                token_indices_to_sample=self.token_indices_to_sample[: batch_size * self.extra_slots_per_request],
                target_positions=model_positions,
                inputs_embeds=inputs_embeds,
                multi_steps_attn_metadata=multi_steps_attn_metadata,
                num_tokens=num_tokens,
            )
            forward_context = get_forward_context()
            if forward_context.cudagraph_runtime_mode == CUDAGraphMode.FULL and not _EXTRA_CTX.capturing:
                self._update_full_graph_params(forward_context, num_tokens, multi_steps_attn_metadata)

    def _propose(
        self,
        target_token_ids: torch.Tensor,
        target_positions: torch.Tensor,
        target_hidden_states: torch.Tensor,
        next_token_ids: torch.Tensor,
        token_indices_to_sample: torch.Tensor | None,
        common_attn_metadata: CommonAttentionMetadata,
        target_model_batch_desc: BatchDescriptor,
        sampling_metadata: SamplingMetadata,
        mm_embed_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
        req_scheduled_tokens=None,
        long_seq_metadata=None,
        num_prefill_reqs=0,
        num_decode_reqs=0,
        scheduler_output: SchedulerOutput = None,
        num_scheduled_tokens: int = 0,
        num_rejected_tokens_gpu: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self._last_draft_probs = None
        batch_size = common_attn_metadata.batch_size()

        if token_indices_to_sample is None:
            token_indices_to_sample = common_attn_metadata.query_start_loc[1:] - 1

        num_tokens, token_indices_to_sample, common_attn_metadata, long_seq_args = self.set_inputs_first_pass(
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
        if self.pcp_size * self.dcp_size > 1:
            assert long_seq_args is not None
        assert self.runner is not None

        has_lora = len(self.runner.input_batch.lora_id_to_lora_request) > 0
        uniform_decode = target_model_batch_desc.uniform

        if self.use_cuda_graph:
            _, batch_descriptor = self.runner.cudagraph_dispatcher.dispatch(
                num_tokens=num_tokens,
                uniform_decode=uniform_decode,
                has_lora=has_lora,
            )
            num_input_tokens = batch_descriptor.num_tokens
        else:
            num_input_tokens = num_tokens

        (
            num_input_tokens,
            num_tokens_across_dp,
            _,
        ) = self.runner._sync_metadata_across_dp(num_input_tokens, is_draft_model=True)

        if self.use_cuda_graph:
            aclgraph_runtime_mode, batch_descriptor = self.runner.cudagraph_dispatcher.dispatch(
                num_tokens=num_input_tokens,
                uniform_decode=uniform_decode,
                has_lora=has_lora,
            )
            num_input_tokens = batch_descriptor.num_tokens
        else:
            aclgraph_runtime_mode = CUDAGraphMode.NONE
            batch_descriptor = None

        if aclgraph_runtime_mode == CUDAGraphMode.FULL:
            num_reqs_padded = self.runner._pad_query_start_loc_for_fia(
                num_input_tokens,
                batch_descriptor.num_reqs if batch_descriptor.num_reqs is not None else common_attn_metadata.num_reqs,
                common_attn_metadata.num_reqs,
                aclgraph_runtime_mode,
                batch_descriptor.num_reqs,
            )
            common_attn_metadata.num_reqs = num_reqs_padded
            common_attn_metadata.query_start_loc = self.runner.query_start_loc.gpu[: num_reqs_padded + 1]
            common_attn_metadata.query_start_loc_cpu = self.runner.query_start_loc.cpu[: num_reqs_padded + 1]
            slicing_length = (
                num_reqs_padded * self.decode_threshold if self.pcp_size * self.dcp_size > 1 else num_reqs_padded
            )
            common_attn_metadata.block_table_tensor = self._adjust_tensor(
                common_attn_metadata.block_table_tensor, slicing_length
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

            if self.pcp_size > 1:
                pcp_allgather_restore_idx = (
                    common_attn_metadata.prefill_context_parallel_metadata.pcp_allgather_restore_idx
                )
                index = torch.arange(
                    pcp_allgather_restore_idx.shape[0],
                    device=pcp_allgather_restore_idx.device,
                )
                mask = (index % (self.pcp_size * self.decode_threshold)) >= self.decode_threshold
                pcp_allgather_restore_idx[mask] = 0
                self.runner.pcp_manager.pcp_allgather_restore_idx.gpu[: pcp_allgather_restore_idx.shape[0]] = (
                    pcp_allgather_restore_idx
                )
                self.runner.pcp_manager.pcp_allgather_restore_idx.gpu[pcp_allgather_restore_idx.shape[0] :] = 0
        else:
            num_reqs_padded = common_attn_metadata.num_reqs
            if not self.vllm_config.model_config.use_mla and self.pcp_size * self.dcp_size == 1:
                common_attn_metadata.block_table_tensor = self._adjust_tensor(
                    common_attn_metadata.block_table_tensor, num_reqs_padded
                )

        if self.supports_mm_inputs:
            inputs_embeds = self.model.embed_input_ids(self.input_ids[:num_tokens])
            self.inputs_embeds[:num_tokens] = inputs_embeds
            inputs_embeds = self.inputs_embeds[:num_input_tokens]
        else:
            inputs_embeds = None

        slot_mapping_len = common_attn_metadata.slot_mapping.shape[0]
        self.slot_mapping_group[0][:slot_mapping_len].copy_(common_attn_metadata.slot_mapping)
        self.slot_mapping_group[0][slot_mapping_len:].fill_(PADDING_SLOT_ID)
        common_attn_metadata.slot_mapping = self.slot_mapping_group[0]
        self._per_group_slot_mappings[self.kv_cache_gid] = self.slot_mapping_group[0][
            : common_attn_metadata.num_actual_tokens
        ]

        self.seq_lens_group[0][:num_reqs_padded].copy_(common_attn_metadata.seq_lens)
        self.seq_lens_group[0][num_reqs_padded:].fill_(0)
        common_attn_metadata.seq_lens = self.seq_lens_group[0][:num_reqs_padded]

        self.query_start_loc_group[0][: num_reqs_padded + 1].copy_(common_attn_metadata.query_start_loc)
        self.query_start_loc_group[0][num_reqs_padded + 1 :].fill_(0)
        common_attn_metadata.query_start_loc = self.query_start_loc_group[0][: num_reqs_padded + 1]
        common_attn_metadata.num_input_tokens = num_input_tokens
        _, multi_steps_attn_metadata = self._build_step_attn_metadatas(common_attn_metadata)
        attn_metadata_i = next(iter(multi_steps_attn_metadata[0].values()))

        if not self.use_cuda_graph:
            common_attn_metadata.block_table_tensor = common_attn_metadata.block_table_tensor.clone()

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
        ):
            forward_context = get_forward_context()
            if forward_context is not None:
                forward_context.moe_layer_index = 0

            model_inputs: dict[str, Any] = {
                "num_input_tokens": num_input_tokens,
                "batch_size": batch_size,
                "token_indices_to_sample": self.token_indices_to_sample[:token_indices_to_sample_len],
                "target_positions": target_positions,
                "inputs_embeds": inputs_embeds,
                "multi_steps_attn_metadata": multi_steps_attn_metadata,
                "num_tokens": num_tokens,
                "is_prefill": attn_metadata_i.num_prefills,
            }
            run_draft = partial(self._runnable, **model_inputs)
            if self.enable_enpu:
                self._update_full_graph_params_if_needed(forward_context, num_input_tokens, multi_steps_attn_metadata)
                draft_token_ids = run_draft()
            else:
                draft_token_ids = run_draft()
                self._update_full_graph_params_if_needed(forward_context, num_input_tokens, multi_steps_attn_metadata)
        return draft_token_ids

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
    ) -> torch.Tensor:
        """Base MTP execution flow with Step3.5 step-aware layer/head selection."""
        self._last_draft_probs = None
        sampling_metadata = self.runner.input_batch.sampling_metadata
        model_input_ids = self.input_ids[:num_input_tokens]
        model_positions = self._get_positions(num_input_tokens)
        model_kwargs = {
            "input_ids": model_input_ids,
            "positions": model_positions,
            "inputs_embeds": inputs_embeds,
            "spec_step_idx": 0,
        }
        if self.pass_hidden_states_to_model:
            model_hidden_states = self.hidden_states[:num_input_tokens]
            model_hidden_states, model_positions = self.maybe_pad_and_reduce(model_hidden_states, model_positions)
            model_kwargs["hidden_states"] = model_hidden_states
            model_kwargs["positions"] = model_positions

        ret_hidden_states = self.model(**model_kwargs)
        if not self.model_returns_tuple():
            last_hidden_states = ret_hidden_states
            hidden_states = last_hidden_states
        else:
            last_hidden_states, hidden_states = ret_hidden_states

        last_hidden_states, model_positions, hidden_states = self.maybe_all_gather_and_unpad(
            last_hidden_states, model_positions, hidden_states
        )

        num_indices = token_indices_to_sample.shape[0]
        if lmhead_tp_enable():
            max_num_reqs_across_dp = (
                self.vllm_config.scheduler_config.max_num_seqs * self.runner.uniform_decode_query_len
            )
            token_indices_to_sample = torch.nn.functional.pad(
                token_indices_to_sample, (0, max_num_reqs_across_dp - num_indices)
            )

        sample_hidden_states = last_hidden_states[token_indices_to_sample]
        draft_token_ids, draft_probs = self._sample_draft_tokens_for_step(
            sample_hidden_states,
            sampling_metadata,
            spec_step_idx=0,
            num_indices=num_indices,
        )
        if self.num_speculative_tokens == 1 or self.parallel_drafting:
            if draft_probs is not None:
                self._last_draft_probs = draft_probs.view(
                    -1, self.num_speculative_tokens, draft_probs.shape[-1]
                ).contiguous()
            return draft_token_ids.view(-1, self.num_speculative_tokens)

        if self.pcp_size * self.dcp_size > 1 and is_prefill:
            draft_token_ids_list = [draft_token_ids for _ in range(self.num_speculative_tokens)]
            return torch.stack(draft_token_ids_list, dim=1)

        return self._run_window_draft_steps(
            first_draft_token_ids=draft_token_ids,
            first_draft_probs=draft_probs,
            first_hidden_states=hidden_states,
            num_input_tokens=num_input_tokens,
            batch_size=batch_size,
            token_indices_to_sample=token_indices_to_sample,
            target_positions=target_positions,
            num_tokens=num_tokens,
            multi_steps_attn_metadata=multi_steps_attn_metadata,
            inputs_embeds=inputs_embeds,
            sampling_metadata=sampling_metadata,
        )

    def _roll_window_inputs_only(
        self,
        *,
        prev_token_ids: torch.Tensor,
        next_token_ids: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        token_indices_to_sample: torch.Tensor,
        num_tokens: int,
        input_batch_size: int,
    ) -> torch.Tensor | None:
        self.input_ids[: num_tokens - 1] = prev_token_ids[1:]
        self.input_ids[token_indices_to_sample] = next_token_ids
        self.hidden_states[:num_tokens] = previous_hidden_states.view(num_tokens, -1)

        if self.supports_mm_inputs:
            self.inputs_embeds[:num_tokens] = self.model.embed_input_ids(self.input_ids[:num_tokens])
            return self.inputs_embeds[:input_batch_size]
        return None

    def _run_window_draft_steps(
        self,
        *,
        first_draft_token_ids: torch.Tensor,
        first_draft_probs: torch.Tensor | None,
        first_hidden_states: torch.Tensor,
        num_input_tokens: int,
        batch_size: int,
        token_indices_to_sample: torch.Tensor,
        target_positions: torch.Tensor,
        num_tokens: int,
        multi_steps_attn_metadata,
        inputs_embeds,
        sampling_metadata: SamplingMetadata,
    ) -> torch.Tensor:
        """Run the Step3.5 MTP full-window layer chain.

        This is the canonical Step3.5 MTP draft path, matching the full-window
        multi-layer proposer loop: every MTP layer reprocesses the whole draft
        window so each independent KV-cache group is seeded over the full
        window. Attention metadata is built once before this loop; layer-to-layer
        state transition only rolls the model input buffers by shifting the
        previous window and placing the newly drafted token at
        token_indices_to_sample.
        """
        draft_probs_list = None if first_draft_probs is None else [first_draft_probs]
        draft_token_ids_list = [first_draft_token_ids]
        full_hidden_states = first_hidden_states[:num_tokens]
        input_batch_size = num_input_tokens

        forward_context = get_forward_context()
        _EXTRA_CTX.num_tokens = input_batch_size
        _EXTRA_CTX.num_accept_tokens = batch_size

        for draft_step in range(self.num_speculative_tokens - 1):
            forward_context = get_forward_context()
            if forward_context is not None:
                forward_context.moe_layer_index = 0

            spec_step_idx = draft_step + 1
            prev_token_ids = self.input_ids[:num_tokens].clone()
            next_token_ids = draft_token_ids_list[-1].int()
            inputs_embeds = self._roll_window_inputs_only(
                prev_token_ids=prev_token_ids,
                next_token_ids=next_token_ids,
                previous_hidden_states=full_hidden_states,
                token_indices_to_sample=token_indices_to_sample,
                num_tokens=num_tokens,
                input_batch_size=input_batch_size,
            )

            model_input_ids = self.input_ids[:input_batch_size]
            model_positions = self._get_positions(input_batch_size)
            model_hidden_states = self.hidden_states[:input_batch_size]
            model_hidden_states, model_positions = self.maybe_pad_and_reduce(
                model_hidden_states,
                model_positions,
            )
            if forward_context is not None and multi_steps_attn_metadata:
                if spec_step_idx >= len(multi_steps_attn_metadata):
                    raise AssertionError("Step3.5 MTP metadata must contain one entry per draft step")
                forward_context.attn_metadata = multi_steps_attn_metadata[spec_step_idx]

            model_kwargs = {
                "input_ids": model_input_ids,
                "positions": model_positions,
                "inputs_embeds": inputs_embeds,
                "spec_step_idx": spec_step_idx,
            }
            if self.pass_hidden_states_to_model:
                model_kwargs["hidden_states"] = model_hidden_states

            ret_hidden_states = self.model(**model_kwargs)
            if not self.model_returns_tuple():
                last_hidden_states = ret_hidden_states
                hidden_states = ret_hidden_states
            else:
                last_hidden_states, hidden_states = ret_hidden_states

            last_hidden_states, model_positions, hidden_states = self.maybe_all_gather_and_unpad(
                last_hidden_states,
                model_positions,
                hidden_states,
            )

            num_indices = token_indices_to_sample.shape[0]
            sample_hidden_states = last_hidden_states[token_indices_to_sample]
            draft_token_ids, draft_probs = self._sample_draft_tokens_for_step(
                sample_hidden_states,
                sampling_metadata,
                spec_step_idx=spec_step_idx,
                num_indices=num_indices,
            )
            if draft_probs is not None:
                assert draft_probs_list is not None
                draft_probs_list.append(draft_probs)
            full_hidden_states = hidden_states[:num_tokens]
            draft_token_ids_list.append(draft_token_ids)

        draft_token_ids = torch.stack(draft_token_ids_list, dim=1)
        if draft_probs_list is not None:
            self._last_draft_probs = torch.stack(draft_probs_list, dim=1).contiguous()
        return draft_token_ids

    # -- overrides matching the GPU Step3p5MTPProposer ----------------------

    def _ensure_draft_layer_types_cover_mtp_layers(self) -> None:
        hf_config = self.draft_model_config.hf_config
        layer_types = getattr(hf_config, "layer_types", None)
        num_hidden_layers = getattr(hf_config, "num_hidden_layers", None)
        num_mtp_layers = getattr(hf_config, "num_nextn_predict_layers", 0)

        if layer_types is None or num_hidden_layers is None or num_mtp_layers is None:
            return

        needed_num_layer_types = num_hidden_layers + num_mtp_layers
        if len(layer_types) >= needed_num_layer_types:
            return

        hf_config.layer_types = list(layer_types) + ["sliding_attention"] * (needed_num_layer_types - len(layer_types))

    def _create_draft_vllm_config(self) -> VllmConfig:
        base = super()._create_draft_vllm_config()
        # Transformers validates Step3.5 layer_types against num_hidden_layers
        # and may leave only the base-layer entries. The MTP draft model builds
        # layers at num_hidden_layers + k, so make the draft config cover those
        # layer indices before the model is constructed.
        self._ensure_draft_layer_types_cover_mtp_layers()
        # Ascend ModelSlim keeps quant metadata in quant_model_description.json,
        # not in config.json's quantization_config, so get_quant_config falls
        # through to the hf_overrides branch. The verifier model_config has
        # hf_overrides={}, but the draft_model_config leaves it None, which that
        # branch rejects. Normalize to {} so the draft takes the verifier's path.
        if not isinstance(getattr(self.draft_model_config, "hf_overrides", None), dict):
            self.draft_model_config.hf_overrides = {}
        return replace(
            base,
            model_config=self.draft_model_config,
            quant_config=get_draft_quant_config(base),
        )

    def validate_same_kv_cache_group(self, kv_cache_config: KVCacheConfig) -> None:
        """Step3.5 MTP draft layers may span multiple KV cache groups."""
        return

    def initialize_attn_backend(
        self,
        kv_cache_config: KVCacheConfig,
        kernel_block_sizes: list[int] | None = None,
    ) -> None:
        all_attn_layers = get_layers_from_vllm_config(
            self.vllm_config,
            AttentionLayerBase,  # type: ignore[type-abstract]
        )

        layer_to_gid: dict[str, int] = {}
        layer_to_spec: dict[str, KVCacheSpec] = {}
        for gid, group in enumerate(kv_cache_config.kv_cache_groups):
            group_spec = group.kv_cache_spec
            for layer_name in group.layer_names:
                layer_to_gid[layer_name] = gid
                if isinstance(group_spec, UniformTypeKVCacheSpecs):
                    if layer_name in group_spec.kv_cache_specs:
                        layer_to_spec[layer_name] = group_spec.kv_cache_specs[layer_name]
                    else:
                        target_layer_name = getattr(
                            all_attn_layers.get(layer_name),
                            "kv_sharing_target_layer_name",
                            None,
                        )
                        if target_layer_name and target_layer_name in group_spec.kv_cache_specs:
                            layer_to_spec[layer_name] = group_spec.kv_cache_specs[target_layer_name]
                        else:
                            layer_to_spec[layer_name] = group_spec
                else:
                    layer_to_spec[layer_name] = group_spec

        attention_groups: dict[tuple[str, int], AttentionGroup] = {}
        for layer_name in sorted(self._draft_attn_layer_names):
            if layer_name not in layer_to_spec:
                continue
            attn_layer = all_attn_layers[layer_name]
            attn_backend = attn_layer.get_attn_backend()
            spec = layer_to_spec[layer_name]
            gid = layer_to_gid[layer_name]
            group_key = (attn_backend.full_cls_name(), gid)

            if group_key not in attention_groups:
                kernel_block_size = (
                    kernel_block_sizes[gid]
                    if kernel_block_sizes is not None and gid < len(kernel_block_sizes)
                    else None
                )
                attn_group = AttentionGroup(
                    backend=attn_backend,
                    layer_names=[layer_name],
                    kv_cache_spec=spec,
                    kv_cache_group_id=gid,
                )
                attn_group.create_metadata_builders(
                    self.vllm_config,
                    self.device,
                    kernel_block_size=kernel_block_size,
                )
                attention_groups[group_key] = attn_group
            else:
                attention_groups[group_key].layer_names.append(layer_name)

        self.draft_attn_groups = list(attention_groups.values())
        if self.draft_attn_groups:
            self.kv_cache_gid = self.draft_attn_groups[0].kv_cache_group_id
            self.block_size = self.draft_attn_groups[0].get_metadata_builder().kv_cache_spec.block_size
        else:
            self.kv_cache_gid = 0
            self.block_size = kv_cache_config.kv_cache_groups[0].kv_cache_spec.block_size
