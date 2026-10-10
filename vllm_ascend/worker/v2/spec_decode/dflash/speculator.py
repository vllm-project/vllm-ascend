# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from dataclasses import replace
from typing import Any, cast

import torch
from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.config.compilation import CUDAGraphMode
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.gpu.spec_decode.dflash import speculator as dflash_speculator
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator

from vllm_ascend.attention.attention_v1 import AscendAttentionBackend, AscendAttentionMetadataBuilder, AscendMetadata
from vllm_ascend.compilation.updatable_graph import UpdatableGraph
from vllm_ascend.ops.triton.v2.spec_decode.prepare_dflash_inputs import DFlashKVGroup, prepare_dflash_inputs_triton
from vllm_ascend.worker.v2.attn_utils import build_attn_metadata_factory, build_attn_metadata_wrapper
from vllm_ascend.worker.v2.spec_decode.lmhead_tp_utils import LmheadTPDraftSamplingMixin
from vllm_ascend.worker.v2.spec_decode.pcp_utils import (
    disable_profiling_chunk_for_draft,
)


def prepare_dflash_inputs_factory(
    kv_cache_block_size: int,
    on_inputs_prepared: Callable[[], None] | None = None,
    get_kv_groups: Callable[[], tuple[DFlashKVGroup, ...] | None] | None = None,
) -> Callable[..., None]:
    # Upstream uses the attention kernel block size for DCP ownership, which is
    # incorrect when physical KV blocks are larger than kernel blocks. Bind the
    # physical size so ownership uses KV cache blocks while slot lookup uses the
    # kernel-sized block table supplied by the upstream caller.
    def prepare_with_block_size(*args: Any, **kwargs: Any) -> None:
        groups = get_kv_groups() if get_kv_groups is not None else None
        # The upstream proposal loops over groups before any input consumer.
        # The first call prepares all groups; subsequent calls need no launch.
        # Outside a proposal, retain the single-group API (None).
        if groups == ():
            return
        prepare_dflash_inputs(*args, **kwargs, kv_cache_block_size=kv_cache_block_size, kv_groups=groups)
        if on_inputs_prepared is not None:
            on_inputs_prepared()

    return prepare_with_block_size


class AscendDFlashSpeculator(LmheadTPDraftSamplingMixin, DFlashSpeculator):
    _deferred_draft_attn_metadata: tuple[Any, ...] | None
    _draft_query_lengths: tuple[tuple[int, int], list[int]] | None

    def load_draft_model(
        self,
        target_model: torch.nn.Module,
        target_attn_layer_names: set[str],
    ) -> torch.nn.Module:
        with disable_profiling_chunk_for_draft(self.vllm_config):
            return super().load_draft_model(target_model, target_attn_layer_names)

    def build_draft_attn_metadatas(self, num_reqs_padded, seq_lens_cpu_upper_bound):
        pending = getattr(self, "_deferred_draft_attn_metadata", None)
        if pending is not None:
            self._deferred_draft_attn_metadata = None
            self._materializing_draft_metadata = True
            try:
                self._build_attn_metadata(*pending)
            finally:
                self._materializing_draft_metadata = False
        num_tokens_padded = num_reqs_padded * self.num_query_per_req
        cached = self._prepared_draft_attn_metadata
        if cached is not None and cached[0] == num_reqs_padded:
            attn_metadata = cached[1]
            self._update_draft_attn_metadata(attn_metadata, num_reqs_padded)
            return [attn_metadata]
        with build_attn_metadata_wrapper():
            # vLLM main (#56181) replaced _build_draft_attn_metadata with
            # _build_uniform_attn_metadata (BatchExecutionDescriptor).
            batch_desc = BatchExecutionDescriptor(
                cg_mode=CUDAGraphMode.FULL,
                num_tokens=num_tokens_padded,
                num_reqs=num_reqs_padded,
            )
            attn_metadata = self._build_uniform_attn_metadata(
                num_reqs=self.input_batch.num_reqs,
                batch_desc=batch_desc,
                num_query_per_req=self.num_query_per_req,
                seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,
                step=self.num_query_per_req,
                causal=self._group_causal,
            )
        self._update_draft_attn_metadata(attn_metadata, num_reqs_padded)
        return [attn_metadata]

    def _update_draft_attn_metadata(self, attn_metadata, num_reqs_padded):
        """Rebuild ``actual_seq_lengths_q`` from the padded request count,
        mirroring Eagle's ``_update_decode_attn_metadata``.

        Upstream ``Speculator._build_draft_attn_metadata`` clamps
        ``query_start_loc`` at the real ``num_reqs`` to keep the cumulative
        series non-decreasing, so when a batch is padded to a capture size
        (``num_reqs_padded > num_reqs``) the cumulative query lengths stop at
        ``num_reqs * num_query_per_req`` instead of ``num_tokens_padded``. The
        Ascend FIA operator requires, in TND layout, that the last element of
        ``actual_seq_lengths_q`` equals the query token count of the graph
        being replayed; otherwise tiling fails with
        ``queryT != last element of actualSequenceLengthQ``.
        """
        query_key = (num_reqs_padded, self.num_query_per_req)
        cached = getattr(self, "_draft_query_lengths", None)
        if cached is None or cached[0] != query_key:
            cached = (query_key, [(i + 1) * self.num_query_per_req for i in range(num_reqs_padded)])
            self._draft_query_lengths = cached
        query_lens_list = cached[1]
        for metadata in attn_metadata.values():
            metadata.actual_seq_lengths_q = query_lens_list

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        super().__init__(vllm_config, device)
        self._lmhead_tp_validate_draft_sampling()
        self._prepared_draft_attn_metadata: tuple[int, dict[str, Any]] | None = None
        self._deferred_draft_attn_metadata = None
        self._draft_metadata_template: tuple[Any, dict[str, AscendMetadata]] | None = None
        self._draft_query_lengths = None
        self._draft_seq_lens_cpu: torch.Tensor | None = None
        self._draft_seq_lens_copy_stream: torch.npu.Stream | None = None
        self._draft_seq_lens_copy_event: torch.npu.Event | None = None
        self._draft_seq_lens_copy_count: int | None = None
        self._draft_inputs_prepared = False
        self._in_draft_proposal = False
        self._materializing_draft_metadata = False
        self._reuse_draft_layout = False
        self._use_cpu_seq_lens = False

    def _get_draft_input_groups(self) -> tuple[DFlashKVGroup, ...] | None:
        if not self._in_draft_proposal:
            return None
        if self._draft_inputs_prepared:
            return ()
        # Read current group views, rather than caching block-table addresses
        # across proposals. Keep the slot buffers used by captured graphs.
        return tuple(
            (
                self.block_tables.slot_mappings[gid],
                self._context_slot_mappings[index],
                self.block_tables.input_block_tables[gid],
                self.block_tables.kernel_block_sizes[gid],
            )
            for index, gid in enumerate(self.draft_kv_cache_group_ids)
        )

    def _on_draft_inputs_prepared(self) -> None:
        if not self._in_draft_proposal:
            return
        self._draft_inputs_prepared = True
        # Grouped preparation writes common lengths once. Snapshot afterward,
        # before context KV work or graph replay is submitted.
        if self._use_cpu_seq_lens:
            self._start_draft_seq_lens_copy(self.input_batch.num_reqs)

    def _start_draft_seq_lens_copy(self, num_reqs: int) -> None:
        source = self.input_buffers.seq_lens
        if self._draft_seq_lens_cpu is None:
            self._draft_seq_lens_cpu = torch.empty(source.numel(), dtype=source.dtype, device="cpu", pin_memory=True)
            self._draft_seq_lens_copy_stream = torch.npu.Stream(device=source.device)
            self._draft_seq_lens_copy_event = torch.npu.Event()
        assert self._draft_seq_lens_copy_stream is not None
        assert self._draft_seq_lens_copy_event is not None
        current_stream = torch.npu.current_stream()
        with torch.npu.stream(self._draft_seq_lens_copy_stream):
            self._draft_seq_lens_copy_stream.wait_stream(current_stream)
            self._draft_seq_lens_cpu[:num_reqs].copy_(source[:num_reqs], non_blocking=True)
            self._draft_seq_lens_copy_event.record()
        self._draft_seq_lens_copy_count = num_reqs

    def _prepare_draft_seq_lens_cpu(self, num_reqs: int, num_reqs_padded: int) -> torch.Tensor:
        if getattr(self, "_draft_seq_lens_copy_count", None) == num_reqs:
            assert self._draft_seq_lens_copy_event is not None
            assert self._draft_seq_lens_cpu is not None
            # Wait only for the snapshot, not the compute stream's context KV
            # work or its newly launched graph (which waits for FIA updates).
            self._draft_seq_lens_copy_event.synchronize()
            self._draft_seq_lens_copy_count = None
            seq_lens_cpu = self._draft_seq_lens_cpu[:num_reqs_padded]
            if seq_lens_cpu.numel() != num_reqs_padded:
                seq_lens_cpu = torch.zeros(num_reqs_padded, dtype=self._draft_seq_lens_cpu.dtype)
                seq_lens_cpu[:num_reqs].copy_(self._draft_seq_lens_cpu[:num_reqs])
            seq_lens_cpu[num_reqs:].zero_()
            return seq_lens_cpu
        # Match GLM5.2/DSpark: transfer the device's valid lengths once, then
        # share the host mirror across FIA builders. The target CPU upper bound
        # still includes rejected tokens and cannot replace this exact mirror.
        seq_lens_cpu = torch.zeros(num_reqs_padded, dtype=torch.int32, device="cpu")
        if num_reqs:
            seq_lens_cpu[:num_reqs].copy_(self.input_buffers.seq_lens[:num_reqs])
        return seq_lens_cpu

    def _can_defer_draft_metadata(self, batch_desc) -> bool:
        if (
            not getattr(self, "_in_draft_proposal", False)
            or not getattr(self, "_reuse_draft_layout", False)
            or getattr(self, "_materializing_draft_metadata", False)
            or getattr(self, "_draft_seq_lens_copy_count", None) is None
            or batch_desc.cg_mode != CUDAGraphMode.FULL
        ):
            return False
        graph = self.query_cudagraph_manager.graphs.get(batch_desc)
        # Every FIA task waits on its own external update event. Graph prefix
        # compute can start before the CPU metadata is ready; attention cannot.
        return isinstance(graph, UpdatableGraph) and bool(graph.tasks)

    def _refresh_draft_layout(self, template, num_reqs_padded, num_tokens):
        refreshed = {}
        group_metadata = {}
        for group_index, groups in enumerate(self.attn_groups):
            for group in groups:
                for name in group.layer_names:
                    previous = template[name]
                    key = (group_index, id(previous))
                    if key not in group_metadata:
                        group_metadata[key] = replace(
                            previous,
                            seq_lens_gpu=self.input_buffers.seq_lens[:num_reqs_padded],
                            query_start_loc_gpu=self.input_buffers.query_start_loc[: num_reqs_padded + 1],
                            block_tables=AscendAttentionMetadataBuilder._pad_block_table(
                                self.block_tables.input_block_tables[group_index][:num_reqs_padded], num_reqs_padded
                            ),
                            slot_mapping=self.block_tables.slot_mappings[group_index][:num_tokens],
                            reshape_cache_event=None,
                            qfa_metadata_cache={},
                        )
                    refreshed[name] = group_metadata[key]
        return refreshed

    def _build_attn_metadata(
        self,
        num_reqs,
        batch_desc,
        query_start_loc_np,
        seq_lens_cpu_upper_bound,
        step,
        causal=True,
        dcp_local_seq_lens=None,
    ):
        if self._can_defer_draft_metadata(batch_desc):
            self._deferred_draft_attn_metadata = (
                num_reqs,
                batch_desc,
                query_start_loc_np,
                seq_lens_cpu_upper_bound,
                step,
                causal,
                dcp_local_seq_lens,
            )
            return None
        context: AbstractContextManager[Any] = nullcontext()
        template_key = None
        if getattr(self, "_use_cpu_seq_lens", False):
            num_reqs_padded = batch_desc.num_reqs or num_reqs
            num_tokens = (
                batch_desc.num_tokens if batch_desc.cg_mode == CUDAGraphMode.FULL else int(query_start_loc_np[-1])
            )
            if getattr(self, "_reuse_draft_layout", False) and batch_desc.cg_mode == CUDAGraphMode.FULL:
                template_key = (
                    num_reqs,
                    num_reqs_padded,
                    num_tokens,
                    step,
                    tuple(query_start_loc_np.tolist()),
                    causal if isinstance(causal, bool) else tuple(sorted(causal.items())),
                )
                cached = self._draft_metadata_template
                if cached is not None and cached[0] == template_key:
                    # Refresh physical cache views while the exact-length D2H
                    # is pending. These are fresh objects, not the cached ones.
                    metadata = self._refresh_draft_layout(cached[1], num_reqs_padded, num_tokens)
                    seq_lens_cpu = self._prepare_draft_seq_lens_cpu(num_reqs, num_reqs_padded)
                    lengths = seq_lens_cpu.tolist()
                    for value in metadata.values():
                        value.seq_lens = value.seq_lens_cpu = seq_lens_cpu
                        value.seq_lens_list = lengths
                    self._prepared_draft_attn_metadata = (num_reqs_padded, metadata)
                    return metadata
            seq_lens_cpu = self._prepare_draft_seq_lens_cpu(num_reqs, num_reqs_padded)
            context = build_attn_metadata_factory(
                self.input_buffers.positions,
                num_tokens,
                is_prefilling=self.draft_is_prefilling[:num_reqs_padded],
                seq_lens_cpu=seq_lens_cpu,
                seq_lens_cpu_is_exact=True,
                parallel_config=self.attn_vllm_config.parallel_config,
            )
        with context:
            metadata = super()._build_attn_metadata(
                num_reqs,
                batch_desc,
                query_start_loc_np,
                seq_lens_cpu_upper_bound,
                step,
                causal,
                dcp_local_seq_lens,
            )
        # Upstream builds this even for FULL replay, refreshing builder state.
        # Reuse that result when the graph manager asks again in this proposal.
        # The result includes group-local cache tensors and exact host lengths.
        if metadata is not None and batch_desc.cg_mode == CUDAGraphMode.FULL:
            self._prepared_draft_attn_metadata = (batch_desc.num_reqs or num_reqs, metadata)
            if template_key is not None and all(type(value) is AscendMetadata for value in metadata.values()):
                self._draft_metadata_template = (template_key, metadata)
        return metadata

    def init_cudagraph_manager(self, cudagraph_mode: CUDAGraphMode) -> None:
        if self.speculative_config.enforce_eager:
            cudagraph_mode = CUDAGraphMode.NONE
        super().init_cudagraph_manager(cudagraph_mode)
        # The Ascend graph manager is patched onto the upstream module and
        # created by super().init_cudagraph_manager without a speculator ref.
        # It needs this speculator to update full-graph params, so set it here.
        self.query_cudagraph_manager.speculator = self
        self.query_cudagraph_manager.update_stream = self.update_stream

    def set_attn(
        self,
        model_state: Any,
        kv_cache_config: Any,
        block_tables: Any,
        target_input_buffers: Any,
        target_attn_groups: Any,
    ) -> None:
        super().set_attn(
            model_state,
            kv_cache_config,
            block_tables,
            target_input_buffers,
            target_attn_groups,
        )
        self._context_slot_mappings = torch.zeros(
            len(self.draft_kv_cache_group_ids),
            self.max_num_tokens,
            dtype=torch.int32,
            device=self.device,
        )
        # npu needs attn_backends to update full graph params in run_fullgraph.
        attn_backends: dict[str, type[AttentionBackend]] = {}
        active_layer_names = self.draft_attn_layer_names
        for kv_cache_group_spec in kv_cache_config.kv_cache_groups:
            layer_names = kv_cache_group_spec.layer_names
            if active_layer_names is not None:
                layer_names = list(active_layer_names.intersection(layer_names))

            layer_type = cast(type[Any], AttentionLayerBase)
            attn_layers = get_layers_from_vllm_config(self.vllm_config, layer_type, layer_names)

            for layer_name in layer_names:
                attn_backends[layer_name] = attn_layers[layer_name].get_attn_backend()

        self.attn_backends = attn_backends
        self._use_cpu_seq_lens = (
            AscendAttentionBackend in attn_backends.values()
            and self.attn_vllm_config.parallel_config.decode_context_parallel_size == 1
            and self.attn_vllm_config.parallel_config.prefill_context_parallel_size == 1
        )
        # Specialized builders own additional per-step state. Keep their full
        # build path; only standard, non-CP FIA layouts are reusable here.
        self._reuse_draft_layout = self._use_cpu_seq_lens and all(
            type(group.get_metadata_builder(0)) is AscendAttentionMetadataBuilder
            and group.get_metadata_builder(0).supports_update_block_table
            for groups in self.attn_groups
            for group in groups
        )
        dflash_speculator.prepare_dflash_inputs = prepare_dflash_inputs_factory(
            self.vllm_config.cache_config.block_size, self._on_draft_inputs_prepared, self._get_draft_input_groups
        )

    def propose(
        self,
        input_batch: InputBatch,
        attn_metadata: dict[str, Any],
        slot_mappings: dict[str, torch.Tensor],
        last_hidden_states: torch.Tensor,
        aux_hidden_states: list[torch.Tensor] | None,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        last_sampled: torch.Tensor,
        next_prefill_tokens: torch.Tensor,
        temperature: torch.Tensor,
        seeds: torch.Tensor,
        dp_sync: Any = None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
        mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
        is_profile: bool = False,
    ) -> torch.Tensor:
        self.input_batch = input_batch
        # Physical cache rows and accepted lengths change between proposals.
        self._prepared_draft_attn_metadata = None
        self._deferred_draft_attn_metadata = None
        self._draft_inputs_prepared = False
        self._in_draft_proposal = True
        sync_state = dp_sync
        if dummy_run and skip_attn_for_dummy_run:
            # Profiling runs the draft with its own query token count, which
            # can differ from the target batch. Let forward_context coordinate
            # the actual draft counts instead of reusing the target DP state.
            # TODO: Remove this guard once main2main includes upstream vLLM
            # #54856 (facd9a74a1), which resets the profiling DP counts.
            sync_state = None
        try:
            with build_attn_metadata_wrapper():
                return super().propose(
                    input_batch,
                    attn_metadata,
                    slot_mappings,
                    last_hidden_states,
                    aux_hidden_states,
                    num_sampled,
                    num_rejected,
                    last_sampled,
                    next_prefill_tokens,
                    temperature,
                    seeds,
                    sync_state,
                    dummy_run,
                    skip_attn_for_dummy_run,
                    mm_inputs,
                    is_profile=is_profile,
                )
        finally:
            self._in_draft_proposal = False


def prepare_dflash_inputs(
    input_buffers: InputBuffers,
    query_slot_mapping: torch.Tensor,
    context_positions: torch.Tensor,
    context_slot_mapping: torch.Tensor,
    sample_indices: torch.Tensor,
    sample_pos: torch.Tensor,
    sample_idx_mapping: torch.Tensor,
    temperature: torch.Tensor,
    seeds: torch.Tensor,
    input_batch: InputBatch,
    num_sampled: torch.Tensor,
    num_rejected: torch.Tensor,
    last_sampled: torch.Tensor,
    next_prefill_tokens: torch.Tensor,
    input_temperature: torch.Tensor,
    input_seeds: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    cp_rank: int,
    cp_size: int,
    cp_interleave: int,
    parallel_drafting_token_id: int,
    num_query_per_req: int,
    num_speculative_steps: int,
    max_num_reqs: int,
    max_num_tokens: int,
    max_model_len: int,
    sample_from_anchor: bool = False,
    *,
    kv_cache_block_size: int,
    kv_groups: tuple[DFlashKVGroup, ...] | None = None,
) -> None:
    prepare_dflash_inputs_triton(
        input_buffers,
        query_slot_mapping,
        context_positions,
        context_slot_mapping,
        sample_indices,
        sample_pos,
        sample_idx_mapping,
        temperature,
        seeds,
        input_batch,
        num_sampled,
        num_rejected,
        last_sampled,
        next_prefill_tokens,
        input_temperature,
        input_seeds,
        block_table,
        block_size,
        cp_rank,
        cp_size,
        cp_interleave,
        parallel_drafting_token_id,
        num_query_per_req,
        num_speculative_steps,
        max_num_reqs,
        max_num_tokens,
        max_model_len,
        sample_from_anchor,
        kv_cache_block_size=kv_cache_block_size,
        kv_groups=kv_groups,
    )
