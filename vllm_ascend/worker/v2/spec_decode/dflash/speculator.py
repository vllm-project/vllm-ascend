# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterator, Mapping
from contextlib import contextmanager, nullcontext
from dataclasses import replace
from typing import Any, cast

import numpy as np
import torch
from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.config.compilation import CUDAGraphMode
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.gpu.spec_decode.dflash import speculator as dflash_speculator
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator

from vllm_ascend.ascend_config import draft_config_context
from vllm_ascend.attention.attention_v1 import AscendAttentionBackend, AscendAttentionState
from vllm_ascend.attention.context_parallel.mla_cp import AscendMlaDCPImpl
from vllm_ascend.attention.mla_v1 import AscendMLABackend, AscendMLAImpl
from vllm_ascend.ops.triton.spec_decode.prepare_dflash_inputs import prepare_dflash_inputs_triton
from vllm_ascend.utils import vllm_version_is
from vllm_ascend.worker.v2.aclgraph_utils import _get_graph_update_backend
from vllm_ascend.worker.v2.attn_utils import build_attn_metadata_factory, build_attn_metadata_wrapper
from vllm_ascend.worker.v2.spec_decode.lmhead_tp_utils import LmheadTPDraftSamplingMixin
from vllm_ascend.worker.v2.spec_decode.pcp_utils import prepare_replicated_draft_profile_inputs


@contextmanager
def draft_metadata_request_count(speculator: Any, num_reqs: int) -> Iterator[None]:
    """Share the live request count and FIA lengths across builds within one proposal."""
    previous_count = getattr(speculator, "_draft_metadata_num_reqs", None)
    previous_lengths = getattr(speculator, "_proposal_host_lengths", None)
    speculator._draft_metadata_num_reqs = num_reqs
    speculator._proposal_host_lengths = {}
    try:
        yield
    finally:
        speculator._draft_metadata_num_reqs = previous_count
        speculator._proposal_host_lengths = previous_lengths


def prepare_draft_fia_lengths(
    speculator: Any, num_reqs: int, num_reqs_padded: int, *, window_step: int | None = None
) -> tuple[torch.Tensor, list[int] | None]:
    """Prepare rejection-adjusted KV lengths and, for a capped window, the valid Q prefix."""
    cache = getattr(speculator, "_proposal_host_lengths", None)
    key = (num_reqs, num_reqs_padded, window_step)
    if cache is not None and key in cache:
        return cache[key]

    kv_lens = torch.zeros(num_reqs_padded, dtype=torch.int32, device="cpu")
    live = kv_lens[:num_reqs]
    valid_queries = None
    if window_step is None:
        live.copy_(speculator.input_buffers.seq_lens[:num_reqs])
    else:
        # The clipped KV length alone cannot distinguish live query rows from padding.
        stride = speculator.num_speculative_steps
        live.copy_(speculator.sample_pos[: num_reqs * stride : stride])
        live.add_(window_step - 1)
        valid_queries = [max(0, window_step - max(0, end - speculator.max_model_len)) for end in live.tolist()]
        live.clamp_(max=speculator.max_model_len)

    result = kv_lens, valid_queries
    if cache is not None:
        cache[key] = result
    return result


def detect_draft_attn_architecture(attn_groups: Any) -> tuple[str | None, bool]:
    backend = _get_graph_update_backend(attn_groups)
    if issubclass(backend, AscendMLABackend):
        mla_groups = [
            group for groups in attn_groups for group in groups if issubclass(group.backend, AscendMLABackend)
        ]
        return "MLA", bool(mla_groups) and all(
            getattr(group.get_metadata_builder(0), "flashmla_state", None) is not None for group in mla_groups
        )
    if issubclass(backend, AscendAttentionBackend):
        return "GQA", False
    return None, False


def install_ascend_dflash_input_preparation() -> None:
    # DFlash and DSpark share the input-preparation symbol in the upstream module.
    if vllm_version_is("0.34.0"):
        dflash_speculator.prepare_dflash_inputs = prepare_dflash_inputs_triton
    else:
        dflash_speculator.prepare_dflash_inputs = _prepare_dflash_inputs_with_block_stride


def _prepare_dflash_inputs_with_block_stride(
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
    block_stride: int,
    *args: Any,
    **kwargs: Any,
) -> None:
    # Upstream main passes block_stride between block_size and cp_rank.
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
        *args,
        block_stride=block_stride,
        **kwargs,
    )


class AscendDFlashSpeculator(LmheadTPDraftSamplingMixin, DFlashSpeculator):
    _windowed_fia_layers: tuple[str, ...] = ()
    _has_windowed_flashmla = False

    if not vllm_version_is("0.33.0"):

        def supports_batch_sharded_draft_logits(self) -> bool:
            # NPU draft-logits kernels index the cache by request state.
            return False

    @draft_config_context()
    def load_draft_model(
        self,
        target_model: torch.nn.Module,
        target_attn_layer_names: set[str],
    ) -> torch.nn.Module:
        model = super().load_draft_model(target_model, target_attn_layer_names)
        # QuaRot targets require draft weights to be aligned after sharing
        # target layers, using the target Ascend configuration.
        if hasattr(model, "post_process"):
            model.post_process(self.vllm_config)
        return model

    def build_draft_attn_metadatas(self, num_reqs_padded, seq_lens_cpu_upper_bound):
        assert self.input_batch is not None
        num_reqs = getattr(self, "_draft_metadata_num_reqs", None)
        if num_reqs is None:
            num_reqs = self.input_batch.num_reqs
        assert 0 <= num_reqs <= num_reqs_padded, "draft requests exceed graph capacity"
        num_tokens_padded = num_reqs_padded * self.num_query_per_req
        # Core delegates the uniform build to the same _build_attn_metadata hook used in eager mode.
        batch_desc = BatchExecutionDescriptor(
            cg_mode=CUDAGraphMode.FULL, num_tokens=num_tokens_padded, num_reqs=num_reqs_padded
        )
        attn_metadata = self._build_uniform_attn_metadata(
            num_reqs=num_reqs,
            batch_desc=batch_desc,
            num_query_per_req=self.num_query_per_req,
            seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,
            step=self.num_query_per_req,
            causal=self._group_causal,
        )
        return [attn_metadata]

    def _build_attn_metadata(
        self,
        num_reqs: int,
        batch_desc: BatchExecutionDescriptor,
        query_start_loc_np: np.ndarray,
        seq_lens_cpu_upper_bound: torch.Tensor,
        step: int,
        causal: bool | Mapping[int, bool] = True,
        dcp_local_seq_lens: torch.Tensor | None = None,
    ) -> dict[str, Any] | None:
        """Build Ascend draft metadata through the upstream attention hook.

        Eager propose() reaches the builder through this hook with only
        ``build_attn_metadata_wrapper()`` outside, so the FIA factory
        context must live here; ``build_draft_attn_metadatas`` alone only
        covers FULL-graph replay. FIA needs rejection-adjusted host lengths.
        A windowed FlashMLA group consumes the device KV lengths directly
        and needs the valid Q prefix when a proposal hits the model limit.
        """
        num_reqs_padded = batch_desc.num_reqs or num_reqs
        # The draft forward may process padded inputs outside FULL graphs, but
        # attention only consumes the tokens owned by real requests there.
        num_tokens = batch_desc.num_tokens if batch_desc.cg_mode == CUDAGraphMode.FULL else int(query_start_loc_np[-1])
        metadata_context = nullcontext()
        valid_queries = None
        if self.attn_architecture == "MLA" and (not self.use_flashmla or self._has_windowed_flashmla):
            seq_lens_cpu = None
            draft_query_lens = None
            if not self.use_flashmla:
                capped_fia = self._windowed_fia_layers and self.draft_max_seq_len == self.max_model_len
                seq_lens_cpu, valid_queries = prepare_draft_fia_lengths(
                    self, num_reqs, num_reqs_padded, window_step=step if capped_fia else None
                )
            if self._has_windowed_flashmla and self.draft_max_seq_len == self.max_model_len:
                # First sample position is the (unclipped) first Q position + 1.
                # KV lengths are already clipped by prepare_dflash_inputs.
                stride = self.num_speculative_steps
                first_sample = self.sample_pos[: num_reqs * stride : stride]
                draft_query_lens = (self.max_model_len + 1 - first_sample).clamp_(0, step).to(torch.int32)
            metadata_context = build_attn_metadata_factory(
                self.input_buffers.positions,
                num_tokens,
                is_prefilling=self.draft_is_prefilling[:num_reqs_padded],
                seq_lens_cpu=seq_lens_cpu,
                attn_state=AscendAttentionState.ChunkedPrefill,
                parallel_config=self.attn_vllm_config.parallel_config,
                draft_query_lens=draft_query_lens,
            )
        with build_attn_metadata_wrapper(), metadata_context:
            attn_metadata = super()._build_attn_metadata(
                num_reqs=num_reqs,
                batch_desc=batch_desc,
                query_start_loc_np=query_start_loc_np,
                seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,
                step=step,
                causal=causal,
                dcp_local_seq_lens=dcp_local_seq_lens,
            )
        if batch_desc.cg_mode == CUDAGraphMode.FULL and attn_metadata is not None:
            # FULL replay uses the captured padded Q, so FIA needs cumulative
            # query lengths through every padded request slot. This covers
            # FIA-backed GQA and MLA drafts alike, as in DSpark.
            self._update_draft_attn_metadata(attn_metadata, num_reqs_padded)
        if attn_metadata and valid_queries is not None and any(count < step for count in valid_queries):
            self._split_window_query_padding(attn_metadata, valid_queries)
        return attn_metadata

    def _split_window_query_padding(self, attn_metadata, valid_queries: list[int]) -> None:
        """Give FIA the valid query prefix and a separate zero-KV sequence for its padding."""
        # TODO: Remove this split and _windowed_fia_layers once FIA device tiling
        # right-aligns the causal window to the valid Q prefix at max_model_len,
        # including fixed-width padded queries during FULL graph replay.
        width = self.num_query_per_req
        by_group = {}
        for name in self._windowed_fia_layers:
            metadata = attn_metadata[name]
            group = id(metadata)
            if group not in by_group:
                decode = metadata.decode
                query_ends, kv_lens, rows = [], [], []
                for req, end in enumerate(decode.actual_seq_lengths_q):
                    count = valid_queries[req] if req < len(valid_queries) else 0
                    if 0 < count < width:
                        query_ends.append(end - width + count)
                        kv_lens.append(decode.seq_lens_list[req])
                        rows.append(req)
                    query_ends.append(end)
                    kv_lens.append(decode.seq_lens_list[req] if count == width else 0)
                    rows.append(req if count == width else 0)
                # Layers in one attention group share metadata. Leave any other
                # consumer of that metadata on its original request layout.
                split = replace(
                    decode, actual_seq_lengths_q=query_ends, seq_lens_list=kv_lens, block_table=decode.block_table[rows]
                )
                by_group[group] = replace(metadata, decode=split)
            attn_metadata[name] = by_group[group]

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
        query_lens_list = [(i + 1) * self.num_query_per_req for i in range(num_reqs_padded)]
        for layer_name, metadata in attn_metadata.items():
            decode_metadata = getattr(metadata, "decode", None)
            if decode_metadata is not None:
                decode_metadata.actual_seq_lengths_q = query_lens_list
            else:
                metadata.actual_seq_lengths_q = query_lens_list
            # Backend subclasses can use different metadata contracts (e.g.
            # device-length CANN FlashAttn). Restrict KV trimming to FIA itself.
            if self.attn_backends[layer_name] is not AscendAttentionBackend:
                continue
            # The builder may append a synthetic request for the padded Q tail.
            # Uniform Q already covers that tail, so discard its extra KV row.
            metadata.seq_lens_list = metadata.seq_lens_list[:num_reqs_padded]
            metadata.seq_lens = metadata.seq_lens[:num_reqs_padded]
            metadata.seq_lens_cpu = metadata.seq_lens_cpu[:num_reqs_padded]
            if metadata.block_tables is not None:
                metadata.block_tables = metadata.block_tables[:num_reqs_padded]

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        super().__init__(vllm_config, device)
        self._lmhead_tp_validate_draft_sampling()

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
        get_layer_names = getattr(self.model, "get_draft_kv_cache_layer_names", None)
        get_causal = getattr(self.model, "get_draft_attn_causal", None)
        # Causality belongs to the model layer even when several layers share a KV group.
        draft_causal = dict(zip(get_layer_names(), get_causal())) if get_layer_names and get_causal else {}
        windowed_fia_layers: list[str] = []
        has_windowed_flashmla = False
        active_layer_names = self.draft_attn_layer_names
        for kv_cache_group_spec in kv_cache_config.kv_cache_groups:
            layer_names = kv_cache_group_spec.layer_names
            if active_layer_names is not None:
                layer_names = list(active_layer_names.intersection(layer_names))

            layer_type = cast(type[Any], AttentionLayerBase)
            attn_layers = get_layers_from_vllm_config(self.vllm_config, layer_type, layer_names)

            for layer_name in layer_names:
                layer = attn_layers[layer_name]
                attn_backends[layer_name] = layer.get_attn_backend()
                impl = getattr(layer, "impl", None)
                if layer_name not in draft_causal or not isinstance(impl, AscendMLAImpl) or impl.sliding_window is None:
                    continue
                if not draft_causal[layer_name]:
                    raise ValueError(f"Windowed MLA draft layers must be causal (layer {layer_name}).")
                if isinstance(impl, AscendMlaDCPImpl):
                    raise ValueError(f"Windowed MLA draft layer {layer_name} cannot use decode context parallelism.")
                if impl.use_flashmla:
                    has_windowed_flashmla = True
                else:
                    windowed_fia_layers.append(layer_name)

        self.attn_backends = attn_backends
        self._windowed_fia_layers = tuple(windowed_fia_layers)
        self._has_windowed_flashmla = has_windowed_flashmla
        install_ascend_dflash_input_preparation()
        self.attn_architecture, self.use_flashmla = detect_draft_attn_architecture(self.attn_groups)

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
        dp_sync_state: Any = None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
        mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
        is_profile: bool = False,
        num_speculative_tokens: int | None = None,
    ) -> torch.Tensor:
        sync_state = dp_sync_state
        if self.pcp_manager is not None:
            # The target's DP sync describes PCP-local requests. The replicated
            # draft consumes the restored global batch and must negotiate its
            # own capacity, including on idle/profile steps.
            sync_state = None
        if dummy_run and skip_attn_for_dummy_run:
            # Profiling runs the draft with its own query token count, which
            # can differ from the target batch. Let forward_context coordinate
            # the actual draft counts instead of reusing the target DP state.
            # TODO: Drop the sync_state reset once main2main includes upstream vLLM
            # #54856 (facd9a74a1), which resets the profiling DP counts.
            sync_state = None
            input_batch, last_hidden_states, aux_hidden_states = prepare_replicated_draft_profile_inputs(
                input_batch, last_hidden_states, aux_hidden_states, self.input_buffers
            )
        self.input_batch = input_batch
        # Upstream trims DP dummy draft requests before selecting its graph,
        # while input_batch still describes the padded target batch. Replay
        # metadata must use the same count without mutating the target batch.
        num_draft_reqs = input_batch.num_reqs
        if dummy_run and sync_state is not None:
            num_draft_reqs = min(num_draft_reqs, sync_state.num_reqs)
        with draft_metadata_request_count(self, num_draft_reqs), build_attn_metadata_wrapper():
            draft_tokens = super().propose(
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
                num_speculative_tokens=num_speculative_tokens,
            )
        return draft_tokens
