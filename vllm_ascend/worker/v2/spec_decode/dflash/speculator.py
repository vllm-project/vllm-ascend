# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Mapping
from contextlib import nullcontext
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
from vllm_ascend.attention.mla_v1 import AscendMLAImpl
from vllm_ascend.ops.triton.spec_decode.prepare_dflash_inputs import prepare_dflash_inputs_triton
from vllm_ascend.utils import vllm_version_is
from vllm_ascend.worker.v2.attn_utils import build_attn_metadata_factory, build_attn_metadata_wrapper
from vllm_ascend.worker.v2.spec_decode.host_lengths import (
    detect_draft_attn_architecture,
    draft_metadata_request_count,
    prepare_draft_host_lengths,
    proposal_host_lengths,
)
from vllm_ascend.worker.v2.spec_decode.lmhead_tp_utils import LmheadTPDraftSamplingMixin
from vllm_ascend.worker.v2.spec_decode.pcp_utils import prepare_replicated_draft_profile_inputs




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


def validate_grouped_draft_attn(model: Any, kv_cache_config: Any, attn_layers: Mapping[str, Any]) -> None:
    """Reject draft layer modes the per-group metadata cannot express.

    The pinned upstream DFlash speculator stores draft causality per KV cache
    group (``_group_causal``), so the last layer's flag wins for the whole
    group; fail fast on mixed causality instead of silently corrupting draft
    attention. Windowed MLA layers must be causal (the FIA pre_tokens band
    is applied only in the causal decode branch) and must not use
    ``AscendMlaDCPImpl``, which passes no band to FIA; it is the same
    restriction upstream vLLM #68214 encodes by refusing DCP for windowed
    DFlash2 drafts. Models without the per-layer hooks keep the upstream
    per-group fallback unchanged.
    """
    get_layer_names = getattr(model, "get_draft_kv_cache_layer_names", None)
    get_causal = getattr(model, "get_draft_attn_causal", None)
    if get_layer_names is None or get_causal is None:
        return
    name_to_gid = {
        layer_name: gid for gid, group in enumerate(kv_cache_config.kv_cache_groups) for layer_name in group.layer_names
    }
    causal_by_gid: dict[int, bool] = {}
    for layer_name, causal in zip(get_layer_names(), get_causal()):
        gid = name_to_gid.get(layer_name)
        if gid is not None:
            causal = bool(causal)
            if gid in causal_by_gid and causal_by_gid[gid] != causal:
                raise ValueError(
                    "DFlash draft layers with mixed attention causality share one "
                    f"KV cache group (group {gid}); the per-group causal flag "
                    "would keep only the last layer's mode and corrupt draft "
                    "attention. Split the causal modes into separate KV cache "
                    "groups or use a checkpoint with homogeneous draft causality."
                )
            causal_by_gid[gid] = causal
        impl = getattr(attn_layers.get(layer_name), "impl", None)
        if not isinstance(impl, AscendMLAImpl) or getattr(impl, "sliding_window", None) is None:
            continue
        if not causal:
            raise ValueError(
                "Windowed MLA draft layers must be causal: the Ascend MLA decode "
                "path applies the sliding-window band only in the causal branch "
                "(FIA sparse_mode=4), so a non-causal windowed draft would "
                f"silently attend the full context (layer {layer_name}). Use a "
                "causal windowed draft, or a non-windowed draft for bidirectional "
                "attention."
            )
        if isinstance(impl, AscendMlaDCPImpl):
            raise ValueError(
                "DFlash draft layers carry a sliding window, but decode context "
                "parallelism routes MLA draft attention through AscendMlaDCPImpl, "
                "which applies no pre_tokens band and would silently drop the "
                "window onto the full DCP-local KV. Run the draft without DCP "
                "(or use a non-windowed draft); this matches upstream vLLM "
                "#68214 refusing DCP for windowed DFlash2 drafts."
            )


class AscendDFlashSpeculator(LmheadTPDraftSamplingMixin, DFlashSpeculator):
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
        # MLA drafts share DSpark's metadata preparation: the factory feeds
        # the rejection-adjusted host lengths and pins
        # attn_state=ChunkedPrefill, so the eager build also takes the
        # sliding-window FIA branch (sparse_mode=4) instead of defaulting
        # attn_state to None and silently dropping the window.
        metadata_context = nullcontext()
        if self.attn_architecture == "MLA" and not self.use_flashmla:
            metadata_context = build_attn_metadata_factory(
                self.input_buffers.positions,
                num_tokens_padded,
                is_prefilling=torch.zeros(num_reqs_padded, dtype=torch.bool, device="cpu"),
                seq_lens_cpu=prepare_draft_host_lengths(self, num_reqs, num_reqs_padded, self.num_query_per_req),
                attn_state=AscendAttentionState.ChunkedPrefill,
                parallel_config=self.attn_vllm_config.parallel_config,
            )
        with build_attn_metadata_wrapper(), metadata_context:
            # vLLM main (#56181) replaced _build_draft_attn_metadata with
            # _build_uniform_attn_metadata (BatchExecutionDescriptor).
            batch_desc = BatchExecutionDescriptor(
                cg_mode=CUDAGraphMode.FULL,
                num_tokens=num_tokens_padded,
                num_reqs=num_reqs_padded,
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
        covers FULL-graph replay. Only FIA-backed MLA drafts enter the
        factory (``attn_state=ChunkedPrefill`` plus the rejection-adjusted
        host lengths); GQA and FlashMLA drafts keep the upstream builder
        inputs and consume the device lengths directly.
        """
        num_reqs_padded = batch_desc.num_reqs or num_reqs
        # The draft forward may process padded inputs outside FULL graphs, but
        # attention only consumes the tokens owned by real requests there.
        num_tokens = (
            batch_desc.num_tokens if batch_desc.cg_mode == CUDAGraphMode.FULL else int(query_start_loc_np[-1])
        )
        metadata_context = nullcontext()
        if self.attn_architecture == "MLA" and not self.use_flashmla:
            metadata_context = build_attn_metadata_factory(
                self.input_buffers.positions,
                num_tokens,
                is_prefilling=torch.zeros(num_reqs_padded, dtype=torch.bool, device="cpu"),
                seq_lens_cpu=prepare_draft_host_lengths(self, num_reqs, num_reqs_padded, step),
                attn_state=AscendAttentionState.ChunkedPrefill,
                parallel_config=self.attn_vllm_config.parallel_config,
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
        return attn_metadata

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
        resolved_layers: dict[str, Any] = {}
        active_layer_names = self.draft_attn_layer_names
        for kv_cache_group_spec in kv_cache_config.kv_cache_groups:
            layer_names = kv_cache_group_spec.layer_names
            if active_layer_names is not None:
                layer_names = list(active_layer_names.intersection(layer_names))

            layer_type = cast(type[Any], AttentionLayerBase)
            attn_layers = get_layers_from_vllm_config(self.vllm_config, layer_type, layer_names)

            for layer_name in layer_names:
                resolved_layers[layer_name] = attn_layers[layer_name]
                attn_backends[layer_name] = attn_layers[layer_name].get_attn_backend()

        self.attn_backends = attn_backends
        install_ascend_dflash_input_preparation()
        self.attn_architecture, self.use_flashmla = detect_draft_attn_architecture(self.attn_groups)
        validate_grouped_draft_attn(self.model, kv_cache_config, resolved_layers)

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
        with proposal_host_lengths(self, num_draft_reqs), build_attn_metadata_wrapper():
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
