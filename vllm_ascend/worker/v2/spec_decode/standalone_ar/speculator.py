# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
from vllm.config import VllmConfig, replace, set_current_vllm_config
from vllm.config.compilation import CUDAGraphMode
from vllm.forward_context import BatchDescriptor, set_forward_context
from vllm.v1.kv_cache_interface import AttentionSpec, CrossAttentionSpec, UniformTypeKVCacheSpecs
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.spec_decode.standalone_ar.speculator import StandaloneARSpeculator

from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.device.hardware_profile import HardwareCapability, get_current_hardware_profile
from vllm_ascend.worker.v2.attn_utils import build_attn_metadata_factory, build_attn_metadata_wrapper
from vllm_ascend.worker.v2.spec_decode.lmhead_tp_utils import LmheadTPDraftSamplingMixin
from vllm_ascend.worker.v2.spec_decode.pcp_utils import disable_profiling_chunk_for_draft


class AscendStandaloneARSpeculator(LmheadTPDraftSamplingMixin, StandaloneARSpeculator):
    """Independent draft LM with exact FIA lengths and upstream eager drafting."""

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        if get_current_hardware_profile().supports(HardwareCapability.COMPATIBILITY_OP_IMPLEMENTATIONS):
            raise NotImplementedError("MRv2 draft_model is not supported on Ascend 310P.")
        parallel = vllm_config.parallel_config
        if (
            parallel.pipeline_parallel_size > 1
            or parallel.prefill_context_parallel_size > 1
            or parallel.decode_context_parallel_size > 1
        ):
            raise NotImplementedError("MRv2 draft_model currently requires PP=PCP=DCP=1 on Ascend.")
        assert vllm_config.speculative_config is not None
        if vllm_config.speculative_config.draft_model_config.is_moe:
            raise NotImplementedError("MRv2 draft_model currently requires a dense draft model on Ascend.")
        draft_model_config = vllm_config.speculative_config.draft_model_config
        if draft_model_config.is_encoder_decoder or draft_model_config.supports_multimodal_inputs:
            raise NotImplementedError("MRv2 draft_model requires a text-only decoder draft model on Ascend.")
        draft_parallel = vllm_config.speculative_config.draft_parallel_config
        if draft_parallel.tensor_parallel_size != parallel.tensor_parallel_size:
            raise NotImplementedError("MRv2 draft_model requires the same draft and target TP size on Ascend.")
        super().__init__(vllm_config, device)
        self._lmhead_tp_validate_draft_sampling()
        self._draft_attn_config = replace(
            vllm_config,
            model_config=self.draft_model_config,
            quant_config=None,
            parallel_config=replace(draft_parallel, rank=parallel.rank),
        )

    @property
    def attn_vllm_config(self) -> VllmConfig:
        # A separate LM can have different heads, dimensions and attention type.
        return self._draft_attn_config

    def set_attn(self, model_state, kv_cache_config, block_tables, target_input_buffers, target_attn_groups) -> None:
        for group in kv_cache_config.kv_cache_groups:
            for name in group.layer_names:
                if name not in self.draft_attn_layer_names:
                    continue
                spec = group.kv_cache_spec
                if isinstance(spec, UniformTypeKVCacheSpecs):
                    spec = spec.kv_cache_specs[name]
                if not isinstance(spec, AttentionSpec) or isinstance(spec, CrossAttentionSpec):
                    raise NotImplementedError("MRv2 draft_model requires an attention-only draft model on Ascend.")
        with set_current_vllm_config(self.attn_vllm_config):
            super().set_attn(model_state, kv_cache_config, block_tables, target_input_buffers, target_attn_groups)
        # Ascend's cache-write and slot-mapping kernels consume int32 slots.
        self.expanded_slot_mappings = torch.empty(
            block_tables.num_kv_cache_groups,
            self.max_num_tokens + self.max_num_reqs,
            dtype=block_tables.slot_mappings.dtype,
            device=self.device,
        )

    def load_draft_model(self, target_model: torch.nn.Module, target_attn_layer_names: set[str]) -> torch.nn.Module:
        with disable_profiling_chunk_for_draft(self.vllm_config):
            return super().load_draft_model(target_model, target_attn_layer_names)

    @torch.inference_mode()
    def _run_model(self, input_ids, positions, attn_metadata, slot_mappings, num_tokens_across_dp) -> torch.Tensor:
        num_tokens = input_ids.shape[0]
        self._prepare_eplb_forward(num_tokens)
        with (
            set_current_vllm_config(self.attn_vllm_config),
            set_forward_context(
                attn_metadata,
                self.attn_vllm_config,
                num_tokens=num_tokens,
                cudagraph_runtime_mode=CUDAGraphMode.NONE,
                num_tokens_across_dp=num_tokens_across_dp,
                slot_mapping=slot_mappings,
                batch_descriptor=BatchDescriptor(num_tokens=num_tokens),
            ),
        ):
            hidden_states = self.model(input_ids=input_ids, positions=positions)
        return hidden_states[0] if isinstance(hidden_states, tuple) else hidden_states

    def _prefill(
        self,
        input_batch: InputBatch,
        num_rejected: torch.Tensor,
        last_sampled: torch.Tensor,
        num_reqs: int,
        skip_attn: bool,
        num_tokens_across_dp: torch.Tensor | None,
    ) -> None:
        if not skip_attn:
            # FIA consumes CPU length lists. Copy once per proposal, not once
            # per draft step. The target CPU mirror may be an upper bound under
            # async scheduling, so derive exact lengths from the device buffers.
            lengths = torch.stack((input_batch.seq_lens[:num_reqs], num_rejected[:num_reqs].int())).cpu()
            self._target_seq_lens_cpu, self._num_rejected_cpu = lengths.unbind(0)
        return super()._prefill(input_batch, num_rejected, last_sampled, num_reqs, skip_attn, num_tokens_across_dp)

    def _build_attn_metadata(
        self,
        num_reqs: int,
        batch_desc: BatchExecutionDescriptor,
        query_start_loc_np: np.ndarray,
        seq_lens_cpu_upper_bound: torch.Tensor,
        step: int,
        causal: bool | Mapping[int, bool] = True,
        dcp_local_seq_lens: torch.Tensor | None = None,
        slot_mappings: torch.Tensor | None = None,
    ) -> dict[str, Any] | None:
        if step == 0:
            # Expanded prefill includes rejected placeholders; causal attention
            # aligns queries using target length + one correction slot.
            seq_lens_cpu = self._target_seq_lens_cpu + 1
            seq_lens_cpu_upper_bound = seq_lens_cpu
            positions = self.expanded_positions
            attn_state = AscendAttentionState.ChunkedPrefill
        else:
            # Upstream passes step + 1 to account for the correction token.
            seq_lens_cpu = self._target_seq_lens_cpu - self._num_rejected_cpu + step
            seq_lens_cpu_upper_bound = self._target_seq_lens_cpu - self._num_rejected_cpu
            positions = self.input_buffers.positions
            attn_state = AscendAttentionState.DecodeOnly
        seq_lens_cpu.clamp_(max=self.max_model_len)
        with (
            set_current_vllm_config(self.attn_vllm_config),
            build_attn_metadata_wrapper(),
            build_attn_metadata_factory(
                positions,
                batch_desc.num_tokens,
                torch.full((num_reqs,), step == 0, dtype=torch.bool),
                seq_lens_cpu=seq_lens_cpu,
                attn_state=attn_state,
                parallel_config=self.attn_vllm_config.parallel_config,
            ),
        ):
            return super()._build_attn_metadata(
                num_reqs,
                batch_desc,
                query_start_loc_np,
                seq_lens_cpu_upper_bound,
                step,
                causal,
                dcp_local_seq_lens,
                slot_mappings,
            )
