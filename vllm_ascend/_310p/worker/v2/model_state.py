# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""310P-specific Model Runner V2 model state."""

from typing import Any

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache
from vllm.v1.worker.gpu.mm.encoder_runner import EncoderRunner
from vllm.v1.worker.utils import AttentionGroup

from vllm_ascend._310p.ops.rotary_embedding import prepare_mrope_cos_sin_slices_from_runner
from vllm_ascend._310p.worker.v2.rope import Ascend310PRopeState, get_310p_rope_state
from vllm_ascend._310p.worker.v2.sampler import Ascend310PGreedySampler
from vllm_ascend.worker.v2.input_batch import AscendInputBatch
from vllm_ascend.worker.v2.model_states.default import AscendModelState
from vllm_ascend.worker.v2.model_states.mamba_hybrid import AscendMambaHybridModelState


class _Ascend310PModelStateMixin:
    """310P RoPE and full-graph state shared by default and hybrid models."""

    def _init_310p_state(
        self,
        vllm_config: VllmConfig,
        model: nn.Module,
        encoder_cache: EncoderCache | None,
        device: torch.device,
    ) -> None:
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.scheduler_config = vllm_config.scheduler_config
        self.model = model
        self.device = device
        self.supports_mm_inputs = encoder_cache is not None
        self.max_model_len = self.model_config.max_model_len
        self.max_num_reqs = self.scheduler_config.max_num_seqs
        self.max_num_tokens = self.scheduler_config.max_num_batched_tokens
        self.inputs_embeds_size = self.model_config.get_inputs_embeds_size()
        self.dtype = self.model_config.dtype
        self._capture_seq_lens_by_ptr: dict[int, torch.Tensor] = {}

        if self.supports_mm_inputs:
            assert encoder_cache is not None
            self.encoder_cache = encoder_cache
            self.encoder_runner = EncoderRunner(
                model=self.model,
                max_num_tokens=self.max_num_tokens,
                hidden_size=self.inputs_embeds_size,
                encoder_cache=encoder_cache,
                dtype=self.dtype,
                device=self.device,
            )

        self._replace_310p_rope_state(encoder_cache)

    def _replace_310p_rope_state(self, encoder_cache: EncoderCache | None) -> None:
        self.rope_state = get_310p_rope_state(
            self.model_config,
            self.model,
            self.max_num_reqs,
            self.max_num_tokens,
            self.max_model_len,
            self.device,
        )
        try:
            from vllm.v1.worker.gpu.model_states.mm_pruning import maybe_create_mm_pruner
        except ImportError:
            self.mm_pruner = None
        else:
            self.mm_pruner = maybe_create_mm_pruner(
                self.model_config, self.model, self.rope_state, encoder_cache
            )

    def _record_capture_seq_lens(self, seq_lens: torch.Tensor) -> None:
        """Keep the largest captured view for each static tensor address."""
        data_ptr = seq_lens.data_ptr()
        recorded = self._capture_seq_lens_by_ptr.get(data_ptr)
        if recorded is None or seq_lens.numel() > recorded.numel():
            self._capture_seq_lens_by_ptr[data_ptr] = seq_lens

    def _refresh_capture_seq_lens(self, runtime_seq_lens: torch.Tensor) -> None:
        """Refresh every static seq-lens buffer before a FULL graph replay."""
        for capture_seq_lens in self._capture_seq_lens_by_ptr.values():
            num_seq_lens = min(capture_seq_lens.numel(), runtime_seq_lens.numel())
            capture_seq_lens[:num_seq_lens].copy_(runtime_seq_lens[:num_seq_lens], non_blocking=True)
            if num_seq_lens < capture_seq_lens.numel():
                capture_seq_lens[num_seq_lens:].zero_()

    def prepare_attn(
        self,
        input_batch: AscendInputBatch,
        cudagraph_mode: CUDAGraphMode,
        block_tables: tuple[torch.Tensor, ...],
        slot_mappings: torch.Tensor,
        attn_groups: list[list[AttentionGroup]],
        kv_cache_config: KVCacheConfig,
        for_capture: bool = False,
    ) -> dict[str, Any]:
        if for_capture:
            self._record_capture_seq_lens(input_batch.seq_lens)
        elif cudagraph_mode == CUDAGraphMode.FULL:
            self._refresh_capture_seq_lens(input_batch.seq_lens)

        attn_metadata = super().prepare_attn(
            input_batch,
            cudagraph_mode,
            block_tables,
            slot_mappings,
            attn_groups,
            kv_cache_config,
            for_capture=for_capture,
        )
        return attn_metadata

    def prepare_inputs(self, input_batch: AscendInputBatch, req_states):
        if self.rope_state is None:
            return super().prepare_inputs(input_batch, req_states)

        assert isinstance(self.rope_state, Ascend310PRopeState)
        self.rope_state.prepare_positions_cpu(
            input_batch.idx_mapping_np,
            input_batch.query_start_loc_np,
            req_states.prefill_len.np,
            req_states.num_computed_tokens_np,
            input_batch.num_tokens_after_padding,
        )
        positions = self.rope_state.get_positions(input_batch.num_tokens_after_padding)
        if self.model_config.uses_mrope:
            prepare_mrope_cos_sin_slices_from_runner(self, positions)
        return {"positions": positions}

    def custom_sampler(self, sampler):
        # Upstream GPUModelRunner.__init__ calls model_state.custom_sampler()
        # when the hook exists. NPUModelRunner310V2 also assigns
        # Ascend310PGreedySampler() on the runner; returning the same class
        # here keeps both call sites on the first-release greedy path.
        del sampler
        return Ascend310PGreedySampler(), None


class Ascend310PModelState(_Ascend310PModelStateMixin, AscendModelState):
    """310P state for standard attention models."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        model: nn.Module,
        encoder_cache: EncoderCache | None,
        device: torch.device,
    ) -> None:
        # Standard attention only needs the 310P RoPE/graph subset. Hybrid
        # models must call the full parent constructor first (see below).
        self._init_310p_state(vllm_config, model, encoder_cache, device)


class Ascend310PMambaHybridModelState(_Ascend310PModelStateMixin, AscendMambaHybridModelState):
    """310P state preserving hybrid/GDN behavior without upstream Triton RoPE."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        model: nn.Module,
        encoder_cache: EncoderCache | None,
        device: torch.device,
    ) -> None:
        # Initialize the complete upstream/Ascend hybrid contract first. This is
        # important because vLLM may add hybrid-only state such as `_align_mode`.
        AscendMambaHybridModelState.__init__(
            self, vllm_config, model, encoder_cache, device
        )
        self._capture_seq_lens_by_ptr = {}
        # The upstream RoPE state is safe to construct but its Triton position
        # preparation cannot run on 310P. Replace it before the first request.
        self._replace_310p_rope_state(encoder_cache)
