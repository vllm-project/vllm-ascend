# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""MRV2 model state for Ascend 310P."""

from typing import Any

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache
from vllm.v1.worker.utils import AttentionGroup

from vllm_ascend._310p.ops.rotary_embedding import prepare_mrope_cos_sin_slices_from_runner
from vllm_ascend._310p.worker.v2.rope import Ascend310PRopeState, get_310p_rope_state
from vllm_ascend.worker.v2.input_batch import AscendInputBatch
from vllm_ascend.worker.v2.model_states.default import AscendModelState

from .sampler import Ascend310PSampler


class Ascend310PModelState(AscendModelState):
    """Model state with Triton-free 310P sampler / MRoPE and encoder support."""

    # TODO: Refactor the sampler override to use Triton Dispatcher after vLLM
    # RFC #45133 lands.

    def __init__(
        self,
        vllm_config: VllmConfig,
        model: nn.Module,
        encoder_cache: EncoderCache | None,
        device: torch.device,
    ) -> None:
        # Initialize the full Ascend/DefaultModelState contract first so
        # attributes such as ``prompt_embeds_state`` / ``encoder_runner`` exist,
        # then swap RoPE to the Triton-free 310P implementation.
        AscendModelState.__init__(self, vllm_config, model, encoder_cache, device)
        # ACLGraph replays the tensor addresses bound during capture. Keep every
        # captured seq_lens buffer so its contents can be refreshed before replay.
        self._capture_seq_lens_by_ptr: dict[int, torch.Tensor] = {}
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
            self.mm_pruner = maybe_create_mm_pruner(self.model_config, self.model, self.rope_state, encoder_cache)

    def _record_capture_seq_lens(self, seq_lens: torch.Tensor) -> None:
        """Record the largest captured view for each physical buffer."""
        data_ptr = seq_lens.data_ptr()
        recorded = self._capture_seq_lens_by_ptr.get(data_ptr)
        if recorded is None or seq_lens.numel() > recorded.numel():
            self._capture_seq_lens_by_ptr[data_ptr] = seq_lens

    def _refresh_capture_seq_lens(self, runtime_seq_lens: torch.Tensor) -> None:
        """Copy runtime lengths to buffers read by FULL graph replay."""
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
            # Updating only input_batch.seq_lens is insufficient when replay is
            # bound to a different capture-time address.
            self._refresh_capture_seq_lens(input_batch.seq_lens)

        return super().prepare_attn(
            input_batch,
            cudagraph_mode,
            block_tables,
            slot_mappings,
            attn_groups,
            kv_cache_config,
            for_capture=for_capture,
        )

    def prepare_inputs(self, input_batch: AscendInputBatch, req_states):
        if self.rope_state is None:
            return super().prepare_inputs(input_batch, req_states)

        assert isinstance(self.rope_state, Ascend310PRopeState)
        # Upstream RopeState.prepare_positions uses Triton; 310P builds positions
        # on CPU from staged prefill tables, then H2D copies.
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
        del sampler
        return Ascend310PSampler(), None
