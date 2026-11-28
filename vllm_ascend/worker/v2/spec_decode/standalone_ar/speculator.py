# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.

from collections.abc import Mapping
from copy import copy
from typing import Any

import numpy as np
import torch
from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
from vllm.v1.worker.gpu.spec_decode.standalone_ar.speculator import StandaloneARSpeculator

from vllm_ascend.ascend_config import draft_config_context
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.compilation.updatable_graph import ContextSource
from vllm_ascend.worker.v2.aclgraph_utils import _get_graph_update_backend
from vllm_ascend.worker.v2.attn_utils import (
    build_attn_metadata_factory,
    build_attn_metadata_wrapper,
)
from vllm_ascend.worker.v2.input_batch import AscendInputBuffers
from vllm_ascend.worker.v2.spec_decode.standalone_ar.aclgraph import StandaloneARAclGraphManager


class AscendStandaloneARSpeculator(StandaloneARSpeculator):
    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        super().__init__(vllm_config, device)
        self.input_buffers = AscendInputBuffers(self.max_num_reqs, self.max_num_tokens, device)
        self._draft_attn_metadata: dict[str, Any] | None = None

    @property
    def draft_vllm_config(self) -> VllmConfig:
        return self._draft_vllm_config

    def supports_batch_sharded_draft_logits(self) -> bool:
        # NPU draft-logits kernels index the cache by request state.
        return False

    @draft_config_context()
    def load_draft_model(self, target_model: torch.nn.Module, target_attn_layer_names: set[str]) -> torch.nn.Module:
        # Fusion patterns infer shapes from registered attention layers. Keep
        # the drafter's compiler view separate from the target's layer shapes.
        target_config = self.vllm_config
        loading_config = copy(target_config)
        loading_config.compilation_config = copy(target_config.compilation_config)
        loading_config.compilation_config.static_forward_context = {}
        self.vllm_config = loading_config
        try:
            draft_model = super().load_draft_model(target_model, target_attn_layer_names)
        finally:
            self.vllm_config = target_config
        # Target-owned KV setup still discovers the loaded draft layers here.
        target_config.compilation_config.static_forward_context.update(
            self._draft_vllm_config.compilation_config.static_forward_context
        )
        return draft_model

    def init_cudagraph_manager(self, cudagraph_mode: CUDAGraphMode) -> None:
        if self.speculative_config.enforce_eager:
            cudagraph_mode = CUDAGraphMode.NONE
        self.prefill_cudagraph_manager = StandaloneARAclGraphManager(
            self.vllm_config,
            self.device,
            cudagraph_mode,
            self.vllm_config.get_decode_cudagraph_query_lens(self.num_speculative_steps + 2),
            varlen_decode=self.enable_adaptive_verification,
        )
        decode_mode = (
            CUDAGraphMode.FULL_DECODE_ONLY if cudagraph_mode.decode_mode() == CUDAGraphMode.FULL else CUDAGraphMode.NONE
        )
        self.decode_cudagraph_manager = StandaloneARAclGraphManager(
            self.vllm_config,
            self.device,
            decode_mode,
            decode_query_lens={1: self.max_num_reqs},
            decode_request_sizes=self.vllm_config.get_draft_decode_cudagraph_request_sizes(),
        )
        for manager in (self.prefill_cudagraph_manager, self.decode_cudagraph_manager):
            manager.speculator = self
            manager.update_stream = self.update_stream

    def set_attn(self, *args, **kwargs) -> None:
        super().set_attn(*args, **kwargs)
        self.attn_backend = _get_graph_update_backend(self.attn_groups)

    def build_draft_attn_metadatas(self, num_reqs_padded, num_tokens_padded, is_draft_model_prefill):
        assert self._draft_attn_metadata is not None
        return [self._draft_attn_metadata]

    def build_param_source(self, num_reqs_padded, draft_attn_metadata, is_draft_model_prefill, *, num_tokens=None):
        return ContextSource(draft_attn_metadata)

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
        num_tokens = int(query_start_loc_np[-1])
        positions = self.expanded_positions if step == 0 else self.input_buffers.positions
        attn_state = AscendAttentionState.DecodeOnly

        with (
            build_attn_metadata_wrapper(),
            build_attn_metadata_factory(
                positions,
                num_tokens,
                self.draft_is_prefilling[:num_reqs],
                attn_state=attn_state,
                parallel_config=self.vllm_config.parallel_config,
            ),
        ):
            metadata = super()._build_attn_metadata(
                num_reqs,
                batch_desc,
                query_start_loc_np,
                seq_lens_cpu_upper_bound,
                step,
                causal,
                dcp_local_seq_lens,
                slot_mappings,
            )
        self._draft_attn_metadata = metadata
        return metadata
