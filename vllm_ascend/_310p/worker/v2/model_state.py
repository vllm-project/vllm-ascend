# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""310P-specific Model Runner V2 model state."""

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache
from vllm.v1.worker.gpu.mm.encoder_runner import EncoderRunner

from vllm_ascend._310p.worker.v2.rope import Ascend310PRopeState, get_310p_rope_state
from vllm_ascend._310p.worker.v2.sampler import Ascend310PGreedySampler
from vllm_ascend.worker.v2.input_batch import AscendInputBatch
from vllm_ascend.worker.v2.model_states.default import AscendModelState


class Ascend310PModelState(AscendModelState):
    """Model state that avoids upstream Triton-backed multimodal RoPE."""

    def __init__(
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

        self.rope_state = get_310p_rope_state(
            self.model_config,
            model,
            self.max_num_reqs,
            self.max_num_tokens,
            self.max_model_len,
            device,
        )
        try:
            from vllm.v1.worker.gpu.model_states.mm_pruning import maybe_create_mm_pruner
        except ImportError:
            self.mm_pruner = None
        else:
            self.mm_pruner = maybe_create_mm_pruner(self.model_config, model, self.rope_state, encoder_cache)

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
        return {"positions": self.rope_state.get_positions(input_batch.num_tokens_after_padding)}

    def custom_sampler(self, sampler):
        del sampler
        return Ascend310PGreedySampler(), None
