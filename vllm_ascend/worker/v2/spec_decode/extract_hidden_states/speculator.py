# Adapt from https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu/spec_decode/extract_hidden_states.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#
"""Ascend Model Runner V2 adaptation of extract_hidden_states speculation.

Mirrors upstream vLLM PR #49811 (ExtractHiddenStatesSpeculator) so Ascend can
dispatch the method through NPUModelRunner V2. Vendored because the currently
pinned upstream release may not yet expose the GPU module.
"""

from typing import Any

import torch
import torch.nn as nn
from vllm.compilation.backends import set_model_tag
from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.forward_context import set_forward_context
from vllm.model_executor.model_loader import get_model
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.spec_decode.speculator import DraftModelSpeculator


class AscendExtractHiddenStatesSpeculator(DraftModelSpeculator):
    """Cache target hidden states while returning always-accepted draft tokens."""

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        assert vllm_config.speculative_config is not None
        if vllm_config.speculative_config.draft_sample_method != "greedy":
            raise ValueError("extract_hidden_states only supports draft_sample_method='greedy'")
        super().__init__(vllm_config, device)

        if self.num_speculative_steps != 1:
            raise ValueError("extract_hidden_states requires num_speculative_tokens to be 1")
        if self.speculative_config.disable_padded_drafter_batch:
            raise ValueError("disable_padded_drafter_batch is not supported with extract_hidden_states method")

        self.supports_mm_inputs = False
        layer_ids = getattr(
            self.draft_model_config.hf_config,
            "eagle_aux_hidden_state_layer_ids",
            None,
        )
        if not layer_ids:
            raise ValueError(
                "eagle_aux_hidden_state_layer_ids must be set in the draft "
                "model config for extract_hidden_states method"
            )

        self.num_hidden_states = len(layer_ids)
        assert isinstance(self.dtype, torch.dtype)
        self.hidden_states = torch.zeros(
            self.max_num_tokens,
            self.num_hidden_states,
            self.vllm_config.model_config.get_hidden_size(),
            dtype=self.dtype,
            device=device,
        )
        # Ascend NPUModelRunner assigns update_stream for draft graph managers;
        # this method does not capture ACL graphs, but keep the attribute.
        self.update_stream = None

    def load_draft_model(
        self,
        target_model: nn.Module,
        target_attn_layer_names: set[str],
    ) -> nn.Module:
        del target_model, target_attn_layer_names
        with set_model_tag("extract_hidden_states"):
            return get_model(
                vllm_config=self.vllm_config,
                model_config=self.draft_model_config,
            )

    def load_model(self, target_model: nn.Module) -> None:
        super().load_model(target_model)
        if len(self.draft_attn_layer_names) != 1:
            raise ValueError(
                "ExtractHiddenStatesModel should have exactly one attention "
                f"layer, found {len(self.draft_attn_layer_names)}"
            )

    def init_cudagraph_manager(self, cudagraph_mode: CUDAGraphMode) -> None:
        del cudagraph_mode

    def capture(self) -> None:
        return None

    @torch.inference_mode()
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
        num_tokens_across_dp: torch.Tensor | None = None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
        mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
        is_profile: bool = False,
    ) -> torch.Tensor:
        del (
            last_hidden_states,
            num_sampled,
            num_rejected,
            next_prefill_tokens,
            temperature,
            seeds,
            dummy_run,
            mm_inputs,
            is_profile,
        )

        draft_tokens = last_sampled[input_batch.idx_mapping, :1]
        if skip_attn_for_dummy_run:
            return draft_tokens
        if aux_hidden_states is None:
            raise ValueError("aux_hidden_states are required when using extract_hidden_states")
        if len(aux_hidden_states) != self.num_hidden_states:
            raise ValueError(
                f"Expected {self.num_hidden_states} auxiliary hidden states, got {len(aux_hidden_states)}"
            )

        stacked_hidden_states = torch.stack(aux_hidden_states, dim=1)
        num_tokens = stacked_hidden_states.shape[0]
        self.hidden_states[:num_tokens].copy_(stacked_hidden_states)

        draft_attn_metadata = {name: attn_metadata[name] for name in self.draft_attn_layer_names}
        draft_slot_mappings = {name: slot_mappings[name][:num_tokens] for name in self.draft_attn_layer_names}
        with set_forward_context(
            draft_attn_metadata,
            self.vllm_config,
            num_tokens=num_tokens,
            num_tokens_across_dp=num_tokens_across_dp,
            cudagraph_runtime_mode=CUDAGraphMode.NONE,
            slot_mapping=draft_slot_mappings,
            is_padding=input_batch.is_padding[:num_tokens],
        ):
            self.model(hidden_states=self.hidden_states[:num_tokens])

        return draft_tokens
