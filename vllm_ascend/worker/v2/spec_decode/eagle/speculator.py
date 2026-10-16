# Adapt from https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu/sample/spec_decode/eagle.py
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
from typing import Any

import torch
from vllm.config import VllmConfig, replace
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.spec_decode.eagle.speculator import EagleSpeculator

from vllm_ascend.worker.v2.spec_decode.autoregressive.speculator import (
    AscendAutoRegressiveSpeculator,
)


class AscendEagleSpeculator(AscendAutoRegressiveSpeculator, EagleSpeculator):
    """Ascend Eagle speculator: the NPU loop from AscendAutoRegressiveSpeculator
    layered on upstream EagleSpeculator (flat/GQA attention)."""

    def _create_draft_vllm_config(self) -> VllmConfig:
        # EAGLE draft models are dense even when the target is an MoE model.
        # The base swaps in the draft model config without re-validating it;
        # the draft-only settings on top turn EP/EPLB off (no experts).
        draft_vllm_config = super()._create_draft_vllm_config()
        draft_vllm_config.parallel_config = replace(
            draft_vllm_config.parallel_config,
            prefill_context_parallel_size=1,
            enable_expert_parallel=False,
            enable_eplb=False,
        )
        return draft_vllm_config

    def load_draft_model(
        self,
        target_model: torch.nn.Module,
        target_attn_layer_names: set[str],
    ) -> torch.nn.Module:
        """Inject the lmhead-TP static row capacity post-load: EAGLE draft
        models are generic model files that never pass ``lmhead_tp_capacity``
        to the head constructor themselves."""
        model = super().load_draft_model(target_model, target_attn_layer_names)
        self._lmhead_tp_inject_static_capacity(model)
        return model

    def _run_model(
        self,
        num_tokens: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
        mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the EAGLE draft backbone and publish the draft LM-head
        capacity before this step's head call. EAGLE drafts one query token
        per request per step and the head runs on exactly those rows, so
        the synced max query rows are the tightest group-uniform capacity
        (``tokens_per_req=1``)."""
        output = super()._run_model(
            num_tokens,
            attn_metadata,
            slot_mappings,
            num_tokens_across_dp,
            cudagraph_runtime_mode,
            mm_inputs,
        )
        self._lmhead_tp_publish_draft_capacity(num_tokens_across_dp, tokens_per_req=1)
        return output
