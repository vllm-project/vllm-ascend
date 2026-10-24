# Adapt from https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu/spec_decode/gemma4/speculator.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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
"""Ascend Gemma4 MTP speculator for Model Runner V2."""

from copy import copy

import torch.nn as nn
from vllm.config import VllmConfig, replace
from vllm.config.compilation import CUDAGraphMode
from vllm.logger import logger
from vllm.v1.worker.gpu.spec_decode.gemma4.speculator import Gemma4Speculator

from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.worker.v2.spec_decode.autoregressive.speculator import (
    AscendAutoRegressiveSpeculator,
)


class AscendGemma4Speculator(AscendAutoRegressiveSpeculator, Gemma4Speculator):
    """Recombines the two methods both bases define (draft config, draft load)."""

    def _prefill(
        self,
        num_reqs: int,
        num_tokens: int,
        attn_metadata,
        slot_mappings,
        num_tokens_across_dp=None,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
        mm_inputs=None,
        skip_sample: bool = False,
    ):
        if attn_metadata and any(
            metadata is not None and metadata.attn_state == AscendAttentionState.PrefillNoCache
            for metadata in attn_metadata.values()
        ):
            attn_metadata = {
                name: copy(metadata) if metadata is not None else None for name, metadata in attn_metadata.items()
            }
            for metadata in attn_metadata.values():
                if metadata is not None and metadata.attn_state == AscendAttentionState.PrefillNoCache:
                    metadata.attn_state = AscendAttentionState.PrefillCacheHit
        super()._prefill(
            num_reqs,
            num_tokens,
            attn_metadata,
            slot_mappings,
            num_tokens_across_dp,
            cudagraph_runtime_mode,
            mm_inputs,
            skip_sample,
        )

    def _create_draft_vllm_config(self) -> VllmConfig:
        draft_vllm_config = super()._create_draft_vllm_config()
        return replace(
            draft_vllm_config,
            parallel_config=replace(
                draft_vllm_config.parallel_config,
                prefill_context_parallel_size=1,
                enable_expert_parallel=False,
                enable_eplb=False,
            ),
        )

    def _setup_gemma4_kv_sharing(
        self,
        model: nn.Module,
        target_attn_layer_names: set[str],
    ) -> None:
        super()._setup_gemma4_kv_sharing(model, target_attn_layer_names)
        self._sync_kv_sharing_target_to_impl(model)

    def _sync_kv_sharing_target_to_impl(self, draft_model: nn.Module) -> None:
        """Propagate the late-bound KV-sharing target onto the Ascend attention impls."""
        synced = 0
        total = 0
        for layer in getattr(draft_model.model, "layers", []):
            attn = getattr(getattr(layer, "self_attn", None), "attn", None)
            if attn is None:
                continue
            total += 1
            target = getattr(attn, "kv_sharing_target_layer_name", None)
            impl = getattr(attn, "impl", None)
            if target is not None and impl is not None:
                impl.kv_sharing_target_layer_name = target
                synced += 1
        logger.info(
            "Gemma4 MTP: propagated KV-sharing target to %d/%d draft layers.",
            synced,
            total,
        )
