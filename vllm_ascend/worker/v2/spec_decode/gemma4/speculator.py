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

import torch.nn as nn
from vllm.config import VllmConfig, replace
from vllm.logger import init_logger
from vllm.v1.worker.gpu.spec_decode.gemma4.speculator import Gemma4Speculator

from vllm_ascend.worker.v2.spec_decode.autoregressive.speculator import (
    AscendAutoRegressiveSpeculator,
)

logger = init_logger(__name__)


class AscendGemma4Speculator(AscendAutoRegressiveSpeculator, Gemma4Speculator):
    """``_create_draft_vllm_config`` and ``load_draft_model`` are the only
    methods defined by both bases, so this subclass recombines them."""

    def _create_draft_vllm_config(self) -> VllmConfig:
        draft_vllm_config = super()._create_draft_vllm_config()
        # The draft is dense even for a MoE target, and Gemma4's heterogeneous
        # head dimensions require the target's forced attention backend.
        draft_vllm_config = replace(
            draft_vllm_config,
            parallel_config=replace(
                draft_vllm_config.parallel_config,
                prefill_context_parallel_size=1,
                enable_expert_parallel=False,
                enable_eplb=False,
            ),
        )
        target_backend = self.vllm_config.attention_config.backend
        if target_backend is None:
            return draft_vllm_config
        return replace(
            draft_vllm_config,
            attention_config=replace(
                draft_vllm_config.attention_config,
                backend=target_backend,
            ),
        )

    def load_draft_model(
        self,
        target_model: nn.Module,
        target_attn_layer_names: set[str],
    ) -> nn.Module:
        # The Ascend base chains into Gemma4Speculator.load_draft_model.
        draft_model = super().load_draft_model(target_model, target_attn_layer_names)
        self._sync_kv_sharing_target_to_impl(draft_model)
        return draft_model

    def _sync_kv_sharing_target_to_impl(self, draft_model: nn.Module) -> None:
        """Copy the late-bound KV-sharing target onto the Ascend attention impls.

        ``AscendAttentionBackendImpl`` snapshots ``kv_sharing_target_layer_name``
        when it is constructed and uses it to skip writing its own KV, but
        ``_setup_gemma4_kv_sharing`` sets that attribute afterwards, on the vLLM
        ``Attention`` wrapper.
        """
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
