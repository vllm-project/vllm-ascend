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
    """Gemma4 MTP speculator for Ascend NPUs.

    Reuses upstream ``Gemma4Speculator`` for draft construction, cross-model KV
    sharing wiring and dimension-mismatched embedding sharing, and layers the
    Ascend draft loop (``AscendAutoRegressiveSpeculator``) on top.

    Note on the MRO: ``AscendAutoRegressiveSpeculator`` precedes
    ``Gemma4Speculator``, so every method defined by both is taken from the
    Ascend base. Only ``_create_draft_vllm_config`` and ``load_draft_model`` are
    actually defined by both -- the rest of the upstream Gemma4 overrides
    (``advance_draft_positions``, ``_setup_gemma4_kv_sharing``,
    ``_share_embeddings``) are inherited untouched.
    """

    def _create_draft_vllm_config(self) -> VllmConfig:
        draft_vllm_config = super()._create_draft_vllm_config()

        # A Gemma4 draft only consumes backbone hidden states, so it stays dense
        # even when the target is MoE: reusing the target's expert flags would
        # make VllmConfig validate the draft as an expert model and fail.
        draft_vllm_config = replace(
            draft_vllm_config,
            parallel_config=replace(
                draft_vllm_config.parallel_config,
                prefill_context_parallel_size=1,
                enable_expert_parallel=False,
                enable_eplb=False,
            ),
        )

        # Gemma4 forces TRITON_ATTN on the target because of its heterogeneous
        # head dimensions (head_dim 256 sliding, 512 full). The base class resets
        # attention_config.backend for draft models, which would drop the sliding
        # draft layers onto a backend that cannot read the KV-shared cache.
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
        # AscendAutoRegressiveSpeculator.load_draft_model chains into
        # Gemma4Speculator.load_draft_model through super(), which builds the
        # draft and wires the cross-model KV sharing.
        draft_model = super().load_draft_model(target_model, target_attn_layer_names)
        self._sync_kv_sharing_target_to_impl(draft_model)
        return draft_model

    def _sync_kv_sharing_target_to_impl(self, draft_model: nn.Module) -> None:
        """Propagate late-bound KV-sharing targets to Ascend attention impls.

        ``Gemma4Speculator._setup_gemma4_kv_sharing`` sets
        ``kv_sharing_target_layer_name`` on the vLLM ``Attention`` wrapper after
        the draft model is built, while ``AscendAttentionBackendImpl`` snapshots
        that attribute when it is constructed and uses it to skip writing its own
        KV. Without this sync the draft would write its dummy K/V into the
        target's cache and every draft token would be rejected.
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
