# SPDX-License-Identifier: Apache-2.0
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

from typing import Any

import torch
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.model_states.interface import ModelSpecificAttnMetadata
from vllm.v1.worker.utils import AttentionGroup

from vllm_ascend.worker.v2.input_batch import AscendInputBatch
from vllm_ascend.worker.v2.model_states.encoder_only import AscendEncoderOnlyModelState
from vllm_ascend.worker.v2.model_states.mamba_hybrid import AscendMambaHybridModelState


class AscendHybridEncoderModelState(AscendMambaHybridModelState, AscendEncoderOnlyModelState):
    """Mamba hybrid state for models that also declare ``ENCODER_ONLY`` layers.

    ColQwen3.5 interleaves GDN layers, whose conv/SSM state is cached as mamba
    KV, with bidirectional full-attention layers advertised as
    ``ENCODER_ONLY``. The encoder layers keep no KV cache and therefore never
    enter ``kv_cache_config.kv_cache_groups``, so the mamba hybrid base builds
    no metadata for them and their attention forward would look up its own
    layer name in ``attn_metadata`` and raise ``KeyError``.

    Both existing Ascend states are needed, so both are inherited: the mamba
    base supplies the cache lifecycle (``preprocess_state`` align pre-copy,
    ``initialize_kv_cache``, ``add_request``, ``finish_execution``) and the
    Ascend mamba/attention metadata; the encoder-only base supplies the
    non-causal metadata for the KV-cache-less layers, including the
    ``encoder_attn_groups`` built by ``EncoderOnlyModelState.__init__`` (which
    the cooperative ``__init__`` chain runs).

    Prompt embeddings stay unsupported, matching the state upstream selects for
    this model family (``EncoderOnlyModelState``).
    """

    supports_prompt_embeds = False

    def prepare_attn(
        self,
        input_batch: AscendInputBatch,
        cudagraph_mode: CUDAGraphMode,
        block_tables: tuple[torch.Tensor, ...],
        slot_mappings: torch.Tensor,
        attn_groups: list[list[AttentionGroup]],
        kv_cache_config: KVCacheConfig,
        for_capture: bool = False,
        ubatch_idx: int = 0,
        model_specific_attn_metadata: ModelSpecificAttnMetadata | None = None,
    ) -> dict[str, Any]:
        # The mamba base builds the KV-backed and mamba metadata without
        # chaining to super(), so the encoder-only groups are added here.
        attn_metadata = super().prepare_attn(
            input_batch,
            cudagraph_mode,
            block_tables,
            slot_mappings,
            attn_groups,
            kv_cache_config,
            for_capture,
            ubatch_idx,
            model_specific_attn_metadata,
        )
        attn_metadata.update(self._build_encoder_attn_metadata(input_batch, cudagraph_mode, for_capture))
        # Ascend reads the latest metadata off the state in the graph/offload
        # paths; keep it in sync with the merged dict.
        self.attn_metadata = attn_metadata
        return attn_metadata
