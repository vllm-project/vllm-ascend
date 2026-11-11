# Adapt from https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu/model_states/encoder_only.py
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

from typing import Any

import torch
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.model_states.encoder_only import EncoderOnlyModelState
from vllm.v1.worker.gpu.model_states.interface import ModelSpecificAttnMetadata
from vllm.v1.worker.utils import AttentionGroup

from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.utils import AscendCommonAttentionMetadata
from vllm_ascend.worker.v2.input_batch import AscendInputBatch


class AscendEncoderOnlyModelState(EncoderOnlyModelState):
    """Ascend adaptation of the encoder-only (BERT/BGE) model state.

    Encoder-only attention is non-causal and keeps no KV cache, so its layers
    return no ``get_kv_cache_spec`` and never enter
    ``kv_cache_config.kv_cache_groups``. The upstream state already builds the
    non-causal metadata for these layers; Ascend only has to feed its own
    attention metadata class instead of the generic upstream one, because
    ``AscendAttentionMetadataBuilder.build`` reads the Ascend-specific
    ``attn_state``/``causal`` fields.

    Graph mode is inherited from the runner: vLLM pins pooling models
    (``pooler_config is not None``) to ``cudagraph_mode=PIECEWISE``, where
    attention stays a splitting op that runs eagerly between the captured
    segments. That is the same shape as the Ascend V1 "aclgraph" path used by
    the pooling tests, so piecewise graphs keep working here. A FULL graph is
    neither requested nor supported: vLLM forces PIECEWISE for pooling, and the
    encoder attention kernel replays from static Python sequence lengths.
    """

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
        assert ubatch_idx == 0, "DBO is not supported on Ascend"
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
        # Keep the Ascend contract: the graph/offload machinery reads the latest
        # attention metadata from the model state.
        self.attn_metadata = attn_metadata
        return attn_metadata

    def _build_encoder_attn_metadata(
        self, input_batch: AscendInputBatch, cudagraph_mode: CUDAGraphMode, for_capture: bool
    ) -> dict[str, Any]:
        """Build non-causal encoder metadata with the Ascend metadata class."""
        if cudagraph_mode == CUDAGraphMode.FULL:
            num_reqs = input_batch.num_reqs_after_padding
            num_tokens = input_batch.num_tokens_after_padding
        else:
            num_reqs = input_batch.num_reqs
            num_tokens = input_batch.num_tokens
        max_query_len = int(input_batch.num_scheduled_tokens.max())
        if for_capture:
            max_seq_len = self.max_model_len
        else:
            max_seq_len = int(input_batch.seq_lens_cpu_upper_bound[:num_reqs].max())

        query_start_loc_cpu = torch.from_numpy(input_batch.query_start_loc_np[: num_reqs + 1])
        seq_lens_cpu = torch.from_numpy(input_batch.seq_lens_np)[:num_reqs]

        common_attn_metadata = AscendCommonAttentionMetadata(
            query_start_loc=input_batch.query_start_loc,
            query_start_loc_cpu=query_start_loc_cpu,
            seq_lens=input_batch.seq_lens[:num_reqs],
            seq_lens_cpu=seq_lens_cpu,
            seq_lens_cpu_upper_bound=input_batch.seq_lens_cpu_upper_bound[:num_reqs],
            num_reqs=num_reqs,
            num_actual_tokens=num_tokens,
            num_input_tokens=num_tokens,
            max_query_len=max_query_len,
            max_seq_len=max_seq_len,
            block_table_tensor=self._dummy_block_table[:num_reqs],
            slot_mapping=self._dummy_slot_mapping[:num_tokens],
            positions=input_batch.positions,
            # Encoder-only attention is bidirectional and uncached, which the
            # Ascend backend turns into a non-causal full self-attention kernel.
            attn_state=AscendAttentionState.PrefillNoCache,
            causal=False,
        )

        attn_metadata: dict[str, Any] = {}
        for group in self.encoder_attn_groups:
            builder = group.get_metadata_builder(0)
            if for_capture:
                metadata = builder.build_for_cudagraph_capture(common_attn_metadata)
            else:
                metadata = builder.build(common_prefix_len=0, common_attn_metadata=common_attn_metadata)
            for name in group.layer_names:
                attn_metadata[name] = metadata
        return attn_metadata
