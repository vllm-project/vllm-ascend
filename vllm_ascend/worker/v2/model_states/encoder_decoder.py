# Adapt from https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu/model_states/encoder_decoder.py
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

from typing import Any

import torch
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.model_states.encoder_decoder import (
    EncoderDecoderAttnMetadata,
    EncoderDecoderModelState,
)
from vllm.v1.worker.utils import AttentionGroup

from vllm_ascend.worker.v2.attn_utils import build_attn_metadata
from vllm_ascend.worker.v2.input_batch import AscendInputBatch


class AscendEncoderDecoderModelState(EncoderDecoderModelState):
    """Encoder-decoder state using Ascend attention metadata."""

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
    ) -> dict[str, Any]:
        assert ubatch_idx == 0, "DBO is not supported"

        # Preserve upstream FULL-mode semantics for future Ascend support.
        if cudagraph_mode == CUDAGraphMode.FULL:
            num_reqs = input_batch.num_reqs_after_padding
            num_input_tokens = input_batch.num_tokens_after_padding
        else:
            num_reqs = input_batch.num_reqs
            num_input_tokens = input_batch.num_tokens

        num_actual_reqs = input_batch.num_reqs
        num_actual_tokens = input_batch.num_tokens
        query_start_loc_cpu = torch.from_numpy(input_batch.query_start_loc_np)
        is_prefilling = torch.from_numpy(input_batch.is_prefilling_np)
        max_query_len = input_batch.num_scheduled_tokens.max().item()
        encoder_decoder_metadata = EncoderDecoderAttnMetadata(
            self._get_encoder_seq_lens(
                input_batch.req_ids,
                attn_groups,
                for_capture,
                num_reqs,
            )
        )

        self.attn_metadata = build_attn_metadata(
            attn_groups=attn_groups,
            num_reqs=num_reqs,
            num_actual_reqs=num_actual_reqs,
            num_tokens=num_input_tokens,
            num_actual_tokens=num_actual_tokens,
            num_input_tokens=num_input_tokens,
            is_prefilling=is_prefilling,
            query_start_loc_gpu=input_batch.query_start_loc,
            query_start_loc_cpu=query_start_loc_cpu,
            max_query_len=max_query_len,
            seq_lens=input_batch.seq_lens,
            max_seq_len=self.max_model_len,
            block_tables=block_tables,
            slot_mappings=slot_mappings,
            kv_cache_config=kv_cache_config,
            seq_lens_cpu_upper_bound=input_batch.seq_lens_cpu_upper_bound,
            dcp_local_seq_lens=input_batch.dcp_local_seq_lens,
            parallel_config=self.vllm_config.parallel_config,
            seq_lens_np=input_batch.seq_lens_np,
            positions=input_batch.positions,
            attn_state=input_batch.attn_state,
            model_specific_attn_metadata=encoder_decoder_metadata,
            for_cudagraph_capture=for_capture,
        )
        return self.attn_metadata
