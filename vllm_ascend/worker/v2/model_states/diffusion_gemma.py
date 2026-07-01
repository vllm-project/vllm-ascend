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
"""Ascend-compatible DiffusionGemma model state.

``DiffusionGemmaModelState`` (in vllm/model_executor/models/diffusion_gemma.py)
was authored against upstream vLLM's attention-metadata API. On Ascend the V2
runner uses ``vllm_ascend.worker.v2.attn_utils.build_attn_metadata`` (different
signature, produces ``AscendMetadata``). This subclass keeps all of the
diffusion canvas/denoising logic and only re-implements ``prepare_attn`` so it:
  * builds Ascend attention metadata via the Ascend builder, and
  * stamps the per-request encoder(causal)/denoise(bidirectional) flag onto the
    resulting ``AscendMetadata.causal_per_req`` field, which the manual
    attention paths consume per request.
"""

from typing import Any

import torch
from vllm.config.compilation import CUDAGraphMode
from vllm.model_executor.models.diffusion_gemma import DiffusionGemmaModelState

from vllm_ascend.worker.v2.attn_utils import build_attn_metadata


class AscendDiffusionGemmaModelState(DiffusionGemmaModelState):
    """DiffusionGemma model state adapted to the Ascend V2 metadata API."""

    def prepare_attn(
        self,
        input_batch,
        cudagraph_mode,
        block_tables,
        slot_mappings,
        attn_groups,
        kv_cache_config,
        for_capture: bool = False,
    ) -> dict[str, Any]:
        if cudagraph_mode == CUDAGraphMode.FULL:
            num_reqs = input_batch.num_reqs_after_padding
            num_tokens = input_batch.num_tokens_after_padding
        else:
            num_reqs = input_batch.num_reqs
            num_tokens = input_batch.num_tokens

        query_start_loc_cpu = torch.from_numpy(input_batch.query_start_loc_np)
        max_query_len = int(input_batch.num_scheduled_tokens.max().item())

        # Per-request causal mode: encoder (commit) = causal, denoise =
        # bidirectional. Mirror DiffusionGemmaModelState.prepare_attn.
        actual_num_reqs = input_batch.num_reqs
        slots = input_batch.idx_mapping[:actual_num_reqs]
        self._causal_buf[:actual_num_reqs] = self.diffusion_states.is_encoder_phase[slots]
        if actual_num_reqs < num_reqs:
            self._causal_buf[actual_num_reqs:num_reqs] = False
        causal = self._causal_buf[:num_reqs]

        attn_metadata = build_attn_metadata(
            attn_groups=attn_groups,
            num_reqs=num_reqs,
            num_tokens=num_tokens,
            query_start_loc_gpu=input_batch.query_start_loc,
            query_start_loc_cpu=query_start_loc_cpu,
            max_query_len=max_query_len,
            seq_lens=input_batch.seq_lens,
            max_seq_len=self.max_model_len,
            block_tables=block_tables,
            slot_mappings=slot_mappings,
            kv_cache_config=kv_cache_config,
            dcp_local_seq_lens=getattr(input_batch, "dcp_local_seq_lens", None),
            seq_lens_np=input_batch.seq_lens_np,
            positions=input_batch.positions,
            attn_state=input_batch.attn_state,
            for_cudagraph_capture=for_capture,
        )
        self.attn_metadata = attn_metadata

        # Stamp the per-request bidirectional/causal flag onto each layer's
        # AscendMetadata.
        #
        # `causal` is a view of the PERSISTENT, fixed-address `_causal_buf`
        # (updated in-place above via slice assignment), so a captured FULL
        # graph that reads it at replay sees the current step's per-request
        # phases -- the same persistence contract as `_persist_seqlens` /
        # `seq_lens_device`. We stamp it onto the new `causal_per_req` field,
        # which both the eager and capturable manual-attention paths consume
        # per request (encoder/commit == causal, denoise == bidirectional).
        #
        # `md.causal` (scalar) is kept as a coarse fallback for any consumer
        # that does not read `causal_per_req`: mark causal only if ALL requests
        # are in the encoder phase (any bidirectional request otherwise
        # dominates the batch-wide value). Correctness for mixed batches comes
        # from `causal_per_req`, not this scalar.
        batch_causal = bool(causal.all().item()) if causal.numel() > 0 else True
        for md in attn_metadata.values():
            md.causal = batch_causal
            md.causal_per_req = causal

        return attn_metadata
