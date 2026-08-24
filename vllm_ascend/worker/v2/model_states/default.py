# Adapt from https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu/model_states/default.py
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

from typing import TYPE_CHECKING, Any
from zlib import adler32

import numpy as np
import torch
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.model_states.default import DefaultModelState
from vllm.v1.worker.utils import AttentionGroup

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.worker.v2.attn_utils import build_attn_metadata
from vllm_ascend.worker.v2.input_batch import AscendInputBatch

if TYPE_CHECKING:
    from vllm_ascend.worker.v2.pcp_manager import AscendPCPManager


class AscendModelState(DefaultModelState):
    """Model state for Ascend NPUs."""

    pcp_manager: "AscendPCPManager | None" = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sparse_kv_offload_enabled = get_ascend_config().sparse_kv_offload_config.enabled
        if self.sparse_kv_offload_enabled:
            self._offload_req_ids_cpu = torch.zeros(
                self.max_num_reqs,
                dtype=torch.int64,
                device="cpu",
                pin_memory=True,
            )
            self._offload_token_to_req_cpu = torch.zeros(
                self.max_num_tokens,
                dtype=torch.int32,
                device="cpu",
                pin_memory=True,
            )
            self._offload_req_ids_tensor = torch.zeros(
                self.max_num_reqs,
                dtype=torch.int64,
                device=self.device,
            )
            self._offload_token_to_req = torch.zeros(
                self.max_num_tokens,
                dtype=torch.int32,
                device=self.device,
            )

    def _prepare_sparse_kv_offload_metadata(
        self,
        input_batch: AscendInputBatch,
        num_reqs: int,
        num_tokens: int,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if not getattr(self, "sparse_kv_offload_enabled", False):
            return None, None

        actual_num_reqs = input_batch.num_reqs
        actual_num_tokens = input_batch.num_tokens
        req_ids_np = self._offload_req_ids_cpu.numpy()
        req_ids_np[:num_reqs].fill(0)
        req_ids_np[:actual_num_reqs] = np.asarray(
            [adler32(req_id.encode("utf-8")) for req_id in input_batch.req_ids],
            dtype=np.int64,
        )

        query_lens = np.diff(input_batch.query_start_loc_np[: actual_num_reqs + 1]).astype(np.int32, copy=False)
        token_to_req = np.repeat(
            np.arange(actual_num_reqs, dtype=np.int32),
            query_lens,
        )
        if token_to_req.shape[0] < actual_num_tokens:
            raise RuntimeError(
                "KV offload token_to_req metadata is shorter than the scheduled "
                f"token batch: metadata={token_to_req.shape[0]}, "
                f"tokens={actual_num_tokens}"
            )
        token_to_req_np = self._offload_token_to_req_cpu.numpy()
        token_to_req_np[:actual_num_tokens] = token_to_req[:actual_num_tokens]
        if num_tokens > actual_num_tokens:
            token_to_req_np[actual_num_tokens:num_tokens].fill(0)

        self._offload_req_ids_tensor[:num_reqs].copy_(
            self._offload_req_ids_cpu[:num_reqs],
            non_blocking=True,
        )
        self._offload_token_to_req[:num_tokens].copy_(
            self._offload_token_to_req_cpu[:num_tokens],
            non_blocking=True,
        )
        return (
            self._offload_req_ids_tensor[:num_reqs],
            self._offload_token_to_req[:num_tokens],
        )

    def prepare_attn(
        self,
        input_batch: AscendInputBatch,
        cudagraph_mode: CUDAGraphMode,
        block_tables: tuple[torch.Tensor, ...],
        slot_mappings: torch.Tensor,
        attn_groups: list[list[AttentionGroup]],
        kv_cache_config: KVCacheConfig,
        for_capture: bool = False,
    ) -> dict[str, Any]:
        """Override prepare_attn method because `build_attn_metadata` is different from vllm."""
        if cudagraph_mode == CUDAGraphMode.FULL:
            # Use padded sizes - padding is handled by model_runner.prepare_attn.
            num_reqs = input_batch.num_reqs_after_padding
        else:
            # Piecewise cudagraphs and eager use the actual request count.
            num_reqs = input_batch.num_reqs

        if cudagraph_mode == CUDAGraphMode.FULL or self.vllm_config.parallel_config.prefill_context_parallel_size > 1:
            # PCP pads each rank to the largest rank-local token count even
            # during eager prefill, so token-shaped metadata must match the
            # padded model input.
            num_input_tokens = input_batch.num_tokens_after_padding
        else:
            num_input_tokens = input_batch.num_tokens

        num_actual_reqs = input_batch.num_reqs
        num_actual_tokens = input_batch.num_tokens
        query_start_loc_cpu = torch.from_numpy(input_batch.query_start_loc_np)
        is_prefilling = torch.from_numpy(input_batch.is_prefilling_np)
        max_query_len = input_batch.num_scheduled_tokens.max().item()
        pcp_context = self.pcp_manager.build_attention_context() if self.pcp_manager is not None else None
        req_ids_tensor, token_to_req = self._prepare_sparse_kv_offload_metadata(
            input_batch,
            num_reqs,
            num_input_tokens,
        )
        # attn_metadata is needed when update_full_graph_params, but no way can get it now.
        # Temporarily store it in model_state.
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
            dcp_local_seq_lens=input_batch.dcp_local_seq_lens,
            # extra attributes for ascend npus.
            seq_lens_np=input_batch.seq_lens_np,
            positions=input_batch.positions,
            attn_state=input_batch.attn_state,
            pcp_context=pcp_context,
            for_cudagraph_capture=for_capture,
            req_ids_tensor=req_ids_tensor,
            token_to_req=token_to_req,
        )
        return self.attn_metadata
