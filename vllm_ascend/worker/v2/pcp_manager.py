# Adapt from https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu/model_runner.py
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

import torch
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.pcp_manager import PCPManager
from vllm.v1.worker.gpu.states import RequestState

from vllm_ascend.worker.v2.attn_utils import build_attn_state
from vllm_ascend.worker.v2.input_batch import AscendInputBatch


class AscendPCPManager(PCPManager):
    """PCP manager that refreshes Ascend-only local-batch metadata."""

    @staticmethod
    def validate_config(
        vllm_config: VllmConfig,
        supports_mm_inputs: bool,
    ) -> None:
        """Validate Ascend MRV2 PCP and the supported PCP+DCP layouts."""
        parallel_config = vllm_config.parallel_config
        model_config = vllm_config.model_config
        if parallel_config.prefill_context_parallel_size <= 1:
            return

        PCPManager.validate_config(vllm_config, supports_mm_inputs)

        cudagraph_mode = vllm_config.compilation_config.cudagraph_mode
        is_sparse_mla = hasattr(getattr(model_config, "hf_text_config", None), "index_topk")
        pcp_size = parallel_config.prefill_context_parallel_size
        dcp_size = parallel_config.decode_context_parallel_size
        if dcp_size > 1:
            tp_size = parallel_config.tensor_parallel_size
            if dcp_size not in (pcp_size, tp_size * pcp_size):
                raise NotImplementedError("Ascend MRV2 PCP+DCP requires DCP to equal PCP or TP * PCP.")
            if is_sparse_mla:
                raise NotImplementedError("Ascend MRV2 PCP+DCP supports dense MLA only.")
            if cudagraph_mode != CUDAGraphMode.NONE:
                raise NotImplementedError("Ascend MRV2 PCP+DCP supports eager mode only. Set -cc.cudagraph_mode=NONE.")
            if getattr(parallel_config, "dcp_comm_backend", "ag_rs") == "a2a":
                raise NotImplementedError("Ascend MRV2 PCP+DCP does not support the A2A DCP backend.")

    def __init__(
        self,
        pcp_world_size: int,
        pcp_rank: int,
        device: torch.device,
        vllm_config: VllmConfig | None = None,
        req_states: RequestState | None = None,
        max_num_reqs: int | None = None,
        max_num_tokens: int | None = None,
        block_tables: BlockTables | None = None,
        dcp_world_size: int = 1,
        dcp_rank: int = 0,
        cp_interleave: int = 1,
    ) -> None:
        super().__init__(
            pcp_world_size,
            pcp_rank,
            device,
            req_states=req_states,
            max_num_reqs=max_num_reqs,
            max_num_tokens=max_num_tokens,
            block_tables=block_tables,
            dcp_world_size=dcp_world_size,
            dcp_rank=dcp_rank,
            cp_interleave=cp_interleave,
        )
        self.vllm_config = vllm_config

    def partition_batch(self, input_batch: AscendInputBatch) -> AscendInputBatch:
        """Partition the batch and update Ascend-specific local metadata."""
        assert self.vllm_config is not None
        local_batch = super().partition_batch(input_batch)
        assert isinstance(local_batch, AscendInputBatch)

        local_seq_lens_np = local_batch.num_computed_tokens_np + local_batch.num_scheduled_tokens
        local_batch.seq_lens_np = local_seq_lens_np
        local_batch.attn_state = build_attn_state(
            self.vllm_config,
            local_seq_lens_np,
            local_batch.num_reqs,
            local_batch.num_scheduled_tokens,
            local_batch.num_scheduled_tokens
            - (local_batch.num_draft_tokens_per_req if local_batch.num_draft_tokens_per_req is not None else 0),
        )
        return local_batch
