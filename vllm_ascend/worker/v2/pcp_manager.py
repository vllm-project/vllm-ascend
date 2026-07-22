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
from vllm.config import VllmConfig
from vllm.distributed.parallel_state import get_dcp_group, get_pcp_group
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.pcp_manager import PCPManager
from vllm.v1.worker.gpu.states import RequestState

from vllm_ascend.worker.v2.attn_utils import build_attn_state
from vllm_ascend.worker.v2.input_batch import AscendInputBatch


class AscendPCPManager(PCPManager):
    """PCP manager that refreshes Ascend-only local-batch metadata."""

    def __init__(self, *args, vllm_config: VllmConfig, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.vllm_config = vllm_config

    def partition_batch(self, input_batch: AscendInputBatch) -> AscendInputBatch:
        """Partition the batch and update Ascend-specific local metadata."""
        local_batch = super().partition_batch(input_batch)
        assert isinstance(local_batch, AscendInputBatch)

        local_seq_lens_np = (
            local_batch.num_computed_tokens_np + local_batch.num_scheduled_tokens
        )
        local_batch.seq_lens_np = local_seq_lens_np
        local_batch.attn_state = build_attn_state(
            self.vllm_config,
            local_seq_lens_np,
            local_batch.num_reqs,
            local_batch.num_scheduled_tokens,
            local_batch.num_scheduled_tokens,
        )
        return local_batch



def maybe_build_ascend_pcp_manager(
    vllm_config: VllmConfig,
    device: torch.device,
    supports_mm_inputs: bool,
    req_states: RequestState,
    block_tables: BlockTables,
) -> AscendPCPManager | None:
    """Build the Ascend PCP manager with community validation semantics."""
    parallel_config = vllm_config.parallel_config
    pcp_size = parallel_config.prefill_context_parallel_size
    if pcp_size <= 1:
        return None

    PCPManager.validate_config(vllm_config, supports_mm_inputs)
    dcp_size = parallel_config.decode_context_parallel_size
    return AscendPCPManager(
        pcp_world_size=pcp_size,
        pcp_rank=get_pcp_group().rank_in_group,
        device=device,
        req_states=req_states,
        max_num_reqs=vllm_config.scheduler_config.max_num_seqs,
        max_num_tokens=vllm_config.scheduler_config.max_num_batched_tokens,
        block_tables=block_tables,
        dcp_world_size=dcp_size,
        dcp_rank=get_dcp_group().rank_in_group if dcp_size > 1 else 0,
        cp_interleave=parallel_config.cp_kv_cache_interleave_size,
        vllm_config=vllm_config,
    )
