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

from dataclasses import replace

import torch
from vllm.config import VllmConfig
from vllm.distributed.parallel_state import get_dcp_group, get_pcp_group
from vllm.utils.math_utils import round_up
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.pcp_manager import PCPManager
from vllm.v1.worker.gpu.states import RequestState

from vllm_ascend.utils import enable_sp
from vllm_ascend.worker.v2.attn_utils import build_attn_state
from vllm_ascend.worker.v2.input_batch import AscendInputBatch


class AscendPCPManager(PCPManager):
    """PCP manager that refreshes Ascend-only local-batch metadata."""

    def __init__(
        self,
        pcp_world_size: int,
        pcp_rank: int,
        device: torch.device,
        vllm_config: VllmConfig,
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

    def _pad_for_sequence_parallelism(self, local_batch: AscendInputBatch) -> AscendInputBatch:
        """Pad each PCP rank's token stride for FlashComm SP collectives.

        PCP=2 and TP=4 example:

            PCP gathered: [A B _ | C D E]
            SP gathered:  [A B _ _ | C D E _]

        The bar separates PCP ranks. SP rounds each rank's stride up to a
        TP-size multiple, so the gather and restore metadata must use the new
        stride.
        """
        tp_size = self.vllm_config.parallel_config.tensor_parallel_size
        if not enable_sp(self.vllm_config) or tp_size <= 1:
            return local_batch

        # Align the per-rank token stride for SP.
        pcp_padded_num_tokens = local_batch.num_tokens_after_padding
        sp_padded_num_tokens = round_up(pcp_padded_num_tokens, tp_size)
        if sp_padded_num_tokens == pcp_padded_num_tokens:
            return local_batch

        assert self._input_buffers is not None
        assert self._hidden_restore_idx is not None
        assert self._padded_gather_idx is not None
        assert self._gathered_kv_write_mask is not None
        input_buffers = self._input_buffers

        # Expand rank-major gather metadata to the new aligned stride.
        pcp_hidden_restore_idx = self._hidden_restore_idx
        pcp_padded_gather_idx = self._padded_gather_idx
        pcp_gathered_kv_write_mask = self._gathered_kv_write_mask
        num_expanded_tokens = sp_padded_num_tokens * self.pcp_world_size
        sp_padded_gather_idx = pcp_padded_gather_idx.new_zeros(num_expanded_tokens)
        sp_gathered_kv_write_mask = pcp_gathered_kv_write_mask.new_zeros(num_expanded_tokens)
        for rank in range(self.pcp_world_size):
            pcp_rank_start = rank * pcp_padded_num_tokens
            sp_rank_start = rank * sp_padded_num_tokens
            sp_padded_gather_idx[sp_rank_start : sp_rank_start + pcp_padded_num_tokens].copy_(
                pcp_padded_gather_idx[pcp_rank_start : pcp_rank_start + pcp_padded_num_tokens]
            )
            sp_gathered_kv_write_mask[sp_rank_start : sp_rank_start + pcp_padded_num_tokens].copy_(
                pcp_gathered_kv_write_mask[pcp_rank_start : pcp_rank_start + pcp_padded_num_tokens]
            )

        # Rebase restore indices from the PCP stride to the SP-aligned stride.
        restore_pcp_rank = torch.div(
            pcp_hidden_restore_idx,
            pcp_padded_num_tokens,
            rounding_mode="floor",
        )
        self._hidden_restore_idx = restore_pcp_rank * sp_padded_num_tokens + torch.remainder(
            pcp_hidden_restore_idx, pcp_padded_num_tokens
        )
        self._padded_gather_idx = sp_padded_gather_idx
        self._gathered_kv_write_mask = sp_gathered_kv_write_mask

        # Initialize the appended local tokens as padding and return the aligned
        # batch view.
        input_buffers.input_ids[pcp_padded_num_tokens:sp_padded_num_tokens].zero_()
        input_buffers.positions[pcp_padded_num_tokens:sp_padded_num_tokens].zero_()
        input_buffers.is_padding[pcp_padded_num_tokens:sp_padded_num_tokens].fill_(True)
        return replace(
            local_batch,
            num_tokens_after_padding=sp_padded_num_tokens,
            input_ids=input_buffers.input_ids[:sp_padded_num_tokens],
            positions=input_buffers.positions[:sp_padded_num_tokens],
            is_padding=input_buffers.is_padding[:sp_padded_num_tokens],
        )

    def partition_batch(self, input_batch: AscendInputBatch) -> AscendInputBatch:
        """Partition the batch and update Ascend-specific local metadata."""
        local_batch = super().partition_batch(input_batch)
        assert isinstance(local_batch, AscendInputBatch)
        local_batch = self._pad_for_sequence_parallelism(local_batch)

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

    AscendPCPManager.validate_config(vllm_config, supports_mm_inputs)
    dcp_size = parallel_config.decode_context_parallel_size
    return AscendPCPManager(
        pcp_world_size=pcp_size,
        pcp_rank=get_pcp_group().rank_in_group,
        device=device,
        vllm_config=vllm_config,
        req_states=req_states,
        max_num_reqs=vllm_config.scheduler_config.max_num_seqs,
        max_num_tokens=vllm_config.scheduler_config.max_num_batched_tokens,
        block_tables=block_tables,
        dcp_world_size=dcp_size,
        dcp_rank=get_dcp_group().rank_in_group if dcp_size > 1 else 0,
        cp_interleave=parallel_config.cp_kv_cache_interleave_size,
    )
