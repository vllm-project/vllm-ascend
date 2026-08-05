# Adapt from https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu/block_table.py
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
import torch
from vllm.model_executor.triton_dispatcher import pluggable_kernel
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.worker.gpu.block_table import BlockTables


class AscendBlockTables(BlockTables):
    """Block table for Ascend NPUs."""

    def __init__(
        self,
        block_sizes: list[int],
        max_num_reqs: int,
        max_num_batched_tokens: int,
        max_num_blocks_per_group: list[int],
        device: torch.device,
        kernel_block_sizes: list[int] | None = None,
        cp_size: int = 1,
        cp_rank: int = 0,
        cp_interleave: int = 1,
    ):
        if kernel_block_sizes is None:
            kernel_block_sizes = block_sizes
        super().__init__(
            block_sizes,
            max_num_reqs,
            max_num_batched_tokens,
            max_num_blocks_per_group,
            device,
            kernel_block_sizes,
            cp_size,
            cp_rank,
            cp_interleave,
        )
        # because we will override these attribute, delete these attribute to
        # make sure it's collected by python gc immediately.
        del self.slot_mappings
        # vllm-ascend' reshape_and_cache function requires slot_mappings to be int32.
        # so we need to redefine slot_mappings to be int32.
        self.slot_mappings: torch.Tensor = torch.zeros(
            self.num_kv_cache_groups,
            self.max_num_batched_tokens,
            dtype=torch.int32,
            device=self.device,
        )

    def compute_slot_mappings(
        self,
        idx_mapping: torch.Tensor,
        query_start_loc: torch.Tensor,
        positions: torch.Tensor,
        num_tokens_padded: int,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        num_reqs = idx_mapping.shape[0]
        slot_mappings = self.slot_mappings if out is None else out
        for group_id, block_table in enumerate(self.block_tables):
            _compute_slot_mappings_kernel[(num_reqs + 1,)](
                slot_mappings.shape[1],
                idx_mapping,
                query_start_loc,
                positions,
                block_table.gpu,
                block_table.gpu.stride(0),
                self.kernel_block_sizes[group_id],
                slot_mappings[group_id],
                self.cp_rank,
                CP_SIZE=self.cp_size,
                CP_INTERLEAVE=self.cp_interleave,
                PAD_ID=PAD_SLOT_ID,
                TRITON_BLOCK_SIZE=1024,  # type: ignore
                TOTAL_BLOCK_SIZE=4096,
            )
        return slot_mappings[:, :num_tokens_padded]


@pluggable_kernel
@triton.jit
def _compute_slot_mappings_kernel(
    max_num_tokens,
    idx_mapping,  # [num_reqs]
    query_start_loc,  # [num_reqs + 1]
    pos,  # [num_tokens]
    block_table_ptr,  # [max_num_reqs, max_num_blocks]
    block_table_stride,
    block_size,
    slot_mapping_ptr,  # [max_num_tokens]
    cp_rank,
    CP_SIZE: tl.constexpr,
    CP_INTERLEAVE: tl.constexpr,
    PAD_ID: tl.constexpr,
    TRITON_BLOCK_SIZE: tl.constexpr,
    TOTAL_BLOCK_SIZE: tl.constexpr,
):
    batch_idx = tl.program_id(0)

    if batch_idx == tl.num_programs(0) - 1:
        actual_num_tokens = tl.load(query_start_loc + batch_idx)
        for i in range(actual_num_tokens, max_num_tokens, TRITON_BLOCK_SIZE):
            offset = i + tl.arange(0, TRITON_BLOCK_SIZE)
            tl.store(slot_mapping_ptr + offset, PAD_ID, mask=offset < max_num_tokens)
        return

    req_state_idx = tl.load(idx_mapping + batch_idx)
    start_idx = tl.load(query_start_loc + batch_idx)
    end_idx = tl.load(query_start_loc + batch_idx + 1)
    for i in range(start_idx, end_idx, TRITON_BLOCK_SIZE):
        offset = i + tl.arange(0, TRITON_BLOCK_SIZE)
        positions = tl.load(pos + offset, mask=offset < end_idx, other=0)

        # Type conversion of 'position' to int32 to be compatible with npu
        # otherwise, it will degrade to scalar computation
        positions = positions.to(tl.int32)
        block_indices = positions // (block_size * CP_SIZE)

        # block_offset = positions % (block_size * CP_SIZE)
        # The % operation on int32 type will degrade to scalar computation
        # replace the % operation with sub and mul instead
        block_offsets = positions - (block_size * CP_SIZE) * block_indices

        # The 'block_indics' variable results in non-contiguous memory assess,
        # which triggers degradation toscalar computation.
        # Mitigate this by loading the complete data block and extracting the required data with tl.gather
        block_numbers = tl.load(block_table_ptr + req_state_idx * block_table_stride + tl.arange(0, TOTAL_BLOCK_SIZE))
        block_numbers = block_numbers.to(tl.float32)
        block_numbers = tl.gather(block_numbers, block_indices, 0)

        if CP_SIZE == 1:
            # Common case: Context parallelism is not used.
            slot_ids = block_numbers * block_size + block_offsets
        else:
            # Context parallelism is used.
            is_local = block_offsets // CP_INTERLEAVE % CP_SIZE == cp_rank
            rounds = block_offsets // (CP_INTERLEAVE * CP_SIZE)
            remainder = block_offsets % CP_INTERLEAVE
            local_offsets = rounds * CP_INTERLEAVE + remainder
            slot_ids = block_numbers * block_size + local_offsets
            slot_ids = tl.where(is_local, slot_ids, PAD_ID)

        tl.store(slot_mapping_ptr + offset, slot_ids, mask=offset < end_idx)
