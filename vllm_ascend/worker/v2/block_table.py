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
from vllm.triton_utils import triton
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.worker.gpu.block_table import BlockTables

from vllm_ascend.ops.triton.v2.block_table.compute_slot_mappings import (
    _compute_slot_mappings_kernel,
)

# Staging a complete block-table row gives substantially faster contiguous GM
# access on Ascend, but the staged fp32 row must fit in UB together with the
# per-token temporaries. PR #15212 validated rows through 16K entries on A3.
# Larger rows use the direct-load path in the same kernel to keep compilation
# resource usage bounded.
_MAX_STAGED_BLOCK_TABLE_PAD_SIZE = 16384


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
        # The kernel block-table row can be wider than
        # max_num_blocks_per_group when one KV block maps to multiple kernel
        # blocks. Use the allocated row stride so the staged row is complete.
        max_block_table_stride = max(block_table.gpu.stride(0) for block_table in self.block_tables)
        # tl.arange needs a compile-time power-of-two size. This value is
        # passed as a constexpr and covers every KV cache group's row.
        self._block_table_pad_size = triton.next_power_of_2(max_block_table_stride)
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
        num_groups = self.num_kv_cache_groups
        slot_mappings = self.slot_mappings if out is None else out
        _compute_slot_mappings_kernel[(num_groups, num_reqs + 1)](
            slot_mappings.shape[1],
            idx_mapping,
            query_start_loc,
            positions,
            self.block_table_ptrs,
            self.block_table_strides,
            self.block_sizes_tensor,
            slot_mappings,
            slot_mappings.stride(0),
            self.cp_rank,
            CP_SIZE=self.cp_size,
            CP_INTERLEAVE=self.cp_interleave,
            PAD_ID=PAD_SLOT_ID,
            TRITON_BLOCK_SIZE=1024,
            BLOCK_TABLE_PAD_SIZE=self._block_table_pad_size,
            USE_BLOCK_TABLE_STAGING=(self._block_table_pad_size <= _MAX_STAGED_BLOCK_TABLE_PAD_SIZE),
        )
        return slot_mappings[:, :num_tokens_padded]
