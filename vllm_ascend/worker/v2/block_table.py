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

from vllm_ascend.ops.triton.block_table.compute_slot_mappings import (
    _compute_slot_mappings_kernel,
)


class AscendBlockTables(BlockTables):
    """Block table for Ascend NPUs."""

    # Main vLLM follows #57169: public block tables stay at manager-block
    # granularity and kernel-block mapping belongs to AttentionGroup.

    block_sizes_tensor: torch.Tensor

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
        slot_mapping_enabled: list[bool] | None = None,
        dcp_sharded: list[bool] | None = None,
    ):
        # vLLM #57169 keeps public block tables at KV-cache-manager
        # granularity. Kernel-specific expansion is handled by AttentionGroup.
        del kernel_block_sizes
        super_kwargs: dict[str, list[bool]] = {}
        if dcp_sharded is not None:
            super_kwargs["dcp_sharded"] = dcp_sharded
        super().__init__(
            block_sizes,
            max_num_reqs,
            max_num_batched_tokens,
            max_num_blocks_per_group,
            device,
            cp_size=cp_size,
            cp_rank=cp_rank,
            cp_interleave=cp_interleave,
            slot_mapping_enabled=slot_mapping_enabled,
            **super_kwargs,
        )
        self._triton_block_size = 1024
        if block_sizes:
            min_block_size = min(block_sizes)
            window_size = (self._triton_block_size + min_block_size - 1) // min_block_size + 1
            self._block_table_window_size = triton.next_power_of_2(window_size)
        else:
            self._block_table_window_size = 1
        # vllm-ascend's reshape_and_cache function requires int32 slot mappings.
        del self.slot_mappings
        self.slot_mappings: torch.Tensor = torch.zeros(  # type: ignore[no-redef]
            self.num_kv_cache_groups,
            self.max_num_batched_tokens,
            dtype=torch.int32,
            device=self.device,
        )

    def append_block_ids(
        self,
        req_index: int,
        new_block_ids: tuple[list[int], ...],
        overwrite: bool,
    ) -> None:
        # Keep public block IDs at KV-cache-manager granularity. The
        # AttentionGroup owns the kernel-granularity mapping.
        return super().append_block_ids(req_index, new_block_ids, overwrite)

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
        if num_groups == 0:
            return slot_mappings[:, :num_tokens_padded]
        slot_mapping_enabled = self.slot_mapping_enabled
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
            TRITON_BLOCK_SIZE=self._triton_block_size,
            BLOCK_TABLE_WINDOW_SIZE=self._block_table_window_size,
            slot_mapping_enabled=slot_mapping_enabled,
            HAS_SLOT_MAPPING_ENABLED=True,
        )
        return slot_mappings[:, :num_tokens_padded]
