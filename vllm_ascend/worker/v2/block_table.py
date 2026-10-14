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
import os

import torch
from vllm.triton_utils import triton
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.worker.gpu.block_table import BlockTables

from vllm_ascend.ops.triton.v2.block_table.compute_slot_mappings import (
    _compute_slot_mappings_kernel,
)


class AscendBlockTables(BlockTables):
    """Block table for Ascend NPUs."""

    block_sizes_tensor: torch.Tensor
    kernel_block_sizes_tensor: torch.Tensor

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
        group_cp_size: list[int] | None = None,
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
            slot_mapping_enabled=slot_mapping_enabled,
        )
        # Per-group DCP world size: replicated groups (e.g. the DeepSeek-V4
        # SWA/state sliding-window caches) pass 1 so their slot mappings keep
        # the full-sequence positions instead of the interleave ownership mask
        # that sharded groups apply. Consumed by the Ascend kernel override in
        # compute_slot_mappings (not forwarded to the upstream constructor).
        self.group_cp_sizes = (
            torch.tensor(group_cp_size, dtype=torch.int32, device=self.device)
            if group_cp_size is not None
            else None
        )
        # Retained for the one-shot width diagnostics below so the per-group
        # table widths land in the request-time log too.
        self._dsa_block_sizes = list(block_sizes)
        self._dsa_row_widths = list(max_num_blocks_per_group)
        self._dsa_group_cp_sizes = list(group_cp_size) if group_cp_size is not None else None
        self._dsa_widths_logged = False
        self._triton_block_size = 1024
        # kernel_block_sizes determine the number of block-table entries
        # touched by one token tile. Use the smallest kernel block size to form
        # one safe constexpr window for all groups, without staging a whole
        # row.
        min_kernel_block_size = min(kernel_block_sizes)
        window_size = (self._triton_block_size + min_kernel_block_size - 1) // min_kernel_block_size + 1
        self._block_table_window_size = triton.next_power_of_2(window_size)
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
        if not self._dsa_widths_logged:
            self._dsa_widths_logged = True
            if os.environ.get("DSA_DCP_DEBUG") == "1":
                print(
                    "[dsa-dcp-kv] widths"
                    f" block_sizes={self._dsa_block_sizes}"
                    f" row_widths={self._dsa_row_widths}"
                    f" group_cp_sizes={self._dsa_group_cp_sizes}"
                    f" cp_size={self.cp_size} cp_interleave={self.cp_interleave}",
                    flush=True,
                )
        slot_mappings = self.slot_mappings if out is None else out
        slot_mapping_enabled = self.slot_mapping_enabled
        _compute_slot_mappings_kernel[(num_groups, num_reqs + 1)](
            slot_mappings.shape[1],
            idx_mapping,
            query_start_loc,
            positions,
            self.block_table_ptrs,
            self.block_table_strides,
            self.block_sizes_tensor,
            self.kernel_block_sizes_tensor,
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
            group_cp_sizes=self.group_cp_sizes,
            HAS_GROUP_CP_SIZES=self.group_cp_sizes is not None,
        )
        return slot_mappings[:, :num_tokens_padded]
