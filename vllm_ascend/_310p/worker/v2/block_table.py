# Adapt from https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu/block_table.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from __future__ import annotations

import numpy as np
import torch
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.worker.gpu.block_table import BlockTables


class Ascend310PBlockTables(BlockTables):
    """CPU-owned MRV2 block tables matching the 310P MRV1 data path."""

    # TODO: Refactor block-table operations to register 310P implementations
    # through Triton Dispatcher after vLLM RFC #45133 lands.

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
    ) -> None:
        if kernel_block_sizes is None:
            kernel_block_sizes = block_sizes
        if cp_size != 1:
            raise NotImplementedError("310P model runner v2 only supports tensor parallelism.")
        if len(max_num_blocks_per_group) != len(block_sizes):
            raise ValueError("max_num_blocks_per_group must match the number of KV cache groups.")

        self.block_sizes = block_sizes
        self.kernel_block_sizes = kernel_block_sizes
        self.max_num_reqs = max_num_reqs
        self.max_num_batched_tokens = max_num_batched_tokens
        self.device = device
        self.cp_size = cp_size
        self.cp_rank = cp_rank
        self.cp_interleave = cp_interleave
        self.num_kv_cache_groups = len(block_sizes)
        if slot_mapping_enabled is None:
            slot_mapping_enabled = [True] * self.num_kv_cache_groups
        if len(slot_mapping_enabled) != self.num_kv_cache_groups:
            raise ValueError("slot_mapping_enabled must match the number of KV cache groups.")
        self._slot_mapping_enabled = slot_mapping_enabled
        self.blocks_per_kv_block = [
            block_size // kernel_block_size for block_size, kernel_block_size in zip(block_sizes, kernel_block_sizes)
        ]

        table_shapes = [
            (max_num_reqs, max_num_blocks * blocks_per_kv_block)
            for max_num_blocks, blocks_per_kv_block in zip(max_num_blocks_per_group, self.blocks_per_kv_block)
        ]
        self.block_tables_cpu = [np.zeros(shape, dtype=np.int32) for shape in table_shapes]
        self.input_block_tables_cpu = [np.zeros(shape, dtype=np.int32) for shape in table_shapes]
        self.input_block_tables = [torch.zeros(shape, dtype=torch.int32, device=device) for shape in table_shapes]
        self.num_blocks_np = np.zeros((self.num_kv_cache_groups, max_num_reqs), dtype=np.int32)
        self.slot_mappings_cpu = np.full(
            (self.num_kv_cache_groups, max_num_batched_tokens),
            PAD_SLOT_ID,
            dtype=np.int32,
        )
        # Persistent device buffers are reused by eager execution and ACLGraph.
        self.slot_mappings = torch.full(
            self.slot_mappings_cpu.shape,
            PAD_SLOT_ID,
            dtype=torch.int32,
            device=device,
        )

    def init_block_table_layout_tensors(self) -> None:
        """310P does not use Triton pointer tables."""

    def append_block_ids(
        self,
        req_index: int,
        new_block_ids: tuple[list[int], ...],
        overwrite: bool,
    ) -> None:
        for group_id, block_ids in enumerate(new_block_ids):
            start = 0 if overwrite else int(self.num_blocks_np[group_id, req_index])
            blocks_per_kv_block = self.blocks_per_kv_block[group_id]
            if blocks_per_kv_block > 1:
                block_ids = [
                    block_id * blocks_per_kv_block + offset
                    for block_id in block_ids
                    for offset in range(blocks_per_kv_block)
                ]
            end = start + len(block_ids)
            if end > self.block_tables_cpu[group_id].shape[1]:
                raise ValueError(f"Too many block IDs for request {req_index} in KV cache group {group_id}.")
            self.block_tables_cpu[group_id][req_index, start:end] = block_ids
            self.num_blocks_np[group_id, req_index] = end

    def apply_staged_writes(self) -> None:
        """Block IDs are written to their CPU owner immediately."""

    @staticmethod
    def _as_numpy(value: np.ndarray | torch.Tensor) -> np.ndarray:
        if isinstance(value, np.ndarray):
            return value.astype(np.int64, copy=False)
        if value.device.type != "cpu":
            raise TypeError("310P block-table metadata must come from the CPU request-state mirror.")
        return value.detach().numpy().astype(np.int64, copy=False)

    def gather_block_tables(
        self,
        idx_mapping: np.ndarray | torch.Tensor,
        num_reqs_padded: int,
    ) -> tuple[torch.Tensor, ...]:
        idx_mapping_np = self._as_numpy(idx_mapping)
        num_reqs = idx_mapping_np.shape[0]
        if num_reqs_padded < num_reqs:
            raise ValueError(f"num_reqs_padded ({num_reqs_padded}) is smaller than num_reqs ({num_reqs}).")

        for group_id, (source, host_output, device_output) in enumerate(
            zip(
                self.block_tables_cpu,
                self.input_block_tables_cpu,
                self.input_block_tables,
            )
        ):
            host_output[:num_reqs_padded].fill(0)
            for batch_idx, req_idx in enumerate(idx_mapping_np):
                num_blocks = int(self.num_blocks_np[group_id, req_idx])
                host_output[batch_idx, :num_blocks] = source[req_idx, :num_blocks]
            device_output[:num_reqs_padded].copy_(torch.from_numpy(host_output[:num_reqs_padded]), non_blocking=True)

        return tuple(table[:num_reqs_padded] for table in self.input_block_tables)

    def compute_slot_mappings(
        self,
        idx_mapping: np.ndarray | torch.Tensor,
        query_start_loc: np.ndarray | torch.Tensor,
        positions: np.ndarray | torch.Tensor,
        num_tokens_padded: int,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        idx_mapping_np = self._as_numpy(idx_mapping)
        query_start_loc_np = self._as_numpy(query_start_loc)
        positions_np = self._as_numpy(positions)
        if query_start_loc_np.shape[0] < idx_mapping_np.shape[0] + 1:
            raise ValueError("query_start_loc does not contain all request boundaries.")

        self.slot_mappings_cpu.fill(PAD_SLOT_ID)
        for group_id, (block_table, block_size) in enumerate(zip(self.block_tables_cpu, self.kernel_block_sizes)):
            if not self._slot_mapping_enabled[group_id]:
                continue
            for batch_idx, req_idx in enumerate(idx_mapping_np):
                start = int(query_start_loc_np[batch_idx])
                end = int(query_start_loc_np[batch_idx + 1])
                token_positions = positions_np[start:end]
                logical_block_indices = token_positions // block_size
                block_numbers = block_table[req_idx, logical_block_indices]
                block_offsets = token_positions % block_size
                self.slot_mappings_cpu[group_id, start:end] = block_numbers * block_size + block_offsets

        device_slots = self.slot_mappings if out is None else out
        device_slots.copy_(torch.from_numpy(self.slot_mappings_cpu), non_blocking=True)
        return device_slots[:, :num_tokens_padded]

    def get_dummy_block_tables(self, num_reqs: int) -> tuple[torch.Tensor, ...]:
        for block_table in self.input_block_tables:
            block_table[:num_reqs].zero_()
        return tuple(block_table[:num_reqs] for block_table in self.input_block_tables)

    def get_dummy_slot_mappings(self, num_tokens: int) -> torch.Tensor:
        self.slot_mappings.fill_(PAD_SLOT_ID)
        return self.slot_mappings[:, :num_tokens]
