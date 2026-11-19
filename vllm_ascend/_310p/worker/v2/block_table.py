# Adapt from https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu/block_table.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from __future__ import annotations

import numpy as np
import torch
from vllm.utils.platform_utils import is_pin_memory_available
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.worker.gpu.block_table import BlockTables


class Ascend310PBlockTables(BlockTables):
    """CPU-owned MRV2 block tables for Ascend 310P."""

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
        dcp_sharded: list[bool] | None = None,
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
        # vLLM main added a per-group ``dcp_sharded`` flag to BlockTables;
        # expose it for upstream readers when the caller supplies it. 310P
        # never passes it (it builds the tables directly), so this is a guard.
        if dcp_sharded is not None:
            if len(dcp_sharded) != self.num_kv_cache_groups:
                raise ValueError("dcp_sharded must match the number of KV cache groups.")
            self.dcp_sharded = torch.tensor(dcp_sharded, dtype=torch.bool, device=device)
        self.blocks_per_kv_block = [
            block_size // kernel_block_size for block_size, kernel_block_size in zip(block_sizes, kernel_block_sizes)
        ]

        table_shapes = [
            (max_num_reqs, max_num_blocks * blocks_per_kv_block)
            for max_num_blocks, blocks_per_kv_block in zip(max_num_blocks_per_group, self.blocks_per_kv_block)
        ]
        pin_memory = is_pin_memory_available()
        self._block_tables_cpu_tensors = [
            torch.zeros(shape, dtype=torch.int32, device="cpu", pin_memory=pin_memory) for shape in table_shapes
        ]
        self._input_block_tables_cpu_tensors = [
            torch.zeros(shape, dtype=torch.int32, device="cpu", pin_memory=pin_memory) for shape in table_shapes
        ]
        self.block_tables_cpu = [tensor.numpy() for tensor in self._block_tables_cpu_tensors]
        self.input_block_tables_cpu = [tensor.numpy() for tensor in self._input_block_tables_cpu_tensors]
        self.input_block_tables = [torch.zeros(shape, dtype=torch.int32, device=device) for shape in table_shapes]
        self.num_blocks_np = np.zeros((self.num_kv_cache_groups, max_num_reqs), dtype=np.int32)
        self._slot_mappings_cpu_tensor = torch.full(
            (self.num_kv_cache_groups, max_num_batched_tokens),
            PAD_SLOT_ID,
            dtype=torch.int32,
            device="cpu",
            pin_memory=pin_memory,
        )
        self.slot_mappings_cpu = self._slot_mappings_cpu_tensor.numpy()
        # Persistent device buffers are reused by eager execution and ACLGraph.
        self.slot_mappings = torch.full(
            self.slot_mappings_cpu.shape,
            PAD_SLOT_ID,
            dtype=torch.int32,
            device=device,
        )
        # Decode steps only allocate a KV block every ``block_size`` tokens.
        # Reuse the last gather while the tables and request order are unchanged.
        self._table_version = 0
        self._gather_cache_key: tuple[int, int, bytes] | None = None
        self._gather_cache_result: tuple[torch.Tensor, ...] | None = None

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
        # Any append changes the gathered rows; invalidate the reuse cache.
        self._table_version += 1

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

        # Decode keeps the same request order and blocks for many steps; skip the
        # host gather and H2D copies until a block allocation or reorder occurs.
        cache_key = (num_reqs_padded, self._table_version, idx_mapping_np.tobytes())
        if cache_key == self._gather_cache_key:
            assert self._gather_cache_result is not None
            return self._gather_cache_result

        for group_id, (source, host_output, host_tensor, device_output) in enumerate(
            zip(
                self.block_tables_cpu,
                self.input_block_tables_cpu,
                self._input_block_tables_cpu_tensors,
                self.input_block_tables,
            )
        ):
            host_output[:num_reqs_padded].fill(0)
            for batch_idx, req_idx in enumerate(idx_mapping_np):
                num_blocks = int(self.num_blocks_np[group_id, req_idx])
                host_output[batch_idx, :num_blocks] = source[req_idx, :num_blocks]
            device_output[:num_reqs_padded].copy_(host_tensor[:num_reqs_padded], non_blocking=True)

        result = tuple(table[:num_reqs_padded] for table in self.input_block_tables)
        self._gather_cache_key = cache_key
        self._gather_cache_result = result
        return result

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

        if num_tokens_padded > self.max_num_batched_tokens:
            raise ValueError(
                f"num_tokens_padded ({num_tokens_padded}) exceeds slot-mapping capacity "
                f"({self.max_num_batched_tokens})."
            )
        num_reqs = idx_mapping_np.shape[0]
        # Token -> request ownership for the token-aligned positions. One vectorized
        # scatter keeps the per-step host cost flat as the batch grows.
        token_start = int(query_start_loc_np[0])
        token_lens = np.maximum(np.diff(query_start_loc_np[: num_reqs + 1]), 0)
        num_positions = int(token_lens.sum())
        token_end = token_start + num_positions
        rows = idx_mapping_np[np.repeat(np.arange(num_reqs), token_lens)]
        active = rows >= 0
        all_active = bool(active.all())
        # Rows that own no blocks keep PAD; only clear when such rows exist or
        # this batch does not cover every padded token column.
        if not all_active or token_start > 0 or token_end < num_tokens_padded or not all(self._slot_mapping_enabled):
            self.slot_mappings_cpu[:, :num_tokens_padded].fill(PAD_SLOT_ID)
        if num_positions:
            token_positions = positions_np[token_start:token_end]
            if not all_active:
                rows = np.where(active, rows, 0)
            for group_id, (block_table, block_size) in enumerate(zip(self.block_tables_cpu, self.kernel_block_sizes)):
                if not self._slot_mapping_enabled[group_id]:
                    continue
                block_numbers = block_table[rows, token_positions // block_size]
                slots = block_numbers * block_size + token_positions % block_size
                column = self.slot_mappings_cpu[group_id]
                if all_active:
                    column[token_start:token_end] = slots
                else:
                    column[token_start:token_end][active] = slots[active]

        device_slots = self.slot_mappings if out is None else out
        for group_id in range(self.num_kv_cache_groups):
            device_slots[group_id, :num_tokens_padded].copy_(
                self._slot_mappings_cpu_tensor[group_id, :num_tokens_padded],
                non_blocking=True,
            )
        return device_slots[:, :num_tokens_padded]

    def get_dummy_block_tables(self, num_reqs: int) -> tuple[torch.Tensor, ...]:
        for block_table in self.input_block_tables:
            block_table[:num_reqs].zero_()
        # Dummy runs overwrite the gathered device rows used by the cache.
        self._table_version += 1
        return tuple(block_table[:num_reqs] for block_table in self.input_block_tables)

    def get_dummy_slot_mappings(self, num_tokens: int) -> torch.Tensor:
        self.slot_mappings.fill_(PAD_SLOT_ID)
        return self.slot_mappings[:, :num_tokens]
