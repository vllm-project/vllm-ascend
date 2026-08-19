# Adapt from https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu/block_table.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from __future__ import annotations

import numpy as np
import torch
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.worker.gpu.block_table import BlockTables


def _compute_group_slot_mappings(
    block_table_np: np.ndarray,
    idx_mapping_np: np.ndarray,
    query_start_loc_np: np.ndarray,
    positions_np: np.ndarray,
    block_size: int,
    out_np: np.ndarray,
) -> None:
    """Fill one KV cache group's slot IDs from CPU request metadata.

    This is the NumPy equivalent of the default Triton slot-mapping kernel, and
    matches the computation the 310P Model Runner V1 block table already uses.
    """
    out_np.fill(PAD_SLOT_ID)
    num_reqs = idx_mapping_np.shape[0]
    num_tokens = int(query_start_loc_np[num_reqs])
    if num_tokens == 0:
        return
    if positions_np.shape[0] < num_tokens:
        raise ValueError(f"positions holds {positions_np.shape[0]} tokens but query_start_loc describes {num_tokens}.")

    tokens_per_req = np.diff(query_start_loc_np[: num_reqs + 1])
    req_indices = np.repeat(idx_mapping_np, tokens_per_req)
    positions = positions_np[:num_tokens]
    block_numbers = block_table_np[req_indices, positions // block_size].astype(np.int64, copy=False)
    out_np[:num_tokens] = block_numbers * block_size + positions % block_size


class Ascend310PBlockTables(BlockTables):
    """V2 block tables that never launch a Triton kernel.

    V2 already keeps scheduler-owned block IDs and request indices on CPU. 310P
    uses those mirrors to gather input block tables and compute slot mappings,
    then copies the result into persistent NPU tensors used by eager execution
    and ACL Graph replay. No device-to-host synchronization is introduced.

    The whole class replaces the upstream one through
    ``patch/worker/patch_v2/patch_block_table.py``, so 310P needs no per-kernel
    dispatch mechanism: staged writes, gather and slot mapping are all Triton
    free here.
    """

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
    ) -> None:
        if kernel_block_sizes is None:
            kernel_block_sizes = block_sizes
        self.block_sizes = block_sizes
        self.kernel_block_sizes = kernel_block_sizes
        self.max_num_reqs = max_num_reqs
        self.max_num_batched_tokens = max_num_batched_tokens
        self.device = device
        self.cp_size = cp_size
        self.cp_rank = cp_rank
        self.cp_interleave = cp_interleave
        self.num_kv_cache_groups = len(block_sizes)
        if len(max_num_blocks_per_group) != self.num_kv_cache_groups:
            raise ValueError("max_num_blocks_per_group must match the number of KV cache groups.")
        if self.cp_size != 1:
            raise NotImplementedError("310P Model Runner V2 first release only supports tensor parallelism.")

        self.blocks_per_kv_block = [
            block_size // kernel_block_size
            for block_size, kernel_block_size in zip(self.block_sizes, self.kernel_block_sizes)
        ]
        table_shapes = [
            (max_num_reqs, max_num_blocks * blocks_per_kv_block)
            for max_num_blocks, blocks_per_kv_block in zip(max_num_blocks_per_group, self.blocks_per_kv_block)
        ]
        self.block_tables_cpu = [torch.zeros(shape, dtype=torch.int32, device="cpu") for shape in table_shapes]
        self.input_block_tables_cpu = [torch.zeros_like(table) for table in self.block_tables_cpu]
        self.input_block_tables = [torch.zeros(shape, dtype=torch.int32, device=device) for shape in table_shapes]
        self.num_blocks_np = np.zeros((self.num_kv_cache_groups, max_num_reqs), dtype=np.int32)
        self.slot_mappings_cpu = torch.full(
            (self.num_kv_cache_groups, max_num_batched_tokens),
            PAD_SLOT_ID,
            dtype=torch.int32,
            device="cpu",
        )
        # NumPy views over the CPU owners, so gather and slot mapping stay in
        # NumPy without re-wrapping the tensors on every step.
        self.block_tables_np = [table.numpy() for table in self.block_tables_cpu]
        self.input_block_tables_np = [table.numpy() for table in self.input_block_tables_cpu]
        self.slot_mappings_np = self.slot_mappings_cpu.numpy()

        # reshape_and_cache on 310P consumes int32 slot IDs.
        self.slot_mappings = torch.full(
            self.slot_mappings_cpu.shape,
            PAD_SLOT_ID,
            dtype=torch.int32,
            device=self.device,
        )

    def init_block_table_layout_tensors(self) -> None:
        """310P does not use raw device pointers consumed by Triton kernels."""

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
            if block_ids:
                self.block_tables_cpu[group_id][req_index, start:end] = torch.tensor(block_ids, dtype=torch.int32)
            self.num_blocks_np[group_id, req_index] = end

    def apply_staged_writes(self) -> None:
        """Block IDs are written to the CPU owner immediately."""

    @staticmethod
    def _as_numpy(value: np.ndarray | torch.Tensor) -> np.ndarray:
        if isinstance(value, np.ndarray):
            return value.astype(np.int64, copy=False)
        if value.device.type != "cpu":
            raise TypeError("310P V2 block-table metadata must come from the CPU request-state mirror.")
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

        for group_id, (source_np, host_output_np, host_output, device_output) in enumerate(
            zip(
                self.block_tables_np,
                self.input_block_tables_np,
                self.input_block_tables_cpu,
                self.input_block_tables,
            )
        ):
            host_output_np[:num_reqs_padded] = 0
            for batch_idx, req_idx in enumerate(idx_mapping_np):
                num_blocks = int(self.num_blocks_np[group_id, req_idx])
                if num_blocks:
                    host_output_np[batch_idx, :num_blocks] = source_np[req_idx, :num_blocks]
            device_output[:num_reqs_padded].copy_(host_output[:num_reqs_padded], non_blocking=True)

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

        for group_id, block_size in enumerate(self.kernel_block_sizes):
            _compute_group_slot_mappings(
                self.block_tables_np[group_id],
                idx_mapping_np,
                query_start_loc_np,
                positions_np,
                block_size,
                self.slot_mappings_np[group_id],
            )

        device_slots = self.slot_mappings if out is None else out
        device_slots.copy_(self.slot_mappings_cpu, non_blocking=True)
        return device_slots[:, :num_tokens_padded]

    def get_dummy_block_tables(self, num_reqs: int) -> tuple[torch.Tensor, ...]:
        return tuple(block_table[:num_reqs].zero_() for block_table in self.input_block_tables)

    def get_dummy_slot_mappings(self, num_tokens: int) -> torch.Tensor:
        self.slot_mappings_cpu.fill_(PAD_SLOT_ID)
        self.slot_mappings.copy_(self.slot_mappings_cpu, non_blocking=True)
        return self.slot_mappings[:, :num_tokens]
