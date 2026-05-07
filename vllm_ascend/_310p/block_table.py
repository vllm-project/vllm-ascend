import numpy as np
import torch
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.worker.cp_utils import get_total_cp_world_size

from vllm_ascend.worker.block_table import BlockTable as AscendBlockTable
from vllm_ascend.worker.block_table import MultiGroupBlockTable as AscendMultiGroupBlockTable


class BlockTable(AscendBlockTable):
    def compute_slot_mapping(self, *args) -> None:
        if len(args) == 3:
            self._compute_slot_mapping_device(*args)
            return

        if len(args) == 2 and any(isinstance(arg, torch.Tensor) for arg in args):
            req_indices, positions = args
            self._write_slot_mapping_device(
                req_indices.to(dtype=torch.int64, device=self.slot_mapping.gpu.device),
                positions.to(device=self.slot_mapping.gpu.device),
            )
            return

        req_indices, positions = self._normalize_slot_mapping_inputs(*args)
        self._compute_slot_mapping_numpy(req_indices, positions)

    def _compute_slot_mapping_device(
        self,
        num_reqs: int,
        query_start_loc: torch.Tensor,
        positions: torch.Tensor,
    ) -> None:
        num_tokens = positions.shape[0]
        if num_tokens == 0:
            return

        token_indices = torch.arange(num_tokens, dtype=query_start_loc.dtype, device=positions.device)
        req_indices = torch.searchsorted(query_start_loc[1 : num_reqs + 1], token_indices, right=True).to(torch.int64)
        self._write_slot_mapping_device(req_indices, positions)

    def _write_slot_mapping_device(
        self,
        req_indices: torch.Tensor,
        positions: torch.Tensor,
    ) -> None:
        num_tokens = positions.shape[0]
        if num_tokens == 0:
            return

        block_table = self.block_table.gpu.flatten()
        if self.dcp_world_size * self.pcp_world_size > 1:
            virtual_block_size = self.block_size * self.dcp_world_size * self.pcp_world_size
            logical_block_idx = positions // virtual_block_size
            block_table_indices = self._get_block_table_indices(req_indices, logical_block_idx)
            block_numbers = block_table[block_table_indices]
            virtual_block_offsets = positions % virtual_block_size
            current_rank = self.dcp_world_size * self.pcp_rank + self.dcp_rank
            mask = (
                virtual_block_offsets // self.cp_kv_cache_interleave_size % (self.dcp_world_size * self.pcp_world_size)
                == current_rank
            )
            block_offsets = (
                virtual_block_offsets
                // (self.dcp_world_size * self.pcp_world_size * self.cp_kv_cache_interleave_size)
                * self.cp_kv_cache_interleave_size
                + virtual_block_offsets % self.cp_kv_cache_interleave_size
            )
            slot_mapping = block_numbers * self.block_size + block_offsets.to(torch.int32)
            slot_mapping = torch.where(mask, slot_mapping, torch.full_like(slot_mapping, PAD_SLOT_ID))
        else:
            logical_block_idx = positions // self.block_size
            block_table_indices = self._get_block_table_indices(req_indices, logical_block_idx)
            block_numbers = block_table[block_table_indices]
            block_offsets = positions % self.block_size
            slot_mapping = block_numbers * self.block_size + block_offsets.to(torch.int32)

        self.slot_mapping.gpu[:num_tokens].copy_(slot_mapping)

    def _compute_slot_mapping_numpy(self, req_indices: np.ndarray, positions: np.ndarray) -> None:
        num_tokens = positions.shape[0]
        if num_tokens == 0:
            self.slot_mapping.copy_to_gpu(0)
            return

        if self.dcp_world_size * self.pcp_world_size > 1:
            virtual_block_size = self.block_size * self.dcp_world_size * self.pcp_world_size
            logical_block_idx = positions // virtual_block_size
            block_table_indices = self._get_block_table_indices(req_indices, logical_block_idx)
            block_numbers = self.block_table.np.ravel()[block_table_indices]
            virtual_block_offsets = positions % virtual_block_size
            current_rank = self.dcp_world_size * self.pcp_rank + self.dcp_rank
            mask = (
                virtual_block_offsets // self.cp_kv_cache_interleave_size % (self.dcp_world_size * self.pcp_world_size)
                == current_rank
            )
            block_offsets = (
                virtual_block_offsets
                // (self.dcp_world_size * self.pcp_world_size * self.cp_kv_cache_interleave_size)
                * self.cp_kv_cache_interleave_size
                + virtual_block_offsets % self.cp_kv_cache_interleave_size
            )
            slot_mapping = block_numbers * self.block_size + block_offsets
            self.slot_mapping.np[:num_tokens] = np.where(mask, slot_mapping, PAD_SLOT_ID)
        else:
            logical_block_idx = positions // self.block_size
            block_table_indices = self._get_block_table_indices(req_indices, logical_block_idx)
            block_numbers = self.block_table.np.ravel()[block_table_indices]
            block_offsets = positions % self.block_size
            np.add(block_numbers * self.block_size, block_offsets, out=self.slot_mapping.np[:num_tokens])

        self.slot_mapping.copy_to_gpu(num_tokens)

    def _get_block_table_indices(self, req_indices, logical_block_idx):
        row_stride = self.max_num_blocks_per_req * self.blocks_per_phys_block
        return req_indices * row_stride + logical_block_idx

    def _normalize_slot_mapping_inputs(self, *args) -> tuple[np.ndarray, np.ndarray]:
        if len(args) == 2:
            req_indices, positions = args
            return self._to_numpy(req_indices), self._to_numpy(positions)

        raise TypeError("compute_slot_mapping expects either 2 or 3 positional arguments")

    @staticmethod
    def _to_numpy(value) -> np.ndarray:
        if isinstance(value, np.ndarray):
            return value.astype(np.int64, copy=False)
        if isinstance(value, torch.Tensor):
            if value.device.type != "cpu":
                raise TypeError("device tensors should use the device slot-mapping path")
            return value.detach().numpy().astype(np.int64, copy=False)
        return np.asarray(value, dtype=np.int64)


class MultiGroupBlockTable(AscendMultiGroupBlockTable):
    def __init__(
        self,
        max_num_reqs: int,
        max_model_len: int,
        max_num_batched_tokens: int,
        pin_memory: bool,
        device: torch.device,
        block_sizes: list[int],
        num_speculative_tokens: int = 0,
        max_num_blocks: list[int] | None = None,
        kernel_sizes: list[list[int]] | None = None,
        cp_kv_cache_interleave_size: int = 1,
    ) -> None:
        if kernel_sizes is None:
            kernel_sizes = [[0]] * len(block_sizes)
        elif len(kernel_sizes) == 1 and len(block_sizes) > 1:
            kernel_sizes = kernel_sizes * len(block_sizes)
        elif len(kernel_sizes) != len(block_sizes):
            raise ValueError(
                f"kernel_sizes length ({len(kernel_sizes)}) must match block_sizes length ({len(block_sizes)})"
            )

        if max_num_blocks is None:
            total_cp_world_size = get_total_cp_world_size()
            max_num_blocks = [cdiv(max_model_len, block_size * total_cp_world_size) for block_size in block_sizes]

        if len(max_num_blocks) != len(block_sizes):
            raise ValueError(
                f"max_num_blocks length ({len(max_num_blocks)}) must match block_sizes length ({len(block_sizes)})"
            )

        self.block_tables = [
            BlockTable(
                block_size,
                max_num_reqs,
                max_num_blocks_per_req,
                max_num_batched_tokens,
                pin_memory,
                device,
                kernel_size_list,
                cp_kv_cache_interleave_size,
                num_speculative_tokens,
            )
            for block_size, kernel_size_list, max_num_blocks_per_req in zip(block_sizes, kernel_sizes, max_num_blocks)
        ]

    def compute_slot_mapping(self, *args) -> None:
        for block_table in self.block_tables:
            block_table.compute_slot_mapping(*args)
