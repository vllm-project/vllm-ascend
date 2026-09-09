# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from triton.runtime.interpreter import InterpretedFunction

from vllm_ascend.ops.triton.v2.block_table import compute_slot_mappings as slot_kernel
from vllm_ascend.worker.v2.block_table import AscendBlockTables


@pytest.mark.parametrize("block_size", [128, 1536])
@pytest.mark.parametrize("cp_size,cp_rank", [(1, 0), (2, 0), (2, 1), (4, 3)])
@pytest.mark.parametrize("interleave", [1, 128])
def test_slot_mapping_uses_expanded_kernel_blocks(block_size, cp_size, cp_rank, interleave):
    kernel_size = 128
    factor = block_size // kernel_size
    block_ids = [1, 5]
    table = torch.tensor([b * factor + k for b in block_ids for k in range(factor)], dtype=torch.int32)
    positions_list = [0, 1, 127, 128, block_size * cp_size - 1, block_size * cp_size, 2 * block_size * cp_size - 1]
    positions = torch.tensor(positions_list, dtype=torch.int64)
    out = torch.zeros(10, dtype=torch.int32)
    expected = []
    for position in positions_list:
        block, offset = divmod(position, block_size * cp_size)
        if offset // interleave % cp_size != cp_rank:
            expected.append(-1)
        else:
            local_offset = offset // (interleave * cp_size) * interleave + offset % interleave
            expected.append(block_ids[block] * block_size + local_offset)

    # Execute the actual address arithmetic on CPU, without compiling or using an NPU.
    kernel = InterpretedFunction(slot_kernel._compute_slot_mappings_kernel.fn)
    with patch.object(slot_kernel, "_load_ptr", InterpretedFunction(slot_kernel._load_ptr.fn)):
        kernel[(1, 2)](
            out.numel(),
            torch.tensor([0], dtype=torch.int32),
            torch.tensor([0, positions.numel()], dtype=torch.int32),
            positions,
            torch.tensor([table.data_ptr()], dtype=torch.int64),
            torch.tensor([table.numel()], dtype=torch.int32),
            torch.tensor([block_size], dtype=torch.int32),
            torch.tensor([kernel_size], dtype=torch.int32),
            out,
            out.numel(),
            cp_rank,
            CP_SIZE=cp_size,
            CP_INTERLEAVE=interleave,
            PAD_ID=-1,
            TRITON_BLOCK_SIZE=8,
            BLOCK_TABLE_PAD_SIZE=32,
            USE_BLOCK_TABLE_STAGING=False,
        )
    assert out.tolist() == expected + [-1] * (out.numel() - len(expected))


@pytest.mark.parametrize("external_output", [False, True])
def test_slot_mapping_passes_both_block_sizes(external_output):
    state = SimpleNamespace(
        num_kv_cache_groups=1,
        slot_mappings=torch.zeros((1, 8), dtype=torch.int32),
        block_table_ptrs=torch.zeros(1, dtype=torch.int64),
        block_table_strides=torch.tensor([24], dtype=torch.int32),
        block_sizes_tensor=torch.tensor([1536], dtype=torch.int32),
        kernel_block_sizes_tensor=torch.tensor([128], dtype=torch.int32),
        cp_size=1,
        cp_rank=0,
        cp_interleave=1,
        _block_table_pad_size=32,
    )
    out = torch.zeros_like(state.slot_mappings) if external_output else None
    kernel = MagicMock()
    with patch("vllm_ascend.worker.v2.block_table._compute_slot_mappings_kernel", kernel):
        result = AscendBlockTables.compute_slot_mappings(
            state, torch.tensor([0]), torch.tensor([0, 1]), torch.tensor([0]), 1, out
        )
    args = kernel.__getitem__.return_value.call_args.args
    assert args[6] is state.block_sizes_tensor
    assert args[7] is state.kernel_block_sizes_tensor
    assert result.data_ptr() == (state.slot_mappings if out is None else out).data_ptr()
