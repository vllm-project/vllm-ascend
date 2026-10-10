# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Negative positions in the fused multi-group slot mapping.

``positions`` is shared by every KV-cache group of a fused launch. In a
mixed circular/non-circular layout (DSV4.1 compressor state, GLM-5-Next
indexer tail) a negative position must become PAD_ID in every group, and it
must not move the block-table window of the regular tokens in its tile.
"""

import pytest
import torch
from vllm.v1.attention.backends.utils import PAD_SLOT_ID

from vllm_ascend.ops.triton.compute_slot_mapping import (
    _compute_slot_mapping_kernel,
    _next_power_of_2,
    compute_slot_mapping_fused_groups,
)

PAD_ID = PAD_SLOT_ID
GROUP_COUNT = 3
BLOCK_SIZES = (128, 64, 32)
MAX_BLOCKS_PER_REQ = 192
SENTINEL = 123456


class _FusedGroupsHarness:
    """Builds per-group block tables plus the fused-kernel metadata tables."""

    def __init__(self, device: str, is_circular: tuple[int, ...]) -> None:
        self.device = device
        # Distinct values per group/row so wrong-group or wrong-row reads are caught.
        self.block_tables = [
            torch.arange(
                g * 1000,
                g * 1000 + MAX_BLOCKS_PER_REQ * MAX_BLOCKS_PER_REQ,
                dtype=torch.int32,
                device=device,
            ).reshape(MAX_BLOCKS_PER_REQ, MAX_BLOCKS_PER_REQ)
            for g in range(GROUP_COUNT)
        ]
        self.block_table_addrs = torch.tensor(
            [t.data_ptr() for t in self.block_tables], dtype=torch.uint64, device=device
        )
        self.block_table_strides = torch.tensor(
            [t.stride(0) for t in self.block_tables], dtype=torch.int64, device=device
        )
        self.block_sizes = torch.tensor(BLOCK_SIZES, dtype=torch.int32, device=device)
        self.is_circular = torch.tensor(is_circular, dtype=torch.int32, device=device)

    def new_slot_mappings(self, max_num_tokens: int) -> list[torch.Tensor]:
        return [
            torch.full((max_num_tokens,), SENTINEL, dtype=torch.int32, device=self.device) for _ in range(GROUP_COUNT)
        ]

    def fused_call(
        self,
        slot_mappings: list[torch.Tensor],
        query_start_loc: torch.Tensor,
        positions: torch.Tensor,
        max_num_tokens: int,
        *,
        circular_ptr: bool = True,
    ) -> None:
        slot_mapping_addrs = torch.tensor([t.data_ptr() for t in slot_mappings], dtype=torch.uint64, device=self.device)
        compute_slot_mapping_fused_groups(
            GROUP_COUNT,
            query_start_loc.shape[0] - 1,
            positions.shape[0],
            max_num_tokens,
            query_start_loc,
            positions,
            self.block_table_addrs,
            slot_mapping_addrs,
            self.block_table_strides,
            self.block_sizes,
            min(BLOCK_SIZES),
            pad_id=PAD_ID,
            is_circular_ptr=self.is_circular if circular_ptr else None,
        )

    def per_group_reference(
        self,
        query_start_loc: torch.Tensor,
        positions: torch.Tensor,
        max_num_tokens: int,
    ) -> list[torch.Tensor]:
        num_reqs = query_start_loc.shape[0] - 1
        outputs = self.new_slot_mappings(max_num_tokens)
        for g in range(GROUP_COUNT):
            _compute_slot_mapping_kernel[(num_reqs + 1,)](
                positions.shape[0],
                max_num_tokens,
                query_start_loc,
                positions,
                self.block_tables[g],
                self.block_tables[g].stride(0),
                BLOCK_SIZES[g],
                outputs[g],
                KV_CACHE_BLOCK_SIZE=1,
                BLOCKS_PER_KV_BLOCK=1,
                TOTAL_CP_WORLD_SIZE=1,
                TOTAL_CP_RANK=0,
                CP_KV_CACHE_INTERLEAVE_SIZE=1,
                PAD_ID=PAD_ID,
                TILE_BLOCK_SIZE=1024,
                BLOCK_TABLE_WINDOW_SIZE=_next_power_of_2((1024 + BLOCK_SIZES[g] - 1) // BLOCK_SIZES[g] + 1),
                IS_CIRCULAR=bool(self.is_circular[g].item()),
            )
        return outputs


def _make_batch(device: str, seq_lens: list[int], position_offsets: list[int], negatives):
    """Request i covers [offset_i, offset_i + len_i); ``negatives`` overrides (index, value) pairs."""
    lengths = [0]
    positions: list[int] = []
    for seq_len, offset in zip(seq_lens, position_offsets):
        positions.extend(range(offset, offset + seq_len))
        lengths.append(len(positions))
    for idx, value in negatives:
        positions[idx] = value
    query_start_loc = torch.tensor(lengths, dtype=torch.int32, device=device)
    positions_tensor = torch.tensor(positions, dtype=torch.int64, device=device)
    max_num_tokens = (len(positions) + 63) // 64 * 64
    return query_start_loc, positions_tensor, max_num_tokens


# Adaptive small-tile, adaptive large-tile and parallel kernels: negatives
# sharing a tile with regular tokens, at or below -block_size (a negative
# block index), and filling a whole request. Values at or below -block_size
# stay out of row 0, where the per-group reference kernel would read before
# the block table.
NEGATIVE_POSITION_CASES = [
    pytest.param([64, 8], [0, 500], ((64, -1), (67, -2)), id="adaptive_small_tile"),
    pytest.param([64, 300], [0, 500], ((64, -1), (67, -2), (200, -3)), id="adaptive_large_tile"),
    pytest.param([512], [0], ((0, -1), (10, -2)), id="parallel_kernel"),
    pytest.param([64, 8], [0, 500], ((64, -200), (65, -1)), id="adaptive_below_block_size"),
    pytest.param([64, 8], [0, 500], tuple((64 + i, -300 + i) for i in range(8)), id="adaptive_all_negative_request"),
    pytest.param([600, 600], [0, 0], ((600, -200), (601, -2)), id="parallel_below_block_size"),
]


@pytest.mark.parametrize("seq_lens,position_offsets,negatives", NEGATIVE_POSITION_CASES)
@torch.inference_mode()
def test_mixed_circular_layout_pads_negative_positions(seq_lens, position_offsets, negatives):
    """Circular groups match the per-group kernel; non-circular groups PAD negatives.

    The per-group kernel leaves negative positions undefined in non-circular
    groups, so only the non-negative ones are compared against it there.
    """
    device = "npu"
    harness = _FusedGroupsHarness(device, is_circular=(1, 0, 1))
    query_start_loc, positions, max_num_tokens = _make_batch(device, seq_lens, position_offsets, negatives)

    fused = harness.new_slot_mappings(max_num_tokens)
    harness.fused_call(fused, query_start_loc, positions, max_num_tokens)
    reference = harness.per_group_reference(query_start_loc, positions, max_num_tokens)
    torch.npu.synchronize()

    num_tokens = positions.shape[0]
    non_negative = positions >= 0
    for g in range(GROUP_COUNT):
        if harness.is_circular[g]:
            assert torch.equal(fused[g], reference[g])
        else:
            assert torch.equal(fused[g][:num_tokens][non_negative], reference[g][:num_tokens][non_negative])
            assert torch.all(fused[g][:num_tokens][~non_negative] == PAD_ID)
            assert torch.all(fused[g][num_tokens:] == PAD_ID)


@pytest.mark.parametrize("seq_lens,position_offsets,negatives", NEGATIVE_POSITION_CASES)
@torch.inference_mode()
def test_all_circular_layout_matches_per_group(seq_lens, position_offsets, negatives):
    device = "npu"
    harness = _FusedGroupsHarness(device, is_circular=(1, 1, 1))
    query_start_loc, positions, max_num_tokens = _make_batch(device, seq_lens, position_offsets, negatives)

    fused = harness.new_slot_mappings(max_num_tokens)
    harness.fused_call(fused, query_start_loc, positions, max_num_tokens)
    reference = harness.per_group_reference(query_start_loc, positions, max_num_tokens)
    torch.npu.synchronize()

    for g in range(GROUP_COUNT):
        assert torch.equal(fused[g], reference[g])


@pytest.mark.parametrize("seq_lens", [[128], [1] * 63 + [128], [512] * 8, [4096]])
@torch.inference_mode()
def test_without_circular_ptr_matches_all_zero_circular(seq_lens):
    """MultiGroupBlockTable omits is_circular_ptr when no group is circular."""
    device = "npu"
    harness = _FusedGroupsHarness(device, is_circular=(0, 0, 0))
    query_start_loc, positions, max_num_tokens = _make_batch(device, seq_lens, [0] * len(seq_lens), ())

    with_ptr = harness.new_slot_mappings(max_num_tokens)
    harness.fused_call(with_ptr, query_start_loc, positions, max_num_tokens)
    without_ptr = harness.new_slot_mappings(max_num_tokens)
    harness.fused_call(without_ptr, query_start_loc, positions, max_num_tokens, circular_ptr=False)
    torch.npu.synchronize()

    for g in range(GROUP_COUNT):
        assert torch.equal(with_ptr[g], without_ptr[g])
