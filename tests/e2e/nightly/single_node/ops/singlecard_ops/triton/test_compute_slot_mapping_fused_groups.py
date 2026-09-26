# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness and JIT-cache bounds for the fused multi-group slot mapping.

The fused kernels must produce exactly the same slot mapping as launching
the per-group kernel once per KV-cache group, and their Triton
specialization count must stay bounded no matter how the batch shape
changes: a fresh specialization triggers an on-line Triton JIT compilation
that takes seconds on Ascend and stalls every in-flight request.
"""

import pytest
import torch
from vllm.v1.attention.backends.utils import PAD_SLOT_ID

from vllm_ascend.ops.triton.compute_slot_mapping import (
    _FUSED_SLOT_MAPPING_TILE_LADDER,
    _compute_slot_mapping_fused_groups_adaptive_kernel,
    _compute_slot_mapping_fused_groups_kernel,
    _compute_slot_mapping_kernel,
    _next_power_of_2,
    compute_slot_mapping_fused_groups,
    prewarm_fused_slot_mapping_kernels,
)
from vllm_ascend.worker.block_table import MultiGroupBlockTable

# PAD_ID is a constexpr, so it must match production for the prewarm tests.
PAD_ID = PAD_SLOT_ID
# Three KV-cache groups with heterogeneous block sizes; which of them are
# circular is chosen per test (DSV4.1 has one circular compressor-state group).
GROUP_COUNT = 3
BLOCK_SIZES = (128, 64, 32)
MAX_BLOCKS_PER_REQ = 192
SENTINEL = 123456


def _kernel_cache_size(kernel) -> int:
    cache = getattr(kernel, "cache", None)
    assert cache is not None, "cannot read the Triton JIT cache; triton API changed?"
    return sum(len(entries) for entries in cache.values())


def _fused_cache_sizes() -> tuple[int, int]:
    return (
        _kernel_cache_size(_compute_slot_mapping_fused_groups_adaptive_kernel),
        _kernel_cache_size(_compute_slot_mapping_fused_groups_kernel),
    )


def _clear_fused_kernel_caches() -> None:
    """Drop in-memory specializations so a test sees exactly what it compiles."""
    for kernel in (_compute_slot_mapping_fused_groups_adaptive_kernel, _compute_slot_mapping_fused_groups_kernel):
        for entries in kernel.cache.values():
            entries.clear()


class _FusedGroupsHarness:
    """Builds per-group block tables plus the fused-kernel metadata tables."""

    def __init__(self, device: str, is_circular: tuple[int, ...] = (0, 0, 0)) -> None:
        self.device = device
        # Distinct values per group/row so wrong-group writes are caught.
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
        num_reqs = query_start_loc.shape[0] - 1
        num_tokens = positions.shape[0]
        slot_mapping_addrs = torch.tensor([t.data_ptr() for t in slot_mappings], dtype=torch.uint64, device=self.device)
        kwargs = {"is_circular_ptr": self.is_circular} if circular_ptr else {}
        compute_slot_mapping_fused_groups(
            GROUP_COUNT,
            num_reqs,
            num_tokens,
            max_num_tokens,
            query_start_loc,
            positions,
            self.block_table_addrs,
            slot_mapping_addrs,
            self.block_table_strides,
            self.block_sizes,
            min(BLOCK_SIZES),
            pad_id=PAD_ID,
            **kwargs,
        )

    def per_group_reference(
        self,
        query_start_loc: torch.Tensor,
        positions: torch.Tensor,
        max_num_tokens: int,
    ) -> list[torch.Tensor]:
        """Launch the (already shape-stable) per-group kernel once per group."""
        num_reqs = query_start_loc.shape[0] - 1
        num_tokens = positions.shape[0]
        outputs = self.new_slot_mappings(max_num_tokens)
        for g in range(GROUP_COUNT):
            _compute_slot_mapping_kernel[(num_reqs + 1,)](
                num_tokens,
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


def _make_batch(device: str, seq_lens: list[int], position_offsets: list[int] | None = None):
    """Prefill-like positions: request i covers [offset_i, offset_i + len_i)."""
    if position_offsets is None:
        position_offsets = [(i * 37) % 512 for i in range(len(seq_lens))]
    lengths = [0]
    positions: list[int] = []
    for seq_len, offset in zip(seq_lens, position_offsets):
        positions.extend(range(offset, offset + seq_len))
        lengths.append(len(positions))
    query_start_loc = torch.tensor(lengths, dtype=torch.int32, device=device)
    positions_tensor = torch.tensor(positions, dtype=torch.int64, device=device)
    # Tail padding region the kernels must fill with PAD_ID.
    max_num_tokens = (len(positions) + 63) // 64 * 64
    return query_start_loc, positions_tensor, max_num_tokens


# Covers the #15289 msprof matrix, mixed prefill/decode batches, uneven
# splits, both fused kernels and every in-kernel tile size.
SHAPE_CASES = [
    pytest.param([4096], id="1x4096_parallel_tiles_4"),
    pytest.param([2048], id="1x2048_parallel_tiles_2"),
    pytest.param([1536], id="1x1536_adaptive_1024"),
    pytest.param([128], id="1x128_adaptive_256"),
    pytest.param([16], id="1x16_adaptive_16"),
    pytest.param([2048, 2048], id="2x2048_parallel_tiles_2"),
    pytest.param([512] * 8, id="8x512_adaptive_1024"),
    pytest.param([16] * 8, id="8x16_adaptive_16"),
    pytest.param([1] * 64, id="64x1_adaptive_16"),
    pytest.param([1] * 63 + [128], id="mixed_b64_decode_plus_prefill"),
    pytest.param([1, 2000], id="uneven_2_uneven_2001_tokens"),
    pytest.param([300, 1, 1, 77, 1, 512, 1, 4096 - 893], id="uneven_8"),
]


@pytest.mark.parametrize("seq_lens", SHAPE_CASES)
def test_fused_groups_match_per_group_kernel(seq_lens):
    device = "npu"
    harness = _FusedGroupsHarness(device)
    query_start_loc, positions, max_num_tokens = _make_batch(device, seq_lens)

    fused = harness.new_slot_mappings(max_num_tokens)
    harness.fused_call(fused, query_start_loc, positions, max_num_tokens)
    reference = harness.per_group_reference(query_start_loc, positions, max_num_tokens)
    torch.npu.synchronize()

    for g in range(GROUP_COUNT):
        assert torch.equal(fused[g], reference[g]), (
            f"group {g} mismatch for shape {seq_lens}: {(fused[g] != reference[g]).sum()} differing slots"
        )


def _set_negative_positions(positions: torch.Tensor, negatives) -> None:
    for idx, value in negatives:
        positions[idx] = value


# Negative positions across the adaptive small-tile, adaptive large-tile and
# parallel kernels: sharing a tile with regular tokens, at or below
# -block_size (a negative block index), and filling a whole request. Values
# at or below -block_size stay out of row 0: the per-group reference kernel
# would read before the block table there.
CIRCULAR_CASES = [
    pytest.param([64, 8], [0, 500], ((64, -1), (67, -2)), id="adaptive_small_tile"),
    pytest.param([64, 300], [0, 500], ((64, -1), (67, -2), (200, -3)), id="adaptive_large_tile"),
    pytest.param([512], [0], ((0, -1), (10, -2)), id="parallel_kernel"),
    pytest.param([64, 8], [0, 500], ((64, -200), (65, -1)), id="adaptive_below_block_size"),
    pytest.param([64, 8], [0, 500], tuple((64 + i, -300 + i) for i in range(8)), id="adaptive_all_negative_request"),
    pytest.param([600, 600], [0, 0], ((600, -200), (601, -2)), id="parallel_below_block_size"),
]


@pytest.mark.parametrize("seq_lens,position_offsets,negatives", CIRCULAR_CASES)
def test_fused_groups_circular_tail_cache_positions(seq_lens, position_offsets, negatives):
    """Negative positions must not disturb the regular tokens' slot mapping.

    Circular groups PAD them exactly like the per-group kernel.  In
    non-circular groups the per-group kernel leaves them undefined, so only
    the non-negative positions are compared against it and the fused kernels
    must PAD the negative ones.
    """
    device = "npu"
    harness = _FusedGroupsHarness(device, is_circular=(1, 0, 1))
    query_start_loc, positions, max_num_tokens = _make_batch(device, seq_lens, position_offsets)
    _set_negative_positions(positions, negatives)

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


@pytest.mark.parametrize("seq_lens,position_offsets,negatives", CIRCULAR_CASES)
def test_fused_groups_all_circular_matches_per_group(seq_lens, position_offsets, negatives):
    """Every group circular: full equality with the per-group kernel."""
    device = "npu"
    harness = _FusedGroupsHarness(device, is_circular=(1, 1, 1))
    query_start_loc, positions, max_num_tokens = _make_batch(device, seq_lens, position_offsets)
    _set_negative_positions(positions, negatives)

    fused = harness.new_slot_mappings(max_num_tokens)
    harness.fused_call(fused, query_start_loc, positions, max_num_tokens)
    reference = harness.per_group_reference(query_start_loc, positions, max_num_tokens)
    torch.npu.synchronize()

    for g in range(GROUP_COUNT):
        assert torch.equal(fused[g], reference[g])


def test_fused_groups_without_circular_ptr_matches_noncircular():
    """Production omits is_circular_ptr when no group is circular."""
    device = "npu"
    harness = _FusedGroupsHarness(device)
    for seq_lens in ([128], [1] * 63 + [128], [512] * 8, [4096]):
        query_start_loc, positions, max_num_tokens = _make_batch(device, seq_lens)
        fused_tensor = harness.new_slot_mappings(max_num_tokens)
        harness.fused_call(fused_tensor, query_start_loc, positions, max_num_tokens)
        fused_none = harness.new_slot_mappings(max_num_tokens)
        harness.fused_call(fused_none, query_start_loc, positions, max_num_tokens, circular_ptr=False)
        for tensor_out, none_out in zip(fused_tensor, fused_none):
            assert torch.equal(tensor_out, none_out)


def _sweep_batch_shapes(harness: _FusedGroupsHarness, circular_ptr: bool) -> None:
    """Run the fused path over every tile size and many (num_reqs, tokens) pairs."""
    for num_reqs in (1, 2, 3, 5, 8, 16, 33, 64):
        for tokens_per_req in (1, 7, 16, 63, 128, 300, 512, 2048):
            seq_lens = [tokens_per_req] * num_reqs
            # Uneven splits exercise mixed in-kernel tile selection.
            if num_reqs > 2:
                seq_lens[-1] = max(1, tokens_per_req - 1)
            query_start_loc, positions, max_num_tokens = _make_batch(harness.device, seq_lens)
            fused = harness.new_slot_mappings(max_num_tokens)
            harness.fused_call(fused, query_start_loc, positions, max_num_tokens, circular_ptr=circular_ptr)
    torch.npu.synchronize()


def test_fused_groups_kernel_cache_bounded():
    """Sweeping batch shapes must not grow the fused kernels' JIT caches."""
    device = "npu"
    harness = _FusedGroupsHarness(device)
    # Both HAS_CIRCULAR variants stay bounded; production uses the None one
    # unless some group really is circular.
    for circular_ptr in (False, True):
        before_adaptive, before_parallel = _fused_cache_sizes()
        _sweep_batch_shapes(harness, circular_ptr)
        after_adaptive, after_parallel = _fused_cache_sizes()
        grown_adaptive = after_adaptive - before_adaptive
        grown_parallel = after_parallel - before_parallel
        # One specialization per tile size in the ladder plus one for a
        # hypothetical device-index split, but never one per batch shape.
        assert grown_adaptive <= len(_FUSED_SLOT_MAPPING_TILE_LADDER) + 1, (
            f"adaptive kernel compiled {grown_adaptive} new specializations"
        )
        assert grown_parallel <= 2, f"parallel-tiles kernel compiled {grown_parallel} new specializations"


@pytest.mark.parametrize("circular_ptr", [False, True], ids=["no_circular", "has_circular"])
def test_prewarm_covers_every_specialization(circular_ptr):
    """After the prewarm, no batch shape may trigger a JIT compile."""
    device = "npu"
    harness = _FusedGroupsHarness(device, is_circular=(0, 1, 0) if circular_ptr else (0, 0, 0))
    _clear_fused_kernel_caches()
    prewarm_fused_slot_mapping_kernels(
        min(BLOCK_SIZES),
        8192,
        PAD_ID,
        device,
        is_circular_ptr=harness.is_circular if circular_ptr else None,
    )
    prewarmed = _fused_cache_sizes()
    assert prewarmed[0] <= len(_FUSED_SLOT_MAPPING_TILE_LADDER) and prewarmed[1] <= 1, prewarmed

    _sweep_batch_shapes(harness, circular_ptr)
    assert _fused_cache_sizes() == prewarmed


def test_multi_group_block_table_prewarm_uses_production_arguments():
    """The block-table hook must prewarm the same keys the serving path uses."""
    device = "npu"
    # One circular group, like DSV4.1's compressor state.
    harness = _FusedGroupsHarness(device, is_circular=(0, 1, 0))
    block_table = MultiGroupBlockTable.__new__(MultiGroupBlockTable)
    block_table._can_fuse_slot_mapping = True
    block_table._fused_min_block_size = min(BLOCK_SIZES)
    block_table._fused_max_num_batched_tokens = 8192
    block_table._fused_block_table_addrs = harness.block_table_addrs
    block_table._fused_is_circular = harness.is_circular
    block_table._fused_any_circular = True

    _clear_fused_kernel_caches()
    block_table.prewarm_fused_slot_mapping_kernels()
    prewarmed = _fused_cache_sizes()
    assert prewarmed[0] <= len(_FUSED_SLOT_MAPPING_TILE_LADDER) and prewarmed[1] <= 1, prewarmed

    _sweep_batch_shapes(harness, circular_ptr=True)
    assert _fused_cache_sizes() == prewarmed
