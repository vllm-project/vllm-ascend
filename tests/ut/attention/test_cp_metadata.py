# SPDX-License-Identifier: Apache-2.0

"""Tests for the backend-agnostic DCP metadata math in the attention layer."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from vllm.distributed.parallel_state import GroupCoordinator

from vllm_ascend.attention.context_parallel import dcp_metadata
from vllm_ascend.attention.utils import AscendDCPMetadata
from vllm_ascend.worker.dcp_utils import DCPManager

_BLOCK_SIZE = 128


def _create_block_table(
    dcp_world_size: int,
    dcp_rank: int,
    cp_kv_cache_interleave_size: int,
    max_num_batched_tokens: int = 512,
):
    """Build a real v1 BlockTable on CPU following tests/ut/worker/test_block_table.py."""
    with patch("vllm_ascend.worker.block_table.get_dcp_group") as mock_get_dcp_group:
        mock_dcp_group = MagicMock(spec=GroupCoordinator)
        mock_dcp_group.world_size = dcp_world_size
        mock_dcp_group.rank_in_group = dcp_rank
        mock_get_dcp_group.return_value = mock_dcp_group

        from vllm_ascend.worker.block_table import BlockTable

        return BlockTable(
            block_size=_BLOCK_SIZE,
            max_num_reqs=4,
            max_num_blocks_per_req=128,
            max_num_batched_tokens=max_num_batched_tokens,
            pin_memory=False,
            device=torch.device("cpu"),
            kernel_sizes=[_BLOCK_SIZE],
            cp_kv_cache_interleave_size=cp_kv_cache_interleave_size,
        )


@pytest.mark.parametrize("dcp_size", [2, 4])
@pytest.mark.parametrize("cp_kv_cache_interleave_size", [1, 2, 128])
def test_reference_dcp_slot_offsets_matches_doc_formula(dcp_size, cp_kv_cache_interleave_size) -> None:
    block_size = _BLOCK_SIZE
    positions = torch.arange(0, 3 * block_size * dcp_size, dtype=torch.int64)
    virtual_block_size = block_size * dcp_size
    for dcp_rank in range(dcp_size):
        owned_mask, logical_block_idx, block_offset = dcp_metadata.reference_dcp_slot_offsets(
            positions,
            block_size=block_size,
            dcp_size=dcp_size,
            cp_kv_cache_interleave_size=cp_kv_cache_interleave_size,
            dcp_rank=dcp_rank,
        )
        # Hand-computed from the design-doc formula:
        # interleave_index = offset_in_virtual_block // interleave,
        # target_rank = interleave_index % dcp_size, and the rank-local tokens
        # packed contiguously inside the virtual block.
        virtual_block_index = positions // virtual_block_size
        offset_in_virtual_block = positions % virtual_block_size
        interleave_index = offset_in_virtual_block // cp_kv_cache_interleave_size
        target_rank = interleave_index % dcp_size
        local_physical_offsets = (
            offset_in_virtual_block // (dcp_size * cp_kv_cache_interleave_size) * cp_kv_cache_interleave_size
            + offset_in_virtual_block % cp_kv_cache_interleave_size
        )
        expected_block_idx = virtual_block_index + local_physical_offsets // block_size
        expected_block_offset = local_physical_offsets % block_size

        torch.testing.assert_close(owned_mask, target_rank == dcp_rank)
        torch.testing.assert_close(logical_block_idx, expected_block_idx)
        torch.testing.assert_close(block_offset, expected_block_offset)


@pytest.mark.parametrize("dcp_size", [2, 4])
@pytest.mark.parametrize("cp_kv_cache_interleave_size", [1, 128])
def test_reference_matches_v1_block_table_dcp_slot_mapping(dcp_size, cp_kv_cache_interleave_size) -> None:
    block_size = _BLOCK_SIZE
    num_reqs = 3
    tokens_per_req = 2 * block_size * dcp_size
    positions = torch.cat([torch.arange(tokens_per_req, dtype=torch.int64) for _ in range(num_reqs)])
    num_tokens = positions.shape[0]
    query_lens = torch.full((num_reqs,), tokens_per_req, dtype=torch.int64)
    query_start_loc = torch.cat([torch.zeros(1, dtype=torch.int64), query_lens.cumsum(0)])
    req_indices = torch.repeat_interleave(torch.arange(num_reqs, dtype=torch.int64), query_lens)

    block_ids_per_req = [list(range(10 * (req + 1), 10 * (req + 1) + 8)) for req in range(num_reqs)]

    for dcp_rank in range(dcp_size):
        block_table = _create_block_table(
            dcp_size,
            dcp_rank,
            cp_kv_cache_interleave_size,
            max_num_batched_tokens=max(512, num_tokens),
        )
        # The v1 slot_mapping buffer is allocated exactly to its token capacity.
        assert block_table.slot_mapping.np.shape[0] >= num_tokens
        for req, block_ids in enumerate(block_ids_per_req):
            block_table.add_row(block_ids, req)
        block_table.compute_slot_mapping(num_reqs, query_start_loc, positions)
        actual = block_table.slot_mapping.cpu[:num_tokens].to(torch.int64)

        owned_mask, logical_block_idx, block_offset = dcp_metadata.reference_dcp_slot_offsets(
            positions,
            block_size=block_size,
            dcp_size=dcp_size,
            cp_kv_cache_interleave_size=cp_kv_cache_interleave_size,
            dcp_rank=dcp_rank,
        )
        block_matrix = torch.from_numpy(np.array(block_table.block_table.np[:num_reqs], dtype=np.int64))
        block_numbers = block_matrix[req_indices, logical_block_idx]
        expected = torch.where(
            owned_mask,
            block_numbers * block_size + block_offset,
            torch.full_like(block_offset, -1),
        )
        torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("dcp_size", [2, 4])
@pytest.mark.parametrize("cp_kv_cache_interleave_size", [1, 2, 128])
def test_local_seq_lens_counts_only_owned_tokens(dcp_size, cp_kv_cache_interleave_size) -> None:
    seq_lens = torch.tensor([0, 1, 63, 64, 65, 127, 128, 129, 255, 256, 257, 1024], dtype=torch.int32)
    all_ranks = dcp_metadata.dcp_local_seq_lens(
        seq_lens,
        dcp_size=dcp_size,
        cp_kv_cache_interleave_size=cp_kv_cache_interleave_size,
    )
    assert all_ranks.shape == (seq_lens.shape[0], dcp_size)
    # Every token is owned by exactly one rank (the integer sum promotes to int64).
    torch.testing.assert_close(all_ranks.sum(dim=1), seq_lens.to(torch.int64))

    cpu_mirror = seq_lens.clone()
    for dcp_rank in range(dcp_size):
        rank_lens = all_ranks[:, dcp_rank]
        expected_rank_lens = dcp_metadata.dcp_local_seq_lens(
            seq_lens,
            dcp_size=dcp_size,
            cp_kv_cache_interleave_size=cp_kv_cache_interleave_size,
            dcp_rank=dcp_rank,
        )
        torch.testing.assert_close(expected_rank_lens, rank_lens)
        device_lens, cpu_lens = dcp_metadata.local_context_lens(
            seq_lens,
            cpu_mirror,
            dcp_size=dcp_size,
            dcp_rank=dcp_rank,
            cp_kv_cache_interleave_size=cp_kv_cache_interleave_size,
        )
        torch.testing.assert_close(device_lens, rank_lens)
        torch.testing.assert_close(cpu_lens, rank_lens)


def test_legacy_gqa_metadata_fields_match_current_contract() -> None:
    manager = object.__new__(DCPManager)
    manager.dcp_world_size = 2
    manager.dcp_world_rank = 0
    manager.vllm_config = MagicMock()
    manager.vllm_config.parallel_config.cp_kv_cache_interleave_size = 1

    common_attn_metadata = SimpleNamespace(
        num_reqs=2,
        _seq_lens_cpu=torch.tensor([133, 261], dtype=torch.int32),
        seq_lens=torch.tensor([133, 261], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 5, 10], dtype=torch.int32),
        max_query_len=5,
        context_parallel_metadata=None,
    )
    manager.prepare_legacy_dcp_metadata(common_attn_metadata)
    manager_metadata = common_attn_metadata.context_parallel_metadata
    assert isinstance(manager_metadata, AscendDCPMetadata)

    helper_metadata = dcp_metadata.build_legacy_gqa_dcp_metadata(
        common_attn_metadata._seq_lens_cpu[: common_attn_metadata.num_reqs],
        common_attn_metadata.query_start_loc_cpu,
        common_attn_metadata.max_query_len,
        dcp_size=2,
        cp_kv_cache_interleave_size=1,
    )

    # Pin the contract shared with test_dcp_manager.py.
    np.testing.assert_array_equal(
        manager_metadata.num_computed_tokens_of_dcp,
        np.array([[67, 66], [131, 130]], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        helper_metadata.num_computed_tokens_of_dcp,
        manager_metadata.num_computed_tokens_of_dcp,
    )
    torch.testing.assert_close(helper_metadata.query_lens_cpu, manager_metadata.query_lens_cpu)
    torch.testing.assert_close(helper_metadata.query_lens_cpu, torch.tensor([5, 5], dtype=torch.int32))
    assert helper_metadata.max_query_len == manager_metadata.max_query_len == 5
    assert helper_metadata.dcp_mtp_attn_mask is None
    assert manager_metadata.dcp_mtp_attn_mask is None


def test_slot_reference_supports_empty_and_single_token() -> None:
    empty_positions = torch.empty(0, dtype=torch.int64)
    empty_owned, empty_block_idx, empty_block_offset = dcp_metadata.reference_dcp_slot_offsets(
        empty_positions,
        block_size=_BLOCK_SIZE,
        dcp_size=2,
        cp_kv_cache_interleave_size=1,
        dcp_rank=0,
    )
    for tensor in (empty_owned, empty_block_idx, empty_block_offset):
        assert tensor.numel() == 0
        assert tensor.dtype in (torch.bool, torch.int64)

    single_position = torch.tensor([0], dtype=torch.int64)
    owned_mask, logical_block_idx, block_offset = dcp_metadata.reference_dcp_slot_offsets(
        single_position,
        block_size=_BLOCK_SIZE,
        dcp_size=2,
        cp_kv_cache_interleave_size=1,
        dcp_rank=0,
    )
    assert owned_mask.tolist() == [True]
    assert logical_block_idx.tolist() == [0]
    assert block_offset.tolist() == [0]
    other_rank_owned, _, _ = dcp_metadata.reference_dcp_slot_offsets(
        single_position,
        block_size=_BLOCK_SIZE,
        dcp_size=2,
        cp_kv_cache_interleave_size=1,
        dcp_rank=1,
    )
    assert other_rank_owned.tolist() == [False]

    # The reference is pure element-wise math, so unsorted positions work too.
    positions = torch.tensor([2047, 0, 3, 128, 129, 1024, 511, 2, 256, 1], dtype=torch.int64)
    dcp_size, cp_kv_cache_interleave_size = 2, 2
    virtual_block_size = _BLOCK_SIZE * dcp_size
    for dcp_rank in range(dcp_size):
        owned_mask, logical_block_idx, block_offset = dcp_metadata.reference_dcp_slot_offsets(
            positions,
            block_size=_BLOCK_SIZE,
            dcp_size=dcp_size,
            cp_kv_cache_interleave_size=cp_kv_cache_interleave_size,
            dcp_rank=dcp_rank,
        )
        offset_in_virtual_block = positions % virtual_block_size
        interleave_index = offset_in_virtual_block // cp_kv_cache_interleave_size
        local_physical_offsets = (
            offset_in_virtual_block // (dcp_size * cp_kv_cache_interleave_size) * cp_kv_cache_interleave_size
            + offset_in_virtual_block % cp_kv_cache_interleave_size
        )
        torch.testing.assert_close(owned_mask, interleave_index % dcp_size == dcp_rank)
        torch.testing.assert_close(
            logical_block_idx,
            positions // virtual_block_size + local_physical_offsets // _BLOCK_SIZE,
        )
        torch.testing.assert_close(block_offset, local_physical_offsets % _BLOCK_SIZE)
