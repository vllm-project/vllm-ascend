# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Direct P2P owner staging: complete logical blocks, not owner chunks."""

from __future__ import annotations

import pytest
import torch

from vllm_ascend.distributed.kv_transfer.utils import owner_transfer

# The owner slab maps two-megabyte chunks. A block larger than one chunk (and
# not chunk-aligned) must still be staged as one complete logical block.
OWNER_CHUNK_BYTES = 2 << 20


def _contiguous_stager(fields, capacity):
    buffer = torch.zeros(capacity, dtype=torch.int8)
    return owner_transfer.OwnerBlockStageBuffer(torch.device("cpu"), fields, buffer=buffer)


def _recording_copy(monkeypatch):
    copies: list[tuple[list[int], list[int], list[int]]] = []

    def copy(src, dst, sizes):
        copies.append((list(src), list(dst), list(sizes)))

    monkeypatch.setattr(owner_transfer, "_device_copy", copy)
    return copies


def test_a_block_crossing_owner_chunks_is_staged_whole(monkeypatch):
    copies = _recording_copy(monkeypatch)
    block_bytes = 3 * OWNER_CHUNK_BYTES + 4096  # spans four 2 MiB chunks
    stager = _contiguous_stager([("kv", block_bytes)], block_bytes)
    controller = owner_transfer.OwnerP2PStageController(
        stager,
        group_fields={0: ["kv"]},
        field_sources={"kv": (0x1000_0000, block_bytes)},
    )

    first_slot, slot_count = controller.stage_request("req-0", [(0, [0])])

    assert (first_slot, slot_count) == (0, 1)
    # One copy range per field, sized as the complete block. It is never
    # clipped to the owner's local 2 MiB chunk.
    assert len(copies) == 1
    src, dst, sizes = copies[0]
    assert src == [0x1000_0000]
    assert sizes == [block_bytes]
    assert sizes[0] > OWNER_CHUNK_BYTES * 2
    # This is the address range the decode-side transport is told to read:
    # one complete block starting at the block's first staged slot.
    slot_map = owner_transfer.owner_stage_slot_map(first_slot, [(0, 0)])
    assert dst == [stager.slot_address("kv", slot_map[(0, 0)])]


def test_slots_and_field_regions_are_field_major(monkeypatch):
    copies = _recording_copy(monkeypatch)
    small, large = 1024, 4 * OWNER_CHUNK_BYTES
    stager = _contiguous_stager([("k", small), ("v", large)], 2 * (small + large))
    controller = owner_transfer.OwnerP2PStageController(
        stager,
        group_fields={0: ["k", "v"]},
        field_sources={"k": (0x100, 4096), "v": (0x200, 8192)},
    )

    first_slot, slot_count = controller.stage_request("req-1", [(0, [5, 6])])

    assert (first_slot, slot_count) == (0, 2)
    src, dst, sizes = copies[0]
    # Sources follow the logical owner VA and include the block stride.
    assert src == [0x100 + 5 * 4096, 0x200 + 5 * 8192, 0x100 + 6 * 4096, 0x200 + 6 * 8192]
    # Each field owns a contiguous region; consecutive slots are adjacent for
    # that field, so the decode side can merge consecutive block transfers.
    assert dst == [
        stager.field_ptr("k") + 0 * small,
        stager.field_ptr("v") + 0 * large,
        stager.field_ptr("k") + 1 * small,
        stager.field_ptr("v") + 1 * large,
    ]
    assert sizes == [small, large, small, large]
    assert stager.field_ptr("v") == stager.field_ptr("k") + stager.slot_count * small


def test_slot_map_matches_the_shared_key_order():
    group_blocks = [(0, [7, 8]), (0, [8]), (1, [3])]
    keys = owner_transfer.owner_stage_keys(group_blocks)
    assert keys == [(0, 7), (0, 8), (1, 3)]
    assert owner_transfer.owner_stage_slot_map(11, keys) == {(0, 7): 11, (0, 8): 12, (1, 3): 13}


def test_release_reclaims_slots_and_capacity_is_enforced(monkeypatch):
    _recording_copy(monkeypatch)
    stager = _contiguous_stager([("kv", 64)], 2 * 64)
    controller = owner_transfer.OwnerP2PStageController(
        stager,
        group_fields={0: ["kv"]},
        field_sources={"kv": (0, 64)},
    )

    first_slot, _ = controller.stage_request("req-a", [(0, [0, 1])])
    assert first_slot == 0
    with pytest.raises(RuntimeError, match="staging buffer is full"):
        controller.stage_request("req-b", [(0, [2])])

    controller.release_request("req-a")
    first_slot, slot_count = controller.stage_request("req-b", [(0, [2])])
    assert (first_slot, slot_count) == (0, 1)


def test_one_request_cannot_be_staged_twice_with_a_different_size(monkeypatch):
    _recording_copy(monkeypatch)
    stager = _contiguous_stager([("kv", 64)], 4 * 64)
    controller = owner_transfer.OwnerP2PStageController(
        stager,
        group_fields={0: ["kv"]},
        field_sources={"kv": (0, 64)},
    )

    assert controller.stage_request("req", [(0, [0])]) == (0, 1)
    assert controller.stage_request("req", [(0, [0])]) == (0, 1)
    with pytest.raises(RuntimeError, match="different block counts"):
        controller.stage_request("req", [(0, [0, 1])])


def test_unknown_group_is_rejected(monkeypatch):
    _recording_copy(monkeypatch)
    stager = _contiguous_stager([("kv", 64)], 4 * 64)
    controller = owner_transfer.OwnerP2PStageController(
        stager,
        group_fields={0: ["kv"]},
        field_sources={"kv": (0, 64)},
    )

    with pytest.raises(ValueError, match="unknown KV cache groups"):
        controller.stage_request("req", [(3, [0])])


def test_a_block_larger_than_the_staging_buffer_is_rejected():
    with pytest.raises(ValueError, match="needs .* bytes per complete block"):
        _contiguous_stager([("kv", 4096)], 1024)
