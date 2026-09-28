# SPDX-License-Identifier: Apache-2.0
"""Unit tests for fused_copy_sfa top-k slot binding and tail geometry."""

import pytest

from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.copy_sfa_topk_slots import (
    CopySfaTopkSlotAllocator,
    copy_sfa_pool_capacity,
    copy_sfa_prefill_dest_geometry,
    copy_sfa_tail_geometry,
)


def test_copy_sfa_pool_capacity_includes_padding_rows():
    assert copy_sfa_pool_capacity(8) == 10


def test_copy_sfa_tail_geometry_skips_aligned_prefix():
    assert copy_sfa_tail_geometry(10240, 128) == (0, 0)
    assert copy_sfa_tail_geometry(0, 128) == (0, 0)


def test_copy_sfa_tail_geometry_keeps_incomplete_last_block():
    assert copy_sfa_tail_geometry(10367, 128) == (127, 80)
    assert copy_sfa_tail_geometry(129, 128) == (1, 1)


def test_copy_sfa_prefill_dest_geometry_dense_for_short_prompts():
    # Whole prompt fits the hot region: dense full-row D2D, including the
    # 128-aligned case that the tail path would skip entirely.
    assert copy_sfa_prefill_dest_geometry(129, 128, 8192) == (True, 0, 0)
    assert copy_sfa_prefill_dest_geometry(4096, 128, 8192) == (True, 0, 0)
    assert copy_sfa_prefill_dest_geometry(8192, 128, 8192) == (True, 0, 0)  # boundary
    assert copy_sfa_prefill_dest_geometry(0, 128, 8192) == (True, 0, 0)


def test_copy_sfa_prefill_dest_geometry_tail_for_long_prompts():
    # Longer than the hot budget: keep the circular-tail-only prefetch.
    assert copy_sfa_prefill_dest_geometry(10367, 128, 8192) == (False, 127, 80)
    # Block-aligned long prompt still has no tail to prefetch.
    assert copy_sfa_prefill_dest_geometry(8320, 128, 8192) == (False, 0, 0)


def test_copy_sfa_slot_allocator_reuses_and_releases():
    allocator = CopySfaTopkSlotAllocator(2)
    first = allocator.bind("req-a")
    second = allocator.bind("req-b")
    assert {first, second} == {0, 1}
    assert allocator.bind("req-a") == first
    allocator.release("req-a")
    assert allocator.get("req-a") is None
    reused = allocator.bind("req-c")
    assert reused == first


def test_copy_sfa_slot_allocator_exhausts_capacity():
    allocator = CopySfaTopkSlotAllocator(1)
    allocator.bind("req-a")
    with pytest.raises(RuntimeError, match="exhausted"):
        allocator.bind("req-b")
