# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Validate logical launch-grid extents before unguarded model calls."""

import pytest
import torch

torch_npu = pytest.importorskip("torch_npu")

from vllm.triton_utils import triton  # noqa: E402

from vllm_ascend.ops.triton.v2.spec_decode.draft_window import (  # noqa: E402
    PLAN_BLOCK_SIZE,
    _gather_draft_kv_fp32,
    _plan_draft_window,
    gather_draft_kv_fp32,
)

pytestmark = pytest.mark.skipif(not torch.npu.is_available(), reason="requires an Ascend NPU")
GUARD_PROGRAM_LIMIT = 1024
SLOT_SENTINEL = -123456789
MASK_SENTINEL = 127


def test_multidimensional_grid_planner_with_output_canaries():
    capacity, query_width, n = 1184, 4, 2
    table_cpu = torch.arange(128, dtype=torch.int32).reshape(2, 64)
    rows_cpu = torch.tensor([1, 0], dtype=torch.int64)
    exact_cpu = torch.tensor([1151, 2305], dtype=torch.int32)
    frozen_cpu = torch.tensor([0, 1152], dtype=torch.int32)
    table, rows, exact, frozen = (x.to("npu") for x in (table_cpu, rows_cpu, exact_cpu, frozen_cpu))
    # Guard the possible stride errors in an unverified grid-count lowering.
    # The actual wrapper still allocates exactly the original logical shape.
    guarded_capacity = GUARD_PROGRAM_LIMIT * PLAN_BLOCK_SIZE
    slots_storage = torch.full((n * guarded_capacity,), SLOT_SENTINEL, device="npu", dtype=torch.int64)
    mask_storage = torch.full((n * query_width * guarded_capacity,), MASK_SENTINEL, device="npu", dtype=torch.uint8)
    slots = slots_storage[: n * capacity].view(n, capacity)
    mask = mask_storage[: n * query_width * capacity].view(torch.bool).view(n, 1, query_width, capacity)
    starts, visible = torch.empty_like(exact), torch.empty_like(exact)
    grid = (n, triton.cdiv(capacity, PLAN_BLOCK_SIZE), query_width)
    _plan_draft_window[grid](
        table,
        rows,
        exact,
        exact,
        frozen,
        exact,
        exact,
        slots,
        starts,
        visible,
        mask,
        64,
        1,
        1,
        1,
        0,
        1,
        128,
        128,
        1035,
        -capacity % PLAN_BLOCK_SIZE,
        query_width,
        True,
        True,
        False,
        False,
        False,
        PLAN_BLOCK_SIZE,
    )
    torch.npu.synchronize()
    columns = torch.arange(capacity)
    expected_slots, expected_masks = [], []
    for row in range(n):
        end, start = int(exact_cpu[row]), int(frozen_cpu[row])
        length = end - start
        absolute = start + columns.clamp(max=length - 1)
        expected_slots.append(table_cpu[rows_cpu[row], absolute // 128] * 128 + absolute % 128)
        expected_masks.append(
            torch.stack(
                [(columns >= length) | (columns > length - query_width + query) for query in range(query_width)]
            )
        )
    assert torch.equal(starts.cpu(), frozen_cpu)
    assert torch.equal(visible.cpu(), exact_cpu - frozen_cpu)
    assert torch.equal(slots.cpu(), torch.stack(expected_slots))
    assert torch.equal(mask.cpu(), torch.stack(expected_masks)[:, None])
    assert torch.equal(table.cpu(), table_cpu)
    assert bool((slots_storage[n * capacity :].cpu() == SLOT_SENTINEL).all())
    assert bool((mask_storage[n * query_width * capacity :].cpu() == MASK_SENTINEL).all())


@pytest.mark.parametrize("tokens", [1, 7, 8, 9, 1151, 1152])
def test_gather_grid_extent_preserves_arbitrary_tail(tokens):
    source = torch.arange(12 * 128 * 2 * 32, dtype=torch.float32).reshape(12, 128, 2, 32)
    keys, values = source.to("npu"), (-source).to("npu")
    slots_cpu = torch.arange(tokens - 1, -1, -1, dtype=torch.int64)[None]
    slots = slots_cpu.to("npu")
    actual_keys, actual_values = gather_draft_kv_fp32(slots, keys, values)
    torch.npu.synchronize()
    expected = source.flatten(0, 1)[slots_cpu]
    torch.testing.assert_close(actual_keys.cpu(), expected, atol=0, rtol=0)
    torch.testing.assert_close(actual_values.cpu(), -expected, atol=0, rtol=0)


def test_aligned_capacity_changes_reuse_gather_binary():
    driver = triton.runtime.driver

    keys = torch.zeros((135, 128, 1, 32), device="npu", dtype=torch.bfloat16)
    values = torch.ones_like(keys)
    torch.npu.synchronize()
    cache = _gather_draft_kv_fp32.device_caches[driver.active.get_current_device()][0]
    # Clear only this test process's drained in-memory kernel cache. Disk
    # cache remains valid; distinct specializations still enter the compiler
    # and return distinct hashes even when their files already exist.
    cache.clear()
    original = _gather_draft_kv_fp32.compile
    hashes = []

    def compiled(source, *args, **kwargs):
        kernel = original(source, *args, **kwargs)
        hashes.append(kernel.hash)
        return kernel

    _gather_draft_kv_fp32.compile = compiled
    try:
        for capacity in (17216, 17248, 17280):
            slots = torch.arange(capacity, device="npu", dtype=torch.int64)[None]
            output = gather_draft_kv_fp32(slots, keys, values)
            torch.npu.synchronize()
            torch.testing.assert_close(output[1].cpu(), torch.ones_like(output[1].cpu()), atol=0, rtol=0)
        assert len(hashes) == 1, "Full capacity still creates separate gather binaries"
    finally:
        _gather_draft_kv_fp32.compile = original
