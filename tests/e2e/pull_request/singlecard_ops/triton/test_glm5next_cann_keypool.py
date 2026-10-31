# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.device import HardwareCapability, get_current_hardware_profile
from vllm_ascend.device.device_config import get_ascend_device_type
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.ops.triton.glm5_next_cann_keypool import write_keypool_caches

pytestmark = pytest.mark.skipif(
    get_ascend_device_type() not in (AscendDeviceType.A2, AscendDeviceType.A3, AscendDeviceType.A5),
    reason="Requires Ascend A2/A3/A5",
)

HEAD_DIM = 128
BLOCK_SIZE = 32
CACHE_BLOCKS = 3
GUARD_BYTES = 256
GUARD_VALUE = 0x55
TAIL_PAGE_ROWS = 80
TAIL_CAPACITY = 7


def _cache_views(storage, dtype, page_bytes):
    item_bytes = torch.empty((), dtype=dtype).element_size()
    cache = storage.view(dtype).as_strided(
        (CACHE_BLOCKS, BLOCK_SIZE, 1, HEAD_DIM),
        (page_bytes // item_bytes, HEAD_DIM, HEAD_DIM, 1),
        GUARD_BYTES // item_bytes,
    )
    scales = None
    if dtype == torch.float8_e4m3fn:
        scales = storage.view(torch.float32).as_strided(
            (CACHE_BLOCKS, BLOCK_SIZE),
            (page_bytes // 4, 1),
            (GUARD_BYTES + BLOCK_SIZE * HEAD_DIM) // 4,
        )
    return cache, scales


def _assert_cache_writes(storage, dtype, page_bytes, pooled, slots):
    expected = torch.full_like(storage, GUARD_VALUE, device="cpu")
    cache, scales = _cache_views(expected, dtype, page_bytes)
    scale_bytes_written = torch.zeros(expected.shape, dtype=torch.bool)
    values = pooled.float()
    if scales is not None:
        maximum = values.abs().amax(dim=-1)
        descales = torch.where(maximum > 0, maximum / torch.finfo(dtype).max, 1.0)
        values = values / descales[:, None]
    for row, slot in enumerate(slots.tolist()):
        if not 0 <= slot < CACHE_BLOCKS * BLOCK_SIZE:
            continue
        block, offset = divmod(slot, BLOCK_SIZE)
        cache[block, offset, 0] = values[row].to(dtype)
        if scales is not None:
            scales[block, offset] = descales[row]
            begin = GUARD_BYTES + block * page_bytes + BLOCK_SIZE * HEAD_DIM + offset * 4
            scale_bytes_written[begin : begin + 4] = True
    actual = storage.cpu()
    # Compare FP8 encodings and all untouched bytes, including the shared page padding.
    torch.testing.assert_close(actual[~scale_bytes_written], expected[~scale_bytes_written], rtol=0, atol=0)
    if scales is not None:
        _, actual_scales = _cache_views(actual, dtype, page_bytes)
        torch.testing.assert_close(actual_scales, scales, rtol=1e-6, atol=0)


@pytest.mark.parametrize(
    "cache_dtype",
    [
        torch.bfloat16,
        pytest.param(
            torch.float8_e4m3fn,
            marks=pytest.mark.skipif(
                not get_current_hardware_profile().supports(HardwareCapability.POOL_KEY_INDEXER_FP8),
                reason="FP8 LI cache requires Ascend 950",
            ),
        ),
    ],
)
@pytest.mark.parametrize("padded", [False, True])
@pytest.mark.parametrize("speculative", [False, True])
@pytest.mark.parametrize("use_graph", [False, True])
@torch.inference_mode()
def test_cache_write_values_scales_invalid_slots_and_graph(cache_dtype, padded, speculative, use_graph):
    item_bytes = torch.empty((), dtype=cache_dtype).element_size()
    page_bytes = BLOCK_SIZE * HEAD_DIM * item_bytes
    if cache_dtype == torch.float8_e4m3fn:
        page_bytes += BLOCK_SIZE * 4
    if padded:
        page_bytes += GUARD_BYTES
    storage = torch.full(
        (GUARD_BYTES + CACHE_BLOCKS * page_bytes + GUARD_BYTES,), GUARD_VALUE, dtype=torch.uint8, device="npu"
    )
    cache, scales = _cache_views(storage, cache_dtype, page_bytes)
    tail_storage = torch.full((2 * TAIL_PAGE_ROWS * 2 * HEAD_DIM + 2 * HEAD_DIM,), 17.0, device="npu")
    tail = tail_storage.as_strided(
        (2, TAIL_CAPACITY, 2 * HEAD_DIM), (TAIL_PAGE_ROWS * 2 * HEAD_DIM, 2 * HEAD_DIM, 1), HEAD_DIM
    )
    writes_storage = torch.full((2, 6), -1, dtype=torch.int64, device="npu")
    writes = writes_storage[:, :3]
    state_rows_cpu = torch.tensor([[1, 0, 2], [3, 4, 5]])
    state_storage = torch.zeros_like(writes_storage)
    state_rows = state_storage[:, :3]
    state_rows.copy_(state_rows_cpu)
    key = torch.empty((6, HEAD_DIM), device="npu")
    gate = torch.empty_like(key)
    state = torch.empty((6, 2 * HEAD_DIM), device="npu")
    pooled_storage = torch.full((7, HEAD_DIM + 16), 8192, dtype=torch.bfloat16, device="npu")
    pooled = pooled_storage[:, :HEAD_DIM]
    slots = torch.empty(7, dtype=torch.int64, device="npu")
    metadata = SimpleNamespace(pooled_slot_mapping=slots)
    tail_metadata = SimpleNamespace(
        cann_inputs=SimpleNamespace(write_indices=writes, state_indices=state_rows, page_rows=TAIL_PAGE_ROWS)
    )

    def run():
        write_keypool_caches(
            tail,
            cache,
            key if speculative else None,
            gate if speculative else None,
            pooled,
            metadata,
            tail_metadata,
            None if speculative else state,
            scale_cache=scales,
        )

    graph = None
    for step in (0, 1):
        key_cpu = torch.arange(6 * HEAD_DIM, dtype=torch.float32).view(6, HEAD_DIM) / 16 + step
        gate_cpu = -key_cpu - 1000
        state_cpu = torch.cat((key_cpu, gate_cpu), dim=-1)
        # Keep nonrepresentable inputs away from FP8 rounding ties.
        pool_cpu = torch.tensor([-223.0, -27.25, -6.9, -2.45, -0.24, 0.0, 0.13, 1.03, 5.1, 11.1, 20.4, 223.0])
        pool_cpu = pool_cpu.roll(step)[torch.arange(HEAD_DIM).remainder(pool_cpu.numel())]
        pool_cpu[0] = torch.finfo(torch.float8_e4m3fn).max
        pool_cpu = (pool_cpu[None] * (2.0 ** torch.arange(-3, 4))[:, None] * (2**step)).bfloat16()
        pool_cpu[2].zero_()
        slots_cpu = torch.tensor(
            [
                step,
                BLOCK_SIZE - 1 - step,
                BLOCK_SIZE + step,
                2 * BLOCK_SIZE + 3 - step,
                -1,
                CACHE_BLOCKS * BLOCK_SIZE,
                CACHE_BLOCKS * BLOCK_SIZE + 3,
            ]
        )
        writes_cpu = torch.tensor(
            [[6 - step, TAIL_PAGE_ROWS + 6 - step, 2 * TAIL_PAGE_ROWS], [TAIL_CAPACITY, -1, step]]
        )
        key.copy_(key_cpu)
        gate.copy_(gate_cpu)
        state.copy_(state_cpu)
        pooled.copy_(pool_cpu)
        slots.copy_(slots_cpu)
        writes.copy_(writes_cpu)
        if use_graph and graph is None:
            for _ in range(3):
                run()
            torch.npu.synchronize()
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                run()
        storage.fill_(GUARD_VALUE)
        tail_storage.fill_(17)
        if graph is None:
            run()
        else:
            graph.replay()
        _assert_cache_writes(storage, cache_dtype, page_bytes, pool_cpu, slots_cpu)
        expected_tail = torch.full_like(tail_storage, 17, device="cpu")
        for row, dest in enumerate(writes_cpu.flatten().tolist()):
            if dest < 0 or dest // TAIL_PAGE_ROWS >= 2 or dest % TAIL_PAGE_ROWS >= TAIL_CAPACITY:
                continue
            source = row if speculative else state_rows_cpu.flatten()[row].item()
            begin = HEAD_DIM + dest * 2 * HEAD_DIM
            expected_tail[begin : begin + 2 * HEAD_DIM] = state_cpu[source]
        torch.testing.assert_close(tail_storage.cpu(), expected_tail, rtol=0, atol=0)
