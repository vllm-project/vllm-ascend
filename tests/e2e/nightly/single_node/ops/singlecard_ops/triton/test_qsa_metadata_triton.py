# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass, replace

import pytest
import torch
import torch_npu  # noqa: F401
from vllm.models.qwen4_exp.common import qsa_cache
from vllm.triton_utils import HAS_TRITON
from vllm.v1.attention.backend import CommonAttentionMetadata

from vllm_ascend.device.device_config import get_ascend_device_type
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.ops.triton.qwen4_exp.qsa_metadata import build_qsa_metadata_triton

pytestmark = [
    pytest.mark.skipif(
        get_ascend_device_type() not in (AscendDeviceType.A2, AscendDeviceType.A3), reason="Requires Ascend A2/A3"
    ),
    pytest.mark.skipif(not HAS_TRITON, reason="Triton is not available"),
]

DEVICE = "npu:0"
STORAGE_BLOCK_SIZE = 192
RING_SIZE = 8
TAIL_GUARD = 17
SENTINEL = -12345


@dataclass(frozen=True)
class MetadataCase:
    query_lens: tuple[int, ...]
    prefixes: tuple[int, ...]
    ratio: int = 4
    ring: int = 0
    padding: int = 0
    cpu_query_lens: tuple[int, ...] | None = None
    columns: int | None = None
    strided: bool = False
    invalid_slots: bool = False
    physical_base: int = 7


def _query_starts(query_lens: tuple[int, ...]) -> list[int]:
    starts = [0]
    for length in query_lens:
        starts.append(starts[-1] + length)
    return starts


def _make_common(case: MetadataCase) -> CommonAttentionMetadata:
    query_starts = _query_starts(case.query_lens)
    cpu_query_starts = _query_starts(case.cpu_query_lens or case.query_lens)
    assert query_starts[-1] == cpu_query_starts[-1]
    num_tokens = query_starts[-1] + case.padding
    num_reqs = len(case.query_lens)
    seq_lens = [prefix + length for prefix, length in zip(case.prefixes, case.query_lens)]
    storage = RING_SIZE if case.ring else STORAGE_BLOCK_SIZE
    default_columns = 1 if case.ring else max(1, (max(seq_lens) // case.ratio + storage - 1) // storage)
    columns = default_columns if case.columns is None else case.columns
    block_table = torch.arange(num_reqs * max(columns, 1), dtype=torch.int32, device="cpu")
    block_table = block_table.reshape(num_reqs, max(columns, 1)) + case.physical_base
    block_table = block_table[:, :columns]
    if case.invalid_slots and columns:
        block_table[0, :] = -1
        if num_reqs > 1:
            block_table[1, 0] = 0
    if case.strided:
        backing = torch.full((num_reqs, columns * 2), -99, dtype=torch.int32, device=DEVICE)
        backing[:, ::2].copy_(block_table.to(DEVICE))
        block_table_device = backing[:, ::2]
    else:
        block_table_device = block_table.to(DEVICE)
    slots = torch.arange(num_tokens, dtype=torch.int64, device=DEVICE) + 17
    if case.invalid_slots:
        slots[::3] = -1
    return CommonAttentionMetadata(
        query_start_loc=torch.tensor(query_starts, dtype=torch.int32, device=DEVICE),
        query_start_loc_cpu=torch.tensor(cpu_query_starts, dtype=torch.int32, device="cpu"),
        seq_lens=torch.tensor(seq_lens, dtype=torch.int32, device=DEVICE),
        num_reqs=num_reqs,
        num_actual_tokens=num_tokens,
        max_query_len=max(case.query_lens),
        max_seq_len=max(seq_lens),
        block_table_tensor=block_table_device,
        slot_mapping=slots,
    )


def _make_buffers(num_tokens: int) -> tuple[torch.Tensor, ...]:
    return tuple(
        torch.full((num_tokens + TAIL_GUARD,), SENTINEL, dtype=dtype, device=DEVICE)
        for dtype in (torch.int32, torch.int64, torch.int32, torch.int64)
    )


def _assert_tail_unchanged(buffer: torch.Tensor, live_count: int) -> None:
    torch.testing.assert_close(buffer[live_count:], torch.full_like(buffer[live_count:], SENTINEL), rtol=0, atol=0)


@pytest.mark.parametrize(
    "case",
    [
        pytest.param(MetadataCase((4096,), (0,)), id="chunked-prefill-first"),
        pytest.param(MetadataCase((3904,), (4096,)), id="chunked-prefill-continuation"),
        pytest.param(MetadataCase((2048, 2048), (0, 0)), id="multi-request-prefill"),
        pytest.param(MetadataCase((4, 4), (8000, 8003)), id="mtp-decode-compressed"),
        pytest.param(MetadataCase((4, 4), (8000, 8003), ratio=1, ring=RING_SIZE), id="mtp-decode-ring"),
        pytest.param(
            MetadataCase((0, 4, 0, 4, 0), (0, 8000, 0, 8003, 0), padding=8, strided=True),
            id="empty-requests-padded-strided-compressed",
        ),
        pytest.param(
            MetadataCase((0, 4, 0, 4, 0), (0, 8000, 0, 8003, 0), ratio=1, ring=RING_SIZE, padding=8, strided=True),
            id="empty-requests-padded-strided-ring",
        ),
        pytest.param(MetadataCase((3, 5), (8000, 8003), cpu_query_lens=(4, 4)), id="device-query-boundaries"),
        pytest.param(MetadataCase((4, 4), (8000, 8003), invalid_slots=True), id="invalid-compressed-slots"),
        pytest.param(
            MetadataCase((4, 4, 4), (8000, 8003, 8005), ratio=1, ring=RING_SIZE, invalid_slots=True),
            id="null-and-negative-ring-blocks",
        ),
        pytest.param(MetadataCase((17,), (6,), ratio=1, ring=RING_SIZE), id="ring-keeps-only-current-suffix"),
        pytest.param(MetadataCase((4,), (8000,), columns=0), id="zero-column-compressed-table"),
        pytest.param(MetadataCase((4,), (8000,), ratio=1, ring=RING_SIZE, columns=0), id="zero-column-ring-table"),
        pytest.param(MetadataCase((4,), (8000,), physical_base=33554437), id="int64-physical-slots"),
        pytest.param(MetadataCase((0, 0), (0, 0), padding=8), id="all-empty-padded-requests"),
        pytest.param(MetadataCase((4, 4), (8000, 8003), ratio=1), id="ratio-one-common-slot-alias"),
        pytest.param(
            MetadataCase(
                (0, 1, 4, 0, 3, 2, 0, 5, 1), (0, 8001, 8000, 0, 8003, 8002, 0, 7999, 8001), padding=8, strided=True
            ),
            id="rolled-loop-nine-requests",
        ),
        pytest.param(
            MetadataCase(
                (0, 1, 4, 2, 0, 3, 1, 5) * 2,
                (0, 8001, 8002, 8003, 0, 8001, 8002, 8003) * 2,
                cpu_query_lens=(2,) * 16,
                padding=8,
                strided=True,
            ),
            id="rolled-loop-sixteen-device-boundaries",
        ),
        pytest.param(
            MetadataCase(
                (0, 1, 2, 0, 4, 3, 1, 5) * 4,
                (0, 8001, 8002, 0, 8004, 8005, 8006, 8007) * 4,
                cpu_query_lens=(2,) * 32,
                ratio=1,
                ring=RING_SIZE,
                padding=16,
                strided=True,
            ),
            id="rolled-loop-thirty-two-ring-requests",
        ),
        pytest.param(
            MetadataCase(
                (1, 0, 0, 2, 0, 3, 0, 0) * 4,
                (8003, 0, 0, 8006, 0, 8004, 0, 0) * 4,
                cpu_query_lens=(1, 1, 1, 1, 1, 1, 0, 0) * 4,
                padding=16,
                strided=True,
                invalid_slots=True,
            ),
            id="rolled-loop-thirty-two-compressed-empty-requests",
        ),
    ],
)
@torch.inference_mode()
def test_qsa_metadata_matches_torch(case: MetadataCase) -> None:
    common = _make_common(case)
    reference_common = replace(common, _token_to_req_indices_cache=None, _num_computed_tokens_cache=None)
    num_tokens = common.num_actual_tokens
    buffers = _make_buffers(num_tokens)
    reference_buffers = _make_buffers(num_tokens)
    work_capacity = (num_tokens + (case.ratio - 1) * common.num_reqs) // case.ratio
    work_storage = reference_work = None
    if not case.ring and case.ratio != 1:
        work_storage = torch.full((work_capacity + TAIL_GUARD, 2), SENTINEL, dtype=torch.int32, device=DEVICE)
        reference_work = torch.empty((work_capacity, 2), dtype=torch.int32, device=DEVICE)
    kwargs = {
        "storage_block_size": RING_SIZE if case.ring else STORAGE_BLOCK_SIZE,
        "compress_ratio": case.ratio,
        "circular_buffer_size": case.ring,
        "request_capacity": 32,
    }
    # Exercise the upstream reference's cold and cached token-to-request paths.
    for _ in range(2):
        work = None if work_storage is None else work_storage[:work_capacity]
        actual = build_qsa_metadata_triton(common, *buffers, k_work_metadata_buffer=work, **kwargs)
        expected = qsa_cache._build_qsa_metadata_torch(
            reference_common, *reference_buffers, k_work_metadata_buffer=reference_work, **kwargs
        )
        for result, reference in zip(actual, expected):
            torch.testing.assert_close(result, reference, rtol=0, atol=0)
        for result, owner in zip(actual[:3], buffers[:3]):
            assert result.data_ptr() == owner.data_ptr()
        slot_owner = common.slot_mapping if not case.ring and case.ratio == 1 else buffers[3]
        assert actual[3].data_ptr() == slot_owner.data_ptr()
        for buffer in buffers:
            _assert_tail_unchanged(buffer, num_tokens)
        if work_storage is not None:
            torch.testing.assert_close(work, reference_work, rtol=0, atol=0)
            _assert_tail_unchanged(work_storage, work_capacity)


@pytest.mark.parametrize("with_work", [False, True])
@torch.inference_mode()
def test_qsa_metadata_zero_tokens(with_work: bool) -> None:
    common = _make_common(MetadataCase((0, 0), (0, 0)))
    buffers = _make_buffers(0)
    work_capacity = 3
    work_storage = torch.full((work_capacity + TAIL_GUARD, 2), SENTINEL, dtype=torch.int32, device=DEVICE)
    work = work_storage[:work_capacity] if with_work else None
    actual = build_qsa_metadata_triton(
        common,
        *buffers,
        storage_block_size=STORAGE_BLOCK_SIZE,
        compress_ratio=4,
        k_work_metadata_buffer=work,
        request_capacity=32,
    )
    assert all(result.numel() == 0 for result in actual)
    for buffer in buffers:
        _assert_tail_unchanged(buffer, 0)
    if with_work:
        torch.testing.assert_close(work, torch.full_like(work, -1), rtol=0, atol=0)
    _assert_tail_unchanged(work_storage, work_capacity if with_work else 0)
