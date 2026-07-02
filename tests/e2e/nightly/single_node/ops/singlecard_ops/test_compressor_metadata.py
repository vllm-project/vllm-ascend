# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Any

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.utils import bootstrap_custom_op_env

bootstrap_custom_op_env(include_vendor_lib=True)
import vllm_ascend.vllm_ascend_C  # type: ignore[import-untyped] # noqa: E402,F401

KV_BLOCK_SIZE = 128
SLOT_MAPPING_FLAT = 1
SLOT_MAPPING_BLOCK_OFFSET = 2
ROPE_DIM = 64
ROPE_ROWS = 2048


@dataclass(frozen=True)
class CompressorMetadataCase:
    name: str
    compress_ratio: int
    query_start_loc: tuple[int, ...]
    start_pos: tuple[int, ...]
    block_table: tuple[tuple[int, ...], ...]
    expected_slot_mapping: tuple[tuple[int, int], ...] | tuple[int, ...] | None = None
    slot_mapping_format: int = SLOT_MAPPING_BLOCK_OFFSET
    num_rows: int | None = None


def _invalid_block_offset_rows(count: int) -> tuple[tuple[int, int], ...]:
    return tuple((-1, KV_BLOCK_SIZE - 1) for _ in range(count))


def _case_with_format(
    name: str,
    slot_mapping_format: int,
    **kwargs,
) -> CompressorMetadataCase:
    suffix = "a5_flat" if slot_mapping_format == SLOT_MAPPING_FLAT else "a2a3_block_offset"
    return CompressorMetadataCase(
        name=f"{name}_{suffix}",
        slot_mapping_format=slot_mapping_format,
        **kwargs,
    )


DSV4_FLASH_MIXED_CASES: list[dict[str, Any]] = [
    {
        "name": "dsv4_flash_c4_four_request_prefill_mixed_lengths_padded",
        "compress_ratio": 4,
        "query_start_loc": (0, 131, 151, 171, 178),
        "start_pos": (0, 125, 508, 640),
        "block_table": (
            (10, 11, 12, 13, 14, 15),
            (20, 21, 22, 23, 24, 25),
            (30, 31, 32, 33, 34, 35),
            (40, 41, 42, 43, 44, 45),
        ),
        "num_rows": 48,
    },
    {
        "name": "dsv4_flash_c128_four_request_prefill_mixed_lengths_padded",
        "compress_ratio": 128,
        "query_start_loc": (0, 131, 391, 521, 531),
        "start_pos": (0, 127, 254, 512),
        "block_table": (
            (50, 51, 52, 53),
            (60, 61, 62, 63),
            (70, 71, 72, 73),
            (80, 81, 82, 83),
        ),
        "num_rows": 10,
    },
    {
        "name": "dsv4_flash_c4_ten_request_decode_mixed_boundaries_padded",
        "compress_ratio": 4,
        "query_start_loc": (0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20),
        "start_pos": (126, 127, 128, 129, 130, 131, 132, 133, 134, 135),
        "block_table": (
            (90, 91, 92, 93),
            (100, 101, 102, 103),
            (110, 111, 112, 113),
            (120, 121, 122, 123),
            (130, 131, 132, 133),
            (140, 141, 142, 143),
            (150, 151, 152, 153),
            (160, 161, 162, 163),
            (170, 171, 172, 173),
            (180, 181, 182, 183),
        ),
        "num_rows": 12,
    },
    {
        "name": "dsv4_flash_c128_ten_request_decode_mixed_boundaries_padded",
        "compress_ratio": 128,
        "query_start_loc": (0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20),
        "start_pos": (126, 127, 128, 254, 255, 256, 382, 383, 384, 510),
        "block_table": (
            (190, 191, 192, 193),
            (200, 201, 202, 203),
            (210, 211, 212, 213),
            (220, 221, 222, 223),
            (230, 231, 232, 233),
            (240, 241, 242, 243),
            (250, 251, 252, 253),
            (260, 261, 262, 263),
            (270, 271, 272, 273),
            (280, 281, 282, 283),
        ),
        "num_rows": 10,
    },
]


DSV4_FLASH_CASES = [
    CompressorMetadataCase(
        name="dsv4_flash_c4_single_request_prefill_127",
        compress_ratio=4,
        query_start_loc=(0, 131),
        start_pos=(0,),
        block_table=((1, 0, 0, 0, 0, 0),),
        expected_slot_mapping=tuple((1, offset) for offset in range(32)) + _invalid_block_offset_rows(1),
    ),
    CompressorMetadataCase(
        name="dsv4_flash_c4_two_request_prefill_padded_127",
        compress_ratio=4,
        query_start_loc=(0, 131, 262),
        start_pos=(0, 0),
        block_table=((1, 0, 0, 0, 0, 0), (3, 0, 0, 0, 0, 0)),
        expected_slot_mapping=(
            tuple((1, offset) for offset in range(32))
            + tuple((3, offset) for offset in range(32))
            + _invalid_block_offset_rows(3)
        ),
    ),
    CompressorMetadataCase(
        name="dsv4_flash_c4_two_request_decode_127",
        compress_ratio=4,
        query_start_loc=(0, 2, 4, 6, 8),
        start_pos=(133, 0, 0, 0),
        block_table=(
            (1, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0),
        ),
        expected_slot_mapping=_invalid_block_offset_rows(2),
    ),
    CompressorMetadataCase(
        name="dsv4_flash_c4_decode_one_valid_one_invalid_127",
        compress_ratio=4,
        query_start_loc=(0, 2, 4),
        start_pos=(130, 133),
        block_table=((5, 0, 0, 0, 0, 0), (6, 0, 0, 0, 0, 0)),
        expected_slot_mapping=((5, 32),) + _invalid_block_offset_rows(2),
    ),
    CompressorMetadataCase(
        name="dsv4_flash_c4_single_request_prefill_flat_127",
        compress_ratio=4,
        query_start_loc=(0, 131),
        start_pos=(0,),
        block_table=((1, 0, 0, 0, 0, 0),),
        expected_slot_mapping=tuple(128 + offset for offset in range(32)) + (-1,),
        slot_mapping_format=1,
    ),
    CompressorMetadataCase(
        name="dsv4_flash_c128_single_request_prefill_127",
        compress_ratio=128,
        query_start_loc=(0, 131),
        start_pos=(0,),
        block_table=((2,),),
        expected_slot_mapping=((2, 0),) + _invalid_block_offset_rows(1),
    ),
    CompressorMetadataCase(
        name="dsv4_flash_c128_two_request_prefill_padded_127",
        compress_ratio=128,
        query_start_loc=(0, 131, 262),
        start_pos=(0, 0),
        block_table=((2,), (4,)),
        expected_slot_mapping=((2, 0), (4, 0)) + _invalid_block_offset_rows(2),
    ),
    CompressorMetadataCase(
        name="dsv4_flash_c128_two_request_decode_127",
        compress_ratio=128,
        query_start_loc=(0, 2, 4, 6, 8),
        start_pos=(133, 0, 0, 0),
        block_table=((2,), (0,), (0,), (0,)),
        expected_slot_mapping=_invalid_block_offset_rows(2),
    ),
    CompressorMetadataCase(
        name="dsv4_flash_c128_decode_one_valid_one_invalid_127",
        compress_ratio=128,
        query_start_loc=(0, 2, 4),
        start_pos=(254, 257),
        block_table=((7,), (8,)),
        expected_slot_mapping=((7, 1),) + _invalid_block_offset_rows(1),
    ),
    *(
        _case_with_format(case["name"], slot_mapping_format, **{k: v for k, v in case.items() if k != "name"})
        for case in DSV4_FLASH_MIXED_CASES
        for slot_mapping_format in (SLOT_MAPPING_BLOCK_OFFSET, SLOT_MAPPING_FLAT)
    ),
]


def _make_rope() -> tuple[torch.Tensor, torch.Tensor]:
    values = torch.arange(ROPE_ROWS * ROPE_DIM, dtype=torch.float32).reshape(ROPE_ROWS, ROPE_DIM)
    return values.to(torch.bfloat16), (values * 0.25).to(torch.bfloat16)


def _reference_outputs(
    case: CompressorMetadataCase,
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    query_start_loc = torch.tensor(case.query_start_loc, dtype=torch.int32)
    start_pos = torch.tensor(case.start_pos, dtype=torch.int32)
    block_table = torch.tensor(case.block_table, dtype=torch.int32)
    if case.expected_slot_mapping is None:
        assert case.num_rows is not None
        num_rows = case.num_rows
    else:
        num_rows = len(case.expected_slot_mapping)

    prefix = [0]
    total_rows = 0
    for req_idx, start in enumerate(start_pos.tolist()):
        seq_len = int(query_start_loc[req_idx + 1] - query_start_loc[req_idx])
        compressed_rows = 0
        if start >= 0 and seq_len > 0:
            compressed_rows = ((start + seq_len) // case.compress_ratio) - (start // case.compress_ratio)
        total_rows += compressed_rows
        prefix.append(total_rows)

    valid_rows = min(total_rows, num_rows)
    ref_cos = torch.empty((num_rows, 1, 1, rope_cos.shape[1]), dtype=rope_cos.dtype)
    ref_sin = torch.empty_like(ref_cos)
    if case.slot_mapping_format == SLOT_MAPPING_BLOCK_OFFSET:
        ref_slot = torch.empty((num_rows, 2), dtype=torch.int32)
    else:
        ref_slot = torch.empty((num_rows,), dtype=torch.int32)

    req_idx = 0
    for row in range(num_rows):
        valid = row < valid_rows
        if valid:
            while req_idx < len(start_pos) and prefix[req_idx + 1] <= row:
                req_idx += 1
            compressed_pos = int(start_pos[req_idx]) // case.compress_ratio + row - prefix[req_idx]
            block_id_offset = compressed_pos // KV_BLOCK_SIZE
            rope_pos = compressed_pos * case.compress_ratio
            if block_id_offset >= block_table.shape[1] or rope_pos >= rope_cos.shape[0]:
                valid = False
            else:
                block_id = int(block_table[req_idx, block_id_offset])
                valid = block_id >= 0

        if valid:
            ref_cos[row, 0, 0].copy_(rope_cos[rope_pos])
            ref_sin[row, 0, 0].copy_(rope_sin[rope_pos])
            slot_offset = compressed_pos % KV_BLOCK_SIZE
            if case.slot_mapping_format == SLOT_MAPPING_BLOCK_OFFSET:
                ref_slot[row, 0] = block_id
                ref_slot[row, 1] = slot_offset
            else:
                ref_slot[row] = block_id * KV_BLOCK_SIZE + slot_offset
        else:
            ref_cos[row, 0, 0].fill_(1)
            ref_sin[row, 0, 0].fill_(0)
            if case.slot_mapping_format == SLOT_MAPPING_BLOCK_OFFSET:
                ref_slot[row, 0] = -1
                ref_slot[row, 1] = KV_BLOCK_SIZE - 1
            else:
                ref_slot[row] = -1

    return ref_cos, ref_sin, ref_slot


@pytest.mark.parametrize("case", DSV4_FLASH_CASES, ids=[case.name for case in DSV4_FLASH_CASES])
def test_compressor_metadata_dsv4_flash_real_metadata(case: CompressorMetadataCase):
    torch.npu.set_device(0)
    rope_cos, rope_sin = _make_rope()
    expected_cos, expected_sin, reference_slot = _reference_outputs(case, rope_cos, rope_sin)
    if case.expected_slot_mapping is None:
        expected_slot = reference_slot
    else:
        expected_slot = torch.tensor(case.expected_slot_mapping, dtype=torch.int32)
        assert torch.equal(reference_slot, expected_slot)
    num_rows = expected_slot.shape[0]

    rope_cos_npu = rope_cos.npu()
    rope_sin_npu = rope_sin.npu()
    query_start_loc_npu = torch.tensor(case.query_start_loc, dtype=torch.int32, device="npu")
    start_pos_npu = torch.tensor(case.start_pos, dtype=torch.int32, device="npu")
    block_table_npu = torch.tensor(case.block_table, dtype=torch.int32, device="npu")

    actual_cos, actual_sin, actual_slot = torch.ops._C_ascend.compressor_metadata(
        rope_cos_npu,
        rope_sin_npu,
        query_start_loc_npu,
        start_pos_npu,
        block_table_npu,
        KV_BLOCK_SIZE,
        case.slot_mapping_format,
        case.compress_ratio,
        num_rows,
        len(case.start_pos),
    )
    out_cos = torch.empty_like(actual_cos)
    out_sin = torch.empty_like(actual_sin)
    out_slot = torch.empty_like(actual_slot)
    out_cos, out_sin, out_slot = torch.ops._C_ascend.compressor_metadata_out(
        rope_cos_npu,
        rope_sin_npu,
        query_start_loc_npu,
        start_pos_npu,
        block_table_npu,
        KV_BLOCK_SIZE,
        case.slot_mapping_format,
        case.compress_ratio,
        len(case.start_pos),
        out_cos,
        out_sin,
        out_slot,
    )
    torch.npu.synchronize()

    assert torch.equal(actual_slot.cpu(), expected_slot)
    assert torch.equal(actual_cos.cpu(), expected_cos)
    assert torch.equal(actual_sin.cpu(), expected_sin)
    assert torch.equal(out_slot.cpu(), expected_slot)
    assert torch.equal(out_cos.cpu(), expected_cos)
    assert torch.equal(out_sin.cpu(), expected_sin)
