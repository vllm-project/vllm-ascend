# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import gc
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from vllm.v1.attention.backends.utils import PAD_SLOT_ID

from vllm_ascend.ops.triton.v2.spec_decode.prepare_dflash_inputs import prepare_dflash_inputs_triton

ACCURACY_CASES = [
    {
        "name": "business_b8_t1626",
        "req_lens": [192, 197, 201, 205, 209, 211, 215, 196],
        "position_starts": [0, 256, 512, 768, 1024, 1280, 1536, 1792],
        "idx_mapping": [7, 1, 12, 3, 20, 5, 31, 9],
        "max_num_reqs": 64,
        "max_num_tokens": 8192,
        "max_model_len": 8192,
        "block_size": 128,
        "num_query_per_req": 9,
        "num_speculative_steps": 8,
        "parallel_drafting_token_id": 151669,
    },
    {
        "name": "business_b64_t576",
        "req_lens": [9] * 64,
        "position_starts": [i * 64 for i in range(64)],
        "idx_mapping": list(range(64)),
        "max_num_reqs": 64,
        "max_num_tokens": 8192,
        "max_model_len": 8192,
        "block_size": 128,
        "num_query_per_req": 9,
        "num_speculative_steps": 8,
        "parallel_drafting_token_id": 151669,
    },
    {
        # Regression for the Triton-Ascend 1-element-vector lowering issue.
        "name": "min_context_compile_boundary",
        "req_lens": [1],
        "position_starts": [10],
        "idx_mapping": [1],
        "max_num_reqs": 4,
        "max_num_tokens": 16,
        "max_model_len": 64,
        "block_size": 4,
        "num_query_per_req": 3,
        "num_speculative_steps": 2,
        "parallel_drafting_token_id": 123,
    },
    {
        "name": "ragged_nondefault_block",
        "req_lens": [3, 17, 29, 61],
        "position_starts": [0, 64, 160, 320],
        "idx_mapping": [6, 1, 7, 3],
        "max_num_reqs": 8,
        "max_num_tokens": 512,
        "max_model_len": 1024,
        "block_size": 16,
        "num_query_per_req": 5,
        "num_speculative_steps": 4,
        "parallel_drafting_token_id": 321,
    },
    {
        "name": "mixed_prefill_rejected",
        "req_lens": [6, 9, 12, 15],
        "position_starts": [0, 32, 64, 96],
        "idx_mapping": [3, 0, 7, 2],
        "num_sampled": [1, 0, 1, 0],
        "num_rejected": [0, 1, 2, 3],
        "max_num_reqs": 8,
        "max_num_tokens": 128,
        "max_model_len": 256,
        "block_size": 8,
        "num_query_per_req": 5,
        "num_speculative_steps": 4,
        "parallel_drafting_token_id": 456,
    },
    {
        # Crosses both QUERY_BLOCK_SIZE=16 and SAMPLE_BLOCK_SIZE=16.
        "name": "query_sample_tile_boundary",
        "req_lens": [2] * 32,
        "position_starts": [i * 32 for i in range(32)],
        "idx_mapping": list(reversed(range(32))),
        "max_num_reqs": 32,
        "max_num_tokens": 1024,
        "max_model_len": 2048,
        "block_size": 16,
        "num_query_per_req": 18,
        "num_speculative_steps": 17,
        "parallel_drafting_token_id": 789,
    },
    {
        "name": "query_position_clamp",
        "req_lens": [2],
        "position_starts": [60],
        "idx_mapping": [1],
        "max_num_reqs": 2,
        "max_num_tokens": 8,
        "max_model_len": 64,
        "block_size": 4,
        "num_query_per_req": 5,
        "num_speculative_steps": 4,
        "parallel_drafting_token_id": 654,
    },
    {
        # No seq/sample/query-slot padding except the terminal query_start_loc entry.
        "name": "full_capacity_no_padding",
        "req_lens": [3, 4, 5, 6],
        "position_starts": [0, 8, 16, 24],
        "idx_mapping": [3, 2, 1, 0],
        "max_num_reqs": 4,
        "max_num_tokens": 20,
        "max_model_len": 128,
        "block_size": 8,
        "num_query_per_req": 5,
        "num_speculative_steps": 4,
        "parallel_drafting_token_id": 987,
    },
]


def _build_query_start_loc(req_lens):
    values = [0]
    for length in req_lens:
        values.append(values[-1] + length)
    return values


def _build_positions(req_lens, position_starts):
    values: list[int] = []
    for length, start in zip(req_lens, position_starts):
        values.extend(range(start, start + length))
    return values


def _allocate_outputs(max_num_reqs, max_num_tokens, num_speculative_steps, device):
    input_buffers = SimpleNamespace(
        input_ids=torch.full((max_num_tokens,), -12345, dtype=torch.int32, device=device),
        positions=torch.full((max_num_tokens,), -12345, dtype=torch.int64, device=device),
        query_start_loc=torch.full((max_num_reqs + 1,), -12345, dtype=torch.int32, device=device),
        seq_lens=torch.full((max_num_reqs,), -12345, dtype=torch.int32, device=device),
    )
    sample_capacity = max_num_reqs * num_speculative_steps
    return SimpleNamespace(
        input_buffers=input_buffers,
        query_slot_mapping=torch.full((max_num_tokens,), -12345, dtype=torch.int32, device=device),
        context_positions=torch.full((max_num_tokens,), -12345, dtype=torch.int64, device=device),
        context_slot_mapping=torch.full((max_num_tokens,), -12345, dtype=torch.int32, device=device),
        sample_indices=torch.full((sample_capacity,), -12345, dtype=torch.int64, device=device),
        sample_pos=torch.full((sample_capacity,), -12345, dtype=torch.int64, device=device),
        sample_idx_mapping=torch.full((sample_capacity,), -12345, dtype=torch.int32, device=device),
        temperature=torch.full((max_num_reqs,), float("nan"), dtype=torch.float32, device=device),
        seeds=torch.full((max_num_reqs,), -12345, dtype=torch.int64, device=device),
    )


def _build_inputs(case, device):
    req_lens = case["req_lens"]
    position_starts = case["position_starts"]
    idx_values = case["idx_mapping"]
    num_reqs = len(req_lens)
    max_num_reqs = case["max_num_reqs"]
    max_num_tokens = case["max_num_tokens"]

    assert len(position_starts) == num_reqs
    assert len(idx_values) == num_reqs
    assert num_reqs <= max_num_reqs
    assert sum(req_lens) <= max_num_tokens
    assert num_reqs * case["num_query_per_req"] <= max_num_tokens

    query_start_loc = _build_query_start_loc(req_lens)
    positions = _build_positions(req_lens, position_starts)

    idx_mapping = torch.zeros((max_num_reqs,), dtype=torch.int32, device=device)
    idx_mapping[:num_reqs] = torch.tensor(idx_values, dtype=torch.int32, device=device)
    input_batch = SimpleNamespace(
        num_reqs=num_reqs,
        num_scheduled_tokens=np.asarray(req_lens, dtype=np.int32),
        positions=torch.tensor(positions, dtype=torch.int64, device=device),
        query_start_loc=torch.tensor(query_start_loc, dtype=torch.int32, device=device),
        idx_mapping=idx_mapping,
    )

    num_sampled_values = case.get("num_sampled", [1] * num_reqs)
    num_rejected_values = case.get("num_rejected", [0] * num_reqs)
    assert len(num_sampled_values) == num_reqs
    assert len(num_rejected_values) == num_reqs
    assert all(0 <= rejected < length for rejected, length in zip(num_rejected_values, req_lens))

    num_sampled = torch.tensor(num_sampled_values, dtype=torch.int32, device=device)
    num_rejected = torch.tensor(num_rejected_values, dtype=torch.int32, device=device)
    last_sampled = torch.arange(1000, 1000 + max_num_reqs, dtype=torch.int64, device=device)
    next_prefill_tokens = torch.arange(2000, 2000 + max_num_reqs, dtype=torch.int32, device=device)
    input_temperature = torch.linspace(0.5, 1.5, max_num_reqs, dtype=torch.float32, device=device)
    input_seeds = torch.arange(10000, 10000 + max_num_reqs, dtype=torch.int64, device=device)

    max_query_position = max(
        start + length - 1 + case["num_query_per_req"] for start, length in zip(position_starts, req_lens)
    )
    block_table_width = max(64, max_query_position // case["block_size"] + 2)
    block_table = torch.arange(
        1,
        max_num_reqs * block_table_width + 1,
        dtype=torch.int32,
        device=device,
    ).view(max_num_reqs, block_table_width)

    outputs = _allocate_outputs(max_num_reqs, max_num_tokens, case["num_speculative_steps"], device)
    return SimpleNamespace(
        input_batch=input_batch,
        outputs=outputs,
        num_sampled=num_sampled,
        num_rejected=num_rejected,
        last_sampled=last_sampled,
        next_prefill_tokens=next_prefill_tokens,
        input_temperature=input_temperature,
        input_seeds=input_seeds,
        block_table=block_table,
    )


def _build_reference(data, case):
    sample_from_anchor = case.get("sample_from_anchor", False)
    sample_off = 0 if sample_from_anchor else 1
    expected_steps = case["num_query_per_req"] - sample_off
    assert case["num_speculative_steps"] == expected_steps

    num_reqs = data.input_batch.num_reqs
    max_num_reqs = case["max_num_reqs"]
    max_num_tokens = case["max_num_tokens"]
    num_query_per_req = case["num_query_per_req"]
    num_speculative_steps = case["num_speculative_steps"]
    block_size = case["block_size"]
    max_model_len = case["max_model_len"]

    positions = data.input_batch.positions.cpu().tolist()
    query_start_loc = data.input_batch.query_start_loc.cpu().tolist()
    idx_mapping = data.input_batch.idx_mapping.cpu().tolist()
    num_sampled = data.num_sampled.cpu().tolist()
    num_rejected = data.num_rejected.cpu().tolist()
    last_sampled = data.last_sampled.cpu().tolist()
    next_prefill_tokens = data.next_prefill_tokens.cpu().tolist()
    input_temperature = data.input_temperature.cpu().tolist()
    input_seeds = data.input_seeds.cpu().tolist()
    block_table = data.block_table.cpu().tolist()

    ref = SimpleNamespace(
        input_ids=[None] * (num_reqs * num_query_per_req),
        query_positions=[None] * (num_reqs * num_query_per_req),
        query_start_loc=[None] * (max_num_reqs + 1),
        seq_lens=[None] * max_num_reqs,
        query_slot_mapping=[None] * max_num_tokens,
        context_positions=[None] * len(positions),
        context_slot_mapping=[None] * len(positions),
        sample_indices=[None] * (max_num_reqs * num_speculative_steps),
        sample_pos=[None] * (max_num_reqs * num_speculative_steps),
        sample_idx_mapping=[None] * (max_num_reqs * num_speculative_steps),
        temperature={},
        seeds={},
    )

    for req_idx in range(num_reqs):
        state_idx = idx_mapping[req_idx]
        ctx_start = query_start_loc[req_idx]
        ctx_end = query_start_loc[req_idx + 1]
        valid_ctx_end = ctx_end - num_rejected[req_idx]
        last_valid_pos = positions[valid_ctx_end - 1]
        bonus_token = last_sampled[state_idx] if num_sampled[req_idx] > 0 else next_prefill_tokens[state_idx]

        for ctx_idx in range(ctx_start, ctx_end):
            ctx_pos = positions[ctx_idx]
            logical_block = min(ctx_pos // block_size, len(block_table[req_idx]) - 1)
            physical_block = block_table[req_idx][logical_block]
            ref.context_positions[ctx_idx] = ctx_pos
            ref.context_slot_mapping[ctx_idx] = physical_block * block_size + ctx_pos % block_size

        query_base = req_idx * num_query_per_req
        ref.query_start_loc[req_idx] = query_base
        ref.seq_lens[req_idx] = last_valid_pos + 1 + num_query_per_req
        ref.temperature[state_idx] = input_temperature[state_idx]
        ref.seeds[state_idx] = input_seeds[state_idx]

        for query_off in range(num_query_per_req):
            query_idx = query_base + query_off
            query_pos = last_valid_pos + 1 + query_off
            ref.input_ids[query_idx] = bonus_token if query_off == 0 else case["parallel_drafting_token_id"]
            ref.query_positions[query_idx] = min(query_pos, max_model_len - 1)
            logical_block = min(query_pos // block_size, len(block_table[req_idx]) - 1)
            physical_block = block_table[req_idx][logical_block]
            ref.query_slot_mapping[query_idx] = physical_block * block_size + query_pos % block_size

        for sample_local in range(num_speculative_steps):
            query_off = sample_local + sample_off
            sample_idx = req_idx * num_speculative_steps + sample_local
            query_idx = query_base + query_off
            query_pos = last_valid_pos + 1 + query_off
            ref.sample_indices[sample_idx] = query_idx
            ref.sample_pos[sample_idx] = query_pos + 1 if sample_from_anchor else query_pos
            ref.sample_idx_mapping[sample_idx] = state_idx

    last_query_end = num_reqs * num_query_per_req
    for i in range(num_reqs, max_num_reqs + 1):
        ref.query_start_loc[i] = last_query_end
    for i in range(num_reqs, max_num_reqs):
        ref.seq_lens[i] = 0

    sample_pad_start = num_reqs * num_speculative_steps
    sample_pad_end = max_num_reqs * num_speculative_steps
    for i in range(sample_pad_start, sample_pad_end):
        ref.sample_indices[i] = 0
        ref.sample_pos[i] = 0
        ref.sample_idx_mapping[i] = -1

    query_pad_start = num_reqs * num_query_per_req
    for i in range(query_pad_start, max_num_tokens):
        ref.query_slot_mapping[i] = PAD_SLOT_ID

    return ref


def _assert_exact(actual, expected):
    expected_tensor = torch.tensor(expected, dtype=actual.dtype, device=actual.device)
    torch.testing.assert_close(actual, expected_tensor, rtol=0, atol=0, equal_nan=True)


def _validate_outputs(data, case, ref):
    outputs = data.outputs
    num_reqs = data.input_batch.num_reqs
    total_context = int(data.input_batch.query_start_loc[num_reqs].item())
    total_query = num_reqs * case["num_query_per_req"]

    _assert_exact(outputs.input_buffers.input_ids[:total_query], ref.input_ids)
    _assert_exact(outputs.input_buffers.positions[:total_query], ref.query_positions)
    _assert_exact(outputs.input_buffers.query_start_loc, ref.query_start_loc)
    _assert_exact(outputs.input_buffers.seq_lens, ref.seq_lens)
    _assert_exact(outputs.query_slot_mapping, ref.query_slot_mapping)
    _assert_exact(outputs.context_positions[:total_context], ref.context_positions)
    _assert_exact(outputs.context_slot_mapping[:total_context], ref.context_slot_mapping)
    _assert_exact(outputs.sample_indices, ref.sample_indices)
    _assert_exact(outputs.sample_pos, ref.sample_pos)
    _assert_exact(outputs.sample_idx_mapping, ref.sample_idx_mapping)

    for state_idx, expected in ref.temperature.items():
        expected_tensor = torch.tensor(expected, dtype=torch.float32, device=outputs.temperature.device)
        torch.testing.assert_close(outputs.temperature[state_idx], expected_tensor, rtol=0, atol=0)

    for state_idx, expected in ref.seeds.items():
        assert outputs.seeds[state_idx].item() == expected


def _impl_args(data, case):
    return [
        data.outputs.input_buffers,
        data.outputs.query_slot_mapping,
        data.outputs.context_positions,
        data.outputs.context_slot_mapping,
        data.outputs.sample_indices,
        data.outputs.sample_pos,
        data.outputs.sample_idx_mapping,
        data.outputs.temperature,
        data.outputs.seeds,
        data.input_batch,
        data.num_sampled,
        data.num_rejected,
        data.last_sampled,
        data.next_prefill_tokens,
        data.input_temperature,
        data.input_seeds,
        data.block_table,
        case["block_size"],
        case["parallel_drafting_token_id"],
        case["num_query_per_req"],
        case["num_speculative_steps"],
        case["max_num_reqs"],
        case["max_num_tokens"],
        case["max_model_len"],
        case.get("sample_from_anchor", False),
    ]


def _cleanup():
    gc.collect()
    torch.npu.empty_cache()


@pytest.mark.parametrize("case", ACCURACY_CASES, ids=lambda case: case["name"])
def test_prepare_dflash_inputs_impl(case):
    data = _build_inputs(case, "npu")
    prepare_dflash_inputs_triton(*_impl_args(data, case))
    _validate_outputs(data, case, _build_reference(data, case))
    _cleanup()
