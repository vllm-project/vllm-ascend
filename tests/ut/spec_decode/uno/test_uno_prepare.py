# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Unit tests for UNO's eager draft-input preparation.

These are the correctness anchor for the Ascend eager port: upstream validates
its fused Triton kernel against the same eager oracle, so matching that oracle
is what makes the Ascend path trustworthy.
"""

from types import SimpleNamespace

import pytest
import torch
from vllm.v1.attention.backends.utils import PAD_SLOT_ID

from vllm_ascend.worker.v2.spec_decode.uno.prepare import (
    fill_uno_noise,
    prepare_uno_inputs,
)

MAX_MODEL_LEN = 64
BLOCK_SIZE = 8
NOISE_HIGH = 100
NOISE_SEED = 7


def _make_buffers(max_num_reqs, max_num_tokens, device="cpu"):
    return SimpleNamespace(
        device=torch.device(device),
        input_ids=torch.zeros(max_num_tokens, dtype=torch.int32, device=device),
        positions=torch.zeros(max_num_tokens, dtype=torch.int64, device=device),
        seq_lens=torch.zeros(max_num_reqs, dtype=torch.int32, device=device),
        query_start_loc=torch.zeros(max_num_reqs + 1, dtype=torch.int32, device=device),
    )


def _make_batch(num_reqs, tokens_per_req, first_positions, block_table):
    """Build a minimal InputBatch look-alike.

    ``tokens_per_req`` is the target query length of each request; the last
    position of each request is what the draft block starts after.
    """
    positions = torch.cat([torch.arange(first, first + tokens_per_req, dtype=torch.int64) for first in first_positions])
    qsl = torch.arange(num_reqs + 1, dtype=torch.int32) * tokens_per_req
    return SimpleNamespace(
        num_reqs=num_reqs,
        idx_mapping=torch.arange(num_reqs, dtype=torch.int64),
        query_start_loc=qsl,
        positions=positions,
        seq_lens_cpu_upper_bound=(torch.tensor(first_positions, dtype=torch.int64) + tokens_per_req),
    )


def _run(
    num_reqs=2,
    k=4,
    tokens_per_req=3,
    first_positions=(0, 10),
    num_rejected=None,
    num_sampled=None,
    max_model_len=MAX_MODEL_LEN,
):
    max_num_tokens = 32
    buffers = _make_buffers(max_num_reqs=4, max_num_tokens=max_num_tokens)
    block_table = torch.tensor(
        [[1, 2, 3, 4, 5, 6, 7, 8]] * num_reqs,
        dtype=torch.int32,
    )
    batch = _make_batch(num_reqs, tokens_per_req, list(first_positions), block_table)
    slot_mapping = torch.full((max_num_tokens,), -1, dtype=torch.int64)
    sample_idx_mapping = torch.full((max_num_tokens,), -1, dtype=torch.int64)
    seeds = torch.arange(num_reqs + 2, dtype=torch.int64)
    last_sampled = torch.arange(100, 100 + (num_reqs + 2) * 1, dtype=torch.int64).reshape(-1, 1)
    next_prefill = torch.arange(200, 200 + (num_reqs + 2), dtype=torch.int64).reshape(1, -1)
    if num_sampled is None:
        num_sampled = torch.ones(num_reqs, dtype=torch.int32)
    prepare_uno_inputs(
        buffers,
        slot_mapping,
        sample_idx_mapping,
        batch,
        num_sampled,
        num_rejected,
        last_sampled,
        next_prefill,
        seeds,
        block_table,
        BLOCK_SIZE,
        k,
        max_model_len,
        NOISE_SEED,
        NOISE_HIGH,
        1,
    )
    return buffers, slot_mapping, sample_idx_mapping


def test_seed_row_replays_last_sampled_token_against_base():
    buffers, _, _ = _run(num_reqs=2, k=4, tokens_per_req=3)
    # Request 0's last sampled token is last_sampled[0] = 100; row 0 of its
    # draft block must be that token, and rows 1..k-1 must be noise.
    assert buffers.input_ids[0].item() == 100
    for row in (1, 2, 3):
        assert 1 <= buffers.input_ids[row].item() < NOISE_HIGH


def test_noise_rows_are_within_range_and_deterministic():
    a, _, _ = _run()
    b, _, _ = _run()
    torch.testing.assert_close(a.input_ids, b.input_ids)
    count = 2 * 4
    noise = a.input_ids[:count]
    # Row 0 of each request is the seed token; every other row is noise.
    for req in range(2):
        base = req * 4
        for row in range(1, 4):
            assert 1 <= noise[base + row].item() < NOISE_HIGH


def testPositions_start_after_last_verified_and_are_contiguous():
    buffers, _, _ = _run(num_reqs=2, k=4, tokens_per_req=3, first_positions=(0, 10))
    # Request 0 occupies positions 0,1,2 -> draft block starts at 3.
    assert buffers.positions[:4].tolist() == [3, 4, 5, 6]
    # Request 1 occupies 10,11,12 -> draft block starts at 13.
    assert buffers.positions[4:8].tolist() == [13, 14, 15, 16]


def test_positions_are_clamped_for_the_model():
    buffers, _, _ = _run(
        num_reqs=1,
        k=4,
        tokens_per_req=2,
        first_positions=(MAX_MODEL_LEN - 3,),
        max_model_len=MAX_MODEL_LEN,
    )
    # Unclamped positions would be MAX_MODEL_LEN-1 .. MAX_MODEL_LEN+2; the model
    # input must never exceed the context.
    assert buffers.positions[:4].tolist() == [MAX_MODEL_LEN - 1] * 4


def test_slot_mapping_pads_rows_beyond_the_context():
    _, slot_mapping, _ = _run(
        num_reqs=1,
        k=4,
        tokens_per_req=2,
        first_positions=(MAX_MODEL_LEN - 3,),
        max_model_len=MAX_MODEL_LEN,
    )
    # Only the position inside the context gets a real slot; the rest are PAD.
    assert slot_mapping[0].item() != PAD_SLOT_ID
    assert slot_mapping[1:4].tolist() == [PAD_SLOT_ID] * 3


def test_slot_mapping_marks_null_block_as_nonresident():
    buffers = _make_buffers(max_num_reqs=2, max_num_tokens=16)
    block_table = torch.zeros((1, 8), dtype=torch.int32)  # block 0 == null block
    batch = _make_batch(1, 2, [0], block_table)
    slot_mapping = torch.full((16,), -1, dtype=torch.int64)
    sample_idx_mapping = torch.full((16,), -1, dtype=torch.int64)
    prepare_uno_inputs(
        buffers,
        slot_mapping,
        sample_idx_mapping,
        batch,
        torch.ones(1, dtype=torch.int32),
        None,
        torch.zeros(2, 1, dtype=torch.int64),
        torch.zeros(1, 2, dtype=torch.int64),
        torch.zeros(2, dtype=torch.int64),
        block_table,
        BLOCK_SIZE,
        2,
        MAX_MODEL_LEN,
        NOISE_SEED,
        NOISE_HIGH,
        1,
    )
    assert slot_mapping[:2].tolist() == [PAD_SLOT_ID, PAD_SLOT_ID]


def test_sample_idx_mapping_is_int64_and_marks_padding_negative():
    _, _, sample_idx_mapping = _run(num_reqs=2, k=4)
    assert sample_idx_mapping.dtype == torch.int64
    assert sample_idx_mapping[:8].tolist() == [0, 0, 0, 0, 1, 1, 1, 1]
    assert (sample_idx_mapping[8:] == -1).all()


def test_sample_idx_mapping_must_be_int64():
    buffers = _make_buffers(max_num_reqs=2, max_num_tokens=8)
    block_table = torch.ones((1, 8), dtype=torch.int32)
    batch = _make_batch(1, 2, [0], block_table)
    with pytest.raises(ValueError, match="int64"):
        prepare_uno_inputs(
            buffers,
            torch.full((8,), -1, dtype=torch.int64),
            torch.full((8,), -1, dtype=torch.int32),  # wrong dtype
            batch,
            torch.ones(1, dtype=torch.int32),
            None,
            torch.zeros(2, 1, dtype=torch.int64),
            torch.zeros(1, 2, dtype=torch.int64),
            torch.zeros(2, dtype=torch.int64),
            block_table,
            BLOCK_SIZE,
            2,
            MAX_MODEL_LEN,
            NOISE_SEED,
            NOISE_HIGH,
            1,
        )


def test_query_start_loc_is_k_per_request():
    buffers, _, _ = _run(num_reqs=2, k=4)
    assert buffers.query_start_loc[:3].tolist() == [0, 4, 8]
    # Trailing entries are pinned to the end of the real block.
    assert buffers.query_start_loc[3:].tolist() == [8] * (buffers.query_start_loc.numel() - 3)


def test_seq_lens_include_the_draft_block():
    buffers, _, _ = _run(num_reqs=2, k=4, tokens_per_req=3, first_positions=(0, 10))
    # Request 0: last position 2 -> 3 + k = 7.
    assert buffers.seq_lens[0].item() == 7
    # Request 1: last position 12 -> 13 + k = 17.
    assert buffers.seq_lens[1].item() == 17
    assert (buffers.seq_lens[2:] == 0).all()


def test_rejected_tokens_shorten_the_start_position():
    accepted, _, _ = _run(num_reqs=1, k=2, tokens_per_req=4, first_positions=(0,))
    rejected, _, _ = _run(
        num_reqs=1,
        k=2,
        tokens_per_req=4,
        first_positions=(0,),
        num_rejected=torch.tensor([2], dtype=torch.int32),
    )
    # With 2 of the 4 target rows rejected, the draft starts two positions
    # earlier.
    assert accepted.positions[0].item() - rejected.positions[0].item() == 2


def test_uniform_prefill_token_used_when_target_sampled_nothing():
    buffers, _, _ = _run(num_reqs=1, k=3, tokens_per_req=2, num_sampled=torch.zeros(1, dtype=torch.int32))
    # next_prefill_tokens[0, 0] == 200 is the bonus token for request 0.
    assert buffers.input_ids[0].item() == 200


def test_noise_hash_mixes_the_flat_row_offset():
    """Different batch-row orderings must produce different noise."""
    ids_a = torch.zeros(8, dtype=torch.int32)
    ids_b = torch.zeros(8, dtype=torch.int32)
    mask = torch.tensor([False, True, True, True] * 2)
    seeds_a = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.int64)
    seeds_b = torch.tensor([1, 1, 1, 1, 0, 0, 0, 0], dtype=torch.int64)
    fill_uno_noise(ids_a, mask, seeds_a, step=1, low=1, high=NOISE_HIGH)
    fill_uno_noise(ids_b, mask, seeds_b, step=1, low=1, high=NOISE_HIGH)
    # Row 1 has seed 0 in batch A and seed 1 in batch B at the same flat
    # offset; the flat-offset term alone must not make them collide for every
    # row.
    assert ids_a[1:4].tolist() != ids_b[1:4].tolist()


def test_noise_step_changes_the_sequence():
    ids_a = torch.zeros(4, dtype=torch.int32)
    ids_b = torch.zeros(4, dtype=torch.int32)
    mask = torch.tensor([False, True, True, True])
    seeds = torch.zeros(4, dtype=torch.int64)
    fill_uno_noise(ids_a, mask, seeds, step=1, low=1, high=NOISE_HIGH)
    fill_uno_noise(ids_b, mask, seeds, step=2, low=1, high=NOISE_HIGH)
    assert ids_a[1:].tolist() != ids_b[1:].tolist()


def test_fill_uno_noise_preserves_non_noise_rows():
    ids = torch.tensor([11, 22, 33], dtype=torch.int32)
    fill_uno_noise(
        ids,
        torch.tensor([False, True, False]),
        torch.zeros(3, dtype=torch.int64),
        step=0,
        low=1,
        high=NOISE_HIGH,
    )
    assert ids[0].item() == 11
    assert ids[2].item() == 33


def test_fill_uno_noise_rejects_empty_range():
    with pytest.raises(ValueError):
        fill_uno_noise(
            torch.zeros(2, dtype=torch.int32),
            torch.tensor([True, True]),
            torch.zeros(2, dtype=torch.int64),
            step=0,
            low=5,
            high=5,
        )


@pytest.mark.parametrize("bad_step", [-1, 1.5, True])
def test_step_must_be_a_nonnegative_int(bad_step):
    buffers = _make_buffers(max_num_reqs=2, max_num_tokens=8)
    block_table = torch.ones((1, 8), dtype=torch.int32)
    batch = _make_batch(1, 2, [0], block_table)
    with pytest.raises(ValueError, match="step"):
        prepare_uno_inputs(
            buffers,
            torch.full((8,), -1, dtype=torch.int64),
            torch.full((8,), -1, dtype=torch.int64).zero_(),
            batch,
            torch.ones(1, dtype=torch.int32),
            None,
            torch.zeros(2, 1, dtype=torch.int64),
            torch.zeros(1, 2, dtype=torch.int64),
            torch.zeros(2, dtype=torch.int64),
            block_table,
            BLOCK_SIZE,
            2,
            MAX_MODEL_LEN,
            NOISE_SEED,
            NOISE_HIGH,
            bad_step,
        )
