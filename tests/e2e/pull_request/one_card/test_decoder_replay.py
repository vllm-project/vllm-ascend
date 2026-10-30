# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm_ascend.ops.triton.decoder_replay import gather_replay_batch


@pytest.mark.parametrize("has_kv_floor", [False, True])
def test_decoder_replay_read_floor_preserves_encoder_bounds(has_kv_floor):
    device = torch.device("npu")
    query = torch.arange(4, dtype=torch.int32, device=device)
    dropped = torch.zeros_like(query)
    seq_lens = torch.tensor([7, 11, 15], dtype=torch.int32, device=device)
    encoder_start = torch.tensor([0, 10, 0], dtype=torch.int32, device=device)
    kv_floor = torch.tensor([6, 8, 0], dtype=torch.int32, device=device) if has_kv_floor else None
    positions = seq_lens.long() - 1
    source_slots = torch.arange(3, dtype=torch.int64, device=device).view(1, -1)
    rows = torch.empty(3, dtype=torch.int64, device=device)
    offsets = torch.empty_like(query)
    replay_positions = torch.empty_like(positions)
    slots = torch.empty_like(source_slots)
    starts = torch.empty_like(seq_lens)

    gather_replay_batch(
        query,
        dropped,
        seq_lens,
        encoder_start,
        positions,
        source_slots,
        rows,
        offsets,
        replay_positions,
        slots,
        starts,
        3,
        2,
        kv_floor=kv_floor,
    )

    assert starts.cpu().tolist() == ([6, 10, 13] if has_kv_floor else [5, 10, 13])
    assert rows.cpu().tolist() == [0, 1, 2]
    torch.testing.assert_close(replay_positions, positions)
    torch.testing.assert_close(slots, source_slots)
