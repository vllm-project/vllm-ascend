# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm_ascend.ops.triton.decoder_replay import gather_replay_batch


def test_decoder_replay_gather_preserves_slots_and_prefix_floor():
    device = torch.device("npu")
    query = torch.tensor([0, 1, 7, 13], dtype=torch.int32, device=device)
    dropped = torch.tensor([0, 0, 4, 4], dtype=torch.int32, device=device)
    seq_lens = torch.tensor([20, 16, 13], dtype=torch.int32, device=device)
    encoder_start = torch.tensor([19, 12, 9], dtype=torch.int32, device=device)
    positions = torch.arange(13, device=device)
    source_slots = torch.arange(32, device=device).reshape(2, 16)[:, :13]
    rows = torch.empty(12, dtype=torch.int64, device=device)
    offsets = torch.empty(4, dtype=torch.int32, device=device)
    replay_positions = torch.empty_like(rows)
    slots = torch.empty((2, 16), dtype=torch.int64, device=device)[:, :12]
    starts = torch.empty(3, dtype=torch.int32, device=device)

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
        12,
        2,
    )

    retained = torch.tensor([0, 5, 6, 7, 8, 9, 10, 11, 12], device=device)
    torch.testing.assert_close(rows[:9], retained)
    torch.testing.assert_close(rows[9:], torch.zeros(3, dtype=torch.int64, device=device))
    torch.testing.assert_close(replay_positions[:9], positions[retained])
    torch.testing.assert_close(slots[:, :9], source_slots[:, retained])
    torch.testing.assert_close(slots[:, 9:], torch.full((2, 3), -1, dtype=torch.int64, device=device))
    assert offsets.cpu().tolist() == [0, 1, 3, 9]
    assert starts.cpu().tolist() == [19, 14, 9]
