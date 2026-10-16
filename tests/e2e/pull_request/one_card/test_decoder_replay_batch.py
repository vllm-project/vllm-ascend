# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.ops.triton.decoder_replay import gather_replay_batch


@pytest.mark.parametrize("prefill_len", [127, 128, 129, 2048])
def test_decoder_replay_keeps_window_and_decode_rows(prefill_len):
    window = 128
    kept = min(prefill_len, window)
    # A decode request precedes a prefill; output includes two padding rows.
    total = 1 + prefill_len
    padded = 1 + kept + 2
    offsets = torch.tensor([0, 1, total], dtype=torch.int32, device="npu")
    dropped = torch.tensor([0, 0, prefill_len - kept], dtype=torch.int32, device="npu")
    seq_lens = torch.tensor([301, prefill_len + 100], dtype=torch.int32, device="npu")
    encoder_replay_start = torch.zeros(2, dtype=torch.int32, device="npu")
    positions = torch.cat((torch.tensor([300]), torch.arange(100, 100 + prefill_len))).npu()
    # Noncontiguous views exercise independent source/destination group strides.
    slots = torch.arange(2 * (total + 3), device="npu").reshape(2, total + 3)[:, :total]
    out_slots = torch.empty((2, padded + 5), dtype=torch.int64, device="npu")[:, :padded]
    rows = torch.empty(padded, dtype=torch.int64, device="npu")
    out_offsets = torch.empty(3, dtype=torch.int32, device="npu")
    out_positions = torch.empty_like(rows)
    replay_start = torch.empty(2, dtype=torch.int32, device="npu")
    gather_replay_batch(
        offsets,
        dropped,
        seq_lens,
        encoder_replay_start,
        positions,
        slots,
        rows,
        out_offsets,
        out_positions,
        out_slots,
        replay_start,
        padded,
        window,
    )
    expected_rows = torch.cat((torch.tensor([0]), torch.arange(total - kept, total))).long()
    actual = 1 + kept
    assert torch.equal(rows[:actual].cpu(), expected_rows)
    assert rows[actual:].cpu().tolist() == [0, 0]
    assert out_offsets.cpu().tolist() == [0, 1, actual]
    assert torch.equal(out_positions[:actual].cpu(), positions.cpu()[expected_rows])
    assert torch.equal(out_slots[:, :actual].cpu(), slots.cpu()[:, expected_rows])
    assert (out_slots[:, actual:].cpu() == -1).all()
    prefill_start = 100 + prefill_len - kept - (window - 1 if prefill_len <= window else 0)
    assert replay_start.cpu().tolist() == [173, max(prefill_start, 0)]
