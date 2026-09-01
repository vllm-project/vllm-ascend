# SPDX-License-Identifier: Apache-2.0

import torch

from vllm_ascend.worker.v2.spec_decode.autoregressive.speculator import (
    AscendAutoRegressiveSpeculator,
)


def _make_speculator(max_model_len: int = 128) -> AscendAutoRegressiveSpeculator:
    speculator = AscendAutoRegressiveSpeculator.__new__(AscendAutoRegressiveSpeculator)
    speculator.max_model_len = max_model_len
    speculator.num_rejected_cpu = torch.zeros(4, dtype=torch.int32)
    return speculator


def test_next_draft_seq_lens_subtract_rejected_tokens_and_zero_padding():
    speculator = _make_speculator()
    speculator._copy_num_rejected_to_cpu(torch.tensor([2, 0, 3], dtype=torch.int32), 3)
    seq_lens = torch.tensor([10, 20, 127, 99], dtype=torch.int32)

    next_seq_lens = speculator._calc_next_seq_lens_cpu(
        seq_lens,
        num_reqs=3,
        num_reqs_padded=4,
        step=1,
    )

    torch.testing.assert_close(next_seq_lens, torch.tensor([9, 21, 125, 0], dtype=torch.int32))
    torch.testing.assert_close(seq_lens, torch.tensor([10, 20, 127, 99], dtype=torch.int32))


def test_next_draft_seq_lens_clamp_to_model_len():
    speculator = _make_speculator(max_model_len=32)
    speculator._copy_num_rejected_to_cpu(torch.tensor([0, 1], dtype=torch.int32), 2)

    next_seq_lens = speculator._calc_next_seq_lens_cpu(
        torch.tensor([31, 1], dtype=torch.int32),
        num_reqs=2,
        num_reqs_padded=2,
        step=3,
    )

    torch.testing.assert_close(next_seq_lens, torch.tensor([32, 3], dtype=torch.int32))
