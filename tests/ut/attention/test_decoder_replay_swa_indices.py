# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm_ascend.attention.dsa_v41 import build_decoder_replay_swa_indices


def test_trimmed_prefill_excludes_unwritten_keys_and_preserves_decode_history():
    window = 4
    positions = torch.tensor([10, 97, 98, 99, 100])
    indices, lengths = build_decoder_replay_swa_indices(
        positions, window, torch.tensor([0, 1, 5]), torch.tensor([11, 101]), torch.tensor([7, 97])
    )
    assert lengths.tolist() == [[4], [1], [2], [3], [4]]
    assert indices[:, 0].tolist() == [
        [7, 8, 9, 10],
        [97, -1, -1, -1],
        [97, 98, -1, -1],
        [97, 98, 99, -1],
        [97, 98, 99, 100],
    ]
