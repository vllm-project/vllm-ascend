# SPDX-License-Identifier: Apache-2.0

import torch

from vllm_ascend.worker.v2.sample.gumbel import flipped_gumbel_noise


def test_flipped_gumbel_noise_matches_fp32_formula():
    uniform = torch.tensor(
        [4.6566127342e-10, 1e-6, 0.25, 0.5, 0.999],
        dtype=torch.float32,
    )

    actual = flipped_gumbel_noise(uniform)
    expected = -torch.log(-torch.log1p(-uniform.clamp_min(4.6566127342e-10)))

    torch.testing.assert_close(actual, expected)


def test_flipped_gumbel_noise_preserves_winning_tail():
    smallest_uniform = torch.tensor([4.6566127342e-10], dtype=torch.float32)
    rounded_complement = torch.tensor([1.0 - 2.0**-24], dtype=torch.float32)

    flipped_tail = flipped_gumbel_noise(smallest_uniform)
    naive_tail = -torch.log(-torch.log(rounded_complement))

    assert flipped_tail.item() > 20.0
    assert naive_tail.item() < 17.0
