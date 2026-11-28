# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from vllm.models.hy_v4_flash.reference import hash_ngrams

from vllm_ascend.models.hy_v4_flash.engram_hash import batched_hashes


@pytest.mark.parametrize("lengths", [[0, 0], [1, 6, 0, 3], [6] * 8])
def test_ragged_hashes_preserve_request_history_and_padding(lengths):
    torch.manual_seed(1024)
    count = sum(lengths)
    ids = torch.randint(-2, 34, (count + 4,))
    lookback = torch.randint(-2, 34, (len(lengths), 2))
    token_map = torch.arange(32).flip(0)
    multipliers = torch.tensor([[1000000007, 1000000009, 1000000033]]).expand(4, -1)
    primes = torch.tensor([[[13, 17], [19, 23]]]).expand(4, -1, -1)
    offsets = torch.tensor([0, 13, 30, 49]).expand(4, -1)
    starts = [0]
    expected = torch.full((4, count + 4, 4), -1, dtype=torch.int32)
    for request, length in enumerate(lengths):
        start = starts[-1]
        stop = start + length
        starts.append(stop)
        if length:
            expected[:, start:stop] = hash_ngrams(
                ids[start:stop],
                lookback[request],
                token_map,
                multipliers,
                primes,
                offsets,
                0,
                31,
                [30],
            )
    actual = torch.empty_like(expected)
    batched_hashes(ids, lookback, starts, token_map, multipliers, primes, offsets, 0, 31, [30], actual)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
