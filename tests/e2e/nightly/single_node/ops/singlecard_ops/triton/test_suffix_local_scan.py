# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MRV2 suffix local scan NPU tests adapted from vLLM #60339."""

import numpy as np
import pytest
import torch

from .suffix_test_utils import (
    DEVICE,
)

# isort: split
# Shared setup skips unsupported platforms before importing the NPU kernel.
from vllm_ascend.ops.triton.spec_decode.suffix_local_scan import _suffix_local_scan_kernel  # noqa: E402


@pytest.mark.parametrize("depth", [8, 24])
@pytest.mark.parametrize("max_len", [300, 2048])
def test_local_scan_matches_reference(depth, max_len):
    """Exercise loop compilation, boundaries and mismatch persistence directly."""
    block = 128 if max_len < 2048 else 256
    lengths = [0, 1, 2, block - 1, block, block + 1, max_len - 1, max_len, max_len]
    slots = [13, 2, 11, 4, 9, 6, 7, 0, 15]
    sampled = [1] * (len(slots) - 1) + [0]
    rng = np.random.default_rng(42)
    tokens = rng.integers(0, 4, size=(16, max_len), dtype=np.int32)
    # Repeated tokens cover full-depth matches and rightmost-position ties.
    tokens[slots[-3:]] = 1
    total_len = np.zeros(16, dtype=np.int32)
    total_len[slots] = lengths
    num_blocks = (max_len + block - 1) // block
    expected = np.zeros((len(slots), num_blocks), dtype=np.int64)
    for batch_idx, (slot, length, num_sampled) in enumerate(zip(slots, lengths, sampled)):
        if not num_sampled:
            continue
        for pos in range(1, length):
            matched = 0
            while matched < min(depth, pos) and tokens[slot, pos - 1 - matched] == tokens[slot, length - 1 - matched]:
                matched += 1
            if matched:
                block_idx = pos // block
                expected[batch_idx, block_idx] = max(expected[batch_idx, block_idx], matched * (max_len + 1) + pos)

    scratch = torch.full(expected.shape, -1, dtype=torch.int64, device=DEVICE)
    _suffix_local_scan_kernel[(len(slots), num_blocks)](
        torch.tensor(tokens, device=DEVICE),
        max_len,
        torch.tensor(slots, dtype=torch.int32, device=DEVICE),
        torch.tensor(total_len, device=DEVICE),
        torch.tensor(sampled, dtype=torch.int32, device=DEVICE),
        scratch,
        num_blocks,
        max_len + 1,
        MAX_DEPTH=depth,
        BLOCK=block,
        num_warps=4,
    )
    np.testing.assert_array_equal(scratch.cpu().numpy(), expected)
