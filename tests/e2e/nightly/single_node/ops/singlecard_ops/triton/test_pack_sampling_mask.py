# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Precision test for the sampling-mask packing kernel (mask_reply).

Validates ``_pack_sampling_mask_kernel`` from
``vllm_ascend.ops.triton.v2.sample.pack_sampling_mask`` against a pure NumPy
reference. This is the NPU replacement for the upstream kernel replaced by
``patch/worker/patch_v2/patch_triton.py``; the one-line fix (cast ``keep`` to
int32 before the ``tl.sum`` reduction) is exercised by the ``counts``
comparison, which fails on the upstream form when any request keeps more than
one token.
"""

import numpy as np
import pytest
import torch
from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON

from vllm_ascend.ops.triton.v2.sample.pack_sampling_mask import (
    SAMPLING_MASK_BLOCK_SIZE,
    _pack_sampling_mask_kernel,
)

DEVICE_TYPE = current_platform.device_type


def _launch(logits: torch.Tensor, num_sampled_tokens: torch.Tensor):
    """Launch the kernel, mirroring ``SamplingMaskTensors.from_logits``."""
    num_reqs, vocab_size = logits.shape
    packed_width = (vocab_size + 7) // 8
    packed_mask = torch.empty((num_reqs, packed_width), dtype=torch.uint8, device=logits.device)
    counts = torch.empty(num_reqs, dtype=torch.int32, device=logits.device)
    _pack_sampling_mask_kernel[(num_reqs,)](
        logits,
        logits.stride(0),
        logits.stride(1),
        num_sampled_tokens,
        packed_mask,
        packed_mask.stride(0),
        counts,
        vocab_size,
        BLOCK_SIZE=SAMPLING_MASK_BLOCK_SIZE,
    )
    return packed_mask, counts


def _reference(logits: torch.Tensor, num_sampled_tokens: torch.Tensor):
    """Pure NumPy reference for ``packed_mask``/``counts`` (little-endian)."""
    num_reqs, vocab_size = logits.shape
    packed_width = (vocab_size + 7) // 8
    logits_np = logits.detach().float().cpu().numpy()
    num_sampled_np = num_sampled_tokens.detach().cpu().numpy()

    keep = np.isfinite(logits_np) & (num_sampled_np[:, None] > 0)
    counts = keep.sum(axis=1).astype(np.int32)

    padded = np.zeros((num_reqs, packed_width * 8), dtype=np.uint8)
    padded[:, :vocab_size] = keep.astype(np.uint8)
    packed = (padded.reshape(num_reqs, packed_width, 8) * (1 << np.arange(8))).sum(axis=2).astype(np.uint8)
    return packed, counts


@pytest.mark.skipif(not HAS_TRITON, reason="Triton not available on this platform")
class TestPackSamplingMask:
    @pytest.mark.parametrize(
        "num_reqs,vocab_size",
        [
            (2, 8),  # exact multiple of 8, smallest useful case
            (4, 100),  # non-multiple of 8, tail byte high bits are zero
            (2, 8193),  # just above 2*BLOCK_SIZE, exercises the multi-block tail
            (3, 32000),  # realistic vocab, multiple blocks
        ],
    )
    def test_matches_reference(self, num_reqs, vocab_size):
        torch.manual_seed(0)
        logits = torch.randn(num_reqs, vocab_size, dtype=torch.float32, device=DEVICE_TYPE)
        num_sampled_tokens = torch.randint(0, 4, (num_reqs,), dtype=torch.int32, device=DEVICE_TYPE)
        # Always include at least one inactive request (num_sampled == 0).
        num_sampled_tokens[0] = 0

        packed, counts = _launch(logits, num_sampled_tokens)
        torch.npu.synchronize()

        ref_packed, ref_counts = _reference(logits, num_sampled_tokens)
        assert torch.equal(packed.cpu(), torch.from_numpy(ref_packed))
        assert torch.equal(counts.cpu(), torch.from_numpy(ref_counts))

    def test_non_finite_logits_filtered(self):
        vocab_size = 16
        logits = torch.full((1, vocab_size), -float("inf"), dtype=torch.float32, device=DEVICE_TYPE)
        logits[0, 2] = 1.0
        logits[0, 5] = float("inf")
        logits[0, 7] = float("nan")
        num_sampled_tokens = torch.tensor([1], dtype=torch.int32, device=DEVICE_TYPE)

        packed, counts = _launch(logits, num_sampled_tokens)
        torch.npu.synchronize()

        # Only token 2 is finite (excludes -inf, +inf and NaN) and active.
        assert counts.cpu().item() == 1
        assert packed.cpu()[0, 0].item() == (1 << 2)

    def test_num_sampled_zero_clears_row(self):
        vocab_size = 32
        logits = torch.randn(2, vocab_size, dtype=torch.float32, device=DEVICE_TYPE)
        num_sampled_tokens = torch.tensor([0, 1], dtype=torch.int32, device=DEVICE_TYPE)

        packed, counts = _launch(logits, num_sampled_tokens)
        torch.npu.synchronize()

        # Inactive request: count 0 and fully-zero mask.
        assert counts.cpu()[0].item() == 0
        assert packed.cpu()[0].eq(0).all().item()
        # Active request with all-finite logits: every token is in the support.
        assert counts.cpu()[1].item() == vocab_size
        assert packed.cpu()[1].eq(0xFF).all().item()

    def test_non_contiguous_logits(self):
        vocab_size = 32
        base = torch.randn(1, vocab_size * 2, dtype=torch.float32, device=DEVICE_TYPE)
        logits = base[:, ::2]  # stride-2 view exercises logits_col_stride
        num_sampled_tokens = torch.tensor([1], dtype=torch.int32, device=DEVICE_TYPE)

        packed, counts = _launch(logits, num_sampled_tokens)
        torch.npu.synchronize()

        ref_packed, ref_counts = _reference(logits, num_sampled_tokens)
        assert torch.equal(packed.cpu(), torch.from_numpy(ref_packed))
        assert torch.equal(counts.cpu(), torch.from_numpy(ref_counts))
