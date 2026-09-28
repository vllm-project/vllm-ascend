# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ascend replacement for the sampling-mask packing kernel.

This is the NPU-ported ``_pack_sampling_mask_kernel`` from
``vllm.v1.worker.gpu.sample.output``. It packs the finite-logit support of
each request (the set of token ids whose logits are not ``±inf``/``NaN``) into
a bit-packed ``uint8`` buffer plus a per-request ``counts`` tensor.

The only difference from the upstream kernel is a single line: triton-ascend
does not upcast the result of ``tl.sum`` on an ``int1`` tensor to ``int32``
(unlike native CUDA Triton), which truncates ``counts`` to 0/1. We therefore
cast ``keep`` to ``int32`` *before* the reduction.
"""

import torch
from vllm.triton_utils import tl, triton

# NPU backend cannot launch the kernel with the upstream BLOCK_SIZE=8192.
# The tightest constraint is the strided-load path: when logits is
# non-contiguous (logits_col_stride != 1) the masked ``tl.load`` lowers to a
# gather, whose max block on Ascend is far smaller than a contiguous load's.
# 2048/4096 fail on that case even though contiguous inputs work; 1024 keeps
# both contiguous and strided cases under the limit. Temporary reduction
# pending an NPU-side fix.
SAMPLING_MASK_BLOCK_SIZE = 1024


@triton.jit
def _pack_sampling_mask_kernel(
    logits_ptr,
    logits_row_stride,
    logits_col_stride,
    num_sampled_tokens_ptr,
    packed_mask_ptr,
    packed_mask_row_stride,
    counts_ptr,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
):
    req_idx = tl.program_id(0)
    is_active = tl.load(num_sampled_tokens_ptr + req_idx) > 0
    count = tl.zeros((), dtype=tl.int32)

    for start_idx in range(0, vocab_size, BLOCK_SIZE):
        offsets = start_idx + tl.arange(0, BLOCK_SIZE)
        valid = offsets < vocab_size
        logits = tl.load(
            logits_ptr + req_idx * logits_row_stride + offsets * logits_col_stride,
            mask=valid,
            other=-float("inf"),
        )
        keep = (logits > -float("inf")) & (logits < float("inf")) & is_active
        # triton-ascend keeps the int1 sum in int1, truncating large counts;
        # cast to int32 first so the reduction accumulates the full count.
        count += tl.sum(keep.to(tl.int32))
        keep = tl.reshape(keep.to(tl.int32), (BLOCK_SIZE // 8, 8))
        bit_shifts = tl.arange(0, 8)[None, :]
        packed = tl.sum(keep << bit_shifts, axis=1).to(tl.uint8)
        byte_offsets = start_idx // 8 + tl.arange(0, BLOCK_SIZE // 8)
        tl.store(
            packed_mask_ptr + req_idx * packed_mask_row_stride + byte_offsets,
            packed,
            mask=byte_offsets < tl.cdiv(vocab_size, 8),
        )

    tl.store(counts_ptr + req_idx, count)


def sampling_mask_from_logits(cls, logits, num_sampled_tokens):
    """NPU replacement for ``SamplingMaskTensors.from_logits``.

    Identical to the upstream classmethod except it launches with the reduced
    ``SAMPLING_MASK_BLOCK_SIZE`` instead of the upstream hard-coded 8192, which
    the Ascend backend cannot launch. Wired in via
    ``patch/worker/patch_v2/patch_triton.py``.
    """
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

    return cls(packed_mask, counts, vocab_size)
