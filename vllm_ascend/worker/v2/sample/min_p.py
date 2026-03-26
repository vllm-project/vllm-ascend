# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num


@triton.jit
def _min_p_kernel(
    logits_ptr,
    logits_out_ptr,
    logits_stride,
    min_p_ptr,
    num_reqs,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
    SUB_BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    start_req = pid * SUB_BLOCK_SIZE
    end_req = tl.minimum(start_req + SUB_BLOCK_SIZE, num_reqs)

    for req_idx in range(start_req, end_req):
        min_p = tl.load(min_p_ptr + req_idx).to(tl.float32)

        if min_p > 0.0:
            max_val_vec = tl.full([BLOCK_SIZE], float("-inf"), dtype=tl.float32)
            for i in range(0, vocab_size, BLOCK_SIZE):
                offsets = i + tl.arange(0, BLOCK_SIZE)
                mask = offsets < vocab_size

                logits = tl.load(logits_ptr + req_idx * logits_stride + offsets, mask=mask, other=float("-inf"))

                max_val_vec = tl.maximum(logits, max_val_vec)
            max_val = tl.max(max_val_vec).to(tl.float32)

            threshold = max_val + tl.log(min_p)
            for i in range(0, vocab_size, BLOCK_SIZE):
                offsets = i + tl.arange(0, BLOCK_SIZE)
                mask = offsets < vocab_size

                logits = tl.load(logits_ptr + req_idx * logits_stride + offsets, mask=mask, other=float("-inf"))

                logits = tl.where(logits < threshold, float("-inf"), logits)
                tl.store(logits_out_ptr + req_idx * logits_stride + offsets, logits, mask=mask)


def apply_min_p(logits: torch.Tensor, min_p: torch.Tensor) -> None:
    if logits.numel() == 0:
        return

    num_reqs, vocab_size = logits.shape

    assert logits.stride(-1) == 1, "The last dimension of logits (vocab_size) must be contiguous in memory."
    assert min_p.is_contiguous(), "The min_p tensor must be contiguous."
    assert min_p.dim() == 1 and min_p.size(0) == num_reqs, "The shape of min_p must be (num_reqs,)."

    vec_core = get_vectorcore_num()
    core_nums = min(num_reqs, vec_core)

    BLOCK_SIZE = 8 * 1024

    BLOCK_SIZE = min(triton.next_power_of_2(vocab_size), BLOCK_SIZE)

    _min_p_kernel[(core_nums,)](
        logits,
        logits,
        logits.stride(0),
        min_p,
        num_reqs,
        vocab_size,
        BLOCK_SIZE,
        SUB_BLOCK_SIZE=triton.cdiv(num_reqs, core_nums),
        multibuffer=False,
    )
