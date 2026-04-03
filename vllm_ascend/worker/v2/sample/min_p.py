# Adapt from https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu/sample/min_p.py.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.

import torch
from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num


@triton.jit(do_not_specialize=["num_reqs", "SUB_BLOCK_SIZE"])
def _min_p_kernel(
    logits_ptr,
    logits_out_ptr,
    logits_stride,
    min_p_ptr,
    num_reqs,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
    SUB_BLOCK_SIZE,
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

    BLOCK_SIZE = min(triton.next_power_of_2(vocab_size), 8192)
    SUB_BLOCK_SIZE = triton.cdiv(num_reqs, core_nums)

    _min_p_kernel[(core_nums,)](
        logits,
        logits,
        logits.stride(0),
        min_p,
        num_reqs,
        vocab_size,
        BLOCK_SIZE=BLOCK_SIZE,
        SUB_BLOCK_SIZE=SUB_BLOCK_SIZE,
        multibuffer=False,
    )
