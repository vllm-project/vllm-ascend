#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
# Adapted from vllm-project/vllm tests/v1/distributed/test_async_llm_dp.py
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
#

"""Device-busy helpers for the "a completed pause implies an idle device" contract.

Upstream vLLM proves that contract in ``tests/v1/distributed/test_async_llm_dp.py``
(vllm-project/vllm#52914) by keeping a CUDA stream busy *after* the work has been
launched and then requiring a quiet stream as soon as ``pause_generation()``
returns. It keeps the stream busy with ``torch.cuda._sleep(200_000_000)``, which
is CUDA-only: on an Ascend host ``torch.cuda`` is unusable and ``torch.npu``
exposes no ``_sleep``, so the call fails with
``AttributeError: module 'torch._C' has no attribute '_cuda_sleep'``.

The NPU equivalent is to queue real kernels and never wait for them, which
leaves the device in exactly the state the upstream test manufactures: the host
has returned while the stream still owns queued work. ``npu_stream_idle()``
reports the same fact as ``torch.cuda.current_stream().query()``.
"""

import torch

# A 4096x4096 fp16 matmul drains in ~0.43 ms on Atlas A2/A3 (measured on
# Ascend910_9382), so this queues roughly 110 ms of device work -- the same
# order of magnitude as the ~110 ms the upstream CUDA test simulates with
# torch.cuda._sleep(200_000_000).
NPU_BUSY_MATMUL_SIZE = 4096
NPU_BUSY_MATMUL_ITERATIONS = 256


def enqueue_npu_busy_work(
    iterations: int = NPU_BUSY_MATMUL_ITERATIONS,
    size: int = NPU_BUSY_MATMUL_SIZE,
) -> None:
    """Queue device work on the current NPU stream without waiting for it.

    Returns while the work is still executing, mirroring the observable effect
    of ``torch.cuda._sleep()`` on a CUDA stream.
    """
    lhs = torch.randn(size, size, dtype=torch.float16, device="npu")
    rhs = torch.randn(size, size, dtype=torch.float16, device="npu")
    out = torch.empty_like(lhs)
    for _ in range(iterations):
        # torch.mm is asynchronous. Never call .cpu()/.item()/synchronize() in
        # this loop: the host would drain the queue and leave the device idle.
        torch.mm(lhs, rhs, out=out)


def npu_stream_idle() -> bool:
    """True when every kernel queued on the current NPU stream has finished."""
    return bool(torch.npu.current_stream().query())
