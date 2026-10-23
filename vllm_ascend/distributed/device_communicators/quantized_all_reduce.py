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
"""Quantized all-reduce for large bf16 TP collectives on Atlas A2/A3.

Halves the all-reduce payload (bf16 -> int8 -> reduce -> bf16). Opt-in, and it
only pays off for large messages, so it is gated by a payload threshold.

Configuration
-------------
VLLM_ASCEND_QUANTIZED_ALL_REDUCE=1
    Enable it. Default 0; only "0"/"1" are accepted. Atlas A2/A3, 1 < TP <= 8.
    Read when the TP communicator is created, so restart after changing it.
    Active when every TP worker logs
    "Quantized all-reduce enabled (min_numel=..., max_q=...)".

VLLM_ASCEND_QUANTIZED_ALL_REDUCE_MIN_NUMEL=<elements>    (default 33554432)
    Payload threshold in elements: payload = MIN_NUMEL * 2 bytes (bf16), so the
    default is 64 MiB == 16384 tokens at hidden_size 2048 (a full
    max_num_batched_tokens chunk). A collective is quantized only when one
    scheduler step holds at least MIN_NUMEL / hidden_size tokens; concurrency
    matters only through batching (16 concurrent 1k-token requests form one
    16k-token step and qualify, a single 1k-token request does not). Decode
    steps (a few thousand elements) never qualify.

    Each collective carries a fixed ~0.75 ms cost, hence a ~36 MiB break-even:
    4 MiB +359%, 16 MiB +67%, 32 MiB +8%, 64 MiB -23%, 128 MiB -23% (vs bf16).

    Keep the default and do not go below 16777216 (32 MiB). If
    max_num_batched_tokens * hidden_size * 2 < 64 MiB the threshold is
    unreachable, so leave the feature disabled.

Design notes
------------
  * Integer safety without a clamp: max_q = 127 // TP and scale = per-channel
    global amax / max_q, so |x/scale| <= max_q by construction and the TP-way
    sum is bounded by TP * max_q <= 127 (HCCL wraps rather than saturates).
  * Never add an int8 clamp: int8 clamp_/clamp/hardtanh are ~250x slower than
    minimum/maximum on Ascend (Ascend/pytorch#178) and made this pipeline ~8x
    slower than the bf16 all-reduce it replaces.
  * Precision: ~6 bit/channel at TP=4 (rel-L2 3.9% vs an exact bf16 reduce),
    ~5 bit at TP=8; TP>8 is rejected (too few levels).
  * Measured on 4x 910B3, 64 MiB payload: 1.73 ms (bf16) -> 1.21 ms (-30%).
    End to end (1k-32k input, 1k output, 32-way concurrency): +1.3% .. +3.6%
    output throughput, -5% .. -6% mean TTFT; GPQA-diamond (198 q): 81.82% vs
    80.81% baseline, i.e. no measurable accuracy loss.
"""

import os

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401
from vllm.logger import logger

from vllm_ascend.device.device_config import get_ascend_device_type
from vllm_ascend.device.hardware import AscendDeviceType

# Validated on A2 (910B, measured end to end); A3 (910C) supports the kernels
# but its int8 reduce behaviour is unmeasured.
_VALID_DEVICE_TYPES = (AscendDeviceType.A2, AscendDeviceType.A3)
# TP>8 leaves too few int8 levels (ws=16 -> max_q=7), so refuse instead.
_MAX_VALID_WORLD_SIZE = 8


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        logger.warning("Invalid %s, falling back to %s", name, default)
        return default


class QuantizedAllReduce:
    """Wrapper adding int8 quantized all-reduce around a base all_reduce."""

    def __init__(self, group: "dist.ProcessGroup") -> None:
        self.group = group
        self.world_size = dist.get_world_size(group)
        device_type = get_ascend_device_type()
        # Fail fast on unvalidated families: a silent bf16 fallback would hide
        # a misconfiguration, and 310P lacks bf16 npu_quantize support anyway.
        if device_type not in _VALID_DEVICE_TYPES:
            raise RuntimeError(
                "VLLM_ASCEND_QUANTIZED_ALL_REDUCE=1 is only validated on "
                f"{'/'.join(t.name for t in _VALID_DEVICE_TYPES)} devices, "
                f"but the current device family is {device_type.name}. "
                "Unset the environment variable or use a supported device."
            )
        if self.world_size > _MAX_VALID_WORLD_SIZE:
            raise RuntimeError(
                f"VLLM_ASCEND_QUANTIZED_ALL_REDUCE=1 supports TP size up to "
                f"{_MAX_VALID_WORLD_SIZE} (the integer-safe bound leaves "
                f"too few int8 levels beyond that), but the current TP size "
                f"is {self.world_size}."
            )
        # Integer-safe bound: scale = amax / max_q keeps every rank within
        # +/-max_q, so the TP-way sum is <= 127 and int8 cannot wrap.
        # ws=2 -> 63 (sum 126), ws=4 -> 31 (sum 124), ws=8 -> 15 (sum 120).
        self.max_q = 127 // self.world_size
        # Payload gate in elements (bf16 payload = 2 bytes per element);
        # default 64 MiB, the measured break-even rounded up. See docstring.
        self.min_numel = _env_int("VLLM_ASCEND_QUANTIZED_ALL_REDUCE_MIN_NUMEL", 32 * 1024 * 1024)

    def all_reduce(self, input_: torch.Tensor) -> torch.Tensor:
        if input_.dtype != torch.bfloat16 or input_.numel() < self.min_numel:
            dist.all_reduce(input_, group=self.group)
            return input_

        # [*, C] view: the last dim carries the channels that share a scale.
        flat = input_.reshape(-1, input_.shape[-1])

        # 1. per-channel global amax (tiny fp32 all-reduce, [C] elements).
        amax = flat.abs().amax(dim=0).float()
        dist.all_reduce(amax, op=dist.ReduceOp.MAX, group=self.group)
        # Exact-bound scale: rounding a value <= max_q cannot exceed max_q, so
        # the int8 sum over TP ranks stays in range and no clamp is needed.
        scale = (amax / float(self.max_q)).clamp_min(1e-8)

        # 2. int8 payload reduce. Do not add a clamp here (see the docstring).
        q = torch_npu.npu_quantize(flat, scale, None, torch.int8, 1, True)
        dist.all_reduce(q, group=self.group)

        # 3. dequantize and write back in place, like the bf16 path.
        out = torch_npu.npu_anti_quant(q, scale, dst_dtype=input_.dtype)
        input_.copy_(out.reshape(input_.shape))
        return input_
