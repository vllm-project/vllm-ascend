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

Halves the all-reduce payload (bf16 -> int8 -> reduce -> bf16). It is opt-in
and only pays off for large messages, so it is gated by a payload threshold.

Configuration
-------------
VLLM_ASCEND_QUANTIZED_ALL_REDUCE=1
    Enable the feature. Default 0 (off); only "0"/"1" are accepted.
    Supported on Atlas A2/A3 with 1 < TP <= 8. The switch is read when the
    TP communicator is created, so restart the server after changing it.
    It is active when every TP worker logs
    "Quantized all-reduce enabled (min_numel=..., max_q=...)".

VLLM_ASCEND_QUANTIZED_ALL_REDUCE_MIN_NUMEL=<elements>    (default 33554432)
    Smallest all-reduce that may take the int8 path, in elements. The
    reduced tensor is the batch hidden state, numel = step_tokens *
    hidden_size, so this is really a payload threshold:

        payload = MIN_NUMEL * 2 bytes (bf16)
        default 33554432 elements = 64 MiB = 16384 tokens at hidden 2048

    A collective is quantized only when one scheduler step holds at least
    MIN_NUMEL / hidden_size tokens, i.e. normally a full
    max_num_batched_tokens chunk. Concurrency matters only through
    batching: 16 concurrent 1k-token requests pack into one 16k-token step
    and qualify, while a single 1k-token request does not. Decode steps
    (a few thousand elements) never qualify and stay on bf16.

    Small messages lose: each collective carries a fixed ~0.75 ms cost
    (amax sync + quantize + dequantize) and the measured break-even is
    ~36 MiB of payload:

        payload    4 MiB   16 MiB   32 MiB   64 MiB   128 MiB
        vs bf16    +359%    +67%     +8%      -23%     -23%

    Guidelines:
      * keep the default (64 MiB) unless your collectives are larger;
      * do not go below 16777216 (32 MiB);
      * if max_num_batched_tokens * hidden_size * 2 < 64 MiB the threshold
        is unreachable, so leave the feature disabled instead of tuning it.

Implementation notes
--------------------
  * Integer safety without a clamp: max_q = 127 // TP and
    scale = per-channel global amax / max_q. Rounding a value that is
    already <= max_q cannot exceed max_q, so the TP-way sum is bounded by
    TP * max_q <= 127 and int8 cannot wrap (HCCL wraps instead of
    saturating), for any input distribution.
  * Do not bound the int8 payload with torch's clamp family: int8
    clamp_/clamp/hardtanh are ~250x slower than minimum/maximum on Ascend
    (see Ascend/pytorch#178), and one such call made this pipeline ~8x
    slower than the plain bf16 all-reduce it is meant to replace.
  * Measured on 4x 910B3, TP=4, 64 MiB payload: 1.73 ms (bf16) -> 1.21 ms
    (-30%); end to end (16k in / 1k out, 32 concurrent): +3.6% output
    throughput and -6% mean TTFT.
"""

import os

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401
from vllm.logger import logger

from vllm_ascend.device.device_config import get_ascend_device_type
from vllm_ascend.device.hardware import AscendDeviceType

# Hardware families where the scheme is validated or considered safe:
# - A2 (910B): measured end to end (latency, throughput, GPQA accuracy).
# - A3 (910C): npu_quantize/npu_anti_quant support the SoC, but the int8
#   reduce behavior is not yet measured there; use with care above TP=8.
_VALID_DEVICE_TYPES = (AscendDeviceType.A2, AscendDeviceType.A3)
# Above this TP size the integer-safe bound eats too many bits (ws=16 ->
# max_q=7, i.e. 3 effective bits) and accuracy degrades; fall back to bf16.
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
        # Fail fast on unvalidated device families: silently falling back to
        # bf16 would hide a misconfiguration, and 310P additionally lacks
        # bf16 npu_quantize support, so the quantized kernels cannot run.
        if device_type not in _VALID_DEVICE_TYPES:
            raise RuntimeError(
                "VLLM_ASCEND_QUANTIZED_ALL_REDUCE=1 is only validated on "
                f"{'/'.join(t.name for t in _VALID_DEVICE_TYPES)} devices, "
                f"but the current device family is {device_type.name}. "
                "Unset the environment variable or use a supported device."
            )
        # Above this TP size the integer-safe bound eats too many bits (ws=16
        # -> max_q=7, i.e. 3 effective levels) and accuracy degrades; refuse
        # instead of silently degrading.
        if self.world_size > _MAX_VALID_WORLD_SIZE:
            raise RuntimeError(
                f"VLLM_ASCEND_QUANTIZED_ALL_REDUCE=1 supports TP size up to "
                f"{_MAX_VALID_WORLD_SIZE} (the integer-safe bound leaves "
                f"too few int8 levels beyond that), but the current TP size "
                f"is {self.world_size}."
            )
        # Integer-safe per-rank bound. With scale = global_amax / max_q every
        # rank's quantized magnitude is <= max_q, so the reduced sum is at
        # most ws * max_q <= 127 and int8 cannot wrap: no clamp is needed.
        # ws=2 -> 63 (sum 126), ws=4 -> 31 (sum 124), ws=8 -> 15 (sum 120).
        self.max_q = 127 // self.world_size
        # Payload gate: only messages with >= MIN_NUMEL elements
        # (MIN_NUMEL * 2 bytes of bf16 payload) take the int8 path. The
        # default is 64 MiB, the measured break-even rounded up; see the
        # module docstring for the sizing table and configuration guide.
        self.min_numel = _env_int(
            "VLLM_ASCEND_QUANTIZED_ALL_REDUCE_MIN_NUMEL", 32 * 1024 * 1024
        )

    def all_reduce(self, input_: torch.Tensor) -> torch.Tensor:
        if input_.dtype != torch.bfloat16 or input_.numel() < self.min_numel:
            dist.all_reduce(input_, group=self.group)
            return input_

        # [*, C] view: the last dim carries the channels that share a scale.
        flat = input_.reshape(-1, input_.shape[-1])

        # 1. per-channel global amax (tiny fp32 all-reduce, [C] elements).
        amax = flat.abs().amax(dim=0).float()
        dist.all_reduce(amax, op=dist.ReduceOp.MAX, group=self.group)
        # Exact-bound scale: |x| <= amax => |x/scale| <= max_q, and rounding
        # never exceeds max_q, so ws * max_q <= 127 and no clamp is needed.
        scale = (amax / float(self.max_q)).clamp_min(1e-8)

        # 2. symmetric int8 quantization, then the int8 payload all-reduce.
        # Do NOT add a clamp here: int8 clamp_/clamp/hardtanh are ~250x
        # slower than minimum/maximum on Ascend and would dominate this
        # pipeline (see the module docstring). The scale above already
        # guarantees the reduced sum cannot overflow int8.
        q = torch_npu.npu_quantize(flat, scale, None, torch.int8, 1, True)
        dist.all_reduce(q, group=self.group)

        # 3. dequantize back to the original dtype/shape, in place, to keep
        # the same mutation contract as the plain all-reduce path.
        out = torch_npu.npu_anti_quant(q, scale, dst_dtype=input_.dtype)
        input_.copy_(out.reshape(input_.shape))
        return input_
