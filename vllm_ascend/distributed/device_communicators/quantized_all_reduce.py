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
"""Quantized all-reduce for Atlas A2 TP inference.

Large bf16 all-reduces (prefill o_proj / MoE combine, typically tens of MiB)
are bandwidth-bound on HCCS. Quantizing the payload to int8 halves the
communication volume with a negligible compute cost on the Vector cores.

Scheme (integer-safe by construction; no clamp kernel is needed):
  1. per-channel symmetric scale with an exact world-size bound:
        max_q    = 127 // world_size      (ws=2 -> 63, ws=4 -> 31, ws=8 -> 15)
        scale[c] = global_amax(channel c) / max_q
     |x| <= global_amax implies |x / scale| <= max_q, and rounding a value
     that is already <= max_q can never exceed max_q, so every rank stays
     within +/-max_q and the element-wise sum over `ws` ranks is bounded by
     ws * max_q <= 127. The int8 reduce therefore cannot wrap (HCCL wraps
     modulo 256 instead of saturating) for any input distribution, including
     the same-sign worst case where every rank holds x == amax on every
     channel.
     Note: the headroom form scale = global_amax * ws / 127 does NOT give
     this guarantee, because round(127/ws) can land above 127/ws (ws=4:
     round(31.75) == 32, and 4 * 32 == 128 wraps to -128). That is why an
     extra clamp was previously needed; the exact-bound scale above removes
     the need for it entirely.
  2. quantize bf16 -> int8, all-reduce in int8 (half payload)
  3. dequantize the reduced sum back to bf16

The per-channel scale needs a tiny fp32 sync (numel == hidden_size, a few KB)
before the payload all-reduce; it is an order of magnitude cheaper than the
bandwidth saved on large messages.

Measured on 4x Atlas 910B3 (TP=4, hidden=2048, 16k-token prefill -> 64 MiB
payload per collective, HCCL_OP_EXPANSION_MODE=AIV, HCCL_BUFFSIZE=1024):
  - per-AR latency: 1.73 ms (bf16) -> 1.21 ms (int8, including the amax sync,
    quantize, reduce and dequantize) => -30%
  - e2e (16k in / 64 out, 32 prompts @ 2 qps): output throughput +3.2%,
    mean TTFT -9%, mean TPOT -25%
  - accuracy (64 MiB payload vs an exact bf16 reduce): rel-L2 3.9%,
    cosine 0.9993; no int8 wrap on i.i.d., same-sign correlated or
    worst-case (all elements == amax) data
  - the win only materialises for large payloads: the pipeline has a fixed
    per-call cost of ~0.75 ms, so the break-even payload is ~36 MiB and
    smaller messages lose (see the min_numel note below)

WARNING: do not bound the int8 payload with torch's clamp family here.
On Ascend, int8 clamp_/clamp/hardtanh fall back to a very slow path
(32 MiB int8: 14.2 ms for clamp_ vs 0.056 ms for minimum/maximum, i.e.
~250x), which turns the -30% win into a +800% regression. This scheme is
safe without any clamp; if a bound is ever required, use
torch.minimum(torch.maximum(q, lo_i8), hi_i8) with preallocated int8
scalars instead.

Enable with VLLM_ASCEND_QUANTIZED_ALL_REDUCE=1.

VLLM_ASCEND_QUANTIZED_ALL_REDUCE_MIN_NUMEL (optional, unit: elements):
  An all-reduce takes the quantized path only when its tensor holds at
  least this many elements. The tensor is the hidden state of the whole
  batch, so numel == num_tokens * hidden_size, and the default
  (32 * 1024 * 1024 = 33,554,432 elements == 64 MiB of bf16 payload)
  maps to the following token count:

      min_tokens = min_numel / hidden_size = 33,554,432 / hidden_size

      hidden_size   tokens to trigger the int8 path
      1024          32768
      2048          16384   (e.g. Qwen3.6-35B-A3B: 16k-token prefill)
      4096           8192
      8192           4096

  The default is sized from the measured break-even (~36 MiB of bf16
  payload): below it the fixed per-call cost of the scheme outweighs the
  halved transfer, so smaller messages are left on the bf16 path.

  Notes:
  - For a single-sequence prefill, num_tokens == sequence length; for
    batched requests it is the summed token count of the batch, so
    several short prompts together can still exceed the threshold.
  - Decode steps usually have num_tokens == batch size (a few to a few
    hundred, far below the threshold) and keep the plain bf16 path by
    design: small messages are latency-bound and cannot amortize the
    extra per-channel scale sync.
  - Lower the value to quantize smaller messages, or raise it to leave
    only the very large prefill collectives on the int8 path.
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
        # Messages smaller than this stay on the bf16 path: they are
        # latency-bound, and the fixed per-call cost of the scheme (~0.75 ms
        # for the local amax, the small fp32 MAX all-reduce and the
        # quantize/dequantize pair) outweighs the halved transfer. Default
        # 32 Mi elements == 64 MiB of bf16 payload, i.e. the measured
        # break-even (~36 MiB) rounded up; maps to
        # min_tokens = 33,554,432 / hidden_size (2048 -> 16384).
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
