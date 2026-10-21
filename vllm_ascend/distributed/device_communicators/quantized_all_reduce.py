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

Scheme (mathematically safe with HCCL wrapping int8 arithmetic):
  1. per-channel symmetric scale with a world-size headroom:
        scale[c] = global_amax(channel c) * world_size / 127
     The headroom guarantees every rank's quantized values stay in
     [-127/ws, 127/ws], so the element-wise sum over `ws` ranks can never
     overflow int8 (HCCL int8 reduce wraps modulo 256 instead of saturating).
  2. quantize bf16 -> int8, all-reduce in int8 (half payload)
  3. dequantize the reduced sum back to bf16

The per-channel scale needs a tiny fp32 sync (numel == hidden_size, a few KB)
before the payload all-reduce; it is an order of magnitude cheaper than the
bandwidth saved on large messages.

Measured on 4x Atlas 910B3 (TP=4, hidden=2048, 16k-token prefill batches):
  - per-AR latency: ~1.9 ms (bf16) -> ~1.4 ms (int8), including quant/dequant
  - e2e TTFT: -25%, throughput: +4.5%
  - accuracy: GPQA-diamond 70.7% vs 69.7% baseline (within noise)

Enable with VLLM_ASCEND_QUANTIZED_ALL_REDUCE=1.

VLLM_ASCEND_QUANTIZED_ALL_REDUCE_MIN_NUMEL (optional, unit: elements):
  An all-reduce takes the quantized path only when its tensor holds at
  least this many elements. The tensor is the hidden state of the whole
  batch, so numel == num_tokens * hidden_size, and the default
  (4 * 1024 * 1024 // 2 = 2,097,152 elements == 4 MiB of bf16 payload)
  maps to the following token count:

      min_tokens = min_numel / hidden_size = 2,097,152 / hidden_size

      hidden_size   tokens to trigger the int8 path
      1024          2048
      2048          1024   (e.g. Qwen3.6-35B-A3B: 1k-token prefill)
      4096           512
      8192           256

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
# Above this TP size the headroom eats too many bits (ws=16 -> max_q=7,
# i.e. 3 effective bits) and accuracy degrades; fall back to bf16.
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
        # Above this TP size the headroom eats too many bits (ws=16 -> max_q=7,
        # i.e. 3 effective levels) and accuracy degrades; refuse instead of
        # silently degrading.
        if self.world_size > _MAX_VALID_WORLD_SIZE:
            raise RuntimeError(
                f"VLLM_ASCEND_QUANTIZED_ALL_REDUCE=1 supports TP size up to "
                f"{_MAX_VALID_WORLD_SIZE} (the integer-safe headroom leaves "
                f"too few int8 levels beyond that), but the current TP size "
                f"is {self.world_size}."
            )
        # Integer-safe per-rank bound. Rounding can push |q| above 127/ws
        # (e.g. x == amax on every rank with ws=4: round(31.75) == 32, and
        # 4 * 32 == 128 overflows int8, wrapping to -128). Clamping to
        # +/-floor(127/ws) guarantees ws * max_q <= 127 for any input:
        # ws=2 -> 126, ws=4 -> 124, ws=8 -> 120, ws=16 -> 112.
        self.max_q = 127 // self.world_size
        # Messages smaller than this stay on the bf16 path: they are
        # latency-bound, and the extra scale sync would cost more than the
        # halved transfer saves. Default 4 MiB of bf16 payload, i.e.
        # 2,097,152 elements. The all-reduce tensor is the batch hidden
        # state (numel == num_tokens * hidden_size), so this maps to
        # min_tokens = 2,097,152 / hidden_size (2048 -> 1024, 4096 -> 512,
        # 8192 -> 256); see the module docstring for the full table.
        self.min_numel = _env_int(
            "VLLM_ASCEND_QUANTIZED_ALL_REDUCE_MIN_NUMEL", 4 * 1024 * 1024 // 2
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
        scale = (amax * (self.world_size / 127.0)).clamp_min(1e-8)

        # 2. symmetric int8 quantization, then clamp to the integer-safe
        # bound so the reduced sum cannot wrap, whatever the input
        # distribution (same-sign worst case included).
        q = torch_npu.npu_quantize(flat, scale, None, torch.int8, 1, True)
        q.clamp_(-self.max_q, self.max_q)
        dist.all_reduce(q, group=self.group)

        # 3. dequantize back to the original dtype/shape, in place, to keep
        # the same mutation contract as the plain all-reduce path.
        out = torch_npu.npu_anti_quant(q, scale, dst_dtype=input_.dtype)
        input_.copy_(out.reshape(input_.shape))
        return input_
