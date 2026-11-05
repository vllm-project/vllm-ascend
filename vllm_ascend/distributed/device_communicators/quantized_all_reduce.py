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
"""Quantized all-reduce for large bf16 TP collectives on Atlas A2/A3/A5.

Halves the all-reduce payload (bf16 -> int8 -> reduce -> bf16). Opt-in, and it
only pays off for large messages, so it is gated by a payload threshold.

Group scope
-----------
The wrapper uses the same routing rules on A2, A3 and A5. It is constructed
for every parallel group, and groups with at most 8 ranks can take the int8
path. Larger EP/MC2 groups keep bf16. Eligible bf16 tensors are selected by
shape and payload size, without inspecting the forward phase or graph mode.

Configuration
-------------
Enable it through ``--additional-config`` (see vllm_ascend.ascend_config):

    --additional-config '{"quantized_all_reduce_config": {"enable": true, "min_numel": 16777216}}'

  enable     bool, default false.
  min_numel  Payload threshold in elements: a collective is quantized only when
             its payload reaches min_numel elements, i.e. min_numel * 2 bytes of
             bf16 data. The default 16,777,216 is 32 MiB, i.e. 8192 tokens at
             hidden_size 2048 or 3277 tokens at hidden_size 5120. Concurrency
             matters only through the collective payload. Mixed batches and
             decode take the int8 path if they meet the same size/shape gates.

  The optimized A2/A3 implementation breaks even at ~20-24 MiB: 16 MiB +13%,
  32 MiB -29%, 42 MiB -32%, 64 MiB -37%, 128 MiB -37% versus BF16 on
  4x910B3. A5 TP=2/4 measurements also support a conservative 32 MiB gate.
  Do not lower min_numel without target-hardware and model validation.

  The setting is read when the TP communicator is created (server start-up) and
  is active when every TP worker logs
  "INT8 quantized all-reduce enabled (group=..., min_numel=..., max_q=...)".
  To diagnose actual calls, add the following to the same --additional-config:
  "runtime_config": {"ascend_log": {"debug":
  ["distributed.device_communicators.quantized_all_reduce"]}}
  Look for QAR-HIT and QAR-SKIP. These Python logs may also appear during
  warmup or graph capture and do not count graph replays. Disable debug logging
  for performance measurements.
"""

import logging
import math

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401

from vllm_ascend.device.device_config import get_ascend_device_type
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.logger import init_logger_ascend

logger = init_logger_ascend(__name__)

# A2 is measured end to end; A3 supports the kernels; A5 TP=2/4 has standalone
# numerical and performance evidence. All families share the A2 routing rules.
_VALID_DEVICE_TYPES = (AscendDeviceType.A2, AscendDeviceType.A3, AscendDeviceType.A5)
# Groups larger than 8 ranks leave too few int8 levels (ws=16 -> max_q=7), so
# they keep bf16 instead of failing. The TP degree is rejected at config time.
_MAX_VALID_WORLD_SIZE = 8


class QuantizedAllReduce:
    """Wrapper adding int8 quantized all-reduce around a base all_reduce."""

    def __init__(self, group: "dist.ProcessGroup", min_numel: int, group_name: str = "") -> None:
        self.group = group
        self.group_name = group_name
        self.world_size = dist.get_world_size(group)
        device_type = get_ascend_device_type()
        self.device_type = device_type
        # Fail fast on unvalidated families: a silent bf16 fallback would hide
        # a misconfiguration, and 310P lacks bf16 npu_quantize support anyway.
        if device_type not in _VALID_DEVICE_TYPES:
            raise RuntimeError(
                "additional_config.quantized_all_reduce_config is only validated on "
                f"{'/'.join(t.name for t in _VALID_DEVICE_TYPES)} devices, "
                f"but the current device family is {device_type.name}. "
                "Disable it or use a supported device."
            )
        # Match A2 routing on every family, including non-TP groups. Oversized
        # EP/MC2 groups fall back without failing communicator construction.
        self.supported = self.world_size <= _MAX_VALID_WORLD_SIZE
        # Integer-safe bound: scale = amax / max_q keeps every rank within
        # +/-max_q, so the TP-way sum is <= 127 and int8 cannot wrap.
        # ws=2 -> 63 (sum 126), ws=4 -> 31 (sum 124), ws=8 -> 15 (sum 120).
        self.max_q = 127 // self.world_size if self.supported else 0
        # Payload gate in elements (bf16 payload = 2 bytes per element);
        # 32 MiB by default, the measured break-even rounded up. See docstring.
        self.min_numel = min_numel

    def _skip_reason(self, input_: torch.Tensor) -> str | None:
        if not self.supported:
            return "unsupported_group"
        if input_.dtype != torch.bfloat16:
            return "dtype"
        if input_.ndim < 2 or input_.shape[-1] < 64:
            return "degenerate_shape"
        if input_.numel() < self.min_numel:
            return "below_threshold"
        return None

    def all_reduce(self, input_: torch.Tensor) -> torch.Tensor:
        reason = self._skip_reason(input_)
        if reason == "unsupported_group":
            # Unsupported, oversized groups stay on the plain BF16 path.
            logger.warning_once(
                "quantized all-reduce disabled for unsupported group %s "
                "(world_size=%d, device=%s); its collectives stay bf16.",
                self.group_name,
                self.world_size,
                self.device_type.name,
            )
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "[QAR-SKIP] reason=unsupported_group group=%s world_size=%d shape=%s dtype=%s "
                    "numel=%d min_numel=%d",
                    self.group_name,
                    self.world_size,
                    tuple(input_.shape),
                    input_.dtype,
                    input_.numel(),
                    self.min_numel,
                )
            dist.all_reduce(input_, group=self.group)
            return input_
        if reason is not None:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "[QAR-SKIP] reason=%s group=%s world_size=%d shape=%s dtype=%s "
                    "numel=%d min_numel=%d",
                    reason,
                    self.group_name,
                    self.world_size,
                    tuple(input_.shape),
                    input_.dtype,
                    input_.numel(),
                    self.min_numel,
                )
            dist.all_reduce(input_, group=self.group)
            return input_

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "[QAR-HIT] group=%s world_size=%d shape=%s dtype=%s numel=%d "
                "payload_bytes=%d min_numel=%d max_q=%d",
                self.group_name,
                self.world_size,
                tuple(input_.shape),
                input_.dtype,
                input_.numel(),
                input_.numel() * input_.element_size(),
                self.min_numel,
                self.max_q,
            )

        # [*, C] view: the last dim carries the channels that share a scale.
        flat = input_.reshape(-1, input_.shape[-1])

        # 1. per-channel global amax (tiny fp32 all-reduce, [C] elements).
        # vector_norm(ord=inf) fuses abs+max and avoids a payload-sized
        # temporary tensor. This is the optimized PR #19000 implementation.
        amax = torch.linalg.vector_norm(flat, ord=math.inf, dim=0).float()
        dist.all_reduce(amax, op=dist.ReduceOp.MAX, group=self.group)
        # Exact-bound scale: rounding a value <= max_q cannot exceed max_q, so
        # the int8 sum over TP ranks stays in range and no clamp is needed.
        scale = (amax / float(self.max_q)).clamp_min(1e-8)

        # 2. int8 payload reduce. Do not add a clamp here (see the docstring).
        q = torch_npu.npu_quantize(flat, scale, None, torch.int8, 1, True)
        dist.all_reduce(q, group=self.group)

        # 3. Dequantize to the original dtype and return out of place. The
        # GroupCoordinator all-reduce contract requires callers to consume the
        # returned tensor, so copying the full payload back is unnecessary.
        out = torch_npu.npu_anti_quant(q, scale, dst_dtype=input_.dtype)
        return out.reshape(input_.shape)
