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
"""Quantized all-reduce for large bf16 TP collectives on Atlas A2/A5.

Halves the all-reduce payload (bf16 -> int8 -> reduce -> bf16). Opt-in, and it
only pays off for large messages, so it is gated by a payload threshold.

A3 (910C) is rejected. Its bf16 HCCS bandwidth is about 2.6x that of A2 while
the fixed per-collective cost of the int8 path is unchanged, so the quantized
path is a net loss at every payload size (measured 5-320 MiB across four
different groups of 910C cards).

Group scope
-----------
The wrapper uses the same routing rules on A2 and A5. It is constructed
for every parallel group, and groups with at most 8 ranks can take the int8
path. Larger EP/MC2 groups keep bf16. Eligible bf16 tensors are selected by
shape and payload size, without inspecting the forward phase or graph mode.

Configuration
-------------
Enable it through ``--additional-config`` (see vllm_ascend.ascend_config):

    --additional-config '{"quantized_all_reduce_config": {"enable": true, "quant_dtype": "int8"}}'

  enable     bool, default false.
  min_numel  Payload threshold in elements: a collective is quantized only when
             its payload reaches min_numel elements, i.e. min_numel * 2 bytes of
             bf16 data. Left at ``None`` (the default) the int8 gate is chosen
             per tensor-parallel degree from the A2 measurements below, but only
             on A2: A5 has no int8 payload calibration, so forcing the int8
             backend there requires an explicit ``min_numel``. MXFP8 keeps its
             own 32 MiB default. Concurrency matters only
             through the collective payload. Mixed batches and decode take the
             int8 path if they meet the same size/shape gates.

  The int8 break-even payload is strongly TP-dependent, because the bf16
  bandwidth grows with the group size while the fixed cost of the int8 path
  (absmax, tiny fp32 amax all-reduce, quantize, dequantize) does not. Measured
  on 910B3 with a dependent chain of 16 collectives, i.e. the timing that
  reproduces serving (bf16 payload, so numel = MiB * 524,288):

    TP   break-even   default min_numel   payload   gain at the default
    2    9-11 MiB     8,388,608           16 MiB    1.45x
    4    21-23 MiB    16,777,216          32 MiB    1.30x
    8    31-38 MiB    31,457,280          60 MiB    1.11x (1.24x at 160 MiB)

  TP=8 is the only degree whose end-to-end crossing sits well above the
  collective one: the isolated chain lets the 8 KiB fp32 amax all-reduce
  overlap perfectly, while a real forward pass does not, so end to end keeps
  40-90% of the collective gain. A 37-point scan in 64-token steps puts the
  chained crossing at 31-38 MiB (31 MiB fitting the >=46 MiB region, 38 MiB
  from the 40/80 MiB endpoints), yet serving is still neutral at 50 MiB
  (+0.17% TTFT) and first pays at 60 MiB (-1.89%). The gate is therefore
  60 MiB, the lowest measured payload that is strictly profitable, which
  leaves 1.6-1.9x headroom over the collective crossing; the gain then grows
  with the payload (1.11x at 60 MiB, 1.16x at 80 MiB, 1.24x at the deployed
  160 MiB). The TP=2 and TP=4 gates sit ~1.5x above their break-even and gain
  1.3-1.5x as soon as they trigger. Degrees that were not measured (3, 5-7)
  take the next larger default, so the gate never undercuts a measured
  break-even.

  MXFP8 has no per-TP calibration yet and keeps 16,777,216 (32 MiB); benchmark
  it on the intended A5 workload before lowering it. The same rule applies to
  the int8 gate: re-measure it on the target hardware instead of guessing.
  Note the gate is a payload threshold, not a token one: if
  max_num_batched_tokens * hidden_size * 2 stays below it the feature can never
  trigger, so either lower min_numel explicitly or leave the feature disabled.

  The setting is read when the TP communicator is created (server start-up) and
  is active when every TP worker logs
  "INT8 quantized all-reduce enabled (group=..., min_numel=..., max_q=...)".
  To diagnose actual calls, add the following to the same --additional-config:
  "runtime_config": {"ascend_log": {"debug":
  ["distributed.device_communicators.int8_all_reduce"]}}
  Look for QAR-HIT and QAR-SKIP. These Python logs may also appear during
  warmup or graph capture and do not count graph replays. Disable debug logging
  for performance measurements.
"""

import logging

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401

from vllm_ascend.device.device_config import get_ascend_device_type
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.logger import init_logger_ascend

logger = init_logger_ascend(__name__)

# A2 is validated end to end; A5 TP=2/4 has standalone numerical and
# performance evidence, but its int8 payload gate is uncalibrated, so the
# per-TP defaults below are A2-only (see default_min_numel). A3 remains
# disabled due to its measured regression.
_VALID_DEVICE_TYPES = (AscendDeviceType.A2, AscendDeviceType.A5)
# Groups larger than 8 ranks leave too few int8 levels (ws=16 -> max_q=7), so
# they keep bf16 instead of failing. The TP degree is rejected at config time.
_MAX_VALID_WORLD_SIZE = 8

# Default payload gate in elements, per tensor-parallel degree, for A2 (910B).
# A bf16 payload of N elements is N * 2 bytes, so N = MiB * 524,288 and these
# entries are 16 / 32 / 60 MiB. TP=2 and TP=4 are their measured break-even
# (~10 / ~22 MiB) rounded up 1.5x, so the quantized path is a clear win right at
# the gate. TP=8 is set 1.6-1.9x above its 31-38 MiB collective crossing, at the
# lowest payload that is strictly profitable end to end (60 MiB, -1.89% TTFT);
# 50 MiB is neutral (+0.17%) because a real forward keeps only 40-90% of the
# collective gain. See the module docstring for the measurement and the
# protocol.
_DEFAULT_MIN_NUMEL_A2: dict[int, int] = {
    2: 8_388_608,
    4: 16_777_216,
    8: 31_457_280,
}


def default_min_numel(world_size: int, device_type: AscendDeviceType) -> int:
    """Return the measured A2 payload gate for a ``world_size``-rank group.

    The per-TP gates were measured on A2 (910B) only. A5 has a different bf16
    HCCS bandwidth, so its int8 break-even is not covered by this table and
    reusing an A2 number there could gate the quantized path in below its own
    crossing. An A5 deployment must pass an explicit ``min_numel`` instead.

    On A2, unmeasured degrees take the next larger measured one, so the gate
    never undercuts a measured break-even; degrees above the supported maximum
    are not reachable here (those groups stay bf16).
    """
    if device_type != AscendDeviceType.A2:
        raise ValueError(
            "the per-TP min_numel defaults are measured on Atlas A2 only, but "
            f"the current device family is {device_type.name}. Set "
            "additional_config.quantized_all_reduce_config.min_numel explicitly "
            "or use the MXFP8 backend."
        )
    for degree in sorted(_DEFAULT_MIN_NUMEL_A2):
        if world_size <= degree:
            return _DEFAULT_MIN_NUMEL_A2[degree]
    return _DEFAULT_MIN_NUMEL_A2[max(_DEFAULT_MIN_NUMEL_A2)]


class QuantizedAllReduce:
    """Wrapper adding int8 quantized all-reduce around a base all_reduce."""

    def __init__(self, group: "dist.ProcessGroup", min_numel: int | None = None, group_name: str = "") -> None:
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
        # Payload gate in elements (bf16 payload = 2 bytes per element). None
        # means "use the measured A2 per-TP default"; see default_min_numel().
        if min_numel is None:
            self.min_numel = default_min_numel(self.world_size, device_type)
        else:
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
                    "[QAR-SKIP] reason=%s group=%s world_size=%d shape=%s dtype=%s numel=%d min_numel=%d",
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
                "[QAR-HIT] group=%s world_size=%d shape=%s dtype=%s numel=%d payload_bytes=%d min_numel=%d max_q=%d",
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
        # vector_norm(ord=inf) is abs+max fused into one kernel: it avoids the
        # transient abs tensor (80 MiB at an 84 MiB payload) and is ~30% faster.
        amax = torch.linalg.vector_norm(flat, dim=0, ord=float("inf")).float()
        dist.all_reduce(amax, op=dist.ReduceOp.MAX, group=self.group)
        # Exact-bound scale: rounding a value <= max_q cannot exceed max_q, so
        # the int8 sum over TP ranks stays in range and no clamp is needed.
        scale = (amax / float(self.max_q)).clamp_min(1e-8)

        # 2. int8 payload reduce. Do not add a clamp here (see the docstring).
        q = torch_npu.npu_quantize(flat, scale, None, torch.int8, 1, True)
        dist.all_reduce(q, group=self.group)

        # 3. dequantize back to the original dtype/shape and return out-of-place:
        # GroupCoordinator.all_reduce treats the device communicator as
        # out-of-place by contract (vLLM custom ops cannot mutate and return in
        # the same op), so callers always consume the returned tensor. This
        # skips an 80 MiB read+write copy per collective.
        out = torch_npu.npu_anti_quant(q, scale, dst_dtype=input_.dtype)
        return out.reshape(input_.shape)
