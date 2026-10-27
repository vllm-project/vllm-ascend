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

Group scope
-----------
The wrapper is constructed for every parallel group (tp, ep, mc2, dp, pp, ...),
because the same ``NPUCommunicator`` backs all of them, but only groups with at
most 8 ranks take the int8 path. EP/MC2 groups are ``DP x PCP x TP`` sized, so
they can be larger than any valid TP degree; those groups keep the plain bf16
path and never fail at start-up. The tensor-parallel degree itself is validated
when the config is parsed (``QuantizedAllReduceConfig._validate_preconditions``),
so TP > 8 is still rejected up front.

Configuration
-------------
Enable it through ``--additional-config`` (see vllm_ascend.ascend_config):

    --additional-config '{"quantized_all_reduce_config": {"enable": true, "min_numel": 33554432}}'

  enable     bool, default false.
  min_numel  Payload threshold in elements: a collective is quantized only when
             its payload reaches min_numel elements, i.e. min_numel * 2 bytes of
             bf16 data. The default 33554432 is 64 MiB == 16384 tokens at
             hidden_size 2048, i.e. a full max_num_batched_tokens chunk.
             Concurrency matters only through batching (16 concurrent 1k-token
             requests form one 16k-token step and qualify, a single 1k-token
             request does not); decode steps never qualify.

  Each collective carries a fixed ~0.75 ms cost, hence a ~36 MiB break-even:
  4 MiB +359%, 16 MiB +67%, 32 MiB +8%, 64 MiB -23%, 128 MiB -23% (vs bf16).
  Keep the default and do not go below 16777216 (32 MiB); if
  max_num_batched_tokens * hidden_size * 2 < 64 MiB the threshold is
  unreachable, so leave the feature disabled.

  The setting is read when the TP communicator is created (server start-up) and
  is active when every TP worker logs
  "Quantized all-reduce enabled (min_numel=..., max_q=...)".
"""

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401
from vllm.logger import logger

from vllm_ascend.device.device_config import get_ascend_device_type
from vllm_ascend.device.hardware import AscendDeviceType

# Validated on A2 (910B, measured end to end); A3 (910C) supports the kernels
# but its int8 reduce behaviour is unmeasured.
_VALID_DEVICE_TYPES = (AscendDeviceType.A2, AscendDeviceType.A3)
# Groups larger than 8 ranks leave too few int8 levels (ws=16 -> max_q=7), so
# they keep bf16 instead of failing. The TP degree is rejected at config time.
_MAX_VALID_WORLD_SIZE = 8


class QuantizedAllReduce:
    """Wrapper adding int8 quantized all-reduce around a base all_reduce."""

    def __init__(self, group: "dist.ProcessGroup", min_numel: int) -> None:
        self.group = group
        self.world_size = dist.get_world_size(group)
        device_type = get_ascend_device_type()
        # Fail fast on unvalidated families: a silent bf16 fallback would hide
        # a misconfiguration, and 310P lacks bf16 npu_quantize support anyway.
        if device_type not in _VALID_DEVICE_TYPES:
            raise RuntimeError(
                "additional_config.quantized_all_reduce_config is only validated on "
                f"{'/'.join(t.name for t in _VALID_DEVICE_TYPES)} devices, "
                f"but the current device family is {device_type.name}. "
                "Disable it or use a supported device."
            )
        # This communicator is also built for non-TP groups (ep/mc2/dp/...),
        # whose sizes are not a tensor-parallel degree; those groups keep bf16.
        self.supported = self.world_size <= _MAX_VALID_WORLD_SIZE
        # Integer-safe bound: scale = amax / max_q keeps every rank within
        # +/-max_q, so the TP-way sum is <= 127 and int8 cannot wrap.
        # ws=2 -> 63 (sum 126), ws=4 -> 31 (sum 124), ws=8 -> 15 (sum 120).
        self.max_q = 127 // self.world_size if self.supported else 0
        # Payload gate in elements (bf16 payload = 2 bytes per element);
        # 64 MiB by default, the measured break-even rounded up. See docstring.
        self.min_numel = min_numel

    def all_reduce(self, input_: torch.Tensor) -> torch.Tensor:
        if not self.supported:
            # Oversized groups (ep/mc2 = DP x PCP x TP) are out of scope: keep
            # the plain bf16 path rather than failing at start-up.
            logger.warning_once(
                "quantized all-reduce disabled for this %d-rank group "
                "(supported groups have at most %d ranks); its collectives stay bf16.",
                self.world_size,
                _MAX_VALID_WORLD_SIZE,
            )
            dist.all_reduce(input_, group=self.group)
            return input_
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
