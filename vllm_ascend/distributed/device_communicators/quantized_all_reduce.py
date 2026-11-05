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
On A2/A3 the existing wrapper behavior is preserved: groups with at most 8
ranks can take the int8 path. On A5, only the standard TP group with TP=2/4 is
eligible, and only during eager pure prefill. Decode, mixed batches, graph
capture and non-TP groups keep the plain bf16 path.

Configuration
-------------
Enable it through ``--additional-config`` (see vllm_ascend.ascend_config):

    --additional-config '{"quantized_all_reduce_config": {"enable": true, "min_numel": 16777216}}'

  enable     bool, default false.
  min_numel  Payload threshold in elements: a collective is quantized only when
             its payload reaches min_numel elements, i.e. min_numel * 2 bytes of
             bf16 data. The default 16,777,216 is 32 MiB, i.e. 8192 tokens at
             hidden_size 2048 or 3277 tokens at hidden_size 5120. Concurrency
             matters only through batching; decode steps never qualify on A5.

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
from numbers import Integral

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401
from vllm.config import CUDAGraphMode
from vllm.forward_context import get_forward_context, is_forward_context_available

from vllm_ascend.device.device_config import get_ascend_device_type
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.logger import init_logger_ascend

logger = init_logger_ascend(__name__)

# A2 is measured end to end; A3 supports the kernels; A5 TP=2/4 has standalone
# numerical and performance evidence and is gated to eager pure prefill.
_VALID_DEVICE_TYPES = (AscendDeviceType.A2, AscendDeviceType.A3, AscendDeviceType.A5)
# Groups larger than 8 ranks leave too few int8 levels (ws=16 -> max_q=7), so
# they keep bf16 instead of failing. The TP degree is rejected at config time.
_MAX_VALID_WORLD_SIZE = 8
_A5_VALID_WORLD_SIZES = (2, 4)


def _is_eager_pure_prefill() -> bool:
    """Use host forward metadata; unknown, mixed and graph batches fall back."""
    if not is_forward_context_available():
        return False
    ctx = get_forward_context()
    if ctx.cudagraph_runtime_mode != CUDAGraphMode.NONE:
        return False
    additional_kwargs = getattr(ctx, "additional_kwargs", None) or {}
    for flag in ("capturing", "in_profile_run", "is_draft_model"):
        if getattr(ctx, flag, False) is True or additional_kwargs.get(flag, False) is True:
            return False
    metadata = ctx.attn_metadata
    if not isinstance(metadata, dict) or not metadata:
        return False
    seen = set()
    for layer_metadata in metadata.values():
        if id(layer_metadata) in seen:
            continue
        seen.add(id(layer_metadata))
        counts = (
            getattr(layer_metadata, "num_prefills", None),
            getattr(layer_metadata, "num_decodes", None),
            getattr(layer_metadata, "num_spec_decodes", 0),
        )
        if not all(isinstance(count, Integral) and not isinstance(count, bool) for count in counts):
            return False
        prefills, decodes, spec_decodes = counts
        if prefills <= 0 or decodes != 0 or spec_decodes != 0:
            return False
    return True


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
        # A5 is deliberately narrower than A2/A3: only the standard TP group
        # and the measured TP sizes are eligible. Other groups stay BF16.
        if device_type == AscendDeviceType.A5:
            is_standard_tp = group_name.split(":", 1)[0] == "tp"
            self.supported = is_standard_tp and self.world_size in _A5_VALID_WORLD_SIZES
        else:
            # Existing A2/A3 behavior: non-TP groups of at most 8 ranks retain
            # the historical quantized path; oversized EP/MC2 groups fall back.
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
        if self.device_type == AscendDeviceType.A5:
            if not input_.is_contiguous():
                return "layout"
            if not _is_eager_pure_prefill():
                return "not_eager_pure_prefill"
            is_capturing = getattr(torch.npu, "is_current_stream_capturing", None)
            if callable(is_capturing) and is_capturing():
                return "capturing"
        return None

    def all_reduce(self, input_: torch.Tensor) -> torch.Tensor:
        reason = self._skip_reason(input_)
        if reason == "unsupported_group":
            # Unsupported groups stay on the plain BF16 path. For A2/A3 these
            # are oversized groups; A5 additionally excludes non-TP groups.
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
