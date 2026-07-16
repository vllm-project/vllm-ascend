#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
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
"""Patch CpuGpuBuffer with double-buffer rotation for async H2D safety.

Under async scheduling, the next iteration's CPU writes to ``buffer.cpu``
can race with the previous iteration's in-flight ``non_blocking=True``
H2D copy, corrupting the data seen by GPU consumers (e.g. query_start_loc).

This patch keeps ``self.cpu`` / ``self.gpu`` / ``self.np`` addresses fixed
(graph-compatible, long-term-ref safe), but routes ``copy_to_gpu`` through
two pre-allocated pinned CPU shadow buffers that rotate each call. Each
shadow is guarded by an NPU event: before reusing a shadow (two iterations
after its previous use), ``synchronize()`` ensures the prior H2D has
completed before the CPU writes to it. In the common case where one
iteration's CPU preparation outlasts the H2D, ``synchronize()`` returns
immediately with zero overhead.
"""

import torch
import torch_npu
from vllm.v1.utils import CpuGpuBuffer

_orig_init = CpuGpuBuffer.__init__
_orig_copy_to_gpu = CpuGpuBuffer.copy_to_gpu


def _patched_init(self, *args, **kwargs):
    _orig_init(self, *args, **kwargs)
    pin_memory = self.cpu.is_pinned()
    self._shadow_buffers = [
        torch.empty_like(self.cpu, pin_memory=pin_memory) for _ in range(2)
    ]
    self._shadow_events = [torch.npu.Event() for _ in range(2)]
    self._shadow_idx = 0


def _patched_copy_to_gpu(self, n: int | None = None) -> torch.Tensor:
    # Ensure this shadow's previous H2D has completed before CPU writes to it.
    self._shadow_events[self._shadow_idx].synchronize()

    shadow = self._shadow_buffers[self._shadow_idx]
    if n is None:
        shadow.copy_(self.cpu)
        result = self.gpu.copy_(shadow, non_blocking=True)
    else:
        shadow[:n].copy_(self.cpu[:n])
        result = self.gpu[:n].copy_(shadow[:n], non_blocking=True)

    self._shadow_events[self._shadow_idx].record()
    self._shadow_idx = (self._shadow_idx + 1) % 2
    return result


CpuGpuBuffer.__init__ = _patched_init
CpuGpuBuffer.copy_to_gpu = _patched_copy_to_gpu
