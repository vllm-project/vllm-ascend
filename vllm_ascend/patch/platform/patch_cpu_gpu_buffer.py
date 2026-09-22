# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# mypy: ignore-errors

"""Keep ``CpuGpuBuffer.copy_to_gpu`` allocation-free on Ascend.

vLLM main made ``copy_to_gpu`` call ``cpu.pin_memory()`` on every copy. Ascend
already creates the host buffer with the requested pinning
(``CpuGpuBuffer(..., pin_memory=PIN_MEMORY)``), so that extra call is redundant,
and in CPU-only environments it raises ``PrivateUse1HooksInterface`` errors
because the NPU pin backend is not registered. Restore the plain copy.
"""

from vllm.v1.utils import CpuGpuBuffer


def _ascend_copy_to_gpu(self, n=None):
    cpu, gpu = self.cpu, self.gpu
    if n is not None:
        cpu, gpu = cpu[:n], gpu[:n]
    return gpu.copy_(cpu, non_blocking=True)


CpuGpuBuffer.copy_to_gpu = _ascend_copy_to_gpu
