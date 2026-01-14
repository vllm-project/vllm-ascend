# Adapt from https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu/block_table.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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
#
import numpy as np
import torch
import vllm.v1.worker.gpu.block_table
import vllm.v1.worker.gpu.states


class UvaBufferWrapper:
    """Ascend NPU doesn't support UVA tensors directly. This is a wrapper class
    that provides CPU and NPU views of a UVA tensor."""

    def __init__(self, *size: int | torch.SymInt, dtype: torch.dtype):
        self.cpu: torch.Tensor = torch.zeros(*size,
                                             dtype=dtype,
                                             device="cpu",
                                             pin_memory=True)
        self.np: np.ndarray = self.cpu.numpy()
        self._gpu: torch.Tensor | None = None

    @property
    def gpu(self) -> torch.Tensor:
        """Get the NPU view of the buffer."""
        if self._gpu is None:
            # use pin_memory and non_blocking copy to NPU, because in npu,
            # non_blocking copy only works for pinned memory.
            self._gpu = self.cpu.pin_memory().to("npu", non_blocking=True)
        return self._gpu


vllm.v1.worker.gpu.states.UvaBuffer = UvaBufferWrapper
vllm.v1.worker.gpu.block_table.UvaBuffer = UvaBufferWrapper
