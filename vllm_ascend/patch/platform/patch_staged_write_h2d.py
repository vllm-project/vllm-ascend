# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
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
# TEMPORARY: keep this patch out of the final test PR. It orders the staged
# request-metadata host-to-device copies with the following apply-write kernel
# on Ascend.

import torch
from vllm.v1.worker.gpu import buffer_utils

# vLLM <= v0.33 stages the metadata as Python lists and uploads them through
# async_tensor_h2d, whose final `.to(non_blocking=True)` is stream-ordered on
# CUDA but not guaranteed to be on Ascend.
_original_async_tensor_h2d = getattr(buffer_utils, "async_tensor_h2d", None)
if _original_async_tensor_h2d is not None:

    def _async_tensor_h2d_ordered_for_staged_lists(data, device=None, dtype=None, out=None):
        if isinstance(data, (list, tuple)):
            target_dtype = dtype if dtype is not None else (out.dtype if out is not None else None)
            staged = torch.tensor(data, dtype=target_dtype, device="cpu")
            if out is not None:
                return out.copy_(staged)
            assert device is not None, "must provide destination tensor or device"
            return staged.to(device=device, dtype=target_dtype)
        return _original_async_tensor_h2d(data, device=device, dtype=dtype, out=out)

    buffer_utils.async_tensor_h2d = _async_tensor_h2d_ordered_for_staged_lists


# Newer vLLM falls back to NonUvaBuffer without pinned-memory UVA; its
# non-blocking copy has the same ordering hazard. The staged payload is only a
# few integers per step, so a blocking copy is negligible.
_non_uva_buffer = getattr(buffer_utils, "NonUvaBuffer", None)
if _non_uva_buffer is not None:

    def _uva_copy_blocking(self, n: int | None = None) -> torch.Tensor:
        if n is None:
            return self._uva.copy_(self.cpu)
        return self._uva[:n].copy_(self.cpu[:n])

    _non_uva_buffer.uva = _uva_copy_blocking
