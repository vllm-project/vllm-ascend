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
#
import torch
from torch import Tensor
from vllm import ir

rms_no_var_size = lambda x, weight, epsilon, variance_size=None: variance_size is None  # noqa: E731


@ir.ops.rms_norm.register_impl(
    "npu_kernels",
    supports_args=rms_no_var_size,
    supported=True,
)
def rms_norm(
    x: Tensor,
    weight: Tensor | None,
    epsilon: float,
    variance_size: int | None = None,
) -> Tensor:
    import torch_npu

    if weight is None:
        weight = torch.ones(x.shape[-1], device=x.device, dtype=x.dtype)
    assert variance_size is None
    x, _ = torch_npu.npu_rms_norm(x, weight, epsilon)
    return x, _
