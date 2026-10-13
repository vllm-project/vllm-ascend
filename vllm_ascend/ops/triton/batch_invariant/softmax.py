# Adapt from https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/batch_invariant.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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


def softmax_batch_invariant(input_, dim, dtype=None):
    # ``aten::softmax`` passes an optional output dtype, while
    # ``aten::_softmax`` passes a ``half_to_float`` boolean as its third
    # argument. This implementation is registered for both operators.
    if isinstance(dtype, bool):
        if dtype:
            input_ = input_.float()
    elif dtype is not None:
        input_ = input_.to(dtype=dtype)

    # Compute softmax in a deterministic way
    # First subtract max for numerical stability (standard practice)
    input_max = torch.amax(input_, dim=dim, keepdim=True)
    input_ = input_ - input_max
    exp_x = torch.exp(input_)
    sum_exp_x = torch.sum(exp_x, dim=dim, keepdim=True)
    return exp_x / sum_exp_x
