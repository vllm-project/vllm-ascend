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

"""
Weak reference utilities for vLLM Ascend.

This module provides functionality for:
- Creating weak references to tensors
- Preventing memory leaks in graph capture scenarios
"""

from typing import Any, Union

import torch
from vllm.sequence import IntermediateTensors


def weak_ref_tensor(tensor: Any) -> Any:
    """
    Create a weak reference to a tensor.
    The new tensor will share the same data as the original tensor,
    but will not keep the original tensor alive.
    """
    if isinstance(tensor, torch.Tensor):
        return torch.ops._C_ascend.weak_ref_tensor(tensor)
    else:
        return tensor


def weak_ref_tensors(
    tensors: Union[torch.Tensor, list[torch.Tensor], tuple[torch.Tensor]]
) -> Union[torch.Tensor, list[Any], tuple[Any], Any]:
    """
    Convenience function to create weak references to tensors,
    for single tensor, list of tensors or tuple of tensors.

    This function should be used in the following scenario:
    When a tensor is created during graph capture, and it's held by a method
    that's not part of the graph, we don't really need to store it, but we
    **do need** its buffer pointer. If we don't handle this, it cannot
    be garbage collected, leading to a memory leak. To avoid this,
    we should create a weak reference to the tensor.
    """
    if isinstance(tensors, torch.Tensor):
        return weak_ref_tensor(tensors)
    if isinstance(tensors, list):
        return [weak_ref_tensor(t) for t in tensors]
    if isinstance(tensors, tuple):
        return tuple(weak_ref_tensor(t) for t in tensors)
    # For IntermediateTensors used in pipeline parallelism
    if isinstance(tensors, IntermediateTensors):
        ret = IntermediateTensors({
            key: weak_ref_tensor(val)
            for key, val in tensors.tensors.items()
        })
        return ret
    raise ValueError("Invalid type for tensors")
