#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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

from __future__ import annotations

import torch

try:
    from vllm.model_executor.models.gemma4 import Gemma4Router
except ImportError:
    Gemma4Router = None


def _cached_to_dtype(module: torch.nn.Module, name: str, tensor: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    if tensor.dtype == x.dtype and tensor.device == x.device:
        return tensor

    cache_key = (
        x.device,
        x.dtype,
        tensor.data_ptr(),
        getattr(tensor, "_version", None),
    )
    key_name = f"_ascend_{name}_cache_key"
    cache_name = f"_ascend_{name}_cache"
    if getattr(module, key_name, None) != cache_key:
        setattr(module, cache_name, tensor.to(device=x.device, dtype=x.dtype))
        setattr(module, key_name, cache_key)
    return getattr(module, cache_name)


if Gemma4Router is not None:

    def _forward(self: Gemma4Router, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = x * _cached_to_dtype(self, "root_size", self.root_size, x)
        x = x * _cached_to_dtype(self, "scale", self.scale, x)
        router_logits, _ = self.proj(x)
        return router_logits

    Gemma4Router.forward = _forward
