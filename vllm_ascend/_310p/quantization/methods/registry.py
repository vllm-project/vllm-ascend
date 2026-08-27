#
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

from typing import Any

# 310P-local registry: maps (quant_type, layer_type) -> SchemeClass
_SCHEME_REGISTRY: dict[tuple[str, str], type[Any]] = {}


def register_scheme(quant_type: str, layer_type: str):
    """Decorator to register a 310P quantization scheme."""

    def decorator(cls: type[Any]) -> type[Any]:
        key = (quant_type, layer_type)
        if key in _SCHEME_REGISTRY:
            raise ValueError(
                f"[310P] Scheme already registered for {quant_type}/{layer_type}: {_SCHEME_REGISTRY[key].__name__}"
            )
        _SCHEME_REGISTRY[key] = cls
        return cls

    return decorator


def get_scheme_class(quant_type: str, layer_type: str) -> type[Any] | None:
    """Get 310P scheme class for given quant_type and layer_type."""
    return _SCHEME_REGISTRY.get((quant_type, layer_type))
