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

from typing import Any

# Registry: maps (quant_type, layer_type) -> SchemeClass
_SCHEME_REGISTRY: dict[tuple[str, str], type[Any]] = {}


def register_scheme(quant_type: str, layer_type: str):
    """Decorator to register a quantization scheme.

    Args:
        quant_type: Quantization type (e.g., "W8A8", "W8A8_DYNAMIC").
        layer_type: Layer type (e.g., "linear", "moe").

    Returns:
        Decorator function that registers the class.

    Example:
        @register_scheme("W8A8_DYNAMIC", "linear")
        class W8A8DynamicLinearScheme(AscendLinearScheme):
            ...
    """

    def decorator(cls: type[Any]) -> type[Any]:
        key = (quant_type, layer_type)
        if key in _SCHEME_REGISTRY:
            raise ValueError(
                f"Scheme already registered for {quant_type}/{layer_type}: {_SCHEME_REGISTRY[key].__name__}"
            )
        _SCHEME_REGISTRY[key] = cls
        return cls

    return decorator


def get_scheme_class(quant_type: str, layer_type: str) -> type[Any] | None:
    """Get scheme class for given quant_type and layer_type.

    Args:
        quant_type: Quantization type (e.g., "W8A8", "W8A8_DYNAMIC").
        layer_type: Layer type (e.g., "linear", "moe").

    Returns:
        The registered scheme class, or None if not found.
    """
    return _SCHEME_REGISTRY.get((quant_type, layer_type))
