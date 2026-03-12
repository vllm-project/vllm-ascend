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
"""Ascend quantization scheme implementations.

This module provides all quantization scheme implementations for Ascend NPU.
Schemes are automatically registered via the @register_scheme decorator.

Usage:
    from vllm_ascend.quantization.methods import get_scheme_class

    # Get a scheme class by quant_type and layer_type
    scheme_cls = get_scheme_class("W8A8_DYNAMIC", "linear")
    scheme = scheme_cls()
"""

from typing import Any

# Import base classes
from .base import AscendAttentionScheme, AscendLinearScheme, AscendMoEScheme, QuantType

# Import registry functions
from .registry import get_scheme_class, register_scheme

# Import all scheme classes for external access
from .w4a4_flatquant import AscendW4A4FlatQuantDynamicLinearMethod
from .w4a4_laos_dynamic import AscendW4A4LaosDynamicLinearMethod
from .w4a8 import AscendW4A8DynamicFusedMoEMethod, AscendW4A8DynamicLinearMethod
from .w4a16 import AscendW4A16FusedMoEMethod
from .w8a8_dynamic import AscendW8A8DynamicFusedMoEMethod, AscendW8A8DynamicLinearMethod
from .w8a8_mxfp8 import AscendW8A8MXFP8DynamicLinearMethod
from .w8a8_pdmix import AscendW8A8PDMixFusedMoeMethod, AscendW8A8PDMixLinearMethod
from .w8a8_static import AscendW8A8LinearMethod
from .w8a16 import AscendW8A16LinearMethod


def is_mx_quant_type(instance: Any) -> bool:
    """Checks if the quantization method is a microscaling (MX) type."""
    MX_QUANT_TYPES = (AscendW8A8MXFP8DynamicLinearMethod,)
    return isinstance(instance, MX_QUANT_TYPES)


__all__ = [
    # Base classes
    "AscendAttentionScheme",
    "AscendLinearScheme",
    "AscendMoEScheme",
    "QuantType",
    # Registry functions
    "register_scheme",
    "get_scheme_class",
    # Utility functions
    "is_mx_quant_type",
    # Scheme classes
    "AscendW8A8LinearMethod",
    "AscendW8A8DynamicLinearMethod",
    "AscendW8A8DynamicFusedMoEMethod",
    "AscendW8A8MXFP8DynamicLinearMethod",
    "AscendW8A8PDMixLinearMethod",
    "AscendW8A8PDMixFusedMoeMethod",
    "AscendW8A16LinearMethod",
    "AscendW4A8DynamicLinearMethod",
    "AscendW4A8DynamicFusedMoEMethod",
    "AscendW4A16FusedMoEMethod",
    "AscendW4A4FlatQuantDynamicLinearMethod",
    "AscendW4A4LaosDynamicLinearMethod",
]
