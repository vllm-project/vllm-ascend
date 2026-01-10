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

# Import all scheme modules to trigger registration via @register_scheme decorator
# Note: Add new quantization modules here to register them
from . import w4a4_flatquant  # noqa: F401
from . import w4a8  # noqa: F401
from . import w4a16  # noqa: F401
from . import w8a8_dynamic  # noqa: F401
from . import w8a8_mxfp8  # noqa: F401
from . import w8a8_pdmix  # noqa: F401
from . import w8a8_static  # noqa: F401
from . import w8a16  # noqa: F401
# Import base classes
from .base import AscendLinearScheme, AscendMoEScheme, QuantType
# Import registry functions
from .registry import get_scheme_class, register_scheme


def is_mx_quant_type(instance: Any) -> bool:
    """Checks if the quantization method is a microscaling (MX) type.
    
    Args:
        instance: The quantization method instance to check.
        
    Returns:
        True if the instance is an MX quantization type, False otherwise.
    """
    from .w8a8_mxfp8 import AscendW8A8MXFP8DynamicLinearMethod
    MX_QUANT_TYPES = (AscendW8A8MXFP8DynamicLinearMethod, )
    return isinstance(instance, MX_QUANT_TYPES)


__all__ = [
    # Base classes
    "AscendLinearScheme",
    "AscendMoEScheme",
    "QuantType",
    # Registry functions
    "register_scheme",
    "get_scheme_class",
    # Utility functions
    "is_mx_quant_type",
]
