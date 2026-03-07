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
"""Compatibility layer for vLLM internal APIs.

This module wraps vLLM's internal APIs to handle version differences.
When vLLM changes their internal implementation, we can adapt here
instead of breaking the entire codebase.
"""

import warnings
from packaging import version

# Figure out which vLLM version we're dealing with
try:
    import vllm
    VLLM_VERSION = version.parse(getattr(vllm, '__version__', '0.0.0'))
except (ImportError, version.InvalidVersion):
    VLLM_VERSION = version.parse("0.0.0")
    warnings.warn("Could not detect vLLM version", RuntimeWarning)

# We test against these versions
MIN_VLLM_VERSION = "0.11.0"
MAX_VLLM_VERSION = "0.12.0"

# Warn if the vLLM version is too old
if VLLM_VERSION < version.parse(MIN_VLLM_VERSION):
    warnings.warn(
        f"vllm-ascend is designed for vLLM >= {MIN_VLLM_VERSION}, "
        f"but found vLLM {VLLM_VERSION}. Some features may not work correctly.",
        RuntimeWarning
    )

# Pull in the compatibility wrappers
from .config import CUDAGraphMode, VllmConfig
from .forward_context import (
    BatchDescriptor,
    ForwardContext,
    get_forward_context,
    set_forward_context,
)

__all__ = [
    'VLLM_VERSION',
    'MIN_VLLM_VERSION',
    'MAX_VLLM_VERSION',
    'CUDAGraphMode',
    'VllmConfig',
    'BatchDescriptor',
    'ForwardContext',
    'get_forward_context',
    'set_forward_context',
]