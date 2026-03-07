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
"""Compatibility wrappers for vllm.forward_context module.

The forward_context module is internal to vLLM and can change.
This module provides stable imports with fallback implementations.
"""

import warnings
from typing import Any, Optional

# Try to import from vLLM first, fall back to mock if not available
try:
    from vllm.forward_context import (
        BatchDescriptor,
        ForwardContext,
        get_forward_context,
        set_forward_context,
    )
    _USE_MOCK = False
except ImportError:
    warnings.warn(
        "forward_context not found in vllm. "
        "Using mock implementation. Some features may not work correctly.",
        RuntimeWarning
    )
    _USE_MOCK = True

    class BatchDescriptor:
        """Mock BatchDescriptor when vLLM doesn't provide it."""

        def __init__(
            self,
            num_tokens: int,
            batch_size: int,
            max_seq_len: int = 0,
            **kwargs
        ):
            self.num_tokens = num_tokens
            self.batch_size = batch_size
            self.max_seq_len = max_seq_len
            for key, value in kwargs.items():
                setattr(self, key, value)

    class ForwardContext:
        """Mock ForwardContext when vLLM doesn't provide it."""

        def __init__(self, batch_descriptor: Optional[BatchDescriptor] = None):
            self.batch_descriptor = batch_descriptor

    # Global context storage for mock implementation
    _current_context: Optional[ForwardContext] = None

    def get_forward_context() -> Optional[ForwardContext]:
        """Get the current forward context."""
        return _current_context

    def set_forward_context(context: Optional[ForwardContext] = None, **kwargs) -> None:
        """Set the current forward context."""
        global _current_context
        if context is not None:
            _current_context = context


# Wrap real implementations with error handling if not using mock
if not _USE_MOCK:
    from vllm.forward_context import set_forward_context as _original_set_forward_context
    from vllm.forward_context import get_forward_context as _original_get_forward_context

    def set_forward_context(context: Any = None, **kwargs) -> None:
        """Set forward context with error handling."""
        try:
            _original_set_forward_context(context, **kwargs)
        except Exception as e:
            warnings.warn(f"Error setting forward context: {e}", RuntimeWarning)

    def get_forward_context() -> Any:
        """Get forward context with error handling."""
        try:
            return _original_get_forward_context()
        except Exception as e:
            warnings.warn(f"Error getting forward context: {e}", RuntimeWarning)
            return None


__all__ = [
    'BatchDescriptor',
    'ForwardContext',
    'get_forward_context',
    'set_forward_context',
]