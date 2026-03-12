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

The forward_context module is internal to vLLM and can change between
versions. This module re-exports whatever vLLM provides when available,
and supplies minimal mock implementations otherwise so that the rest of
vllm-ascend can import these names without caring whether the underlying
vLLM version ships them or not.

Important: vLLM's set_forward_context is a @contextmanager generator,
and callers (e.g. ascend_forward_context.py) use it with 'with'.
Any wrapper or fallback here must preserve that context-manager protocol.
"""

import warnings
from contextlib import contextmanager
from typing import Any, Optional

# Try to import directly from vLLM. When vLLM is installed and its
# forward_context module exists, we just re-export everything as-is.
# No wrapping needed -- that would break the context-manager protocol
# of set_forward_context and risk signature mismatches.
try:
    from vllm.forward_context import (
        BatchDescriptor,
        ForwardContext,
        get_forward_context,
        set_forward_context,
    )
except ImportError:
    warnings.warn(
        "forward_context not found in vllm. "
        "Using mock implementation -- some features may not work correctly.",
        RuntimeWarning,
    )

    class BatchDescriptor:
        """Stand-in for vLLM's BatchDescriptor dataclass."""

        def __init__(self, num_tokens: int = 0, num_reqs: Optional[int] = None, **kwargs):
            self.num_tokens = num_tokens
            self.num_reqs = num_reqs
            for key, value in kwargs.items():
                setattr(self, key, value)

    class ForwardContext:
        """Stand-in for vLLM's ForwardContext dataclass.

        Unlike the real dataclass this is a plain object that accepts
        any attribute assignment, since downstream code (ascend_forward_context)
        freely sets attributes like moe_comm_type, flash_comm_v1_enabled, etc.
        """

        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    # Module-level storage, mirrors vLLM's _forward_context global.
    _forward_context: Optional[ForwardContext] = None

    def get_forward_context() -> ForwardContext:
        """Return the current forward context (mock)."""
        assert _forward_context is not None, (
            "Forward context is not set. "
            "Please use set_forward_context to set the forward context."
        )
        return _forward_context

    @contextmanager
    def set_forward_context(
        attn_metadata: Any = None,
        vllm_config: Any = None,
        virtual_engine: int = 0,
        num_tokens: Optional[int] = None,
        num_tokens_across_dp: Any = None,
        cudagraph_runtime_mode: Any = None,
        batch_descriptor: Optional[BatchDescriptor] = None,
        skip_compiled: bool = False,
        **kwargs,
    ):
        """Mock context manager matching vLLM's set_forward_context signature.

        Creates a temporary ForwardContext, makes it the active context for
        the duration of the ``with`` block, then restores the previous one.
        """
        global _forward_context
        prev = _forward_context

        ctx = ForwardContext(
            attn_metadata=attn_metadata,
            virtual_engine=virtual_engine,
            dp_metadata=None,
            cudagraph_runtime_mode=cudagraph_runtime_mode,
            batch_descriptor=batch_descriptor,
            skip_compiled=skip_compiled,
        )
        _forward_context = ctx
        try:
            yield
        finally:
            _forward_context = prev


__all__ = [
    "BatchDescriptor",
    "ForwardContext",
    "get_forward_context",
    "set_forward_context",
]
