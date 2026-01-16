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
Stream management utilities for vLLM Ascend.

This module provides functionality for:
- Managing different NPU streams (compute, prefetch, global, etc.)
- Switching between streams
"""

from contextlib import nullcontext

import torch
import torch_npu  # noqa: F401

from vllm_ascend.ascend_config import WeightPrefetchConfig

_CURRENT_STREAM = None
_PREFETCH_STREAM = None
_WEIGHT_PREFETCH_METHOD = None
_GLOBAL_STREAM = None
_SHARED_EXPERTS_CALCULATION_STREAM = None
_CP_CHUNKEDPREFILL_COMM_STREAM = None


def current_stream() -> torch.npu.Stream:
    """
    replace `torch.npu.current_stream()` with `vllm.utils.current_stream()`.
    it turns out that `torch.npu.current_stream()` is quite expensive,
    as it will construct a new stream object at each call.
    here we patch `torch.npu.set_stream` to keep track of the current stream
    directly, so that we can avoid calling `torch.npu.current_stream()`.

    """
    global _CURRENT_STREAM
    if _CURRENT_STREAM is None:
        # when this function is called before any stream is set,
        # we return the default stream.
        _CURRENT_STREAM = torch.npu.current_stream()
    return _CURRENT_STREAM


def prefetch_stream() -> torch.npu.Stream:
    global _PREFETCH_STREAM
    if _PREFETCH_STREAM is None:
        # when this function is called before any stream is set,
        # we return the default stream.
        _PREFETCH_STREAM = torch_npu.npu.Stream()
    return _PREFETCH_STREAM


def set_weight_prefetch_method(weight_prefetch_config: WeightPrefetchConfig):
    global _WEIGHT_PREFETCH_METHOD
    if _WEIGHT_PREFETCH_METHOD is None:
        from vllm_ascend.ops.weight_prefetch import WeightPrefetchMethod
        _WEIGHT_PREFETCH_METHOD = WeightPrefetchMethod(weight_prefetch_config)
    return _WEIGHT_PREFETCH_METHOD


def get_weight_prefetch_method():
    return _WEIGHT_PREFETCH_METHOD


def global_stream() -> torch.npu.Stream:
    global _GLOBAL_STREAM
    if _GLOBAL_STREAM is None:
        # when this function is called before any stream is set,
        # we return the default stream.
        _GLOBAL_STREAM = torch_npu.npu.Stream()
    return _GLOBAL_STREAM


def shared_experts_calculation_stream() -> torch.npu.Stream:
    global _SHARED_EXPERTS_CALCULATION_STREAM
    if _SHARED_EXPERTS_CALCULATION_STREAM is None:
        # when this function is called before any stream is set,
        # we return the default stream.
        _SHARED_EXPERTS_CALCULATION_STREAM = torch_npu.npu.Stream()
    return _SHARED_EXPERTS_CALCULATION_STREAM


def cp_chunkedprefill_comm_stream() -> torch.npu.Stream:
    global _CP_CHUNKEDPREFILL_COMM_STREAM
    if _CP_CHUNKEDPREFILL_COMM_STREAM is None:
        _CP_CHUNKEDPREFILL_COMM_STREAM = torch_npu.npu.Stream()
    return _CP_CHUNKEDPREFILL_COMM_STREAM


def npu_stream_switch(target_stream: torch.npu.Stream,
                      *,
                      enabled: bool = True):
    """
    Switch to the target stream if enabled is True.
    Otherwise, do nothing.
    """
    if not enabled:
        return nullcontext()
    assert target_stream is not None
    return torch.npu.stream(target_stream)
