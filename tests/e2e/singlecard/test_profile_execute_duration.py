#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
# Adapted from vllm/tests/basic_correctness/test_basic_correctness.py
# Copyright 2023 The vLLM team.
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
import gc
import os
import time
from unittest.mock import patch

import torch
import vllm  # noqa: F401

from vllm_ascend.utils import ProfileExecuteDuration


@patch.dict(os.environ, {"VLLM_ASCEND_MODEL_EXECUTE_TIME_OBSERVE": "1"})
def test_execue_duration_enabled_discrepancy():
    a = torch.randn(10000, 10000).npu()
    b = torch.randn(10000, 10000).npu()

    # warmup
    torch.matmul(a, b)
    torch.npu.synchronize()

    cpu_start = time.perf_counter()
    with ProfileExecuteDuration().capture_async("forward"):
        torch.matmul(a, b)
        torch.npu.synchronize()
        cpu_duration = (time.perf_counter() - cpu_start) * 1000
    npu_durations = ProfileExecuteDuration().pop_captured_sync()
    assert npu_durations and 'forward' in npu_durations
    assert not ProfileExecuteDuration._observations

    # Assert discrepancy between CPU and NPU duration is within 50% roughly
    diff = abs(cpu_duration - npu_durations['forward']) / max(
        cpu_duration, npu_durations['forward'])
    assert diff <= 0.5, (
        f"CPU={cpu_duration:.2f}ms, NPU={npu_durations['forward']:.2f}ms")

    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()


def test_execue_duration_disabled():
    a = torch.randn(100, 100).npu()
    b = torch.randn(100, 100).npu()

    with ProfileExecuteDuration().capture_async("forward"):
        torch.matmul(a, b)
        torch.npu.synchronize()
    npu_durations = ProfileExecuteDuration().pop_captured_sync()
    assert not npu_durations

    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()
