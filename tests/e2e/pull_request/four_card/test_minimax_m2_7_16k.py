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
"""MiniMax-M2.7 4-card 16k1k V1 vs V2 benchmark (separate CI job)."""

from __future__ import annotations

import pytest

from tests.e2e.conftest import wait_until_npu_memory_free
from tests.e2e.pull_request.four_card.minimax_m2_7_common import (
    BENCH_16K,
    MINIMAX_M2_7_MODEL,
    _benchmark_pair,
)


@pytest.mark.e2e_model(MINIMAX_M2_7_MODEL)
@pytest.mark.e2e_coverage(
    arch="moe",
    feature="long_sequence",
    parallel="TP,EP,DP",
    deploy="pd_mix",
    hardware="A3",
    quantization="W8A8",
    graph_mode="eager",
)
@wait_until_npu_memory_free()
def test_minimax_m2_7_16k1k_v2_vs_v1() -> None:
    """16k1k: 400 requests, 20 concurrent, 0% prefix hit, V2 >= V1 * 0.97."""
    _benchmark_pair(
        bench_config=BENCH_16K,
        case="16k1k",
    )
