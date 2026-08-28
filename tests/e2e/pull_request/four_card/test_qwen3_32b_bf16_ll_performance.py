#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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
# This file is a part of the vllm-ascend project.
#
"""PR performance guard for the Qwen3-32B short-context exit scenario (bf16).

Short-context low-latency (LL): 16k input / 1k output.

The same benchmark is run once with the V1 model runner
(``VLLM_USE_V2_MODEL_RUNNER=0``) and once with the V2 model runner
(``VLLM_USE_V2_MODEL_RUNNER=1``); V2 TTFT/TPOT must not regress more than +3%
vs V1. Being faster is allowed: the 2026-08-06 measurement showed V2 TTFT is
~2.6% faster than V1 at bs=2, so a two-sided +-3% gate would be fragile.

Scenario (2026-08-19 revision, eagle3 excluded until its accuracy/perf fix
lands upstream):
  TP2 x DP2 + async-scheduling + FULL_DECODE_ONLY (bf16),
  max-model-len=18000, num_prompts=50, output=1024, concurrency=2.
  Prefix caching is not enabled in this low-latency scenario.

The benchmark uses vLLM's built-in ``vllm bench serve`` CLI with the
``random`` synthetic dataset, so it runs in the PR CI environment without
external aisbench resources (see ``qwen3_32b_v2_migration_common.py``).
Any change to the scenario parameters or tolerances must be approved by the
team.
"""

from tests.e2e.pull_request.four_card.qwen3_32b_v2_migration_common import (
    _COMMON_ENV,
    BENCH_LL_ARGS,
    QWEN3_32B_BF16_MODEL,
    _assert_ll_latency,
    _benchmark_pair,
    _ll_server_args,
)


def test_qwen3_32b_bf16_ll_16k_v1_v2_latency_within_3pct():
    v1_result, v2_result = _benchmark_pair(QWEN3_32B_BF16_MODEL, _ll_server_args, _COMMON_ENV, BENCH_LL_ARGS, "LL")
    _assert_ll_latency(v1_result, v2_result, "LL")
