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
"""PR performance guard for the Qwen3-32B W8A8 long-context exit scenario.

Long-context high-throughput (LC): 128k input / 1k output.

The same benchmark is run once with the V1 model runner
(``VLLM_USE_V2_MODEL_RUNNER=0``) and once with the V2 model runner
(``VLLM_USE_V2_MODEL_RUNNER=1``); the V2 total token throughput must stay
within +-3% of V1, so stacking/regression issues are exposed at PR time.

Scenario (2026-08-19 revision):
  TP4 + YaRN + prefix caching (90% shared prefix) + FULL_DECODE_ONLY +
  quantization (W8A8), max-model-len=135000, num_prompts=16, output=1024,
  concurrency=4.

The benchmark uses vLLM's built-in ``vllm bench serve`` CLI with the
``prefix_repetition`` synthetic dataset (all prompts share a ~90% prefix), so
it runs in the PR CI environment without external aisbench resources (see
``qwen3_32b_v2_migration_common.py``). eagle3 is excluded until its
accuracy/perf fix lands upstream. Any change to the scenario parameters or
tolerances must be approved by the team.
"""

from tests.e2e.pull_request.four_card.qwen3_32b_v2_migration_common import (
    _LC_ENV,
    BENCH_LC_ARGS,
    QWEN3_32B_W8A8_MODEL,
    _assert_lc_throughput_ratio,
    _benchmark_pair,
    _lc_server_args,
)


def test_qwen3_32b_w8a8_lc_128k_v1_v2_throughput_within_3pct():
    v1_result, v2_result = _benchmark_pair(QWEN3_32B_W8A8_MODEL, _lc_server_args, _LC_ENV, BENCH_LC_ARGS, "LC")
    _assert_lc_throughput_ratio(v1_result, v2_result, "LC")
