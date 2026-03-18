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

from tests.e2e.conftest import wait_until_npu_memory_free
from tests.e2e.qwen35_mtp_compat import GRAPH_COMPILATION_CONFIG, run_qwen35_mtp_smoke_test


@wait_until_npu_memory_free()
def test_qwen35_mtp_async_tp2_pcp2_eager():
    run_qwen35_mtp_smoke_test(
        "tp2_pcp2_eager",
        tensor_parallel_size=2,
        prefill_context_parallel_size=2,
        decode_context_parallel_size=1,
        enforce_eager=True,
    )


@wait_until_npu_memory_free()
def test_qwen35_mtp_async_tp2_pcp2_full_decode():
    run_qwen35_mtp_smoke_test(
        "tp2_pcp2_full_decode",
        tensor_parallel_size=2,
        prefill_context_parallel_size=2,
        decode_context_parallel_size=1,
        compilation_config=GRAPH_COMPILATION_CONFIG,
    )
