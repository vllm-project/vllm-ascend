#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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

from tests.e2e.conftest import VllmRunner, wait_until_npu_memory_free
from tests.e2e.pull_request.utils import run_pd_disaggregation


@wait_until_npu_memory_free()
def test_moe_w8a8_tp_pp_ep_full_decode_only():
    """Verify W8A8 MoE generation with TP, PP, EP, and full decode only."""
    model = "vllm-ascend/DeepSeek-V3.2-W8A8-Pruning"
    prompts = ["Hello, my name is"]

    with VllmRunner(
        model,
        enable_expert_parallel=True,
        quantization="ascend",
        max_model_len=1024,
        tensor_parallel_size=2,
        pipeline_parallel_size=2,
        gpu_memory_utilization=0.8,
        compilation_config={"cudagraph_capture_sizes": [2, 4, 6, 8, 10, 12], "cudagraph_mode": "FULL_DECODE_ONLY"},
    ) as vllm_model:
        outputs = vllm_model.generate_greedy(prompts, max_tokens=500)

        assert len(outputs) == len(prompts)
        assert len(outputs[0][1]) > len(prompts[0])


@wait_until_npu_memory_free()
def test_pd_disaggregation_w8a8_sfa_dsa_full_decode_only():
    """Verify the existing TP2 1P1D SFA deployment."""
    run_pd_disaggregation(
        model="vllm-ascend/DeepSeek-V3.2-W8A8-Pruning",
        model_args=["--enable-expert-parallel", "--quantization", "ascend"],
        prefill_tp_size=2,
        prefill_pcp_size=1,
        decode_tp_size=2,
        async_scheduling=False,
        use_model_runner_v2=False,
    )


@wait_until_npu_memory_free()
def test_pd_disaggregation_w8a8_sfa_pcp_full_decode_only():
    """Verify PCP2 prefill KV transfer to an async TP1 decode server."""
    run_pd_disaggregation(
        model="vllm-ascend/DeepSeek-V3.2-W8A8-Pruning",
        model_args=["--enable-expert-parallel", "--quantization", "ascend"],
        prefill_tp_size=1,
        prefill_pcp_size=2,
        decode_tp_size=1,
        async_scheduling=True,
        use_model_runner_v2=True,
    )
