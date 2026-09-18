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

import json
import os

import pytest
import requests
from vllm.utils.network_utils import get_open_port

from tests.e2e.conftest import DisaggPDProxy, RemotePDServer, VllmRunner, wait_until_npu_memory_free


@pytest.mark.parametrize("enable_fused_mc2", [0, 2], ids=["default", "megamoe"])
@wait_until_npu_memory_free()
def test_moe_w8a8_tp_pp_ep_full_decode_only(enable_fused_mc2: int):
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
        additional_config={"enable_fused_mc2": enable_fused_mc2},
        compilation_config={"cudagraph_capture_sizes": [2, 4, 6, 8, 10, 12], "cudagraph_mode": "FULL_DECODE_ONLY"},
    ) as vllm_model:
        outputs = vllm_model.generate_greedy(prompts, max_tokens=500)

        assert len(outputs) == len(prompts)
        assert len(outputs[0][1]) > len(prompts[0])


def _run_pd_disaggregation_w8a8_sfa(
    *,
    prefill_tp_size: int,
    prefill_pcp_size: int,
    decode_tp_size: int,
    async_scheduling: bool,
    use_model_runner_v2: bool,
) -> None:
    """Run one W8A8 1P1D SFA request through the disaggregated proxy."""
    prefiller_port = [get_open_port()]
    decoder_port = [get_open_port()]
    proxy_port = get_open_port()

    ld_library_path = os.environ.get("LD_LIBRARY_PATH", "")
    env_dict = {
        "LD_LIBRARY_PATH": f"/usr/local/lib:{ld_library_path}",
    }
    if use_model_runner_v2:
        env_dict["VLLM_USE_V2_MODEL_RUNNER"] = "1"

    vllm_server_args = [
        [
            "--port",
            str(prefiller_port[0]),
            "--model",
            "vllm-ascend/DeepSeek-V3.2-W8A8-Pruning",
            "--trust-remote-code",
            "--enable-request-id-headers",
            "--no-enable-prefix-caching",
            "--enable-expert-parallel",
            "--quantization",
            "ascend",
            "--max-model-len",
            "1024",
            "--max-num-batched-tokens",
            "1024",
            "--max-num-seqs",
            "4",
            "--tensor-parallel-size",
            str(prefill_tp_size),
            "--prefill-context-parallel-size",
            str(prefill_pcp_size),
            "--gpu-memory-utilization",
            "0.9",
            "--kv-transfer-config",
            json.dumps(
                {
                    "kv_connector": "MooncakeConnectorV1",
                    "kv_role": "kv_producer",
                    "kv_port": "30000",
                    "kv_connector_extra_config": {
                        "prefill": {"dp_size": 1, "tp_size": prefill_tp_size},
                        "decode": {"dp_size": 1, "tp_size": decode_tp_size},
                    },
                }
            ),
            "--enforce-eager",
        ],
        [
            "--port",
            str(decoder_port[0]),
            "--model",
            "vllm-ascend/DeepSeek-V3.2-W8A8-Pruning",
            "--trust-remote-code",
            "--enable-request-id-headers",
            "--no-enable-prefix-caching",
            "--enable-expert-parallel",
            "--quantization",
            "ascend",
            "--max-model-len",
            "1024",
            "--max-num-batched-tokens",
            "1024",
            "--max-num-seqs",
            "4",
            "--tensor-parallel-size",
            str(decode_tp_size),
            "--gpu-memory-utilization",
            "0.9",
            "--kv-transfer-config",
            json.dumps(
                {
                    "kv_connector": "MooncakeConnectorV1",
                    "kv_role": "kv_consumer",
                    "kv_port": "30200",
                    "kv_connector_extra_config": {
                        "prefill": {"dp_size": 1, "tp_size": prefill_tp_size},
                        "decode": {"dp_size": 1, "tp_size": decode_tp_size},
                    },
                }
            ),
            "--compilation-config",
            json.dumps(
                {
                    "cudagraph_mode": "FULL_DECODE_ONLY",
                    "cudagraph_capture_sizes": [1, 2, 4, 8],
                }
            ),
            *(["--async-scheduling"] if async_scheduling else []),
        ],
    ]

    with (
        RemotePDServer(vllm_server_args, env_dict=env_dict),
        DisaggPDProxy(
            port=proxy_port,
            prefiller_ports=prefiller_port,
            decoder_ports=decoder_port,
        ) as proxy,
    ):
        response = requests.post(
            proxy.url_for("v1", "chat", "completions"),
            json={
                "model": "vllm-ascend/DeepSeek-V3.2-W8A8-Pruning",
                "messages": [{"role": "user", "content": "Hello, my name is"}],
                "max_tokens": 5,
                "temperature": 0.0,
            },
            timeout=600,
        )
        response.raise_for_status()
        output = response.json()

        assert output["choices"][0]["message"]["content"]


@wait_until_npu_memory_free()
def test_pd_disaggregation_w8a8_sfa_dsa_full_decode_only():
    """Verify the existing TP2 1P1D SFA deployment."""
    _run_pd_disaggregation_w8a8_sfa(
        prefill_tp_size=2,
        prefill_pcp_size=1,
        decode_tp_size=2,
        async_scheduling=False,
        use_model_runner_v2=False,
    )


@wait_until_npu_memory_free()
def test_pd_disaggregation_w8a8_sfa_pcp_full_decode_only():
    """Verify PCP2 prefill KV transfer to an async TP1 decode server."""
    _run_pd_disaggregation_w8a8_sfa(
        prefill_tp_size=1,
        prefill_pcp_size=2,
        decode_tp_size=1,
        async_scheduling=True,
        use_model_runner_v2=True,
    )
