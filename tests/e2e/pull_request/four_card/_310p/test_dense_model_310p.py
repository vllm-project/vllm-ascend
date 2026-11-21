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

import os

import pytest
import requests
from prometheus_client.parser import text_string_to_metric_families
from vllm.utils.network_utils import get_open_port

from tests.e2e.conftest import RemoteOpenAIServer, VllmRunner


def test_qwen3_dense_tp4_w8a8():
    example_prompts = [
        "Hello, my name is",
    ]
    max_tokens = 5
    with VllmRunner(
        "vllm-ascend/Qwen3-32B-W8A8",
        tensor_parallel_size=4,
        enforce_eager=True,
        dtype="float16",
        quantization="ascend",
        max_model_len=16384,
    ) as vllm_model:
        vllm_model.generate_greedy(example_prompts, max_tokens)


@pytest.mark.parametrize("dp_size", [1, 2])
def test_qwen3_5_dense_online_dp_tp2(dp_size: int):
    model = os.environ.get("QWEN35_DENSE_MODEL", "Qwen/Qwen3.5-4B")
    port = get_open_port()
    args = [
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--tensor-parallel-size",
        "2",
        "--data-parallel-size",
        str(dp_size),
        "--distributed-executor-backend",
        "mp",
        "--api-server-count",
        "1",
        "--dtype",
        "float16",
        "--mamba-ssm-cache-dtype",
        "float16",
        "--max-model-len",
        "2048",
        "--max-num-seqs",
        "4",
        "--gpu-memory-utilization",
        "0.35",
        "--language-model-only",
        "--enforce-eager",
    ]
    with RemoteOpenAIServer(
        model,
        args,
        server_host="127.0.0.1",
        server_port=port,
        auto_port=False,
        env_dict={"VLLM_USE_V2_MODEL_RUNNER": "1"},
    ) as server:
        with server.get_client() as client:
            for rank in range(dp_size):
                output = client.completions.create(
                    model=model,
                    prompt="Hello, my name is",
                    temperature=0,
                    max_tokens=8,
                    extra_headers={"X-data-parallel-rank": str(rank)},
                )
                assert output.choices[0].text.strip()

        metrics = requests.get(server.url_for("metrics"), timeout=10)
        metrics.raise_for_status()
        ranks = {
            sample.labels["engine"]
            for family in text_string_to_metric_families(metrics.text)
            for sample in family.samples
            if sample.name == "vllm:request_success_total"
            and sample.labels["finished_reason"] in ("length", "stop")
            and sample.value > 0
        }
        assert ranks == {str(rank) for rank in range(dp_size)}
