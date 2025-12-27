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

import pytest
from vllm.utils import get_open_port

from tests.e2e.conftest import RemoteEPDServer, DisaggEpdProxy
from tools.aisbench import run_aisbench_cases

MODELS = [
    "Qwen/Qwen2.5-VL-7B-Instruct",
]
SHARED_STORAGE_PATH = "/dev/shm/epd/storage"
TENSOR_PARALLELS = [1]

warmup_cases = [{
    "case_type": "performance",
    "dataset_path": "vllm-ascend/textvqa-perf-1080p",
    "request_conf": "vllm_api_stream_chat",
    "dataset_conf": "textvqa/textvqa_gen_base64",
    "num_prompts": 50,
    "max_out_len": 20,
    "batch_size": 32,
    "request_rate": 0,
    "baseline": 1,
    "threshold": 0.97
}]
aisbench_cases = [{
    "case_type": "accuracy",
    "dataset_path": "vllm-ascend/textvqa-lite",
    "request_conf": "vllm_api_stream_chat",
    "dataset_conf": "textvqa/textvqa_gen_base64",
    "max_out_len": 2048,
    "batch_size": 128,
    "baseline": 82.05,
    "threshold": 5
}, {
    "case_type": "performance",
    "dataset_path": "vllm-ascend/textvqa-perf-1080p",
    "request_conf": "vllm_api_stream_chat",
    "dataset_conf": "textvqa/textvqa_gen_base64",
    "num_prompts": 512,
    "max_out_len": 256,
    "batch_size": 128,
    "request_rate": 0,
    "baseline": 1,
    "threshold": 0.97
}]


@pytest.mark.asyncio
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("tp_size", TENSOR_PARALLELS)
async def test_models(model: str, tp_size: int) -> None:
    encode_port = get_open_port()
    pd_port = get_open_port()
    e_server_args = [
        "--port",
        str(encode_port), "--model", model, "--gpu-memory-utilization", "0.01",
        "--tensor-parallel-size",
        str(tp_size), "--enforce-eager", "--no-enable-prefix-caching",
        "--max-model-len", "10000", "--max-num-batched-tokens", "10000",
        "--max-num-seqs", "1", "--ec-transfer-config",
        '{"ec_connector_extra_config":{"shared_storage_path":"' +
        SHARED_STORAGE_PATH +
        '"},"ec_connector":"ECSharedStorageConnector","ec_role": "ec_producer"}'
    ]
    pd_server_args = [
        "--port",
        str(pd_port), "--model", model, "--gpu-memory-utilization", "0.95",
        "--tensor-parallel-size",
        str(tp_size), "--enforce-eager", "--max-model-len", "10000",
        "--max-num-batched-tokens", "10000", "--max-num-seqs", "128",
        "--ec-transfer-config",
        '{"ec_connector_extra_config":{"shared_storage_path":"' +
        SHARED_STORAGE_PATH +
        '"},"ec_connector":"ECSharedStorageConnector","ec_role": "ec_consumer"}'
    ]
    proxy_port = get_open_port()
    proxy_args = [
        "--host", "127.0.0.1", "--port",
        str(proxy_port), "--encode-servers-urls",
        f"http://localhost:{encode_port}", "--decode-servers-urls",
        f"http://localhost:{pd_port}", "--prefill-servers-urls", "disable"
    ]

    async with RemoteEPDServer(e_serve_args=e_server_args,
                               pd_serve_args=pd_server_args) as server:
        async with DisaggEpdProxy(proxy_args=proxy_args) as proxy:
            # warm up
            run_aisbench_cases(model=model,
                               port=proxy_port,
                               aisbench_cases=warmup_cases)
            # aisbench test
            run_aisbench_cases(model, proxy_port, aisbench_cases)
