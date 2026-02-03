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
from vllm.utils.network_utils import get_open_port

from tests.e2e.conftest import DisaggEpdProxy, RemoteEPDServer
from tools.send_mm_request import send_image_request

MODELS = [
    "Qwen/Qwen2.5-VL-7B-Instruct",
]
SHARED_STORAGE_PATH = "/dev/shm/epd/storage"
TENSOR_PARALLELS = [1]

@pytest.mark.asyncio
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("tp_size", TENSOR_PARALLELS)
async def test_models(model: str, tp_size: int) -> None:
    encode_port = get_open_port()
    pd_port = get_open_port()
    vllm_server_args = [
        [
            "--port",
            str(encode_port), "--model", model, "--gpu-memory-utilization",
            "0.01", "--tensor-parallel-size",
            str(tp_size), "--enforce-eager", "--no-enable-prefix-caching",
            "--max-model-len", "10000", "--max-num-batched-tokens", "10000",
            "--max-num-seqs", "1", "--ec-transfer-config",
            '{"ec_connector_extra_config":{"shared_storage_path":"' +
            SHARED_STORAGE_PATH +
            '"},"ec_connector":"ECExampleConnector","ec_role": "ec_producer"}'
        ],
        [
            "--port",
            str(pd_port), "--model", model, "--gpu-memory-utilization", "0.95",
            "--tensor-parallel-size",
            str(tp_size), "--enforce-eager", "--max-model-len", "10000",
            "--max-num-batched-tokens", "10000", "--max-num-seqs", "128",
            "--ec-transfer-config",
            '{"ec_connector_extra_config":{"shared_storage_path":"' +
            SHARED_STORAGE_PATH +
            '"},"ec_connector":"ECExampleConnector","ec_role": "ec_consumer"}'
        ]
    ]
    proxy_port = get_open_port()
    proxy_args = [
        "--host", "127.0.0.1", "--port",
        str(proxy_port), "--encode-servers-urls",
        f"http://localhost:{encode_port}", "--decode-servers-urls",
        f"http://localhost:{pd_port}", "--prefill-servers-urls", "disable"
    ]

    with RemoteEPDServer(vllm_serve_args=vllm_server_args) as _:
        with DisaggEpdProxy(proxy_args=proxy_args) as proxy:
            send_image_request(model, proxy)

