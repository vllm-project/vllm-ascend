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

import json
import os
from collections.abc import Generator
from typing import Any
from unittest.mock import patch

import pytest
from vllm.utils.network_utils import get_open_port

from tests.e2e.conftest import RemoteOpenAIServer, VllmRunner
from tests.e2e.weekly.single_node.features.structured_output.cases import (
    MODEL_NAME,
    SERVED_MODEL_NAME,
)

SERVER_ENV = {
    "VLLM_USE_V1": "1",
    "TASK_QUEUE_ENABLE": "1",
    "HCCL_BUFFSIZE": "1024",
}
ADDITIONAL_CONFIG = {
    "enable_flashcomm1": True,
    "ascend_compilation_config": {"fuse_norm_quant": False},
}


@pytest.fixture(scope="module")
def offline_runner() -> Generator[VllmRunner, None, None]:
    with (
        patch.dict(os.environ, SERVER_ENV, clear=False),
        VllmRunner(
            model_name=MODEL_NAME,
            max_model_len=40960,
            max_num_seqs=64,
            tensor_parallel_size=2,
            distributed_executor_backend="mp",
            quantization="ascend",
            gpu_memory_utilization=0.9,
            compilation_config={"cudagraph_capture_sizes": [64]},
            additional_config=ADDITIONAL_CONFIG,
        ) as runner,
    ):
        yield runner


@pytest.fixture(scope="module")
def openai_server() -> Generator[RemoteOpenAIServer, None, None]:
    server_port = get_open_port()
    server_args = [
        "--trust-remote-code",
        "--port",
        str(server_port),
        "--data-parallel-size",
        "1",
        "--tensor-parallel-size",
        "2",
        "--served-model-name",
        SERVED_MODEL_NAME,
        "--distributed-executor-backend",
        "mp",
        "--max-model-len",
        "40960",
        "--max-num-batched-tokens",
        "16384",
        "--max-num-seqs",
        "64",
        "--gpu-memory-utilization",
        "0.9",
        "--quantization",
        "ascend",
        "--compilation-config",
        json.dumps({"cudagraph_capture_sizes": [64]}),
        "--additional-config",
        json.dumps(ADDITIONAL_CONFIG),
    ]
    with RemoteOpenAIServer(
        MODEL_NAME,
        server_args,
        server_host="127.0.0.1",
        server_port=server_port,
        env_dict=SERVER_ENV,
        auto_port=False,
    ) as server:
        yield server


@pytest.fixture(scope="module")
def openai_client(openai_server: RemoteOpenAIServer) -> Generator[Any, None, None]:
    client = openai_server.get_client()
    try:
        yield client
    finally:
        client.close()
