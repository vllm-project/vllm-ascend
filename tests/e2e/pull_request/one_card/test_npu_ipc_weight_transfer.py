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
"""End-to-end test for the NPU IPC weight transfer engine.

Unlike the HCCL engine, NPU IPC requires the trainer and the inference worker
to be co-located on the *same* physical NPU chip, so only a single NPU is
needed. The trainer model is built from the architecture config with random
weights (download-free); set ``WEIGHT_TRANSFER_TEST_MODEL=/path/to/checkpoint``
to share real weights instead. See ``examples/rl/rlhf_http_npu_ipc.py`` for the
end-user workflow.
"""

import os

import pytest
import requests
import torch
import torch_npu  # noqa: F401  # registers the NPU backend
from transformers import AutoConfig, AutoModelForCausalLM

from tests.e2e.conftest import RemoteOpenAIServer

MODEL_NAME = "Qwen/Qwen3-0.6B"

INFERENCE_DEVICE_INDEX = 0

PROMPTS = [
    "Hello, my name is",
    "The capital of France is",
]

UPDATE_TIMEOUT = 300
CONTROL_TIMEOUT = 60


def _build_trainer_model(device_index: int):
    device = f"npu:{device_index}"
    override_path = os.getenv("WEIGHT_TRANSFER_TEST_MODEL")
    if override_path:
        model = AutoModelForCausalLM.from_pretrained(override_path, dtype=torch.bfloat16)
    else:
        config = AutoConfig.from_pretrained(MODEL_NAME, trust_remote_code=True)
        model = AutoModelForCausalLM.from_config(config)
    model = model.to(device=device, dtype=torch.bfloat16)
    model.eval()
    return model


def _post(server: RemoteOpenAIServer, route: str, *, json=None, timeout=CONTROL_TIMEOUT):
    response = requests.post(server.url_for(route), json=json, timeout=timeout)
    response.raise_for_status()
    return response


def _generate(client, model, prompts):
    completions = []
    for prompt in prompts:
        response = client.completions.create(model=model, prompt=prompt, max_tokens=16, temperature=0)
        completions.append(response.choices[0].text)
    return completions


def _has_lifecycle_endpoints(server: RemoteOpenAIServer) -> bool:
    """Probe ``/start_weight_update``; also performs the actual call when present."""
    try:
        response = requests.post(
            server.url_for("start_weight_update"),
            json={"is_checkpoint_format": True},
            timeout=CONTROL_TIMEOUT,
        )
    except requests.RequestException:
        return False
    if response.status_code == 404:
        return False
    response.raise_for_status()
    return True


@pytest.mark.e2e_features("multimodal", "eager_mode")
@pytest.mark.e2e_model("Qwen/Qwen3-0.6B")
@pytest.mark.skipif(
    torch.npu.device_count() < 1,
    reason="NPU IPC weight transfer e2e test requires at least 1 NPU.",
)
def test_npu_ipc_weight_transfer_updates_server_weights():
    from vllm.utils.network_utils import get_open_port

    port = get_open_port()
    server_args = [
        "--enforce-eager",
        "--load-format",
        "dummy",
        "--weight-transfer-config",
        '{"backend": "ipc"}',
        "--max-model-len",
        "1024",
        # IPC co-locates the trainer on the same NPU, so leave room for it.
        "--gpu-memory-utilization",
        "0.5",
        "--port",
        str(port),
        "--trust-remote-code",
    ]
    # VLLM_SERVER_DEV_MODE registers the dev endpoints; insecure serialization
    # lets the server unpickle the IPC handles sent over HTTP. Pin the server to
    # physical NPU 0 so its IPC UUID matches the trainer below.
    env_dict = {
        "VLLM_SERVER_DEV_MODE": "1",
        "VLLM_ALLOW_INSECURE_SERIALIZATION": "1",
        "ASCEND_RT_VISIBLE_DEVICES": str(INFERENCE_DEVICE_INDEX),
        "VLLM_ASCEND_ENABLE_NZ": "0",
    }

    with RemoteOpenAIServer(
        MODEL_NAME,
        vllm_serve_args=server_args,
        server_host="127.0.0.1",
        server_port=port,
        env_dict=env_dict,
        auto_port=False,
    ) as server:
        client = server.get_client()

        outputs_before = _generate(client, MODEL_NAME, PROMPTS)

        # Trainer shares physical NPU 0 with the server. It leaves
        # ASCEND_RT_VISIBLE_DEVICES unset (identity mapping logical 0 ->
        # physical 0) so both processes resolve to the same IPC UUID.
        torch.npu.set_device(INFERENCE_DEVICE_INDEX)
        train_model = _build_trainer_model(INFERENCE_DEVICE_INDEX)
        os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"

        from vllm_ascend.distributed.weight_transfer.npu_ipc_engine import (
            NPUIPCTrainerSendWeightsArgs,
            NPUIPCWeightTransferEngine,
        )

        _post(server, "init_weight_transfer_engine", json={"init_info": {}})

        _post(server, "pause")
        # The probe performs /start_weight_update when present, so it must not
        # be called again below. Older vLLM without the lifecycle endpoints is
        # out of scope for this IPC test.
        if not _has_lifecycle_endpoints(server):
            _post(server, "resume")
            pytest.skip("vLLM build lacks the /start_weight_update lifecycle endpoints required by NPU IPC.")

        # trainer_send_weights POSTs to /update_weights itself; the server
        # rebuilds tensors locally and loads them before the POST returns (no
        # collective back to the trainer, so no background thread is needed).
        # train_model stays referenced so the shared NPU storage outlives it.
        trainer_args = NPUIPCTrainerSendWeightsArgs(send_mode="http", url=server.url_root)
        NPUIPCWeightTransferEngine.trainer_send_weights(
            iterator=train_model.named_parameters(),
            trainer_args=trainer_args,
        )

        _post(server, "finish_weight_update")
        _post(server, "resume")

        outputs_after = _generate(client, MODEL_NAME, PROMPTS)

    assert outputs_after != outputs_before, "server weights did not change after NPU IPC transfer"
