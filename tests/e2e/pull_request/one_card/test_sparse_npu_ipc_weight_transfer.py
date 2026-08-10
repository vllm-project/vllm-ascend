# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end test for sparse NPU IPC weight transfer.

Like the dense NPU IPC E2E, the trainer and inference worker share one
physical NPU. The server uses dummy weights while the trainer is built from
the model config, keeping the test independent of checkpoint downloads.
"""

import os

import pytest
import requests
import torch
import torch_npu  # noqa: F401
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from tests.e2e.conftest import RemoteOpenAIServer

MODEL_NAME = os.environ.get(
    "VLLM_SPARSE_IPC_TEST_MODEL", "Qwen/Qwen3-0.6B"
)
PARAMETER_NAME = "model.embed_tokens.weight"
PROMPT = "The future of AI is"
INFERENCE_DEVICE_INDEX = 0
CONTROL_TIMEOUT = 60


def build_trainer_model(device_index: int):
    device = f"npu:{device_index}"
    override_path = os.getenv("WEIGHT_TRANSFER_TEST_MODEL")
    if override_path:
        model = AutoModelForCausalLM.from_pretrained(
            override_path, dtype=torch.bfloat16
        )
    else:
        config = AutoConfig.from_pretrained(
            MODEL_NAME, trust_remote_code=True
        )
        model = AutoModelForCausalLM.from_config(config)
    model = model.to(device=device, dtype=torch.bfloat16)
    model.eval()
    return model


def post(server, endpoint, payload=None):
    response = requests.post(
        server.url_for(endpoint), json=payload, timeout=CONTROL_TIMEOUT
    )
    response.raise_for_status()


def generate(client):
    response = client.completions.create(
        model=MODEL_NAME,
        prompt=PROMPT,
        max_tokens=16,
        temperature=0,
    )
    return response.choices[0].text


@pytest.mark.skipif(
    torch.npu.device_count() < 1,
    reason="Sparse NPU IPC E2E requires one NPU.",
)
def test_sparse_npu_ipc_updates_runtime_parameter_and_resumes_generation():
    from vllm.utils.network_utils import get_open_port

    port = get_open_port()
    server_args = [
        "--enforce-eager",
        "--load-format",
        "dummy",
        "--weight-transfer-config",
        '{"backend":"sparse_ipc"}',
        "--gpu-memory-utilization",
        "0.5",
        "--max-model-len",
        "1024",
        "--port",
        str(port),
        "--trust-remote-code",
    ]
    env_dict = {
        "VLLM_SERVER_DEV_MODE": "1",
        "VLLM_ALLOW_INSECURE_SERIALIZATION": "1",
        "VLLM_ASCEND_ENABLE_NZ": "0",
        "ASCEND_RT_VISIBLE_DEVICES": str(INFERENCE_DEVICE_INDEX),
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
        output_before = generate(client)

        torch.npu.set_device(INFERENCE_DEVICE_INDEX)
        os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        trainer = build_trainer_model(INFERENCE_DEVICE_INDEX)
        parameter = trainer.get_parameter(PARAMETER_NAME)
        token_ids = tokenizer(PROMPT, add_special_tokens=False)["input_ids"][:4]
        rows = torch.tensor(token_ids, device=parameter.device, dtype=torch.long)
        with torch.no_grad():
            parameter[rows] = parameter[rows].roll(1, dims=0)

        hidden_size = parameter.shape[1]
        columns = torch.arange(hidden_size, device=parameter.device)
        indices = (rows.unsqueeze(1) * hidden_size + columns).reshape(-1)

        from vllm_ascend.distributed.weight_transfer.sparse_weight_patch import (
            SparseWeightPatch,
        )
        from vllm_ascend.distributed.weight_transfer.sparse_npu_ipc_engine import (
            SparseNPUIPCTrainerSendWeightsArgs,
            SparseNPUIPCWeightTransferEngine,
        )

        sparse_patch = SparseWeightPatch(
            PARAMETER_NAME,
            indices.to(torch.int32),
            parameter[rows].reshape(-1).contiguous(),
        )
        post(server, "init_weight_transfer_engine", {"init_info": {}})
        post(server, "pause")
        post(server, "start_weight_update")
        SparseNPUIPCWeightTransferEngine.trainer_send_weights(
            iter([sparse_patch]),
            SparseNPUIPCTrainerSendWeightsArgs(
                send_mode="http",
                url=server.url_root,
                parameter_shapes={PARAMETER_NAME: list(parameter.shape)},
            ),
        )
        post(server, "finish_weight_update")
        post(server, "resume")

        output_after = generate(client)

    assert output_after != output_before, (
        "server weights did not change after sparse NPU IPC transfer"
    )
