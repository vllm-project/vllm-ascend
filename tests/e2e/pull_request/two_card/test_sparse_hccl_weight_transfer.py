# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end test for sparse HCCL weight transfer.

This follows the dense HCCL E2E workflow: a vLLM server with dummy weights
runs on NPU 0 while a trainer model built from the same config runs on NPU 1.
Only selected embedding rows are sent as sparse ``indices + values`` patches.
"""

import os
import threading

import pytest
import requests
import torch
import torch_npu  # noqa: F401
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from vllm.utils.network_utils import get_ip, get_open_port

from tests.e2e.conftest import RemoteOpenAIServer

MODEL_NAME = os.environ.get(
    "VLLM_SPARSE_HCCL_TEST_MODEL", "Qwen/Qwen3-0.6B"
)
PARAMETER_NAME = "model.embed_tokens.weight"
INFERENCE_WORLD_SIZE = 1
TRAINER_DEVICE_INDEX = INFERENCE_WORLD_SIZE
PROMPTS = [
    "Hello, my name is",
    "The capital of France is",
]
INIT_TIMEOUT = 120
UPDATE_TIMEOUT = 300
CONTROL_TIMEOUT = 60


def _log(message: str) -> None:
    print(f"[trainer] {message}", flush=True)


def _build_trainer_model(device_index: int):
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


def _post(
    server: RemoteOpenAIServer,
    route: str,
    *,
    json=None,
    timeout=CONTROL_TIMEOUT,
):
    response = requests.post(
        server.url_for(route), json=json, timeout=timeout
    )
    response.raise_for_status()
    return response


class _BackgroundPost(threading.Thread):
    def __init__(
        self,
        server: RemoteOpenAIServer,
        route: str,
        *,
        json=None,
        timeout=CONTROL_TIMEOUT,
    ):
        super().__init__(daemon=True)
        self._server = server
        self._route = route
        self._json = json
        self._timeout = timeout
        self.error: BaseException | None = None

    def run(self) -> None:
        try:
            _post(
                self._server,
                self._route,
                json=self._json,
                timeout=self._timeout,
            )
        except BaseException as exc:  # noqa: BLE001
            self.error = exc

    def raise_if_failed(self) -> None:
        if self.error is not None:
            raise RuntimeError(
                f"server-side /{self._route} failed"
            ) from self.error


def _generate(client):
    outputs = []
    for prompt in PROMPTS:
        response = client.completions.create(
            model=MODEL_NAME,
            prompt=prompt,
            max_tokens=16,
            temperature=0,
        )
        outputs.append(response.choices[0].text)
    return outputs


def _build_sparse_patch(model, tokenizer):
    from vllm_ascend.distributed.weight_transfer.sparse_weight_patch import (
        SparseWeightPatch,
    )

    parameter = model.get_parameter(PARAMETER_NAME)
    token_ids = []
    for prompt in PROMPTS:
        token_ids.extend(
            tokenizer(prompt, add_special_tokens=False)["input_ids"]
        )
    rows = torch.tensor(
        sorted(set(token_ids)),
        device=parameter.device,
        dtype=torch.long,
    )
    hidden_size = parameter.shape[1]
    columns = torch.arange(hidden_size, device=parameter.device)
    indices = (rows.unsqueeze(1) * hidden_size + columns).reshape(-1)
    values = parameter.index_select(0, rows).reshape(-1).contiguous()
    return SparseWeightPatch(
        name=PARAMETER_NAME,
        indices=indices.to(torch.int32),
        values=values,
    ), parameter


@pytest.mark.skipif(
    torch.npu.device_count() < 2,
    reason="Sparse HCCL weight transfer E2E requires at least 2 NPUs.",
)
def test_sparse_hccl_weight_transfer_updates_server_weights():
    port = get_open_port()
    server_args = [
        "--enforce-eager",
        "--load-format",
        "dummy",
        "--weight-transfer-config",
        '{"backend":"sparse_hccl"}',
        "--additional-config",
        '{"weight_nz_mode": 0}',
        "--tensor-parallel-size",
        str(INFERENCE_WORLD_SIZE),
        "--max-model-len",
        "1024",
        "--gpu-memory-utilization",
        "0.6",
        "--port",
        str(port),
        "--trust-remote-code",
    ]
    env_dict = {
        "VLLM_SERVER_DEV_MODE": "1",
        "ASCEND_RT_VISIBLE_DEVICES": "0",
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
        outputs_before = _generate(client)

        torch.npu.set_device(TRAINER_DEVICE_INDEX)
        train_model = _build_trainer_model(TRAINER_DEVICE_INDEX)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        patch, parameter = _build_sparse_patch(train_model, tokenizer)

        from vllm_ascend.distributed.weight_transfer.hccl_engine import (
            HCCLTrainerSendWeightsArgs,
        )
        from vllm_ascend.distributed.weight_transfer.sparse_hccl_engine import (
            SparseHCCLWeightTransferEngine,
        )

        master_address = get_ip()
        master_port = get_open_port()
        world_size = INFERENCE_WORLD_SIZE + 1
        init_info = {
            "master_address": master_address,
            "master_port": master_port,
            "rank_offset": 1,
            "world_size": world_size,
        }
        init_thread = _BackgroundPost(
            server,
            "init_weight_transfer_engine",
            json={"init_info": init_info},
            timeout=INIT_TIMEOUT,
        )
        init_thread.start()
        group = SparseHCCLWeightTransferEngine.trainer_init(
            {
                "master_address": master_address,
                "master_port": master_port,
                "world_size": world_size,
            }
        )
        init_thread.join()
        init_thread.raise_if_failed()

        _post(server, "pause")
        _post(server, "start_weight_update")
        update_info = {
            "names": [PARAMETER_NAME],
            "dtype_names": [str(patch.values.dtype).split(".")[-1]],
            "shapes": [list(parameter.shape)],
            "num_updates_list": [patch.indices.numel()],
        }
        update_thread = _BackgroundPost(
            server,
            "update_weights",
            json={"update_info": update_info},
            timeout=UPDATE_TIMEOUT,
        )
        update_thread.start()
        SparseHCCLWeightTransferEngine.trainer_send_weights(
            iter([patch]),
            HCCLTrainerSendWeightsArgs(group=group),
        )
        torch.npu.synchronize()
        update_thread.join()
        update_thread.raise_if_failed()

        _post(server, "finish_weight_update")
        _post(server, "resume")
        outputs_after = _generate(client)
        _log(f"outputs before update: {outputs_before}")
        _log(f"outputs after update: {outputs_after}")

    assert outputs_after != outputs_before, (
        "server weights did not change after sparse HCCL transfer"
    )
