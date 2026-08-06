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
"""End-to-end test for verl-style RL LoRA updates over NPU IPC.

The trainer and vLLM worker share logical NPU 0 because NPU IPC handles are
valid only between processes on the same physical device. Alice and Bob LoRA
tensors are transferred from trainer NPU memory and replace one stable logical
adapter without writing or reloading an adapter checkpoint.
"""

from __future__ import annotations

import base64
import json
import os
import pickle
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

import huggingface_hub
import pytest
import requests
import safetensors
import torch
import torch_npu  # noqa: F401  # registers the NPU backend
from huggingface_hub import snapshot_download
from msgspec import field
from vllm.lora.peft_helper import PEFTHelper
from vllm.lora.request import LoRARequest
from vllm.lora.worker_manager import LRUCacheWorkerLoRAManager
from vllm.utils.network_utils import get_open_port

if TYPE_CHECKING:
    from tests.e2e.conftest import RemoteOpenAIServer


class TensorLoRARequest(LoRARequest):
    """LoRA request carrying live trainer tensors instead of a checkpoint path."""

    peft_config: dict = field(default=None)
    lora_tensors: dict = field(default=None)


def _install_tensor_lora_loader() -> None:
    """Add verl's in-memory LoRA loading branch without patching vLLM files."""
    if getattr(LRUCacheWorkerLoRAManager, "_rl_tensor_lora_patch", False):
        return

    original_load_adapter = LRUCacheWorkerLoRAManager._load_adapter

    def load_adapter(self, lora_request):
        if not isinstance(lora_request, TensorLoRARequest):
            return original_load_adapter(self, lora_request)

        peft_helper = PEFTHelper.from_dict(lora_request.peft_config)
        peft_helper.validate_legal(self.lora_config)

        model = self._adapter_manager.model
        weights_mapper = getattr(model, "hf_to_vllm_mapper", None)
        if weights_mapper is not None and hasattr(
            weights_mapper, "get_unstacked_mapper"
        ):
            weights_mapper = weights_mapper.get_unstacked_mapper()

        lora = self._lora_model_cls.from_lora_tensors(
            lora_model_id=lora_request.lora_int_id,
            tensors=lora_request.lora_tensors,
            peft_helper=peft_helper,
            device="cpu",
            dtype=self.lora_config.lora_dtype,
            model_vocab_size=self.vocab_size,
            weights_mapper=weights_mapper,
            skip_prefixes=getattr(model, "lora_skip_prefixes", None),
        )
        if getattr(lora, "extra_vocab_size", 0) > getattr(
            self.lora_config, "lora_extra_vocab_size", 0
        ):
            raise ValueError(
                "LoRA added vocab exceeds the configured lora_extra_vocab_size"
            )
        return lora

    LRUCacheWorkerLoRAManager._load_adapter = load_adapter
    LRUCacheWorkerLoRAManager._rl_tensor_lora_patch = True


_install_tensor_lora_loader()


class LoRAIPCWorkerExtension:
    """Receive live LoRA tensors through the NPU IPC transfer engine."""

    def update_weights_from_ipc(
        self,
        update_info_json: str,
        peft_config_json: str,
        lora_name: str,
        lora_int_id: str,
    ) -> dict:
        self._check_weight_transfer_engine()
        update_info = self.weight_transfer_engine.parse_update_info(
            json.loads(update_info_json)
        )
        peft_config = json.loads(peft_config_json)
        adapter_id = int(lora_int_id)
        lora_tensors: dict[str, torch.Tensor] = {}
        received_bytes = 0

        def collect_weights(weights: list[tuple[str, torch.Tensor]]) -> None:
            nonlocal received_bytes
            for name, tensor in weights:
                if name in lora_tensors:
                    raise ValueError(f"Received duplicate LoRA tensor {name!r}")
                # The trainer owns IPC storage; keep adapter-owned CPU tensors.
                owned_tensor = tensor.detach().to("cpu").contiguous().clone()
                lora_tensors[name] = owned_tensor
                received_bytes += owned_tensor.numel() * owned_tensor.element_size()

        with torch.device(self.device):
            self.weight_transfer_engine.receive_weights(
                update_info,
                collect_weights,
            )

        expected_names = set(update_info.names)
        if set(lora_tensors) != expected_names:
            missing = sorted(expected_names - set(lora_tensors))
            unexpected = sorted(set(lora_tensors) - expected_names)
            raise RuntimeError(
                f"Incomplete LoRA transfer: missing={missing}, "
                f"unexpected={unexpected}"
            )

        request = TensorLoRARequest(
            lora_name=lora_name,
            lora_int_id=adapter_id,
            lora_path=f"tensor://{lora_name}",
            peft_config=peft_config,
            lora_tensors=lora_tensors,
        )

        if adapter_id in self.list_loras():
            self.remove_lora(adapter_id)
        if not self.add_lora(request):
            raise RuntimeError(f"Failed to add LoRA adapter id {adapter_id}")
        torch.npu.synchronize()

        return {
            "lora_int_id": adapter_id,
            "tensor_count": len(lora_tensors),
            "received_bytes": received_bytes,
        }


MODEL_NAME = "Qwen/Qwen3-0.6B"
ALICE_LORA = "charent/self_cognition_Alice"
BOB_LORA = "charent/self_cognition_Bob"

LORA_NAME = "rl_adapter"
LORA_INT_ID = 1
INFERENCE_DEVICE_INDEX = 0
LORA_RANK = 8
SELF_COGNITION_PROMPT = "Hi, tell me about you"
EXPECTED_ALICE_NAME = "alice"
EXPECTED_BOB_NAME = "bob"

SERVER_START_TIMEOUT = 2800
CONTROL_TIMEOUT = 300
WORKER_EXTENSION = (
    "tests.e2e.pull_request.two_card.test_rl_lora_ipc_weight_transfer."
    "LoRAIPCWorkerExtension"
)

CONFIG_COMPATIBILITY_FIELDS = (
    "r",
    "lora_alpha",
    "peft_type",
    "task_type",
    "bias",
    "fan_in_fan_out",
    "use_dora",
    "use_rslora",
    "modules_to_save",
    "rank_pattern",
    "alpha_pattern",
)


def _log(message: str) -> None:
    print(f"[trainer] {message}", flush=True)


def _download_lora_adapter(repo_id: str) -> Path:
    """Download a LoRA repository and return its cached snapshot directory."""
    adapter_path = Path(
        snapshot_download(
            repo_id=repo_id,
            local_files_only=huggingface_hub.constants.HF_HUB_OFFLINE,
        )
    )
    required_files = ("adapter_config.json", "adapter_model.safetensors")
    missing_files = [
        file_name
        for file_name in required_files
        if not (adapter_path / file_name).is_file()
    ]
    if missing_files:
        raise FileNotFoundError(
            f"LoRA repository {repo_id!r} is missing required files: "
            f"{missing_files}"
        )
    return adapter_path


def _read_adapter_config(adapter_path: Path) -> dict:
    with open(
        adapter_path / "adapter_config.json",
        encoding="utf-8",
    ) as config_file:
        return json.load(config_file)


def _assert_compatible_configs(alice_config: dict, bob_config: dict) -> None:
    for field_name in CONFIG_COMPATIBILITY_FIELDS:
        assert alice_config.get(field_name) == bob_config.get(field_name), (
            f"Alice and Bob LoRA configurations differ in {field_name!r}: "
            f"{alice_config.get(field_name)!r} != "
            f"{bob_config.get(field_name)!r}"
        )

    assert set(alice_config["target_modules"]) == set(
        bob_config["target_modules"]
    ), "Alice and Bob target different base-model modules"


def _load_trainer_lora(source_adapter: Path) -> dict[str, torch.Tensor]:
    source_weights = source_adapter / "adapter_model.safetensors"
    trainer_device = f"npu:{INFERENCE_DEVICE_INDEX}"

    tensors: dict[str, torch.Tensor] = {}
    with safetensors.safe_open(source_weights, framework="pt") as source_file:
        for tensor_name in source_file.keys():
            tensors[tensor_name] = (
                source_file.get_tensor(tensor_name)
                .to(device=trainer_device)
                .contiguous()
            )
    _log(f"loaded {len(tensors)} live LoRA tensors from {source_adapter.name}")
    return tensors


def _post(
    server: RemoteOpenAIServer,
    route: str,
    *,
    json_body=None,
    timeout=CONTROL_TIMEOUT,
):
    response = requests.post(server.url_for(route), json=json_body, timeout=timeout)
    response.raise_for_status()
    return response


def _serialize_update_info(update_info) -> str:
    update_fields = asdict(update_info)
    ipc_handles = update_fields.pop("ipc_handles")
    update_fields.pop("ipc_handles_pickled", None)
    update_fields["ipc_handles_pickled"] = base64.b64encode(
        pickle.dumps(ipc_handles)
    ).decode("ascii")
    return json.dumps(update_fields)


def _transfer_lora(
    server: RemoteOpenAIServer,
    source_adapter: Path,
    peft_config: dict,
) -> dict:
    from vllm_ascend.distributed.weight_transfer.npu_ipc_engine import (
        NPUIPCTrainerSendWeightsArgs,
        NPUIPCWeightTransferEngine,
    )

    tensors = _load_trainer_lora(source_adapter)
    expected_bytes = sum(
        tensor.numel() * tensor.element_size() for tensor in tensors.values()
    )
    results: list[dict] = []

    def send_update(update_info) -> None:
        response = _post(
            server,
            "collective_rpc",
            json_body={
                "method": "update_weights_from_ipc",
                "timeout": CONTROL_TIMEOUT,
                "kwargs": {
                    "update_info_json": _serialize_update_info(update_info),
                    "peft_config_json": json.dumps(peft_config),
                    "lora_name": LORA_NAME,
                    "lora_int_id": str(LORA_INT_ID),
                },
            },
        )
        results.append(response.json()["results"][0])

    _post(server, "pause")
    NPUIPCWeightTransferEngine.trainer_send_weights(
        iterator=iter(tensors.items()),
        trainer_args=NPUIPCTrainerSendWeightsArgs(
            send_mode=send_update,
            packed=False,
        ),
    )

    assert len(results) == 1
    result = results[0]
    assert result["tensor_count"] == len(tensors)
    assert result["received_bytes"] == expected_bytes

    cache_response = _post(server, "reset_prefix_cache").json()
    assert cache_response["success"]
    _post(server, "resume")

    del tensors
    torch.npu.empty_cache()
    return result


def _generate_identity(client, model_name: str) -> str:
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant.",
            },
            {
                "role": "user",
                "content": SELF_COGNITION_PROMPT,
            },
        ],
        max_tokens=64,
        temperature=0,
        extra_body={
            "chat_template_kwargs": {
                "enable_thinking": False,
            }
        },
    )
    content = response.choices[0].message.content
    assert content is not None
    return content


@pytest.mark.skipif(
    torch.npu.device_count() < 1,
    reason="RL LoRA NPU IPC transfer test requires at least 1 NPU.",
)
def test_rl_lora_update_weights_from_ipc_and_hot_update():
    from tests.e2e.conftest import RemoteOpenAIServer

    alice_lora_path = _download_lora_adapter(ALICE_LORA)
    bob_lora_path = _download_lora_adapter(BOB_LORA)
    alice_config = _read_adapter_config(alice_lora_path)
    bob_config = _read_adapter_config(bob_lora_path)
    _assert_compatible_configs(alice_config, bob_config)
    assert alice_config["r"] == LORA_RANK

    torch.npu.set_device(INFERENCE_DEVICE_INDEX)
    os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
    port = get_open_port()

    lora_module = {
        "name": LORA_NAME,
        "path": str(alice_lora_path),
        "base_model_name": MODEL_NAME,
    }
    server_args = [
        "--enforce-eager",
        "--enable-lora",
        "--lora-modules",
        json.dumps(lora_module),
        "--max-loras",
        "1",
        "--max-cpu-loras",
        "1",
        "--max-lora-rank",
        str(LORA_RANK),
        "--weight-transfer-config",
        '{"backend": "ipc"}',
        "--worker-extension-cls",
        WORKER_EXTENSION,
        "--max-model-len",
        "1024",
        "--gpu-memory-utilization",
        "0.09",
        "--port",
        str(port),
        "--trust-remote-code",
    ]
    env_dict = {
        "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "True",
        "VLLM_ALLOW_INSECURE_SERIALIZATION": "1",
        "VLLM_SERVER_DEV_MODE": "1",
        "VLLM_ASCEND_ENABLE_NZ": "0",
    }

    _log("starting vLLM and trainer on physical NPU 0")
    with RemoteOpenAIServer(
        MODEL_NAME,
        vllm_serve_args=server_args,
        server_host="127.0.0.1",
        server_port=port,
        env_dict=env_dict,
        auto_port=False,
        max_wait_seconds=SERVER_START_TIMEOUT,
    ) as server:
        client = server.get_client()
        _post(server, "init_weight_transfer_engine", json_body={"init_info": {}})

        _log("step 1: transferring live Alice LoRA tensors over NPU IPC")
        alice_result = _transfer_lora(server, alice_lora_path, alice_config)
        alice_output = _generate_identity(client, LORA_NAME)
        _log(f"Alice output: {alice_output!r}")

        _log("step 2: transferring live Bob LoRA tensors over NPU IPC")
        bob_result = _transfer_lora(server, bob_lora_path, bob_config)
        bob_output = _generate_identity(client, LORA_NAME)
        _log(f"Bob output: {bob_output!r}")

        assert bob_result["tensor_count"] == alice_result["tensor_count"]
        assert bob_result["received_bytes"] == alice_result["received_bytes"]

    assert EXPECTED_ALICE_NAME in alice_output.lower(), (
        "The initial adapter did not identify as Alice: "
        f"{alice_output!r}"
    )
    assert EXPECTED_BOB_NAME in bob_output.lower(), (
        "The hot-updated adapter did not identify as Bob: "
        f"{bob_output!r}"
    )
    assert alice_output != bob_output, (
        "Alice and Bob produced identical outputs; the LoRA hot update may "
        "not have taken effect."
    )
