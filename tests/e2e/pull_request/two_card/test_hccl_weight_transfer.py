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
"""End-to-end test for the HCCL weight transfer engine.

This test starts a vLLM server with dummy weights and the HCCL weight transfer
backend enabled, then runs the trainer side of an RLHF-style weight sync from a
separate NPU. It exercises the full control plane (HTTP) + data plane (HCCL
packed broadcast + layerwise reload) and asserts the server's weights actually
change after the broadcast.

To keep the test self-contained and download-free, the trainer model is built
from the architecture config with random weights (only the tiny config/tokenizer
are needed, which the server already fetches). The parameter names/shapes/dtypes
match the real checkpoint, so the broadcast pipeline is fully exercised; we just
don't assert "coherent text" since the broadcast weights are random. Set
``WEIGHT_TRANSFER_TEST_MODEL=/path/to/checkpoint`` to instead broadcast real
weights from a local checkpoint.

Topology (requires 2 NPUs):
- NPU 0: vLLM inference worker (rank 1 in the HCCL group)
- NPU 1: trainer / weight source (rank 0 in the HCCL group)

Refer to ``examples/rl/rlhf_http_hccl.py`` for the end-user workflow.

Run with::

    pytest tests/e2e/multicard/2-cards/test_weight_transfer_hccl.py
"""

import os
import threading

import pytest
import requests
import torch
import torch_npu  # noqa: F401  # registers the NPU backend
from transformers import AutoConfig, AutoModelForCausalLM
from vllm.utils.network_utils import get_ip, get_open_port

from tests.e2e.conftest import RemoteOpenAIServer

MODEL_NAME = "Qwen/Qwen3-0.6B"

# Device 0 hosts the inference worker, device 1 hosts the trainer.
INFERENCE_WORLD_SIZE = 1
TRAINER_DEVICE_INDEX = INFERENCE_WORLD_SIZE

PROMPTS = [
    "Hello, my name is",
    "The capital of France is",
]

# HTTP timeouts (seconds). Weight broadcast can take a while for large models.
INIT_TIMEOUT = 120
UPDATE_TIMEOUT = 300
CONTROL_TIMEOUT = 60


def _log(message: str) -> None:
    """Flushed log so step markers show up immediately even when stdout is piped."""
    print(f"[trainer] {message}", flush=True)


def _build_trainer_model(device_index: int):
    """Build the trainer-side model without downloading the checkpoint weights.

    By default the model is instantiated from the architecture config with random
    weights (no ``model.safetensors`` download required); only the tiny config is
    read, which the server already fetches. Its ``named_parameters`` carry the
    same names/shapes/dtypes as the real checkpoint, so the HCCL broadcast +
    layerwise reload path is exercised exactly as with real weights.

    Set ``WEIGHT_TRANSFER_TEST_MODEL=/path/to/checkpoint`` to broadcast real
    weights from a local directory instead.
    """
    device = f"npu:{device_index}"
    override_path = os.getenv("WEIGHT_TRANSFER_TEST_MODEL")
    if override_path:
        _log(f"loading real trainer weights from {override_path}")
        model = AutoModelForCausalLM.from_pretrained(override_path, dtype=torch.bfloat16)
    else:
        _log("building trainer model from config with random weights (download-free)")
        config = AutoConfig.from_pretrained(MODEL_NAME, trust_remote_code=True)
        model = AutoModelForCausalLM.from_config(config)
    model = model.to(device=device, dtype=torch.bfloat16)
    return model


def _post(server: RemoteOpenAIServer, route: str, *, json=None, timeout=CONTROL_TIMEOUT):
    response = requests.post(server.url_for(route), json=json, timeout=timeout)
    response.raise_for_status()
    return response


class _BackgroundPost(threading.Thread):
    """Run an HTTP POST in a thread while keeping its exception visible.

    The trainer side blocks on collective HCCL ops, so the matching server-side
    RPC must run concurrently. If that RPC fails, swallowing the exception would
    deadlock the trainer forever; instead we record it and surface it on join().
    """

    def __init__(self, server: RemoteOpenAIServer, route: str, *, json=None, timeout=CONTROL_TIMEOUT):
        super().__init__(daemon=True)
        self._server = server
        self._route = route
        self._json = json
        self._timeout = timeout
        self.error: BaseException | None = None

    def run(self) -> None:
        try:
            _post(self._server, self._route, json=self._json, timeout=self._timeout)
            _log(f"background POST /{self._route} done")
        except BaseException as exc:  # noqa: BLE001 - re-raised on join via raise_if_failed
            self.error = exc
            _log(f"background POST /{self._route} FAILED: {exc!r}")

    def raise_if_failed(self) -> None:
        if self.error is not None:
            raise RuntimeError(f"server-side /{self._route} failed") from self.error


def _generate(client, model, prompts):
    completions = []
    for prompt in prompts:
        response = client.completions.create(
            model=model,
            prompt=prompt,
            max_tokens=16,
            temperature=0,
        )
        completions.append(response.choices[0].text)
    return completions


def _collect_weight_metadata(train_model):
    """Collect parameter metadata and size the packed buffer for broadcasting."""
    names: list[str] = []
    dtype_names: list[str] = []
    shapes: list[list[int]] = []
    max_tensor_bytes = 0
    for name, parameter in train_model.named_parameters():
        names.append(name)
        dtype_names.append(str(parameter.dtype).split(".")[-1])
        shapes.append(list(parameter.shape))
        tensor_bytes = parameter.numel() * parameter.element_size()
        max_tensor_bytes = max(max_tensor_bytes, tensor_bytes)

    # Keep the 1 GiB default unless a single tensor needs more (+128 MiB headroom).
    packed_buffer_size_bytes = max(max_tensor_bytes + 128 * 2**20, 2**30)
    return names, dtype_names, shapes, packed_buffer_size_bytes


def _has_lifecycle_endpoints(server: RemoteOpenAIServer) -> bool:
    """Detect whether the server exposes the vLLM-main start/finish endpoints.

    On vLLM main, ``/start_weight_update`` and ``/finish_weight_update`` drive
    the layerwise reload lifecycle. On v0.20.2 these endpoints do not exist and
    ``update_weights`` is self-contained, so a probe returns 404.
    """
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


@pytest.mark.e2e_features("multimodal", "eager_mode", "hccl_weight_transfer")
@pytest.mark.e2e_model("Qwen/Qwen3-0.6B")
@pytest.mark.skipif(
    torch.npu.device_count() < 2,
    reason="HCCL weight transfer e2e test requires at least 2 NPUs.",
)
def test_hccl_weight_transfer_updates_server_weights():
    port = get_open_port()
    server_args = [
        "--enforce-eager",
        "--load-format",
        "dummy",
        "--weight-transfer-config",
        '{"backend": "nccl"}',
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
    # The dev-mode endpoints (/init_weight_transfer_engine, /update_weights,
    # /pause, /resume, ...) are only registered when VLLM_SERVER_DEV_MODE=1.
    # Pin the server to NPU 0 so the trainer can own NPU 1 exclusively.
    env_dict = {
        "VLLM_SERVER_DEV_MODE": "1",
        "ASCEND_RT_VISIBLE_DEVICES": "0",
        "VLLM_ASCEND_ENABLE_NZ": "0",
    }

    _log(f"starting server on port {port} (device 0, dummy weights) ...")
    with RemoteOpenAIServer(
        MODEL_NAME,
        vllm_serve_args=server_args,
        # Health check, OpenAI client and control-plane requests all target this
        # host; use loopback explicitly so they reach the local server directly.
        server_host="127.0.0.1",
        server_port=port,
        env_dict=env_dict,
        auto_port=False,
    ) as server:
        client = server.get_client()

        # 1) Baseline generation with dummy weights (expected to be nonsense).
        _log("generating baseline outputs (dummy weights) ...")
        outputs_before = _generate(client, MODEL_NAME, PROMPTS)
        _log(f"outputs BEFORE weight update: {outputs_before}")

        # 2) Build the trainer model on the trainer NPU (download-free by default).
        _log(f"preparing trainer model on npu:{TRAINER_DEVICE_INDEX} ...")
        torch.npu.set_device(TRAINER_DEVICE_INDEX)
        train_model = _build_trainer_model(TRAINER_DEVICE_INDEX)
        _log("trainer model ready")

        # Import after the server is up so the HCCL engine plugin is registered.
        from vllm_ascend.distributed.weight_transfer.hccl_engine import (
            HCCLTrainerSendWeightsArgs,
            HCCLWeightTransferEngine,
        )

        master_address = get_ip()
        master_port = get_open_port()
        rank_offset = 1
        world_size = INFERENCE_WORLD_SIZE + 1  # workers + trainer

        # 3) Build the HCCL process group on both sides. The server side blocks
        #    until the trainer connects, so kick it off in a background thread.
        init_info = dict(
            master_address=master_address,
            master_port=master_port,
            rank_offset=rank_offset,
            world_size=world_size,
        )
        _log(f"HCCL rendezvous at {master_address}:{master_port} (world_size={world_size}) ...")
        init_thread = _BackgroundPost(
            server,
            "init_weight_transfer_engine",
            json={"init_info": init_info},
            timeout=INIT_TIMEOUT,
        )
        init_thread.start()
        model_update_group = HCCLWeightTransferEngine.trainer_init(
            dict(
                master_address=master_address,
                master_port=master_port,
                world_size=world_size,
            ),
        )
        _log("trainer_init returned, waiting for server init RPC ...")
        init_thread.join()
        init_thread.raise_if_failed()
        _log("HCCL process group established")

        # 4) Pause generation and start the weight update lifecycle. On vLLM
        #    main this probe also performs the actual /start_weight_update call,
        #    so we must not call it again below.
        _post(server, "pause")
        use_lifecycle = _has_lifecycle_endpoints(server)
        _log(f"paused; lifecycle endpoints available: {use_lifecycle}")

        names, dtype_names, shapes, packed_buffer_size_bytes = _collect_weight_metadata(train_model)
        update_info = dict(
            names=names,
            dtype_names=dtype_names,
            shapes=shapes,
            packed=True,
            packed_buffer_size_bytes=packed_buffer_size_bytes,
        )
        if not use_lifecycle:
            # v0.20.2 folds the layerwise reload lifecycle into update_weights.
            update_info["is_checkpoint_format"] = True

        # update_weights blocks on the server while it waits for HCCL broadcasts,
        # so run it in a thread while the trainer produces the data.
        _log(f"broadcasting {len(names)} tensors via HCCL (packed) ...")
        update_thread = _BackgroundPost(
            server,
            "update_weights",
            json={"update_info": update_info},
            timeout=UPDATE_TIMEOUT,
        )
        update_thread.start()

        trainer_args = HCCLTrainerSendWeightsArgs(
            group=model_update_group,
            packed=True,
            packed_buffer_size_bytes=packed_buffer_size_bytes,
        )
        HCCLWeightTransferEngine.trainer_send_weights(
            iterator=train_model.named_parameters(),
            trainer_args=trainer_args,
        )
        _log("trainer finished sending weights, waiting for server update RPC ...")
        update_thread.join()
        update_thread.raise_if_failed()
        _log("weight broadcast complete")

        # 5) Finalize the lifecycle and resume generation.
        if use_lifecycle:
            _post(server, "finish_weight_update")
        _post(server, "resume")

        # 6) Generation after the broadcast weights are loaded.
        outputs_after = _generate(client, MODEL_NAME, PROMPTS)
        _log(f"outputs AFTER weight update: {outputs_after}")

    # Reaching here means the full HCCL transfer pipeline succeeded: every
    # control-plane RPC raised on a non-2xx response and each background POST
    # re-raised on join(). The broadcast weights differ from the server's dummy
    # init, so the served model must now produce different generations.
    assert outputs_after != outputs_before, "server weights did not change after HCCL transfer"
