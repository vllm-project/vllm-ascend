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

import pytest
import torch
import torch_npu  # noqa: F401  # registers the NPU backend
from vllm.utils.network_utils import get_ip, get_open_port

from tests.e2e.conftest import RemoteOpenAIServer
from tests.e2e.pull_request.weight_transfer_utils import (
    INIT_TIMEOUT,
    UPDATE_TIMEOUT,
    BackgroundPost,
    build_trainer_model,
    collect_weight_metadata,
    generate,
    log,
    post,
    start_weight_update_if_available,
)

MODEL_NAME = "Qwen/Qwen3-0.6B"

# Device 0 hosts the inference worker, device 1 hosts the trainer.
INFERENCE_WORLD_SIZE = 1
TRAINER_DEVICE_INDEX = INFERENCE_WORLD_SIZE

PROMPTS = [
    "Hello, my name is",
    "The capital of France is",
]

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

    log(f"starting server on port {port} (device 0, dummy weights) ...")
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
        log("generating baseline outputs (dummy weights) ...")
        outputs_before = generate(client, MODEL_NAME, PROMPTS)
        log(f"outputs BEFORE weight update: {outputs_before}")

        # 2) Build the trainer model on the trainer NPU (download-free by default).
        log(f"preparing trainer model on npu:{TRAINER_DEVICE_INDEX} ...")
        torch.npu.set_device(TRAINER_DEVICE_INDEX)
        train_model = build_trainer_model(MODEL_NAME, TRAINER_DEVICE_INDEX)
        log("trainer model ready")

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
        log(f"HCCL rendezvous at {master_address}:{master_port} (world_size={world_size}) ...")
        init_thread = BackgroundPost(
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
        log("trainer_init returned, waiting for server init RPC ...")
        init_thread.join()
        init_thread.raise_if_failed()
        log("HCCL process group established")

        # 4) Pause generation and start the weight update lifecycle when the
        #    server exposes /start_weight_update. Do not call it again below.
        post(server, "pause")
        use_lifecycle = start_weight_update_if_available(server)
        log(f"paused; lifecycle endpoints available: {use_lifecycle}")

        names, dtype_names, shapes, packed_buffer_size_bytes = collect_weight_metadata(train_model)
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
        log(f"broadcasting {len(names)} tensors via HCCL (packed) ...")
        update_thread = BackgroundPost(
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
        log("trainer finished sending weights, waiting for server update RPC ...")
        update_thread.join()
        update_thread.raise_if_failed()
        log("weight broadcast complete")

        # 5) Finalize the lifecycle and resume generation.
        if use_lifecycle:
            post(server, "finish_weight_update")
        post(server, "resume")

        # 6) Generation after the broadcast weights are loaded.
        outputs_after = generate(client, MODEL_NAME, PROMPTS)
        log(f"outputs AFTER weight update: {outputs_after}")

    # Reaching here means the full HCCL transfer pipeline succeeded: every
    # control-plane RPC raised on a non-2xx response and each background POST
    # re-raised on join(). The broadcast weights differ from the server's dummy
    # init, so the served model must now produce different generations.
    assert outputs_after != outputs_before, "server weights did not change after HCCL transfer"
