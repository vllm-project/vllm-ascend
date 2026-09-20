# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Model-aware end-to-end tests for NPU IPC live weight updates.

Qwen3-0.6B cannot expose missing START/FINISH hooks because its checkpoint and
runtime weight representations are compatible. This matrix instead targets
architectures whose correctness depends on the transaction: fused-MoE layout
restoration, derived FP32 routing weights, and SFA source/derived-state
restoration. Each reduced model first proves dummy inference in FULL_DECODE_ONLY
mode, then loads a complete fixed-random BF16 parameter set and compares that
baseline with an exact full-payload reload across a pause/resume boundary.
"""

import os

import pytest
import requests
import torch
import torch_npu  # noqa: F401  # registers the NPU backend

from tests.e2e.conftest import RemoteOpenAIServer
from tests.e2e.pull_request.rlhf.weight_transfer_test_utils import (
    FixedRandomWeightSource,
    WeightUpdateModelCase,
    assert_dummy_then_fixed_reload,
    generation_signature,
    packed_buffer_size_for,
    pytest_model_cases,
)

INFERENCE_DEVICE_INDEX = 0
CONTROL_TIMEOUT = 60


def _post(server: RemoteOpenAIServer, route: str, *, json=None, timeout=CONTROL_TIMEOUT):
    response = requests.post(server.url_for(route), json=json, timeout=timeout)
    response.raise_for_status()
    return response


_ENGINES_REGISTERED = False


def _register_engines_once() -> None:
    """Register the Ascend weight transfer engines exactly once per process.

    ``register_engine()`` is not idempotent: it registers ``hccl`` and
    ``npu_ipc`` and the underlying factory raises ``ValueError: Weight transfer
    engine 'hccl' is already registered`` on a second call. Registering inside
    the test body therefore fails every parametrisation after the first.
    """
    global _ENGINES_REGISTERED
    if _ENGINES_REGISTERED:
        return
    from vllm_ascend.distributed.weight_transfer import register_engine

    register_engine()
    _ENGINES_REGISTERED = True


@pytest.mark.skipif(
    torch.npu.device_count() < 1,
    reason="NPU IPC weight transfer e2e test requires at least 1 NPU.",
)
@pytest.mark.parametrize("case", pytest_model_cases())
@pytest.mark.parametrize("packed", [False, True], ids=["unpacked", "packed"])
def test_npu_ipc_weight_transfer_transaction(case: WeightUpdateModelCase, packed: bool):
    from vllm.utils.network_utils import get_open_port

    port = get_open_port()
    server_args = [
        "--load-format",
        "dummy",
        "--dtype",
        "bfloat16",
        "--compilation-config",
        '{"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [1]}',
        "--weight-transfer-config",
        '{"backend": "npu_ipc"}',
        "--max-model-len",
        "1024",
        "--gpu-memory-utilization",
        "0.45",
        "--port",
        str(port),
        "--trust-remote-code",
        "--additional-config",
        '{"weight_nz_mode": 0}',
        *case.server_args(),
    ]
    env_dict = {
        "VLLM_SERVER_DEV_MODE": "1",
        "VLLM_ALLOW_INSECURE_SERIALIZATION": "1",
        "ASCEND_RT_VISIBLE_DEVICES": str(INFERENCE_DEVICE_INDEX),
    }

    with RemoteOpenAIServer(
        case.model,
        vllm_serve_args=server_args,
        server_host="127.0.0.1",
        server_port=port,
        env_dict=env_dict,
        auto_port=False,
    ) as server:
        client = server.get_client()
        dummy_signature = generation_signature(client, case.model)

        torch.npu.set_device(INFERENCE_DEVICE_INDEX)
        source = FixedRandomWeightSource(case, torch.device("npu", INFERENCE_DEVICE_INDEX))
        os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"

        from vllm.distributed.weight_transfer.clients import HTTPVLLMWeightSyncClient
        from vllm.distributed.weight_transfer.factory import WeightTransferTrainerFactory

        from vllm_ascend.distributed.weight_transfer.npu_ipc_engine import NPUIPCTrainerInitInfo

        _register_engines_once()
        engine = WeightTransferTrainerFactory.trainer_init(
            NPUIPCTrainerInitInfo(
                rank=0,
                packed=packed,
                packed_buffer_size_bytes=packed_buffer_size_for(source),
            ),
            client=HTTPVLLMWeightSyncClient(base_url=server.url_root),
            source=source,
        )

        signatures = []
        for _ in range(2):
            _post(server, "pause")
            # send_weights owns the complete START -> LOAD -> FINISH transaction.
            engine.send_weights()
            _post(server, "resume")
            signatures.append(generation_signature(client, case.model))

    assert_dummy_then_fixed_reload(dummy_signature, signatures[0], signatures[1], case)
