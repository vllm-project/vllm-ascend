# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Model-aware two-card HCCL live weight-update regression tests.

NPU 0 hosts the inference worker and NPU 1 deterministically generates every
parameter of the layer-reduced model. Names and shapes come from a meta-device
HF model; values are fixed-random BF16 and require no checkpoint weights. The
test proves dummy inference in FULL_DECODE_ONLY mode before comparing a complete
baseline with an exact full-payload reload across a pause/resume boundary.
Both unpacked and packed HCCL broadcasts exercise the same transaction.

The trainer side is driven by ``HCCLTrainerWeightTransferEngine``: it opens the
rank-0 HCCL endpoint itself, then ``send_weights()`` owns the whole remote
transaction (init handshake, START -> broadcast -> FINISH), mirroring the
upstream NCCL trainer engine.
"""

import pytest
import requests
import torch
import torch_npu  # noqa: F401  # registers the NPU backend
from vllm.utils.network_utils import get_ip, get_open_port

from tests.e2e.conftest import RemoteOpenAIServer
from tests.e2e.pull_request.rlhf.weight_transfer_test_utils import (
    FixedRandomWeightSource,
    WeightUpdateModelCase,
    assert_dummy_then_fixed_reload,
    generation_signature,
    packed_buffer_size_for,
    pytest_model_cases,
    register_engines_once,
)

INFERENCE_WORLD_SIZE = 1
TRAINER_DEVICE_INDEX = INFERENCE_WORLD_SIZE
CONTROL_TIMEOUT = 60


def _log(message: str) -> None:
    print(f"[trainer] {message}", flush=True)


def _post(server: RemoteOpenAIServer, route: str, *, json=None, timeout=CONTROL_TIMEOUT):
    response = requests.post(server.url_for(route), json=json, timeout=timeout)
    response.raise_for_status()
    return response


@pytest.mark.skipif(
    torch.npu.device_count() < 2,
    reason="HCCL weight transfer e2e test requires at least 2 NPUs.",
)
@pytest.mark.parametrize("case", pytest_model_cases())
@pytest.mark.parametrize("packed", [False, True], ids=["unpacked", "packed"])
def test_hccl_weight_transfer_transaction(case: WeightUpdateModelCase, packed: bool):
    port = get_open_port()
    server_args = [
        "--load-format",
        "dummy",
        "--dtype",
        "bfloat16",
        "--compilation-config",
        '{"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [1]}',
        "--weight-transfer-config",
        '{"backend": "hccl"}',
        "--tensor-parallel-size",
        str(INFERENCE_WORLD_SIZE),
        "--max-model-len",
        "1024",
        "--gpu-memory-utilization",
        "0.75",
        "--port",
        str(port),
        "--trust-remote-code",
        "--additional-config",
        '{"weight_nz_mode": 0}',
        *case.server_args(),
    ]
    env_dict = {
        "VLLM_SERVER_DEV_MODE": "1",
        "ASCEND_RT_VISIBLE_DEVICES": "0",
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

        torch.npu.set_device(TRAINER_DEVICE_INDEX)
        source = FixedRandomWeightSource(case, torch.device("npu", TRAINER_DEVICE_INDEX))

        from vllm.distributed.weight_transfer.clients import HTTPVLLMWeightSyncClient
        from vllm.distributed.weight_transfer.factory import WeightTransferTrainerFactory

        from vllm_ascend.distributed.weight_transfer.hccl_engine import HCCLTrainerInitInfo

        register_engines_once()
        engine = WeightTransferTrainerFactory.trainer_init(
            HCCLTrainerInitInfo(
                rank=0,
                master_address=get_ip(),
                master_port=get_open_port(),
                # The trainer is HCCL rank 0; the single inference worker is
                # rank 1, so the group is the workers plus the sender.
                world_size=INFERENCE_WORLD_SIZE + 1,
                packed=packed,
                packed_buffer_size_bytes=packed_buffer_size_for(source),
            ),
            client=HTTPVLLMWeightSyncClient(base_url=server.url_root),
            source=source,
        )

        signatures = []
        for update_round in range(2):
            _log(
                f"{case.id}: sending {('packed' if packed else 'unpacked')} "
                f"fixed-random update round={update_round + 1}"
            )
            _post(server, "pause")
            # send_weights owns the complete INIT -> START -> LOAD -> FINISH
            # transaction against the inference server.
            engine.send_weights()
            _post(server, "resume")
            signatures.append(generation_signature(client, case.model))

    assert_dummy_then_fixed_reload(dummy_signature, signatures[0], signatures[1], case)
