# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Model-aware two-card HCCL live weight-update regression tests.

NPU 0 hosts the inference worker and NPU 1 deterministically generates every
parameter of the layer-reduced model. Names and shapes come from a meta-device
HF model; values are fixed-random BF16 and require no checkpoint weights. The
test proves dummy inference in FULL_DECODE_ONLY mode before comparing a complete
baseline with an exact full-payload reload across a pause/resume boundary.
"""

import math
import threading

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
    pytest_model_cases,
)

INFERENCE_WORLD_SIZE = 1
TRAINER_DEVICE_INDEX = INFERENCE_WORLD_SIZE
INIT_TIMEOUT = 120
UPDATE_TIMEOUT = 300
CONTROL_TIMEOUT = 60


def _log(message: str) -> None:
    print(f"[trainer] {message}", flush=True)


def _post(server: RemoteOpenAIServer, route: str, *, json=None, timeout=CONTROL_TIMEOUT):
    response = requests.post(server.url_for(route), json=json, timeout=timeout)
    response.raise_for_status()
    return response


class _BackgroundPost(threading.Thread):
    """Run a blocking server-side HCCL RPC and surface its exception."""

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
        except BaseException as exc:  # noqa: BLE001 - re-raised by raise_if_failed
            self.error = exc

    def raise_if_failed(self) -> None:
        if self.error is not None:
            raise RuntimeError(f"server-side /{self._route} failed") from self.error


def _collect_weight_metadata(source: FixedRandomWeightSource):
    metadata = source.metadata()
    max_tensor_bytes = max(math.prod(meta.shape) * meta.dtype.itemsize for meta in metadata)
    return (
        [meta.name for meta in metadata],
        [str(meta.dtype).split(".")[-1] for meta in metadata],
        [list(meta.shape) for meta in metadata],
        max(max_tensor_bytes + 128 * 2**20, 2**30),
    )


def _send_update(server, source, model_update_group) -> None:
    from vllm_ascend.distributed.weight_transfer.hccl_engine import (
        HCCLTrainerSendWeightsArgs,
        HCCLWeightTransferEngine,
    )

    names, dtype_names, shapes, packed_buffer_size_bytes = _collect_weight_metadata(source)
    _post(server, "pause")
    _post(server, "start_weight_update")

    update_thread = _BackgroundPost(
        server,
        "update_weights",
        json={
            "update_info": {
                "names": names,
                "dtype_names": dtype_names,
                "shapes": shapes,
                "packed": True,
                "packed_buffer_size_bytes": packed_buffer_size_bytes,
            }
        },
        timeout=UPDATE_TIMEOUT,
    )
    update_thread.start()
    HCCLWeightTransferEngine.trainer_send_weights(
        iterator=iter(source),
        trainer_args=HCCLTrainerSendWeightsArgs(
            group=model_update_group,
            packed=True,
            packed_buffer_size_bytes=packed_buffer_size_bytes,
        ),
    )
    update_thread.join()
    update_thread.raise_if_failed()
    _post(server, "finish_weight_update")
    _post(server, "resume")


@pytest.mark.skipif(
    torch.npu.device_count() < 2,
    reason="HCCL weight transfer e2e test requires at least 2 NPUs.",
)
@pytest.mark.parametrize("case", pytest_model_cases())
def test_hccl_weight_transfer_transaction(case: WeightUpdateModelCase):
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

        from vllm_ascend.distributed.weight_transfer.hccl_engine import HCCLWeightTransferEngine

        master_address = get_ip()
        master_port = get_open_port()
        world_size = INFERENCE_WORLD_SIZE + 1
        init_thread = _BackgroundPost(
            server,
            "init_weight_transfer_engine",
            json={
                "init_info": {
                    "master_address": master_address,
                    "master_port": master_port,
                    "rank_offset": 1,
                    "world_size": world_size,
                }
            },
            timeout=INIT_TIMEOUT,
        )
        init_thread.start()
        model_update_group = HCCLWeightTransferEngine.trainer_init(
            {
                "master_address": master_address,
                "master_port": master_port,
                "world_size": world_size,
            }
        )
        init_thread.join()
        init_thread.raise_if_failed()

        signatures = []
        for update_round in range(2):
            _log(f"{case.id}: sending complete fixed-random update round={update_round + 1}")
            _send_update(server, source, model_update_group)
            signatures.append(generation_signature(client, case.model))

    assert_dummy_then_fixed_reload(dummy_signature, signatures[0], signatures[1], case)
