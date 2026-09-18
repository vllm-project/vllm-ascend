# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Model-aware two-card HCCL live weight-update regression tests.

NPU 0 hosts the inference worker and NPU 1 deterministically generates every
parameter of the layer-reduced model. Names and shapes come from a meta-device
HF model; values are fixed-random BF16 and require no checkpoint weights.

The correctness oracle is *normal startup loading of that same payload*: the
generator also writes a temporary checkpoint, a reference server loads it at
startup, and the live-update lane has to reproduce its signature exactly —

    normal startup load(W) == first live update(W) == second live update(W)

The dummy-started lane's pre-update signature must differ from the reference, so
a transfer that never ran cannot pass, and the second update covers the layerwise
reload lifecycle, runtime/destructive representations, derived state and the
graph-captured storage of the first update. Both packed modes exercise the same
transaction.

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
    assert_weight_update_matches_reference,
    generation_signature,
    live_update_serve_args,
    packed_buffer_size_for,
    pytest_model_cases,
    reference_signature,
    register_engines_once,
    wait_for_free_device_memory,
)

INFERENCE_WORLD_SIZE = 1
TRAINER_DEVICE_INDEX = INFERENCE_WORLD_SIZE
INFERENCE_DEVICE_INDEX = 0
CONTROL_TIMEOUT = 60
GPU_MEMORY_UTILIZATION = 0.75


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
    torch.npu.set_device(TRAINER_DEVICE_INDEX)
    source = FixedRandomWeightSource(case, torch.device("npu", TRAINER_DEVICE_INDEX))
    env_dict = {
        "VLLM_SERVER_DEV_MODE": "1",
        "ASCEND_RT_VISIBLE_DEVICES": "0",
    }

    # Independent oracle first: a server that loads exactly this payload at
    # startup, on its own lifecycle, before the live path touches anything.
    reference = reference_signature(
        source,
        case,
        port=get_open_port(),
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        tensor_parallel_size=INFERENCE_WORLD_SIZE,
        device_index=INFERENCE_DEVICE_INDEX,
        env_dict=env_dict,
    )

    # Dummy sanity, first live update and the reload regression share one lane.
    # Its pre-update signature is the dummy-sanity check; the *reference* is the
    # part that must come from a separate server. The reference and the lane share
    # one card, so wait for its process tree to hand the HBM back.
    wait_for_free_device_memory(INFERENCE_DEVICE_INDEX, GPU_MEMORY_UTILIZATION)
    port = get_open_port()
    with RemoteOpenAIServer(
        case.model,
        vllm_serve_args=live_update_serve_args(
            case,
            backend="hccl",
            port=port,
            gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
            tensor_parallel_size=INFERENCE_WORLD_SIZE,
        ),
        server_host="127.0.0.1",
        server_port=port,
        env_dict=env_dict,
        auto_port=False,
    ) as server:
        client = server.get_client()
        dummy_signature = generation_signature(client, case.model)

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

    updated_signature, reloaded_signature = signatures
    assert_weight_update_matches_reference(
        dummy_signature,
        reference,
        updated_signature,
        reloaded_signature,
        case,
    )
