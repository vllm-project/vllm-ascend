# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""HCCL transactions with a TP1 rollout server and a separate trainer NPU.

The trainer broadcasts the checkpoint in two pause/update/resume rounds. The
shared server lifecycle supplies an independent startup-load reference and the
dummy pre-update signature. IPC's same-device sleep/wake protocol is not used.
"""

import os

import pytest
import torch
import torch_npu  # noqa: F401  # registers the NPU backend
from vllm.distributed.weight_transfer.clients import HTTPVLLMWeightSyncClient
from vllm.distributed.weight_transfer.factory import WeightTransferTrainerFactory
from vllm.utils.network_utils import get_ip, get_open_port

from tests.e2e.common.rlhf.common import post_control, register_engines_once, server_visible_devices
from tests.e2e.common.rlhf.weight_transfer.case import WeightUpdateModelCase
from tests.e2e.common.rlhf.weight_transfer.reference_check import (
    UPDATE_ROUNDS,
    assert_weight_update_matches_reference,
    generation_signature,
    weight_transfer_servers,
)
from tests.e2e.common.rlhf.weight_transfer.weight_source import (
    make_weight_source,
    packed_buffer_size_for,
)

UPDATE_TIMEOUT = 300

INFERENCE_WORLD_SIZE = 1
# Parent logical devices: inference on the first, trainer on the next.
# Child-server physical visibility is resolved from the parent environment.
INFERENCE_DEVICE_INDEX = int(os.environ.get("VLLM_RL_TEST_DEVICE_INDEX", "0"))
TRAINER_DEVICE_INDEX = INFERENCE_DEVICE_INDEX + 1
GPU_MEMORY_UTILIZATION = 0.75


def run_hccl_weight_transfer_transaction(case: WeightUpdateModelCase, packed: bool):
    if INFERENCE_DEVICE_INDEX < 0 or torch.npu.device_count() <= TRAINER_DEVICE_INDEX:
        pytest.skip("HCCL weight transfer e2e test requires at least 2 NPUs.")

    torch.npu.set_device(TRAINER_DEVICE_INDEX)
    source = make_weight_source(case, torch.device("npu", TRAINER_DEVICE_INDEX))
    env_dict = {
        "VLLM_SERVER_DEV_MODE": "1",
        "ASCEND_RT_VISIBLE_DEVICES": server_visible_devices(INFERENCE_DEVICE_INDEX, INFERENCE_WORLD_SIZE),
    }

    with weight_transfer_servers(
        source,
        case,
        backend="hccl",
        tensor_parallel_size=INFERENCE_WORLD_SIZE,
        device_index=INFERENCE_DEVICE_INDEX,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        env_dict=env_dict,
    ) as (server, reference, dummy_signature):
        client = server.get_client()

        from vllm_ascend.distributed.weight_transfer.hccl_engine import HCCLTrainerInitInfo

        register_engines_once()
        engine = WeightTransferTrainerFactory.trainer_init(
            HCCLTrainerInitInfo(
                rank=0,
                master_address=get_ip(),
                master_port=get_open_port(),
                world_size=INFERENCE_WORLD_SIZE + 1,
                packed=packed,
                packed_buffer_size_bytes=packed_buffer_size_for(source),
            ),
            client=HTTPVLLMWeightSyncClient(base_url=server.url_root, timeout=UPDATE_TIMEOUT),
            source=source,
        )
        signatures = []
        try:
            for update_round in range(UPDATE_ROUNDS):
                print(
                    f"[trainer] {case.id}: sending {('packed' if packed else 'unpacked')} "
                    f"fixed-random update round={update_round + 1}",
                    flush=True,
                )
                post_control(server, "pause")
                try:
                    engine.send_weights()
                finally:
                    post_control(server, "resume")
                signatures.append(generation_signature(client, case.model))
        finally:
            engine.shutdown()

    assert_weight_update_matches_reference(
        dummy_signature,
        reference,
        signatures,
        case,
    )
