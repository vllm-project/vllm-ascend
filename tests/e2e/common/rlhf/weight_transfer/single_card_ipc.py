# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Same-chip IPC full updates using the shared startup oracle and server lifecycle."""

import os

import torch
from vllm.distributed.weight_transfer.clients import HTTPVLLMWeightSyncClient
from vllm.distributed.weight_transfer.factory import WeightTransferTrainerFactory

from tests.e2e.common.rlhf.common import post_control, register_engines_once, server_visible_devices
from tests.e2e.common.rlhf.weight_transfer.reference_check import (
    UPDATE_ROUNDS,
    assert_weight_update_matches_reference,
    generation_signature,
    weight_transfer_servers,
)
from tests.e2e.common.rlhf.weight_transfer.weight_source import FixedRandomWeightSource, packed_buffer_size_for
from vllm_ascend.distributed.weight_transfer.npu_ipc_engine import NPUIPCTrainerInitInfo


def run_single_card_ipc(case, packed, *, device_index, gpu_memory_utilization):
    torch.npu.set_device(device_index)
    source = FixedRandomWeightSource(case, torch.device("npu", device_index))
    os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
    env = {
        "VLLM_SERVER_DEV_MODE": "1",
        "VLLM_ALLOW_INSECURE_SERIALIZATION": "1",
        "ASCEND_RT_VISIBLE_DEVICES": server_visible_devices(device_index, 1),
    }
    with weight_transfer_servers(
        source,
        case,
        backend="npu_ipc",
        tensor_parallel_size=1,
        device_index=device_index,
        gpu_memory_utilization=gpu_memory_utilization,
        env_dict=env,
    ) as (server, reference, dummy):
        register_engines_once()
        engine = WeightTransferTrainerFactory.trainer_init(
            NPUIPCTrainerInitInfo(rank=0, packed=packed, packed_buffer_size_bytes=packed_buffer_size_for(source)),
            client=HTTPVLLMWeightSyncClient(base_url=server.url_root),
            source=source,
        )
        signatures = []
        try:
            for _ in range(UPDATE_ROUNDS):
                post_control(server, "pause")
                post_control(server, "sleep", params={"level": 2})
                post_control(server, "wake_up", params={"tags": ["weights"]})
                engine.send_weights()
                torch.npu.empty_cache()
                post_control(server, "wake_up", params={"tags": ["kv_cache"]})
                signatures.append(generation_signature(server.get_client(), case.model, image_prompt=case.image_prompt))
        finally:
            engine.shutdown()
    assert_weight_update_matches_reference(dummy, reference, signatures, case)
