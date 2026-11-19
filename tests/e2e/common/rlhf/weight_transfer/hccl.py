# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""HCCL transactions with a TP1 rollout server and a separate trainer NPU.

Direct reload alternates two checkpoints and checks each round against cold loading.
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
    generation_signature,
    reference_signature,
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


def run_hccl_weight_transfer_transaction(case: WeightUpdateModelCase, packed: bool, *, reload_mode: str):
    if INFERENCE_DEVICE_INDEX < 0 or torch.npu.device_count() <= TRAINER_DEVICE_INDEX:
        pytest.skip("HCCL weight transfer e2e test requires at least 2 NPUs.")

    torch.npu.set_device(TRAINER_DEVICE_INDEX)
    source = make_weight_source(case, torch.device("npu", TRAINER_DEVICE_INDEX))
    env_dict = {
        "VLLM_SERVER_DEV_MODE": "1",
        "VLLM_RELOAD_MODE": reload_mode,
        "VLLM_USE_V2_MODEL_RUNNER": "1",
        "ASCEND_RT_VISIBLE_DEVICES": server_visible_devices(INFERENCE_DEVICE_INDEX, INFERENCE_WORLD_SIZE),
    }

    server_options = dict(
        tensor_parallel_size=INFERENCE_WORLD_SIZE,
        device_index=INFERENCE_DEVICE_INDEX,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        env_dict=env_dict,
    )
    seed_a = source.seed
    seeds = (seed_a,) * UPDATE_ROUNDS
    references = {}
    if reload_mode == "direct":
        seed_b = seed_a + 1
        seeds = (seed_a, seed_b, seed_a)
        source.seed = seed_b
        references[seed_b] = reference_signature(source, case, port=get_open_port(), **server_options)
        source.seed = seed_a

    with weight_transfer_servers(source, case, backend="hccl", **server_options) as (
        server,
        reference,
        dummy_signature,
    ):
        references[seed_a] = reference
        assert dummy_signature != reference, f"{case.id}: dummy and cold A produced identical signatures"
        if reload_mode == "direct":
            assert reference != references[seed_b], f"{case.id}: cold A and B produced identical signatures"
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
        try:
            for update_round, seed in enumerate(seeds, start=1):
                source.seed = seed
                print(f"[trainer] {case.id}: {reload_mode} update round={update_round} seed={seed}", flush=True)
                post_control(server, "pause")
                engine.send_weights()
                post_control(server, "resume")
                assert generation_signature(client, case.model) == references[seed], (
                    f"{case.id}: {reload_mode} round {update_round} differs from cold loading seed {seed}"
                )
        finally:
            engine.shutdown()
