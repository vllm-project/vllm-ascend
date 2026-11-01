# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""TP2 IPC transactions with one trainer and rollout worker per NPU.

Each trainer generates the full checkpoint locally; Gloo merges both devices'
IPC handles. Before each update the server sleeps at level 2, wakes its weights,
receives the update, and wakes its KV cache. The shared server lifecycle supplies
an independent startup-load reference and the dummy pre-update signature.
"""

import os
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch_npu  # noqa: F401  # registers the NPU backend
from vllm.utils.network_utils import get_open_port

from tests.e2e.pull_request.rlhf.weight_transfer.models.base import WeightUpdateModelCase
from tests.e2e.pull_request.rlhf.weight_transfer.runners.common import (
    CONTROL_TIMEOUT,
    UPDATE_ROUNDS,
    UPDATE_TIMEOUT,
    post_control,
    weight_transfer_servers,
)
from tests.e2e.pull_request.rlhf.weight_transfer.utils import (
    PACKED_LANE_SKIP_REASON,
    FixedRandomWeightSource,
    assert_weight_update_matches_reference,
    generation_signature,
    packed_buffer_size_for,
    register_engines_once,
)

INFERENCE_WORLD_SIZE = 2
# The trainer is co-located with the server on the same chip, so both models
# have to fit; keep the budget low enough to leave room for the payload.
GPU_MEMORY_UTILIZATION = 0.45
# First of the two consecutive inference/trainer cards, as an absolute
# device index inside the container: the harness hands the server
# `ASCEND_RT_VISIBLE_DEVICES` explicitly, so this is a physical chip. The default
# is the first card CI uses; a shared host can select an idle card pair by
# exporting `VLLM_RL_TEST_DEVICE_INDEX` (leaving `ASCEND_RT_VISIBLE_DEVICES`
# unset, so the pytest process's logical indices equal the physical ones).
INFERENCE_DEVICE_INDEX = int(os.environ.get("VLLM_RL_TEST_DEVICE_INDEX", "0"))


def _ipc_trainer_peer(_local_rank: int, case: WeightUpdateModelCase, packed: bool, server_url: str, rendezvous: str):
    from vllm.distributed.weight_transfer.clients import HTTPVLLMWeightSyncClient
    from vllm.distributed.weight_transfer.factory import WeightTransferTrainerFactory

    from vllm_ascend.distributed.weight_transfer.npu_ipc_engine import NPUIPCTrainerInitInfo

    torch.npu.set_device(INFERENCE_DEVICE_INDEX + 1)
    source = FixedRandomWeightSource(case, torch.device("npu", INFERENCE_DEVICE_INDEX + 1))
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=1,
        world_size=INFERENCE_WORLD_SIZE,
        timeout=timedelta(seconds=UPDATE_TIMEOUT),
    )
    try:
        register_engines_once()
        engine = WeightTransferTrainerFactory.trainer_init(
            NPUIPCTrainerInitInfo(
                rank=1,
                packed=packed,
                packed_buffer_size_bytes=packed_buffer_size_for(source),
            ),
            client=HTTPVLLMWeightSyncClient(base_url=server_url, timeout=UPDATE_TIMEOUT),
            source=source,
        )
        try:
            for _ in range(UPDATE_ROUNDS):
                # Rank 0 releases this barrier after both rollout workers wake.
                dist.barrier()
                engine.send_weights()
        finally:
            engine.shutdown()
    finally:
        dist.destroy_process_group()


def run_npu_ipc_weight_transfer_transaction(case: WeightUpdateModelCase, packed: bool):
    if torch.npu.device_count() < INFERENCE_DEVICE_INDEX + INFERENCE_WORLD_SIZE:
        pytest.skip("NPU IPC weight transfer e2e test requires 2 consecutive NPUs for TP=2.")
    if case.skip_reason is not None:
        pytest.skip(f"{case.id}: {case.skip_reason}")
    if packed:
        pytest.skip(f"{case.id}: {PACKED_LANE_SKIP_REASON}")

    torch.npu.set_device(INFERENCE_DEVICE_INDEX)
    source = FixedRandomWeightSource(case, torch.device("npu", INFERENCE_DEVICE_INDEX))
    os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
    env_dict = {
        "VLLM_SERVER_DEV_MODE": "1",
        "VLLM_ALLOW_INSECURE_SERIALIZATION": "1",
        "ASCEND_RT_VISIBLE_DEVICES": ",".join(
            str(INFERENCE_DEVICE_INDEX + rank) for rank in range(INFERENCE_WORLD_SIZE)
        ),
    }

    with weight_transfer_servers(
        source,
        case,
        backend="npu_ipc",
        tensor_parallel_size=INFERENCE_WORLD_SIZE,
        device_index=INFERENCE_DEVICE_INDEX,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        env_dict=env_dict,
    ) as (server, reference, dummy_signature):
        client = server.get_client()

        from vllm.distributed.weight_transfer.clients import HTTPVLLMWeightSyncClient
        from vllm.distributed.weight_transfer.factory import WeightTransferTrainerFactory

        from vllm_ascend.distributed.weight_transfer.npu_ipc_engine import NPUIPCTrainerInitInfo

        rendezvous = f"tcp://127.0.0.1:{get_open_port()}"
        peer = mp.spawn(
            _ipc_trainer_peer,
            args=(case, packed, server.url_root, rendezvous),
            nprocs=1,
            join=False,
        )
        engine = None
        try:
            dist.init_process_group(
                "gloo",
                init_method=rendezvous,
                rank=0,
                world_size=INFERENCE_WORLD_SIZE,
                timeout=timedelta(seconds=UPDATE_TIMEOUT),
            )
            register_engines_once()
            engine = WeightTransferTrainerFactory.trainer_init(
                NPUIPCTrainerInitInfo(
                    rank=0,
                    packed=packed,
                    packed_buffer_size_bytes=packed_buffer_size_for(source),
                ),
                client=HTTPVLLMWeightSyncClient(base_url=server.url_root, timeout=UPDATE_TIMEOUT),
                source=source,
            )

            signatures = []
            for _ in range(UPDATE_ROUNDS):
                post_control(server, "pause")
                # Same-chip RL: hand the engine's HBM to the co-located trainer
                # before generating the payload. Level 2 discards the weights and the
                # KV cache outright; `/sleep` reads its level from the query string,
                # a JSON body is ignored.
                post_control(server, "sleep", params={"level": 2})
                # Level 2 unmapped the weights pool and `send_weights`' layerwise
                # reload copies the payload back into exactly those buffers, so the
                # weights allocation has to be mapped again before the transaction.
                # The KV cache stays asleep until the update is finished.
                post_control(server, "wake_up", params={"tags": ["weights"]})
                # send_weights owns the complete START -> LOAD -> FINISH transaction.
                dist.barrier()
                engine.send_weights()
                # The KV cache is re-allocated last, from the updated weights' state.
                # That wake leaves nothing asleep, so the engine resumes scheduling on
                # its own and no explicit `/resume` is needed.
                post_control(server, "wake_up", params={"tags": ["kv_cache"]})
                signatures.append(generation_signature(client, case.model))
            if not peer.join(timeout=CONTROL_TIMEOUT):
                raise TimeoutError("IPC trainer peer did not exit after the two updates")
        finally:
            if engine is not None:
                engine.shutdown()
            for process in peer.processes:
                if process.is_alive():
                    process.terminate()
                process.join(timeout=CONTROL_TIMEOUT)
            if dist.is_initialized():
                dist.destroy_process_group()

    updated_signature, reloaded_signature = signatures
    assert_weight_update_matches_reference(
        dummy_signature,
        reference,
        updated_signature,
        reloaded_signature,
        case,
    )
