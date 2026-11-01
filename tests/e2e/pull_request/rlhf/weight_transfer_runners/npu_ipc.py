# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Model-aware end-to-end tests for NPU IPC live weight updates.

Qwen3-0.6B cannot expose missing START/FINISH hooks because its checkpoint and
runtime weight representations are compatible. This matrix instead targets
architectures whose correctness depends on the transaction: fused-MoE layout
restoration, derived FP32 routing weights, and SFA source/derived-state
restoration. Cases whose model is not ready for the transaction are skipped in
both lanes with a per-case ``skip_reason`` — DeepSeek-V4-Flash needs the
attention-sink fix that lives in #16355. TP=2 IPC runs Qwen3.5-35B-A3B
GLM-5.3-Flash and MiniMax-M3.

The correctness oracle is *normal startup loading of the same payload*, not the
first live update: the generator also writes a temporary checkpoint, a reference
server loads it at startup, and the live-update lane has to reproduce its
signature exactly —

    normal startup load(W) == first live update(W) == second live update(W)

The dummy-started lane's pre-update signature must differ from the reference, so
a transfer that never ran cannot pass, and the second update covers the layerwise
reload lifecycle, runtime/destructive representations, derived state and the
graph-captured storage of the first update. Enabled models cover unpacked TP=2 IPC;
Qwen3.5 additionally covers packed IPC. The packed lane is currently skipped
on this branch — see ``PACKED_LANE_SKIP_REASON`` — so only the unpacked lane runs.

Two trainer ranks each materialize the same full checkpoint on their local
card and merge IPC handles for both TP workers. Each trainer shares its card
with one rollout worker, so every round releases the engine's HBM
with a level-2 sleep and wakes it again afterwards. Level 2
discards the weights, so the weights allocation is mapped back *before* the
transaction — the layerwise reload copies the payload into those very buffers —
while the KV cache is re-allocated last, once the new weights are in place.
"""

import os
from datetime import timedelta

import pytest
import requests
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch_npu  # noqa: F401  # registers the NPU backend
from vllm.utils.network_utils import get_open_port

from tests.e2e.conftest import RemoteOpenAIServer
from tests.e2e.pull_request.rlhf.weight_transfer_models.base import WeightUpdateModelCase
from tests.e2e.pull_request.rlhf.weight_transfer_test_utils import (
    PACKED_LANE_SKIP_REASON,
    FixedRandomWeightSource,
    assert_weight_update_matches_reference,
    generation_signature,
    live_update_serve_args,
    packed_buffer_size_for,
    reference_signature,
    register_engines_once,
    wait_for_free_device_memory,
)

INFERENCE_WORLD_SIZE = 2
UPDATE_TIMEOUT = 300
CONTROL_TIMEOUT = 60
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


def _post(server: RemoteOpenAIServer, route: str, *, json=None, params=None, timeout=CONTROL_TIMEOUT):
    response = requests.post(server.url_for(route), json=json, params=params, timeout=timeout)
    response.raise_for_status()
    return response


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
            for _ in range(2):
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
        **dict(case.server_env),
        "VLLM_SERVER_DEV_MODE": "1",
        "VLLM_ALLOW_INSECURE_SERIALIZATION": "1",
        "ASCEND_RT_VISIBLE_DEVICES": ",".join(
            str(INFERENCE_DEVICE_INDEX + rank) for rank in range(INFERENCE_WORLD_SIZE)
        ),
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
    # two cards, so wait for its process tree to hand the HBM back.
    for rank in range(INFERENCE_WORLD_SIZE):
        wait_for_free_device_memory(INFERENCE_DEVICE_INDEX + rank, GPU_MEMORY_UTILIZATION)
    port = get_open_port()
    with RemoteOpenAIServer(
        case.model,
        vllm_serve_args=live_update_serve_args(
            case,
            backend="npu_ipc",
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
            for _ in range(2):
                _post(server, "pause")
                # Same-chip RL: hand the engine's HBM to the co-located trainer
                # before generating the payload. Level 2 discards the weights and the
                # KV cache outright; `/sleep` reads its level from the query string,
                # a JSON body is ignored.
                _post(server, "sleep", params={"level": 2})
                # Level 2 unmapped the weights pool and `send_weights`' layerwise
                # reload copies the payload back into exactly those buffers, so the
                # weights allocation has to be mapped again before the transaction.
                # The KV cache stays asleep until the update is finished.
                _post(server, "wake_up", params={"tags": ["weights"]})
                # send_weights owns the complete START -> LOAD -> FINISH transaction.
                dist.barrier()
                engine.send_weights()
                # The KV cache is re-allocated last, from the updated weights' state.
                # That wake leaves nothing asleep, so the engine resumes scheduling on
                # its own and no explicit `/resume` is needed.
                _post(server, "wake_up", params={"tags": ["kv_cache"]})
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
