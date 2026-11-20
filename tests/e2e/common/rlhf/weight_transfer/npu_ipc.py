# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""IPC transactions with one trainer and rollout worker per NPU.

Each trainer streams the full checkpoint locally; Gloo merges the devices'
IPC handles. Existing cases default to sleep level 2. Direct cases may run
without sleep and start from real checkpoint A, with independent cold A/B
references for repeated updates.
"""

import os
import time
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch_npu  # noqa: F401  # registers the NPU backend
from vllm.utils.network_utils import get_open_port

from tests.e2e.common.rlhf.common import (
    CONTROL_TIMEOUT,
    post_control,
    register_engines_once,
    server_visible_devices,
)
from tests.e2e.common.rlhf.weight_transfer.case import WeightUpdateModelCase
from tests.e2e.common.rlhf.weight_transfer.reference_check import (
    UPDATE_ROUNDS,
    assert_weight_update_matches_reference,
    generation_signature,
    reference_signature,
    weight_transfer_servers,
)
from tests.e2e.common.rlhf.weight_transfer.reload_probe import (
    assert_reload_reuses_runtime,
    assert_sleep_allocations,
    reload_snapshots,
)
from tests.e2e.common.rlhf.weight_transfer.weight_source import (
    make_weight_source,
    packed_buffer_size_for,
)

UPDATE_TIMEOUT = 300

INFERENCE_WORLD_SIZE = 2
# The trainer is co-located with the server on the same chip, so both models
# have to fit; keep the budget low enough to leave room for the payload. Cases
# whose payload is much heavier than the default lanes raise it from the runner.
GPU_MEMORY_UTILIZATION = float(os.environ.get("VLLM_RL_TEST_GPU_MEM_UTIL", "0.45"))
# First parent-process logical device; child-server visibility is mapped to
# physical IDs through the parent's ASCEND_RT_VISIBLE_DEVICES, when present.
INFERENCE_DEVICE_INDEX = int(os.environ.get("VLLM_RL_TEST_DEVICE_INDEX", "0"))


def _ipc_trainer_peer(
    local_rank: int,
    case: WeightUpdateModelCase,
    packed: bool,
    server_url: str,
    rendezvous: str,
    seeds: tuple[int, ...],
    world_size: int,
):
    from vllm.distributed.weight_transfer.clients import HTTPVLLMWeightSyncClient
    from vllm.distributed.weight_transfer.factory import WeightTransferTrainerFactory

    from vllm_ascend.distributed.weight_transfer.npu_ipc_engine import NPUIPCTrainerInitInfo

    rank = local_rank + 1  # Rank 0 runs in the parent; spawn supplies peer-local ranks.
    device_index = INFERENCE_DEVICE_INDEX + rank
    torch.npu.set_device(device_index)
    source = make_weight_source(case, torch.device("npu", device_index))
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=UPDATE_TIMEOUT),
    )
    try:
        register_engines_once()
        engine = WeightTransferTrainerFactory.trainer_init(
            NPUIPCTrainerInitInfo(
                rank=rank,
                packed=packed,
                packed_buffer_size_bytes=packed_buffer_size_for(source),
            ),
            client=HTTPVLLMWeightSyncClient(base_url=server_url, timeout=UPDATE_TIMEOUT),
            source=source,
        )
        try:
            for seed in seeds:
                source.seed = seed
                # Rank 0 releases this barrier when rollout is ready to update.
                dist.barrier()
                engine.send_weights()
                # This rank materialised its own copy of the payload, and the NPU
                # caching allocator keeps those blocks. The caller wakes the KV
                # cache on this chip after the barrier below, so hand the memory
                # back here as well, not only on rank 0, and wait there until rank 0
                # has released its own blocks.
                torch.npu.empty_cache()
                dist.barrier()
        finally:
            engine.shutdown()
    finally:
        dist.destroy_process_group()


def run_npu_ipc_weight_transfer_transaction(
    case: WeightUpdateModelCase,
    packed: bool,
    *,
    reload_mode: str = "layerwise",
    execution_mode: str = "graph",
    sleep_level: int | None = 2,
    cold_start: bool = False,
    tensor_parallel_size: int = INFERENCE_WORLD_SIZE,
    gpu_memory_utilization: float = GPU_MEMORY_UTILIZATION,
):
    if sleep_level not in (None, 1, 2):
        raise ValueError(f"Unsupported sleep level: {sleep_level}")
    if tensor_parallel_size < 1:
        raise ValueError("tensor_parallel_size must be positive")
    if INFERENCE_DEVICE_INDEX < 0 or torch.npu.device_count() < INFERENCE_DEVICE_INDEX + tensor_parallel_size:
        pytest.skip("NPU IPC weight transfer requires enough consecutive NPUs for the requested TP size.")

    torch.npu.set_device(INFERENCE_DEVICE_INDEX)
    source = make_weight_source(case, torch.device("npu", INFERENCE_DEVICE_INDEX))
    os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
    env_dict = {
        "VLLM_SERVER_DEV_MODE": "1",
        "VLLM_ALLOW_INSECURE_SERIALIZATION": "1",
        "VLLM_RELOAD_MODE": reload_mode,
        "VLLM_USE_V2_MODEL_RUNNER": "1",
        "ASCEND_RT_VISIBLE_DEVICES": server_visible_devices(INFERENCE_DEVICE_INDEX, tensor_parallel_size),
    }

    options = dict(
        tensor_parallel_size=tensor_parallel_size,
        device_index=INFERENCE_DEVICE_INDEX,
        gpu_memory_utilization=gpu_memory_utilization,
        env_dict=env_dict,
        execution_mode=execution_mode,
        enable_sleep=sleep_level is not None,
    )
    seed_a = source.seed
    seeds = (seed_a,) * UPDATE_ROUNDS
    references = {}
    if reload_mode == "direct":
        seed_b = seed_a + 1
        seeds = (seed_a, seed_b, seed_a)
        source.seed = seed_b
        references[seed_b] = reference_signature(source, case, port=get_open_port(), **options)
        source.seed = seed_a

    with weight_transfer_servers(source, case, backend="npu_ipc", cold_start=cold_start, **options) as (
        server,
        reference,
        dummy_signature,
    ):
        references[seed_a] = reference
        if not cold_start:
            assert dummy_signature != reference, f"{case.id}: dummy and cold A produced identical signatures"
        if reload_mode == "direct":
            assert reference != references[seed_b], f"{case.id}: cold A and B produced identical signatures"
        client = server.get_client()
        snapshots = reload_snapshots(server, execution_mode=execution_mode) if cold_start else None

        from vllm.distributed.weight_transfer.clients import HTTPVLLMWeightSyncClient
        from vllm.distributed.weight_transfer.factory import WeightTransferTrainerFactory

        from vllm_ascend.distributed.weight_transfer.npu_ipc_engine import NPUIPCTrainerInitInfo

        rendezvous = f"tcp://127.0.0.1:{get_open_port()}"
        peer = (
            mp.spawn(
                _ipc_trainer_peer,
                args=(case, packed, server.url_root, rendezvous, seeds, tensor_parallel_size),
                nprocs=tensor_parallel_size - 1,
                join=False,
            )
            if tensor_parallel_size > 1
            else None
        )
        engine = None
        try:
            dist.init_process_group(
                "gloo",
                init_method=rendezvous,
                rank=0,
                world_size=tensor_parallel_size,
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
            for update_round, seed in enumerate(seeds, start=1):
                source.seed = seed
                post_control(server, "pause")
                # Same-chip RL: hand the engine's HBM to the co-located trainer
                # before generating the payload. Level 2 discards the weights and the
                # KV cache outright; `/sleep` reads its level from the query string,
                # a JSON body is ignored.
                if sleep_level is not None:
                    post_control(server, "sleep", params={"level": sleep_level})
                    if snapshots is not None:
                        assert_sleep_allocations(server, sleep_level=sleep_level, asleep_tags=("weights", "kv_cache"))
                # Level 2 unmapped the weights pool and `send_weights`' layerwise
                # reload copies the payload back into exactly those buffers, so the
                # weights allocation has to be mapped again before the transaction.
                # The KV cache stays asleep until the update is finished.
                if sleep_level is not None:
                    post_control(server, "wake_up", params={"tags": ["weights"]})
                    if snapshots is not None:
                        assert_sleep_allocations(server, sleep_level=sleep_level, asleep_tags=("kv_cache",))
                # send_weights owns the complete START -> LOAD -> FINISH transaction.
                dist.barrier()
                transfer_started = time.monotonic()
                engine.send_weights()
                print(
                    f"[ipc-transfer] {case.id} packed={packed} mode={execution_mode} "
                    f"round={update_round} seconds={time.monotonic() - transfer_started:.3f}",
                    flush=True,
                )
                # The trainer shares the chip with the rollout engine, and the NPU
                # caching allocator keeps the transient payload tensors this source
                # materialised. Device-wide "still in use" after the next sleep went
                # 1.10 -> 7.66 GiB across rounds and the engine's wake_up then died
                # in aclrtMallocPhysical with OOM, so hand the memory back
                # explicitly.
                torch.npu.empty_cache()
                # Wait for the peer's release before the engine allocates the KV cache.
                dist.barrier()
                # The KV cache is re-allocated last, from the updated weights' state.
                # That wake leaves nothing asleep, so the engine resumes scheduling on
                # its own and no explicit `/resume` is needed.
                if sleep_level is not None:
                    post_control(server, "wake_up", params={"tags": ["kv_cache"]})
                    if snapshots is not None:
                        assert_sleep_allocations(server, sleep_level=sleep_level, asleep_tags=())
                else:
                    post_control(server, "resume")
                signature = generation_signature(client, case.model)
                signatures.append(signature)
                if reload_mode == "direct":
                    assert signature == references[seed], (
                        f"{case.id}: direct IPC round {update_round} differs from cold loading seed {seed}"
                    )
                if snapshots is not None:
                    current = reload_snapshots(server, execution_mode=execution_mode)
                    assert_reload_reuses_runtime(snapshots, current, execution_mode=execution_mode)
                    snapshots = current
            if peer is not None:
                deadline = time.monotonic() + CONTROL_TIMEOUT
                # ProcessContext.join can return False after joining only the
                # first exited peer. TP4 has three peers; allow the remainder
                # to exit within the same total timeout.
                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError("IPC trainer peers did not exit after the weight updates")
                    if peer.join(timeout=remaining):
                        break
        finally:
            if engine is not None:
                engine.shutdown()
            for process in peer.processes if peer is not None else ():
                if process.is_alive():
                    process.terminate()
                process.join(timeout=CONTROL_TIMEOUT)
            if dist.is_initialized():
                dist.destroy_process_group()

    if reload_mode != "direct":
        assert_weight_update_matches_reference(dummy_signature, reference, signatures, case)
