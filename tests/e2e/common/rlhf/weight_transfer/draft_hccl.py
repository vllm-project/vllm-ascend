# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Draft HCCL client, model snapshots and two-draft/one-target transaction."""

import torch
import torch_npu  # noqa: F401
from vllm.utils.network_utils import get_open_port

from tests.e2e.common.rlhf.common import (
    rlhf_server,
    server_visible_devices,
    wait_for_free_device_memory,
)
from tests.e2e.common.rlhf.weight_transfer.case import QWEN3_DRAFT_CASE, QWEN3_DRAFT_TARGET_CASE
from tests.e2e.common.rlhf.weight_transfer.hccl import (
    INFERENCE_DEVICE_INDEX,
    INFERENCE_WORLD_SIZE,
    TRAINER_DEVICE_INDEX,
    WeightUpdateClient,
    create_draft_hccl_trainer,
    send_update,
)
from tests.e2e.common.rlhf.weight_transfer.reference_check import (
    draft_serve_args,
    generation_signature,
    parameter_addresses,
    parameter_digests,
    snapshot_weight_update_models,
)
from tests.e2e.common.rlhf.weight_transfer.weight_source import (
    FIXED_WEIGHT_SEED,
    FixedRandomWeightSource,
)


def _log(message):
    print(f"[trainer] {message}", flush=True)


def run_draft_hccl_weight_transfer(tmp_path, monkeypatch, packed: bool, enforce_eager: bool):
    """Full draft payload twice, then target payload, against independent loads.

    Greedy output alone cannot observe a changed draft. Match every parameter
    digest to a startup-load reference as well, and preserve runtime addresses.
    """
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    torch.npu.set_device(TRAINER_DEVICE_INDEX)
    device = torch.device("npu", TRAINER_DEVICE_INDEX)
    case, draft_case = QWEN3_DRAFT_TARGET_CASE, QWEN3_DRAFT_CASE
    initial = FixedRandomWeightSource(case, device, draft_case=draft_case)
    draft_updated = FixedRandomWeightSource(case, device, draft_case=draft_case, draft_seed=FIXED_WEIGHT_SEED + 2)
    target_updated = FixedRandomWeightSource(
        case, device, seed=FIXED_WEIGHT_SEED + 3, draft_case=draft_case, draft_seed=FIXED_WEIGHT_SEED + 2
    )
    initial_path = tmp_path / "initial"
    initial.write_checkpoint(initial_path)
    env_dict = {
        "VLLM_SERVER_DEV_MODE": "1",
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "ASCEND_RT_VISIBLE_DEVICES": server_visible_devices(INFERENCE_DEVICE_INDEX, INFERENCE_WORLD_SIZE),
    }
    references = []
    for name, source in (("draft-reference", draft_updated), ("target-reference", target_updated)):
        checkpoint = tmp_path / name
        source.write_checkpoint(checkpoint)
        wait_for_free_device_memory(INFERENCE_DEVICE_INDEX, 0.2)
        port = get_open_port()
        with rlhf_server(
            str(checkpoint),
            vllm_serve_args=draft_serve_args(checkpoint, case, port, enforce_eager=enforce_eager, live=False),
            server_host="127.0.0.1",
            server_port=port,
            env_dict=env_dict,
        ) as server:
            signature = generation_signature(server.get_client(), case.id)
            references.append((parameter_digests(snapshot_weight_update_models(server)), signature))

    wait_for_free_device_memory(INFERENCE_DEVICE_INDEX, 0.2)
    port = get_open_port()
    with rlhf_server(
        str(initial_path),
        vllm_serve_args=draft_serve_args(initial_path, case, port, enforce_eager=enforce_eager, live=True),
        server_host="127.0.0.1",
        server_port=port,
        env_dict=env_dict,
    ) as server:
        client = server.get_client()
        generation_signature(client, case.id)  # Capture/warm up graphs before checking addresses.
        before = snapshot_weight_update_models(server)
        addresses = parameter_addresses(before)
        sync_client = WeightUpdateClient(server.url_root)
        assert draft_updated.draft_source is not None
        engine = create_draft_hccl_trainer(sync_client, draft_updated.draft_source, packed, target_updated)
        try:
            assert parameter_digests(before)["draft"] != references[0][0]["draft"], "Draft payload is not observable"
            for update_round in range(2):
                _log(f"draft round={update_round + 1}")
                send_update(server, engine, sync_client, draft_updated.draft_source, is_draft=True)
                assert generation_signature(client, case.id) == references[0][1]
                after = snapshot_weight_update_models(server)
                assert parameter_digests(after) == references[0][0]
                assert parameter_digests(after)["target"] == parameter_digests(before)["target"]
                assert parameter_addresses(after) == addresses

            send_update(server, engine, sync_client, target_updated, is_draft=False)
            assert generation_signature(client, case.id) == references[1][1]
            after = snapshot_weight_update_models(server)
            assert parameter_digests(after) == references[1][0]
            assert parameter_addresses(after) == addresses
        finally:
            engine.shutdown()
