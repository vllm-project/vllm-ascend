# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Verify two draft HCCL updates and one target update against startup loads."""

import json

import torch
import torch_npu  # noqa: F401
from vllm.utils.network_utils import get_open_port

from tests.e2e.common.rlhf.common import (
    post_control,
    rlhf_server,
    server_visible_devices,
    wait_for_free_device_memory,
)
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


def run_draft_hccl_weight_transfer(tmp_path, monkeypatch, packed: bool, enforce_eager: bool, speculative_case):
    """Compare two draft updates and one target update with independent loads."""
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    torch.npu.set_device(TRAINER_DEVICE_INDEX)
    device = torch.device("npu", TRAINER_DEVICE_INDEX)
    case, draft_case = speculative_case.target, speculative_case.draft
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

    # The draft reference keeps the initial target. The target reference
    # contains the updated target together with the same updated draft.
    references = {}
    for role, source in (("draft", draft_updated), ("target", target_updated)):
        checkpoint = tmp_path / f"{role}-reference"
        source.write_checkpoint(checkpoint)
        wait_for_free_device_memory(INFERENCE_DEVICE_INDEX, 0.2)
        port = get_open_port()
        with rlhf_server(
            str(checkpoint),
            vllm_serve_args=draft_serve_args(
                checkpoint, speculative_case, port, enforce_eager=enforce_eager, live=False
            ),
            server_host="127.0.0.1",
            server_port=port,
            env_dict=env_dict,
        ) as server:
            signature = generation_signature(server.get_client(), case.id)
            projections = post_control(
                server, "collective_rpc", json={"method": "snapshot_draft_context_projections"}
            ).json()["results"]
            references[role] = {
                "parameters": parameter_digests(snapshot_weight_update_models(server)),
                "generation": signature,
                "context_projections": projections,
            }
            (tmp_path / f"{role}-reference.json").write_text(json.dumps(references[role], indent=2))

    wait_for_free_device_memory(INFERENCE_DEVICE_INDEX, 0.2)
    port = get_open_port()
    results = []
    with rlhf_server(
        str(initial_path),
        vllm_serve_args=draft_serve_args(initial_path, speculative_case, port, enforce_eager=enforce_eager, live=True),
        server_host="127.0.0.1",
        server_port=port,
        env_dict=env_dict,
    ) as server:
        client = server.get_client()
        sync_client = WeightUpdateClient(server.url_root)
        assert draft_updated.draft_source is not None
        engine = create_draft_hccl_trainer(sync_client, draft_updated.draft_source, packed, target_updated)
        try:
            # Reuse HCCL for warmup and all updates; capture graphs before
            # recording the addresses which live updates must preserve.
            generation_signature(client, case.id)
            architectures = post_control(server, "collective_rpc", json={"method": "draft_model_architecture"}).json()[
                "results"
            ]
            assert len(architectures) == INFERENCE_WORLD_SIZE and all(
                name.endswith(speculative_case.draft_architecture) for name in architectures
            ), architectures
            before = snapshot_weight_update_models(server)
            original_addresses = parameter_addresses(before)
            (tmp_path / "initial-snapshot.json").write_text(json.dumps(before, indent=2))
            assert parameter_digests(before)["draft"] != references["draft"]["parameters"]["draft"], (
                "Draft payload is not observable"
            )

            updates = (
                ("draft", draft_updated.draft_source),
                ("draft", draft_updated.draft_source),
                ("target", target_updated),
            )
            for update_round, (role, source) in enumerate(updates, start=1):
                print(f"[trainer] {speculative_case.id}: {role} update round={update_round}", flush=True)
                send_update(server, engine, sync_client, source, is_draft=role == "draft")
                after = snapshot_weight_update_models(server)
                projections = post_control(
                    server, "collective_rpc", json={"method": "snapshot_draft_context_projections"}
                ).json()["results"]
                signature = generation_signature(client, case.id)
                observation = {
                    "parameters": parameter_digests(after),
                    "generation": signature,
                    "context_projections": projections,
                }
                (tmp_path / f"update-{update_round}-{role}-snapshot.json").write_text(json.dumps(after, indent=2))
                (tmp_path / f"update-{update_round}-{role}-generation.json").write_text(json.dumps(signature, indent=2))
                results.append((role, observation, parameter_addresses(after)))
        finally:
            engine.shutdown()

    # Full parameter comparison also checks that draft-only updates leave the
    # target unchanged. Greedy generation alone cannot observe draft changes.
    for update_round, (role, observation, addresses) in enumerate(results, start=1):
        assert observation == references[role], (
            f"{speculative_case.id}: {role} update {update_round} differs from startup"
        )
        assert addresses == original_addresses, (
            f"{speculative_case.id}: update {update_round} changed parameter storage"
        )
