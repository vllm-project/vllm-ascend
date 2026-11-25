# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""HCCL transactions with a TP1 rollout server and a separate trainer NPU.

The trainer broadcasts the checkpoint in two pause/update/resume rounds. The
shared server lifecycle supplies an independent startup-load reference and the
dummy pre-update signature. Draft coverage sends two draft updates followed by
one target update. IPC's same-device sleep/wake protocol is not used.
"""

import json
import os

import pytest
import torch
import torch_npu  # noqa: F401  # registers the NPU backend
from vllm.distributed.weight_transfer.clients import HTTPVLLMWeightSyncClient
from vllm.distributed.weight_transfer.factory import WeightTransferTrainerFactory
from vllm.utils.network_utils import get_ip, get_open_port

from tests.e2e.common.rlhf.common import (
    post_control,
    register_engines_once,
    rlhf_server,
    server_visible_devices,
    wait_for_free_device_memory,
)
from tests.e2e.common.rlhf.weight_transfer.case import WeightUpdateModelCase
from tests.e2e.common.rlhf.weight_transfer.reference_check import (
    UPDATE_ROUNDS,
    assert_weight_update_matches_reference,
    draft_serve_args,
    generation_signature,
    parameter_addresses,
    parameter_digests,
    snapshot_weight_update_models,
    weight_transfer_servers,
)
from tests.e2e.common.rlhf.weight_transfer.weight_source import (
    FIXED_WEIGHT_SEED,
    FixedRandomWeightSource,
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

        engine = create_hccl_trainer(
            HTTPVLLMWeightSyncClient(base_url=server.url_root, timeout=UPDATE_TIMEOUT), source, packed
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
                signatures.append(
                    generation_signature(client, case.model, prompts=case.prompts, image_prompt=case.image_prompt)
                )
        finally:
            engine.shutdown()

    assert_weight_update_matches_reference(
        dummy_signature,
        reference,
        signatures,
        case,
    )


class WeightUpdateClient(HTTPVLLMWeightSyncClient):
    """Select the draft START endpoint without changing the trainer protocol."""

    def __init__(self, base_url: str):
        super().__init__(base_url=base_url, timeout=UPDATE_TIMEOUT)
        self.is_draft = False

    def start_weight_update(self) -> None:
        self._post("start_draft_weight_update" if self.is_draft else "start_weight_update")


def create_hccl_trainer(client, source, packed: bool, buffer_source=None):
    """Initialize HCCL for either a target or draft synchronization client."""
    from vllm_ascend.distributed.weight_transfer.hccl_engine import HCCLTrainerInitInfo

    register_engines_once()
    return WeightTransferTrainerFactory.trainer_init(
        HCCLTrainerInitInfo(
            rank=0,
            master_address=get_ip(),
            master_port=get_open_port(),
            world_size=INFERENCE_WORLD_SIZE + 1,
            packed=packed,
            packed_buffer_size_bytes=packed_buffer_size_for(source if buffer_source is None else buffer_source),
        ),
        client=client,
        source=source,
    )


def collect_draft_observation(server, case: WeightUpdateModelCase):
    """Measure startup and updated models after generation refreshes derived state."""
    signature, draft_outputs = [], []
    for prompt in case.prompts:
        post_control(server, "collective_rpc", json={"method": "start_draft_output_capture"})
        try:
            signature.extend(generation_signature(server.get_client(), case.id, prompts=(prompt,)))
        finally:
            outputs = post_control(server, "collective_rpc", json={"method": "finish_draft_output_capture"}).json()[
                "results"
            ]
        assert len(outputs) == INFERENCE_WORLD_SIZE and all(outputs), "No draft proposal was captured"
        draft_outputs.append(outputs)
    snapshot = snapshot_weight_update_models(server)
    projections = post_control(server, "collective_rpc", json={"method": "snapshot_draft_context_projections"}).json()[
        "results"
    ]
    return snapshot, {
        "parameters": parameter_digests(snapshot),
        "generation": signature,
        "draft_outputs": draft_outputs,
        "context_projections": projections,
    }


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
            _, references[role] = collect_draft_observation(server, case)
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
        engine = create_hccl_trainer(sync_client, draft_updated.draft_source, packed, target_updated)
        try:
            # Reuse HCCL for warmup and all updates; capture graphs before
            # recording the addresses which live updates must preserve.
            generation_signature(client, case.id, prompts=case.prompts)
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
                sync_client.is_draft = role == "draft"
                engine.source = source
                post_control(server, "pause")
                # Failed transfers leave partially updated weights; keep generation paused.
                engine.send_weights()
                post_control(server, "resume")
                after, observation = collect_draft_observation(server, case)
                (tmp_path / f"update-{update_round}-{role}-snapshot.json").write_text(json.dumps(after, indent=2))
                (tmp_path / f"update-{update_round}-{role}-generation.json").write_text(
                    json.dumps(observation["generation"], indent=2)
                )
                (tmp_path / f"update-{update_round}-{role}-draft-outputs.json").write_text(
                    json.dumps(observation["draft_outputs"], indent=2)
                )
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
