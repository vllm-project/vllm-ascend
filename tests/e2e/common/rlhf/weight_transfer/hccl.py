# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""HCCL transactions with a TP1 rollout server and a separate trainer NPU.

The trainer broadcasts checkpoints in pause/update/resume rounds. The
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
    capture_update_result,
    hccl_weight_transfer_servers,
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
    device = torch.device("npu", TRAINER_DEVICE_INDEX)
    source_a = make_weight_source(case, device, seed=0)
    source_b = make_weight_source(case, device, seed=1)
    updates = (("A", source_a), ("B", source_b), ("A", source_a))
    audit = case.weight_update_audit_factory() if case.weight_update_audit_factory is not None else None
    env_dict = {
        "VLLM_SERVER_DEV_MODE": "1",
        "ASCEND_RT_VISIBLE_DEVICES": server_visible_devices(INFERENCE_DEVICE_INDEX, INFERENCE_WORLD_SIZE),
    }

    with hccl_weight_transfer_servers(
        {"A": source_a, "B": source_b},
        case,
        backend="hccl",
        tensor_parallel_size=INFERENCE_WORLD_SIZE,
        device_index=INFERENCE_DEVICE_INDEX,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        env_dict=env_dict,
        audit=audit,
    ) as (server, references, dummy_signature):
        from vllm_ascend.distributed.weight_transfer.hccl_engine import HCCLTrainerInitInfo

        initial_state = audit.capture_state(server) if audit is not None else None
        updated_states = []
        register_engines_once()
        engine = WeightTransferTrainerFactory.trainer_init(
            HCCLTrainerInitInfo(
                rank=0,
                master_address=get_ip(),
                master_port=get_open_port(),
                world_size=INFERENCE_WORLD_SIZE + 1,
                packed=packed,
                packed_buffer_size_bytes=packed_buffer_size_for(source_a),
            ),
            client=HTTPVLLMWeightSyncClient(base_url=server.url_root, timeout=UPDATE_TIMEOUT),
            source=source_a,
        )
        signatures = []
        try:
            if dummy_signature is not None:
                assert dummy_signature != references["A"], f"{case.id}: initial payload is not observable"
            for update_round, (reference_name, source) in enumerate(updates, start=1):
                engine.source = source
                print(
                    f"[trainer] {case.id}: sending {('packed' if packed else 'unpacked')} "
                    f"{reference_name} update round={update_round}",
                    flush=True,
                )
                post_control(server, "pause")
                # Match the draft runner: do not resume partially updated weights
                # if a transfer fails.
                engine.send_weights()
                post_control(server, "resume")
                if audit is not None:
                    updated_states.append(audit.capture_state(server))
                observation = capture_update_result(server, case, audit=audit)
                assert observation == references[reference_name], (
                    f"{case.id}: {reference_name} update {update_round} differs from fresh-load"
                )
                signatures.append(observation)
        finally:
            engine.shutdown()

    assert signatures[0] != signatures[1], f"{case.id}: A/B updates have identical observations"
    assert signatures[2] == signatures[0], f"{case.id}: final A did not restore the first A"
    if audit is not None:
        audit.check_updates(
            initial_state=initial_state,
            updated_states=updated_states,
            references=[references["A"], references["B"]],
            updates=signatures,
            case=case,
        )
