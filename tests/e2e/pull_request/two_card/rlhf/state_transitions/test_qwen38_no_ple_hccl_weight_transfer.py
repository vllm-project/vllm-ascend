# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TP1 real four-layer no-PLE HCCL updates; A3 rollout plus trainer."""

import pytest
import torch
import torch_npu  # noqa: F401
from vllm.utils.network_utils import get_ip, get_open_port

from tests.e2e.conftest import RemoteOpenAIServer
from tests.e2e.pull_request.rlhf.qwen38_checkpoint_source import Qwen38CheckpointSource
from tests.e2e.pull_request.rlhf.qwen38_weight_transfer_utils import (
    MEMORY_UTILIZATION,
    checkpoint_case,
    physical_device,
    post,
    worker_rpc,
)
from tests.e2e.pull_request.rlhf.weight_transfer_test_utils import (
    assert_weight_update_matches_reference,
    generation_signature,
    live_update_serve_args,
    packed_buffer_size_for,
    reference_signature,
    wait_for_free_device_memory,
)
from tests.e2e.pull_request.two_card.rlhf.state_transitions.test_hccl_weight_transfer import _BackgroundPost

TRANSFER_TIMEOUT = 600


def finish_rpc(thread) -> None:
    thread.join(timeout=TRANSFER_TIMEOUT + 10)
    if thread.is_alive():
        raise TimeoutError("HCCL RPC thread did not finish")
    thread.raise_if_failed()


def send_update(server, source, group, packed):
    from vllm_ascend.distributed.weight_transfer.hccl_engine import HCCLTrainerSendWeightsArgs, HCCLWeightTransferEngine

    metadata = source.metadata()
    buffer_size = packed_buffer_size_for(source)
    post(server, "pause")
    post(server, "start_weight_update")
    thread = _BackgroundPost(
        server,
        "update_weights",
        timeout=TRANSFER_TIMEOUT,
        json={
            "update_info": {
                "names": [m.name for m in metadata],
                "dtype_names": ["bfloat16"] * len(metadata),
                "shapes": [list(m.shape) for m in metadata],
                "packed": packed,
                "packed_buffer_size_bytes": buffer_size,
            }
        },
    )
    thread.start()
    try:
        HCCLWeightTransferEngine.trainer_send_weights(
            iterator=iter(source),
            trainer_args=HCCLTrainerSendWeightsArgs(
                group=group,
                packed=packed,
                packed_buffer_size_bytes=buffer_size,
            ),
        )
    finally:
        finish_rpc(thread)
    post(server, "finish_weight_update")
    post(server, "resume")


@pytest.mark.e2e_coverage(
    arch="moe",
    feature="weight_transfer,aclgraph,logprobs",
    parallel="",
    deploy="pd_mix",
    hardware="A3",
    quantization="BF16",
    graph_mode="full_decode_only",
)
@pytest.mark.parametrize("packed", [False, True], ids=["unpacked", "packed"])
def test_qwen38_no_ple_hccl_weight_transfer(request, packed):
    if torch.npu.device_count() < 2:
        pytest.skip("HCCL requires two allocated NPUs")
    from vllm_ascend.distributed.weight_transfer.hccl_engine import HCCLWeightTransferEngine

    case, checkpoint, digest, prompts = checkpoint_case(request)
    torch.npu.set_device(1)
    source = Qwen38CheckpointSource(checkpoint, digest, torch.device("npu", 1))
    env = {"VLLM_SERVER_DEV_MODE": "1", "ASCEND_RT_VISIBLE_DEVICES": physical_device(0)}
    reference = reference_signature(
        source,
        case,
        port=get_open_port(),
        gpu_memory_utilization=MEMORY_UTILIZATION,
        tensor_parallel_size=1,
        device_index=0,
        env_dict=env,
        prompts=prompts,
    )
    wait_for_free_device_memory(0, MEMORY_UTILIZATION)
    port = get_open_port()
    with RemoteOpenAIServer(
        case.model,
        vllm_serve_args=live_update_serve_args(
            case,
            backend="hccl",
            port=port,
            gpu_memory_utilization=MEMORY_UTILIZATION,
        ),
        server_port=port,
        env_dict=env,
        auto_port=False,
    ) as server:
        worker_rpc(server, "install_no_ple_probe")
        dummy = generation_signature(server.get_client(), case.model, prompts=prompts)
        worker_rpc(server, "check_no_ple_execution")
        address, group_port = get_ip(), get_open_port()
        thread = _BackgroundPost(
            server,
            "init_weight_transfer_engine",
            timeout=TRANSFER_TIMEOUT,
            json={
                "init_info": {
                    "master_address": address,
                    "master_port": group_port,
                    "rank_offset": 1,
                    "world_size": 2,
                }
            },
        )
        group = None
        thread.start()
        try:
            group = HCCLWeightTransferEngine.trainer_init(
                {
                    "master_address": address,
                    "master_port": group_port,
                    "world_size": 2,
                }
            )
            finish_rpc(thread)
            signatures = []
            for _ in range(2):
                worker_rpc(server, "install_no_ple_probe")
                send_update(server, source, group, packed)
                signatures.append(generation_signature(server.get_client(), case.model, prompts=prompts))
                worker_rpc(server, "check_no_ple_execution")
        finally:
            if group is not None:
                group.close()
            if thread.is_alive():
                finish_rpc(thread)
    assert_weight_update_matches_reference(dummy, reference, *signatures, case)
