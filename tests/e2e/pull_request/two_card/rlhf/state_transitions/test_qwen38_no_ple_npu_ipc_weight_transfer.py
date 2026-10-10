# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A3 two-card runner, using one physical NPU for rollout and IPC trainer."""

import pytest
import torch
import torch_npu  # noqa: F401
from vllm.utils.network_utils import get_open_port

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
    register_engines_once,
    wait_for_free_device_memory,
)


@pytest.mark.e2e_coverage(
    arch="moe",
    feature="weight_transfer,sleep_wake,aclgraph,logprobs",
    parallel="",
    deploy="pd_mix",
    hardware="A3",
    quantization="BF16",
    graph_mode="full_decode_only",
)
@pytest.mark.parametrize("packed", [False, True], ids=["unpacked", "packed"])
def test_qwen38_no_ple_npu_ipc_weight_transfer(request, monkeypatch, packed):
    if torch.npu.device_count() < 1:
        pytest.skip("IPC requires an allocated NPU")
    from vllm.distributed.weight_transfer.clients import HTTPVLLMWeightSyncClient
    from vllm.distributed.weight_transfer.factory import WeightTransferTrainerFactory

    from vllm_ascend.distributed.weight_transfer.npu_ipc_engine import NPUIPCTrainerInitInfo, npu_generate_uuid

    case, checkpoint, digest, prompts = checkpoint_case(request)
    torch.npu.set_device(0)
    source = Qwen38CheckpointSource(checkpoint, digest, torch.device("npu", 0))
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    env = {
        "VLLM_SERVER_DEV_MODE": "1",
        "ASCEND_RT_VISIBLE_DEVICES": physical_device(0),
        "VLLM_ALLOW_INSECURE_SERIALIZATION": "1",
    }
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
            backend="npu_ipc",
            port=port,
            gpu_memory_utilization=MEMORY_UTILIZATION,
        ),
        server_port=port,
        env_dict=env,
        auto_port=False,
    ) as server:
        result = worker_rpc(server, "install_no_ple_probe")
        assert result["results"] == [npu_generate_uuid(0)], result
        dummy = generation_signature(server.get_client(), case.model, prompts=prompts)
        worker_rpc(server, "check_no_ple_execution")
        register_engines_once()
        engine = WeightTransferTrainerFactory.trainer_init(
            NPUIPCTrainerInitInfo(rank=0, packed=packed, packed_buffer_size_bytes=packed_buffer_size_for(source)),
            client=HTTPVLLMWeightSyncClient(base_url=server.url_root),
            source=source,
        )
        signatures = []
        for _ in range(2):
            worker_rpc(server, "install_no_ple_probe")
            post(server, "pause")
            post(server, "sleep", params={"level": 2})
            post(server, "wake_up", params={"tags": ["weights"]})
            engine.send_weights()
            torch.npu.synchronize()
            torch.npu.empty_cache()
            post(server, "wake_up", params={"tags": ["kv_cache"]})
            signatures.append(generation_signature(server.get_client(), case.model, prompts=prompts))
            worker_rpc(server, "check_no_ple_execution")
    assert_weight_update_matches_reference(dummy, reference, signatures[0], signatures[1], case)
