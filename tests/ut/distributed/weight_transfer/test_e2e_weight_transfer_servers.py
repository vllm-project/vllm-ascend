# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reference/live server isolation and environment parity across backends."""

from dataclasses import replace
from unittest.mock import MagicMock, call, patch

import pytest

from tests.e2e.pull_request.rlhf.weight_transfer.models.qwen import QWEN35_CASE
from tests.e2e.pull_request.rlhf.weight_transfer.runners import common


@pytest.mark.parametrize(("backend", "tp"), [("npu_ipc", 2), ("hccl", 1)])
def test_servers_share_environment_and_wait_for_all_devices(backend, tp):
    case = replace(QWEN35_CASE, server_env=(("VLLM_USE_V2_MODEL_RUNNER", "0"),))
    source = MagicMock()
    reference, dummy = ["startup"], ["dummy"]
    env = {"ASCEND_RT_VISIBLE_DEVICES": "6,7" if tp == 2 else "6"}
    expected_env = {"VLLM_USE_V2_MODEL_RUNNER": "0", **env}
    with (
        patch.object(common, "reference_signature", return_value=reference) as startup,
        patch.object(common, "wait_for_free_device_memory") as wait,
        patch.object(common, "RemoteOpenAIServer") as server_factory,
        patch.object(common, "generation_signature", return_value=dummy),
        patch.object(common, "get_open_port", return_value=8123),
    ):
        with common.weight_transfer_servers(
            source,
            case,
            backend=backend,
            tensor_parallel_size=tp,
            device_index=6,
            gpu_memory_utilization=0.45,
            env_dict=env,
        ) as (server, actual_reference, actual_dummy):
            assert server is server_factory.return_value.__enter__.return_value
            assert actual_reference == reference
            assert actual_dummy == dummy
            startup.assert_called_once_with(
                source,
                case,
                port=8123,
                gpu_memory_utilization=0.45,
                tensor_parallel_size=tp,
                device_index=6,
                env_dict=expected_env,
            )
            assert wait.call_args_list == [call(6 + rank, 0.45) for rank in range(tp)]
            assert server_factory.call_args.kwargs["env_dict"] == expected_env
        server_factory.return_value.__exit__.assert_called_once()
    assert env == {"ASCEND_RT_VISIBLE_DEVICES": "6,7" if tp == 2 else "6"}
