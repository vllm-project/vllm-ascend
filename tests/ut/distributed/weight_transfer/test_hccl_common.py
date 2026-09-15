# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from vllm_ascend.distributed.weight_transfer.hccl_common import (
    HCCLWeightTransferInitInfo,
    trainer_init,
    worker_init_process_group,
)


def test_worker_init_process_group_uses_global_dp_rank():
    init_info = HCCLWeightTransferInitInfo(
        master_address="127.0.0.1",
        master_port=12345,
        rank_offset=1,
        world_size=16,
    )
    parallel_config = SimpleNamespace(
        data_parallel_index=2,
        world_size=4,
        rank=3,
    )

    with (
        patch(
            "vllm_ascend.distributed.weight_transfer.hccl_common."
            "stateless_init_process_group",
            return_value="group",
        ) as mock_init,
        patch("torch.accelerator.current_device_index", return_value=5),
    ):
        result = worker_init_process_group(init_info, parallel_config)

    assert result == "group"
    mock_init.assert_called_once_with("127.0.0.1", 12345, 12, 16, device=5)


@pytest.mark.parametrize(
    "init_info",
    [
        {
            "master_address": "127.0.0.1",
            "master_port": 23456,
            "world_size": 2,
        },
        HCCLWeightTransferInitInfo(
            master_address="127.0.0.1",
            master_port=23456,
            rank_offset=1,
            world_size=2,
        ),
    ],
)
def test_trainer_init_always_uses_rank_zero(init_info):
    with (
        patch(
            "vllm_ascend.distributed.weight_transfer.hccl_common."
            "stateless_init_process_group",
            return_value="group",
        ) as mock_init,
        patch("torch.accelerator.current_device_index", return_value=3),
    ):
        assert trainer_init(init_info) == "group"

    mock_init.assert_called_once_with("127.0.0.1", 23456, 0, 2, 3)
