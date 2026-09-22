# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
import types
from unittest.mock import MagicMock, patch

sys.modules.setdefault("torch_npu", types.ModuleType("torch_npu"))

from vllm_ascend.snapshot.distributed import (  # noqa: E402
    _abort_hccl_process_group,
    cleanup_dist_env_for_snapshot,
)


def test_abort_hccl_process_group_uses_npu_backend():
    process_group = MagicMock()
    backend = process_group._get_backend.return_value

    with patch(
        "vllm_ascend.snapshot.distributed.torch.device",
        return_value="npu-device",
    ):
        _abort_hccl_process_group(process_group)

    process_group._get_backend.assert_called_once_with("npu-device")
    backend.abort_hccl_comm.assert_called_once_with("reinit")


def test_snapshot_cleanup_injects_hccl_destroyer():
    with (
        patch("vllm_ascend.snapshot.distributed.destroy_model_parallel") as destroy_model,
        patch("vllm_ascend.snapshot.distributed.destroy_distributed_environment") as destroy_world,
        patch("vllm_ascend.snapshot.distributed.reset_group_name_registry") as reset,
    ):
        cleanup_dist_env_for_snapshot()

    destroy_model.assert_called_once_with(_abort_hccl_process_group)
    destroy_world.assert_called_once_with(_abort_hccl_process_group)
    reset.assert_called_once_with()
