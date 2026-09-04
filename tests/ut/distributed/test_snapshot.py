# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
import types
from unittest.mock import patch

sys.modules.setdefault("torch_npu", types.ModuleType("torch_npu"))

from vllm_ascend.snapshot.distributed import (  # noqa: E402
    cleanup_dist_env_for_snapshot,
    is_snapshot_hccl_teardown_enabled,
    snapshot_hccl_teardown,
)


def test_snapshot_cleanup_uses_standard_distributed_destroy():
    with (
        patch("vllm_ascend.snapshot.distributed.destroy_model_parallel") as destroy_model,
        patch("vllm_ascend.snapshot.distributed.destroy_distributed_environment") as destroy_world,
        patch("vllm_ascend.snapshot.distributed.reset_group_name_registry") as reset,
    ):
        cleanup_dist_env_for_snapshot()

    destroy_model.assert_called_once_with()
    destroy_world.assert_called_once_with()
    reset.assert_called_once_with()


def test_snapshot_hccl_teardown_is_scoped():
    assert not is_snapshot_hccl_teardown_enabled()

    with snapshot_hccl_teardown(True):
        assert is_snapshot_hccl_teardown_enabled()

    assert not is_snapshot_hccl_teardown_enabled()
