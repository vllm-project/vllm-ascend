#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#

"""A5 collective-communication QoS: default world, group lookup, and NetLoader."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from vllm_ascend.ascend_config import CollectiveCommunicationQosConfig
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.utils import (
    SLEEP_LIFECYCLE_ANCHOR_GROUP_NAME,
    get_hccl_config_for_pg_options,
    get_hccl_qos_config,
)

_QOS_MEDIUM = {
    "hccl_sdma_qos": 4,
    "qos_service_level": 4,
    "qos_traffic_class": 128,
}
_QOS_HIGH = {
    "hccl_sdma_qos": 6,
    "qos_service_level": 6,
    "qos_traffic_class": 192,
}


class _HcclOptions:
    def __init__(self):
        self.hccl_config = None


def _qos_ascend_config(
    enabled: bool = True,
    default_priority: str = "medium",
    manual: dict | None = None,
):
    collective = CollectiveCommunicationQosConfig(
        enabled=enabled,
        default_priority=default_priority,
        manual={} if manual is None else manual,
    )
    return SimpleNamespace(ai_qos=SimpleNamespace(collective_communication=collective))


def _load_patch_distributed():
    """Load patch_distributed.py without importing the worker package.

    Importing ``vllm_ascend.patch.worker`` applies every worker patch. This
    loads only the distributed patch, then restores the symbols it replaces.
    """
    import torch.distributed as dist
    import vllm.distributed as vllm_distributed
    import vllm.distributed.parallel_state as parallel_state

    saved_init = dist.init_process_group
    saved_coordinator = parallel_state.GroupCoordinator
    saved_destroy = parallel_state.destroy_distributed_environment
    saved_vllm_destroy = vllm_distributed.destroy_distributed_environment
    module_path = Path(__file__).resolve().parents[3] / "vllm_ascend/patch/worker/patch_distributed.py"
    spec = importlib.util.spec_from_file_location("_ut_ai_qos_patch_distributed", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    finally:
        dist.init_process_group = saved_init
        parallel_state.GroupCoordinator = saved_coordinator
        parallel_state.destroy_distributed_environment = saved_destroy
        vllm_distributed.destroy_distributed_environment = saved_vllm_destroy
    return module


def test_manual_rejects_empty_group_name():
    with pytest.raises(ValidationError):
        CollectiveCommunicationQosConfig(manual={"": "low"})


def test_get_qos_uses_manual_or_default():
    disabled = CollectiveCommunicationQosConfig(enabled=False, manual={"tp": "high"})
    assert disabled.get_qos("tp") is None

    enabled = CollectiveCommunicationQosConfig(
        enabled=True,
        default_priority="medium",
        manual={"tp": "high"},
    )
    assert enabled.get_qos("tp") == 6
    assert enabled.get_qos("pp") == 4


def test_non_a5_skips_ascend_config():
    with (
        patch("vllm_ascend.utils.get_ascend_device_type", return_value=AscendDeviceType.A3),
        patch("vllm_ascend.utils.get_ascend_config") as get_config,
    ):
        assert get_hccl_qos_config("tp") == {}
    get_config.assert_not_called()


def test_a5_disabled_returns_empty_qos():
    with (
        patch("vllm_ascend.utils.get_ascend_device_type", return_value=AscendDeviceType.A5),
        patch("vllm_ascend.utils.get_ascend_config", return_value=_qos_ascend_config(enabled=False)),
    ):
        assert get_hccl_qos_config("tp") == {}


def test_a5_manual_overrides_default_priority():
    config = _qos_ascend_config(manual={"tp": "high"})
    with (
        patch("vllm_ascend.utils.get_ascend_device_type", return_value=AscendDeviceType.A5),
        patch("vllm_ascend.utils.get_ascend_config", return_value=config),
    ):
        assert get_hccl_qos_config("tp") == _QOS_HIGH
        assert get_hccl_qos_config("pp") == _QOS_MEDIUM


def test_p_tp_suffix_uses_normalized_name():
    config = _qos_ascend_config(manual={"p_tp": "low", "p_tp_0": "high"})
    with (
        patch("vllm_ascend.utils.get_ascend_device_type", return_value=AscendDeviceType.A5),
        patch("vllm_ascend.utils.get_ascend_config", return_value=config),
    ):
        qos = get_hccl_qos_config("p_tp_0")
    assert qos["hccl_sdma_qos"] == 2
    assert qos["qos_traffic_class"] == 64


def test_mc2_qos_does_not_add_buffer_size():
    config = _qos_ascend_config()
    with (
        patch("vllm_ascend.utils.get_ascend_device_type", return_value=AscendDeviceType.A5),
        patch("vllm_ascend.utils.get_ascend_config", return_value=config),
    ):
        hccl_config = get_hccl_config_for_pg_options("mc2")
    assert hccl_config == _QOS_MEDIUM
    assert "hccl_buffer_size" not in hccl_config


def test_mc2_without_qos_stays_unset():
    with patch("vllm_ascend.utils.get_ascend_device_type", return_value=AscendDeviceType.A3):
        assert get_hccl_config_for_pg_options("mc2") is None


def test_sleep_anchor_keeps_buffer_and_can_take_qos():
    with patch("vllm_ascend.utils.get_ascend_device_type", return_value=AscendDeviceType.A3):
        buffer_only = get_hccl_config_for_pg_options(SLEEP_LIFECYCLE_ANCHOR_GROUP_NAME)
    assert buffer_only == {"hccl_buffer_size": 1}

    config = _qos_ascend_config(default_priority="high")
    with (
        patch("vllm_ascend.utils.get_ascend_device_type", return_value=AscendDeviceType.A5),
        patch("vllm_ascend.utils.get_ascend_config", return_value=config),
    ):
        with_qos = get_hccl_config_for_pg_options(SLEEP_LIFECYCLE_ANCHOR_GROUP_NAME)
    assert with_qos["hccl_buffer_size"] == 1
    assert with_qos["hccl_sdma_qos"] == 6


def test_default_world_ignores_non_hccl_backend():
    patch_distributed = _load_patch_distributed()
    original = MagicMock(return_value="pg")
    with patch.object(patch_distributed, "get_hccl_qos_config") as get_qos:
        wrapped = patch_distributed._wrap_init_process_group(original)
        assert wrapped(backend="cpu:gloo,npu:hccl") == "pg"
        assert wrapped(backend="gloo") == "pg"
    get_qos.assert_not_called()
    assert original.call_args_list[0].kwargs == {"backend": "cpu:gloo,npu:hccl"}
    assert original.call_args_list[1].kwargs == {"backend": "gloo"}


def test_default_world_hccl_without_qos_keeps_caller_options():
    patch_distributed = _load_patch_distributed()
    original = MagicMock(return_value="pg")
    with patch.object(patch_distributed, "get_hccl_qos_config", return_value={}):
        wrapped = patch_distributed._wrap_init_process_group(original)
        assert wrapped(backend="hccl") == "pg"
    original.assert_called_once_with(backend="hccl")


def test_default_world_hccl_qos_sets_name_without_buffer():
    patch_distributed = _load_patch_distributed()
    original = MagicMock(return_value="pg")
    with (
        patch.object(patch_distributed, "get_hccl_qos_config", return_value=dict(_QOS_MEDIUM)),
        patch("torch_npu._C._distributed_c10d.ProcessGroupHCCL.Options", _HcclOptions),
    ):
        wrapped = patch_distributed._wrap_init_process_group(original)
        assert wrapped(backend="hccl") == "pg"
    pg_options = original.call_args.kwargs["pg_options"]
    assert pg_options.hccl_config == {
        "group_name": "default_world",
        **_QOS_MEDIUM,
    }


def test_default_world_hccl_qos_preserves_existing_buffer():
    patch_distributed = _load_patch_distributed()
    original = MagicMock(return_value="pg")
    existing = SimpleNamespace(hccl_config={"hccl_buffer_size": 200})
    with patch.object(patch_distributed, "get_hccl_qos_config", return_value=dict(_QOS_HIGH)):
        wrapped = patch_distributed._wrap_init_process_group(original)
        assert wrapped(backend="hccl", pg_options=existing) == "pg"
    assert existing.hccl_config == {
        "hccl_buffer_size": 200,
        "group_name": "default_world",
        **_QOS_HIGH,
    }


def test_netloader_writes_group_name_even_when_qos_is_empty():
    from vllm_ascend.model_loader.netloader.executor.netloader_pg import _set_hccl_config

    pg_options = SimpleNamespace(hccl_config=None)
    with patch(
        "vllm_ascend.model_loader.netloader.executor.netloader_pg.get_hccl_qos_config",
        return_value={},
    ):
        _set_hccl_config(pg_options, "netloader")
    assert pg_options.hccl_config == {"group_name": "netloader"}


def test_netloader_qos_merges_without_adding_buffer():
    from vllm_ascend.model_loader.netloader.executor.netloader_pg import _set_hccl_config

    pg_options = SimpleNamespace(hccl_config=None)
    with patch(
        "vllm_ascend.model_loader.netloader.executor.netloader_pg.get_hccl_qos_config",
        return_value=dict(_QOS_MEDIUM),
    ):
        _set_hccl_config(pg_options, "netloader_draft")
    assert pg_options.hccl_config == {
        "group_name": "netloader_draft",
        **_QOS_MEDIUM,
    }
    assert "hccl_buffer_size" not in pg_options.hccl_config
