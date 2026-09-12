# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import json
from types import SimpleNamespace
from unittest import mock

import pytest

from vllm_ascend import utils


@pytest.fixture
def physical_device_lookup():
    with mock.patch("vllm.platforms.current_platform") as platform:
        yield platform.visible_device_id_to_physical_device_id


@pytest.mark.parametrize(
    "visible_devices,user_device_id,physical_device_id",
    [(None, 1, 1), ("5,4", 0, 5), ("5,4", 1, 4), (None, 0, 4), ("1,0", 0, 5)],
)
def test_endpoint_uses_cann_physical_id(
    tmp_path, monkeypatch, physical_device_lookup, visible_devices, user_device_id, physical_device_id
):
    # Include container remapping and visibility reordering. The helper must
    # use CANN's result rather than indexing the environment or using the rank.
    if visible_devices is None:
        monkeypatch.delenv("ASCEND_RT_VISIBLE_DEVICES", raising=False)
    else:
        monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", visible_devices)
    monkeypatch.setenv("ASCEND_LOCAL_COMM_RES", "previous")
    expected = {"endpoint": f"physical-{physical_device_id}"}
    (tmp_path / f"ub_endpoint_npu_{physical_device_id}.json").write_text(json.dumps(expected))
    config = SimpleNamespace(kv_connector_extra_config={"ascend_local_comm_res_path": str(tmp_path)})
    physical_device_lookup.return_value = physical_device_id

    utils.setup_ascend_local_comm_res(user_device_id, config)

    physical_device_lookup.assert_called_once_with(user_device_id)
    assert json.loads(utils.os.environ["ASCEND_LOCAL_COMM_RES"]) == expected


def test_endpoint_mapping_failure_does_not_fall_back(tmp_path, monkeypatch, physical_device_lookup):
    monkeypatch.setenv("ASCEND_LOCAL_COMM_RES", "previous")
    (tmp_path / "ub_endpoint_npu_0.json").write_text('{"wrong": true}')
    config = SimpleNamespace(kv_connector_extra_config={"ascend_local_comm_res_path": str(tmp_path)})
    physical_device_lookup.side_effect = RuntimeError("aclrtGetPhyDevIdByUserDevId failed")

    with pytest.raises(RuntimeError, match="aclrtGetPhyDevIdByUserDevId failed"):
        utils.setup_ascend_local_comm_res(0, config)

    assert utils.os.environ["ASCEND_LOCAL_COMM_RES"] == "previous"


@pytest.mark.parametrize("config", [None, SimpleNamespace(kv_connector_extra_config={})])
def test_no_endpoint_path_does_not_resolve_device(config, physical_device_lookup):
    utils.setup_ascend_local_comm_res(0, config)
    physical_device_lookup.assert_not_called()
