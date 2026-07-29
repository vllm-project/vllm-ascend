# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm_ascend.distributed.weight_transfer import device_mapping


def test_parse_visible_devices_returns_none_for_unset_value():
    assert device_mapping.parse_visible_devices(None) is None
    assert device_mapping.parse_visible_devices("") is None


def test_parse_visible_devices_accepts_spaces():
    assert device_mapping.parse_visible_devices("2, 4,7") == [2, 4, 7]


def test_parse_visible_devices_rejects_empty_token():
    with pytest.raises(ValueError, match="Invalid ASCEND_RT_VISIBLE_DEVICES"):
        device_mapping.parse_visible_devices("2,,4")


def test_parse_visible_devices_rejects_non_integer_token():
    with pytest.raises(ValueError, match="Invalid device id"):
        device_mapping.parse_visible_devices("2,npu4")


def test_logical_to_physical_device_id_uses_identity_without_visible_devices():
    assert device_mapping.logical_to_physical_device_id(3) == 3


def test_logical_to_physical_device_id_uses_visible_device_mapping():
    assert device_mapping.logical_to_physical_device_id(0, "2,3") == 2
    assert device_mapping.logical_to_physical_device_id(1, "2,3") == 3


def test_logical_to_physical_device_id_rejects_out_of_bounds_index():
    with pytest.raises(ValueError, match="out of bounds"):
        device_mapping.logical_to_physical_device_id(2, "2,3")


def test_get_npu_ipc_uuid_uses_host_and_physical_device(monkeypatch):
    device_mapping.get_npu_ipc_uuid.cache_clear()
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "4,7")

    assert device_mapping.get_npu_ipc_uuid(host_ip="1.2.3.4", logical_device=1) == "1.2.3.4-7"
