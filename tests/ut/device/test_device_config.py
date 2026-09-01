from unittest.mock import MagicMock

import pytest

from vllm_ascend import _build_info
from vllm_ascend.device.device_config import (
    check_ascend_device_type,
    get_ascend_device_type,
    get_device_config,
)
from vllm_ascend.device.hardware import (
    AscendDeviceType,
    device_type_from_runtime_soc,
    device_type_from_soc_version,
)


@pytest.fixture(autouse=True)
def clear_device_config_cache():
    get_device_config.cache_clear()
    yield
    get_device_config.cache_clear()


@pytest.mark.parametrize(
    ("soc_version", "expected"),
    [
        ("ascend910b1", AscendDeviceType.A2),
        ("ASCEND910_9391", AscendDeviceType.A3),
        ("ascend310p3vir08", AscendDeviceType._310P),
        ("ascend950_9599", AscendDeviceType.A5),
    ],
)
def test_device_type_from_soc_version(soc_version, expected):
    assert device_type_from_soc_version(soc_version) is expected


@pytest.mark.parametrize(
    ("soc_version", "expected"),
    [
        (220, AscendDeviceType.A2),
        (255, AscendDeviceType.A3),
        (203, AscendDeviceType._310P),
        (260, AscendDeviceType.A5),
    ],
)
def test_device_type_from_runtime_soc(soc_version, expected):
    assert device_type_from_runtime_soc(soc_version) is expected


def test_device_config_uses_build_info(monkeypatch):
    monkeypatch.setattr(_build_info, "__device_type__", "A3")

    assert get_ascend_device_type() is AscendDeviceType.A3


def test_device_config_supports_legacy_soc_build_info(monkeypatch):
    monkeypatch.delattr(_build_info, "__device_type__", raising=False)
    monkeypatch.setattr(_build_info, "__soc_version__", "Ascend310P3", raising=False)

    assert get_ascend_device_type() is AscendDeviceType._310P


def test_import_time_config_does_not_probe_runtime(monkeypatch):
    import torch_npu

    monkeypatch.setattr(_build_info, "__device_type__", "A2")
    get_soc_version = MagicMock(side_effect=AssertionError("runtime probe is not import-safe"))
    monkeypatch.setattr(torch_npu.npu, "get_soc_version", get_soc_version)

    assert get_ascend_device_type() is AscendDeviceType.A2
    get_soc_version.assert_not_called()


def test_runtime_device_match_succeeds(monkeypatch):
    import torch_npu

    monkeypatch.setattr(_build_info, "__device_type__", "A2")
    monkeypatch.setattr(torch_npu.npu, "get_soc_version", lambda: 220)

    check_ascend_device_type()


def test_runtime_device_mismatch_raises_runtime_error(monkeypatch):
    import torch_npu

    monkeypatch.setattr(_build_info, "__device_type__", "A2")
    monkeypatch.setattr(torch_npu.npu, "get_soc_version", lambda: 250)

    with pytest.raises(RuntimeError, match="does not match"):
        check_ascend_device_type()


@pytest.mark.parametrize("soc_version", ["unknown", "ascend910x"])
def test_unknown_build_soc_version_is_rejected(soc_version):
    with pytest.raises(RuntimeError, match="Undefined soc_version"):
        device_type_from_soc_version(soc_version)


def test_unknown_runtime_soc_version_is_rejected():
    with pytest.raises(RuntimeError, match="Cannot support runtime soc_version"):
        device_type_from_runtime_soc(999)
