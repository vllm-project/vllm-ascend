import logging
from unittest.mock import MagicMock

import pytest

from vllm_ascend import _build_info
from vllm_ascend.device.device_config import (
    check_ascend_device_type,
    get_ascend_device_type,
    get_device_config,
)
from vllm_ascend.device.hardware import (
    CPU_ONLY_FALLBACK_SOC_VERSION,
    AscendDeviceType,
    device_type_from_runtime_soc,
    device_type_from_soc_version,
    resolve_build_soc_version,
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
        ("ASCEND910_9363", AscendDeviceType.A3),
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
        (256, AscendDeviceType.A3),
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


@pytest.mark.parametrize("soc_version", [257, 999])
def test_unknown_runtime_soc_version_is_rejected(soc_version):
    with pytest.raises(RuntimeError, match="Cannot support runtime soc_version"):
        device_type_from_runtime_soc(soc_version)


def test_detected_soc_version_wins_over_the_fallback():
    """An NPU is present: npu-smi's output is preferred regardless of the
    value of ``COMPILE_CUSTOM_KERNELS``.
    """
    for compile_custom_kernels in (True, False):
        assert resolve_build_soc_version("ascend910_9391", compile_custom_kernels) == "ascend910_9391"


def test_missing_soc_version_still_fails_a_kernel_build():
    """npu-smi should be installed and return a chip type if
    COMPILE_CUSTOM_KERNELS is true; raise a runtime error if
    this is not the behavior.
    """
    with pytest.raises(RuntimeError, match="Could not determine chip type"):
        resolve_build_soc_version("", compile_custom_kernels=True)


def test_missing_soc_version_falls_back_without_custom_kernels(caplog):
    """Check that SOC version resolves to a dummy value when COMPILE_CUSTOM_KERNELS
    is False and npu-smi is not installed (CPU-only scenario); check that
    warning is logged that mentions the dummy SOC version and the
    COMPILE_CUSTOM_KERNELS flag.
    """
    with caplog.at_level(logging.WARNING):
        assert resolve_build_soc_version("", compile_custom_kernels=False) == CPU_ONLY_FALLBACK_SOC_VERSION

    assert CPU_ONLY_FALLBACK_SOC_VERSION in caplog.text
    assert "COMPILE_CUSTOM_KERNELS=0" in caplog.text


def test_fallback_soc_version_is_valid():
    """Check that the dummy SOC version for CPU-only scenarios resolves to a real
    device.
    """
    assert device_type_from_soc_version(CPU_ONLY_FALLBACK_SOC_VERSION) is AscendDeviceType.A2
