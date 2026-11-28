from unittest.mock import MagicMock

import pytest
import torch

from vllm_ascend.patch.platform.patch_torch_accelerator import _get_npu_memory_info


def test_torch_accelerator_get_memory_info_is_patched():
    assert torch.accelerator.get_memory_info is _get_npu_memory_info


@pytest.mark.parametrize(
    ("device", "expected_device"),
    [
        (3, 3),
        ("npu:2", 2),
        (torch.device("npu:1"), 1),
    ],
)
def test_get_npu_memory_info_uses_requested_device(monkeypatch, device, expected_device):
    mem_get_info = MagicMock(return_value=(1024, 2048))
    monkeypatch.setattr(torch.npu, "mem_get_info", mem_get_info)

    assert _get_npu_memory_info(device) == (1024, 2048)
    mem_get_info.assert_called_once_with(expected_device)


@pytest.mark.parametrize("device", ["cuda:0", torch.device("cpu")])
def test_get_npu_memory_info_rejects_non_npu_devices(device):
    with pytest.raises(RuntimeError, match="Expected 'npu'"):
        _get_npu_memory_info(device)


def test_get_npu_memory_info_uses_current_device(monkeypatch):
    monkeypatch.setattr(torch.npu, "current_device", MagicMock(return_value=4))
    mem_get_info = MagicMock(return_value=(1024, 2048))
    monkeypatch.setattr(torch.npu, "mem_get_info", mem_get_info)

    assert _get_npu_memory_info() == (1024, 2048)
    mem_get_info.assert_called_once_with(4)
