import sys
from types import ModuleType
from unittest.mock import Mock

import pytest
import torch
from vllm.utils import torch_utils as upstream_torch_utils

from vllm_ascend.patch.platform import patch_torch_utils


def test_upstream_uva_helper_is_patched_for_npu() -> None:
    assert upstream_torch_utils.get_accelerator_view_from_cpu_tensor is patch_torch_utils.get_npu_view_from_cpu_tensor


def test_npu_uva_helper_requires_pinned_memory() -> None:
    cpu_tensor = Mock()
    cpu_tensor.is_pinned.return_value = False

    with pytest.raises(RuntimeError, match="pinned host memory"):
        patch_torch_utils.get_npu_view_from_cpu_tensor(cpu_tensor)


@pytest.mark.parametrize("view_available", [False, True])
def test_npu_uva_helper_uses_ascend_operator(monkeypatch, view_available) -> None:
    monkeypatch.setitem(sys.modules, "vllm_ascend.vllm_ascend_C", ModuleType("vllm_ascend_C"))
    cpu_tensor = Mock()
    cpu_tensor.is_pinned.return_value = True
    view = torch.empty(1) if view_available else None
    operator = Mock(return_value=view)
    monkeypatch.setattr(torch.ops._C_ascend, "get_npu_view_from_cpu_tensor", operator, raising=False)

    if view_available:
        assert patch_torch_utils.get_npu_view_from_cpu_tensor(cpu_tensor) is view
    else:
        with pytest.raises(RuntimeError, match="pinned_mem_register:True"):
            patch_torch_utils.get_npu_view_from_cpu_tensor(cpu_tensor)
    operator.assert_called_once_with(cpu_tensor)
