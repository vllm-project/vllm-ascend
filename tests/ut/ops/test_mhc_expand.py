# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from vllm_ascend.models.glm5next.ops import mhc_ops as glm_mhc
from vllm_ascend.ops import mhc


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("shape,mult", [((3, 17), 4), ((0, 17), 4), ((3, 0), 4), ((3, 17), 1)])
def test_cpu_fallback(monkeypatch, dtype, shape, mult):
    loader = MagicMock(side_effect=AssertionError("CPU input must not initialize NPU"))
    monkeypatch.setattr(mhc, "enable_custom_op", loader)
    x = torch.randn(shape, dtype=dtype)
    actual = mhc.mhc_expand(x, mult)
    torch.testing.assert_close(actual, x.unsqueeze(1).repeat(1, mult, 1), rtol=0, atol=0)
    assert actual.is_contiguous()
    loader.assert_not_called()


def test_fallback_gradient_and_noncontiguous():
    x = torch.randn(17, 3, requires_grad=True)
    y = mhc.mhc_expand(x.t(), 4)
    y.sum().backward()
    torch.testing.assert_close(x.grad, torch.full_like(x, 4))


@pytest.mark.parametrize("supported,enabled", [(False, True), (True, False), (True, True)])
def test_npu_routing(monkeypatch, supported, enabled):
    x = MagicMock()
    x.device = SimpleNamespace(type="npu")
    x.ndim = 2
    x.shape = (4, 64)
    x.dtype = torch.bfloat16
    x.requires_grad = False
    x.numel.return_value = 64
    x.is_contiguous.return_value = True
    monkeypatch.setattr(mhc, "MHC_EXPAND_SUPPORTED", supported)
    monkeypatch.setattr(mhc, "enable_custom_op", lambda: enabled)
    kernel = MagicMock()
    monkeypatch.setattr(torch.ops._C_ascend, "npu_mhc_expand_if_supported", kernel, raising=False)
    result = mhc.mhc_expand(x, 4)
    if supported and enabled:
        kernel.assert_called_once_with(x, 4)
        assert result is kernel.return_value
    else:
        kernel.assert_not_called()
        assert result is x.unsqueeze.return_value.expand.return_value.contiguous.return_value


@pytest.mark.parametrize("reason", ["dtype", "noncontiguous", "unaligned", "gradient", "empty", "trivial", "mult"])
def test_npu_fallback_preconditions(monkeypatch, reason):
    x = MagicMock()
    x.device = SimpleNamespace(type="npu")
    x.ndim = 2
    x.shape = (4, 17 if reason == "unaligned" else 64)
    x.dtype = torch.float32 if reason == "dtype" else torch.bfloat16
    x.requires_grad = reason == "gradient"
    x.numel.return_value = 0 if reason == "empty" else 64
    x.is_contiguous.return_value = reason != "noncontiguous"
    mult = {"trivial": 1, "mult": 8}.get(reason, 4)
    monkeypatch.setattr(mhc, "MHC_EXPAND_SUPPORTED", True)
    loader = MagicMock(return_value=True)
    monkeypatch.setattr(mhc, "enable_custom_op", loader)
    kernel = MagicMock(return_value=None)
    monkeypatch.setattr(torch.ops._C_ascend, "npu_mhc_expand_if_supported", kernel, raising=False)
    result = mhc.mhc_expand(x, mult)
    if reason == "gradient":
        kernel.assert_not_called()
        loader.assert_not_called()
    else:
        kernel.assert_called_once_with(x, mult)
    assert result is x.unsqueeze.return_value.expand.return_value.contiguous.return_value


def test_glm_expand_delegates_to_helper(monkeypatch):
    helper = MagicMock()
    monkeypatch.setattr(glm_mhc, "mhc_expand", helper)
    x = torch.randn(3, 64)
    assert glm_mhc.hc_expand(x, 4) is helper.return_value
    helper.assert_called_once_with(x, 4)


def test_glm_expand_contract_preserved():
    x = torch.randn(3, 64)
    expanded = glm_mhc.hc_expand(x, 4)
    torch.testing.assert_close(expanded, x.unsqueeze(1).repeat(1, 4, 1), rtol=0, atol=0)
    torch.testing.assert_close(glm_mhc.hc_contract(expanded, 4), x, rtol=0, atol=0)
