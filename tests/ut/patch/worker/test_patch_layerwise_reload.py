# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm_ascend.patch.worker import patch_layerwise_reload as reload_patch
from vllm_ascend.quantization.online_mx import AscendOnlineMXLinearMethod, AscendOnlineMXMoEMethod
from vllm_ascend.quantization.quant_type import QuantType


@pytest.mark.parametrize("shape", [(8, 4), (2, 8, 4)])
@pytest.mark.parametrize("transposed", [False, True])
def test_copy_preserves_bytes_layout_and_storage(shape, transposed, monkeypatch):
    destination = torch.zeros(shape, dtype=torch.uint8)
    source = torch.arange(destination.numel(), dtype=torch.uint8).reshape(shape)
    if transposed:
        destination = destination.transpose(-1, -2)
        source = source.transpose(-1, -2)
    pointer, stride = destination.data_ptr(), destination.stride()
    original_copy = torch.Tensor.copy_

    def copy_without_views(target, value):
        if transposed:
            assert target._base is None
            assert value._base is None
        return original_copy(target, value)

    monkeypatch.setattr(torch.Tensor, "copy_", copy_without_views)
    reload_patch._copy_mxfp4_tensor(destination, source)
    assert torch.equal(destination, source)
    assert destination.data_ptr() == pointer
    assert destination.stride() == stride


@pytest.mark.parametrize("method_class", [AscendOnlineMXLinearMethod, AscendOnlineMXMoEMethod])
def test_restore_captured_tensors_and_buffer_persistence(method_class):
    method = method_class.__new__(method_class)
    method.quant_method = SimpleNamespace(quant_type=QuantType.W4A8MXFP)
    layer = torch.nn.Module()
    layer.quant_method = method
    original = torch.nn.Parameter(torch.zeros(8, 4, dtype=torch.uint8).t(), requires_grad=False)
    layer.weight = torch.nn.Parameter(torch.ones_like(original), requires_grad=False)
    buffers = {name: torch.zeros(2) for name in ("persistent", "loaded", "scratch", "removed")}
    layer.register_buffer("persistent", torch.ones(2))
    layer.register_buffer("loaded", torch.ones(2), persistent=False)
    layer.register_buffer("scratch", torch.ones(2), persistent=False)
    info = SimpleNamespace(
        kernel_tensors=({"weight": original}, buffers),
        kernel_non_persistent_buffers={"loaded", "scratch", "removed"},
        loaded_weights=[("loaded", None)],
    )
    pointer = original.data_ptr()
    reload_patch._copy_and_restore_kernel_tensors(layer, info)
    assert layer.weight is original
    assert original.data_ptr() == pointer
    assert torch.equal(original, torch.ones_like(original))
    assert torch.equal(layer.persistent, torch.ones(2))
    assert torch.equal(layer.loaded, torch.ones(2))
    assert torch.equal(layer.scratch, torch.zeros(2))
    assert torch.equal(layer.removed, torch.zeros(2))
    assert layer._non_persistent_buffers_set == {"loaded", "scratch", "removed"}


@pytest.mark.parametrize("online", [False, True])
def test_other_methods_delegate_to_upstream(online):
    method = AscendOnlineMXLinearMethod.__new__(AscendOnlineMXLinearMethod) if online else SimpleNamespace()
    method.quant_method = SimpleNamespace(quant_type=QuantType.W8A8MXFP)
    layer, info = SimpleNamespace(quant_method=method), object()
    with patch.object(reload_patch, "_original_copy_and_restore") as upstream:
        reload_patch._copy_and_restore_kernel_tensors(layer, info)
    upstream.assert_called_once_with(layer, info)
