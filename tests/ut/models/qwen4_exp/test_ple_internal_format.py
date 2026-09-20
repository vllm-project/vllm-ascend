# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from unittest.mock import Mock

import pytest
import torch

from vllm_ascend.models.qwen4_exp import ple


def test_ple_internal_format_is_scoped(monkeypatch: pytest.MonkeyPatch) -> None:
    updates: list[bool] = []
    monkeypatch.setattr(ple, "_get_allow_internal_format", lambda: True)
    monkeypatch.setattr(ple, "_set_allow_internal_format", updates.append)
    output = torch.tensor([2.0])
    original = Mock(return_value=output)
    layer = object()
    inputs = torch.tensor([1.0])

    wrapped = ple._wrap_ple_short_conv(original)

    assert wrapped(layer, inputs) is output
    original.assert_called_once_with(layer, inputs)
    assert updates == [False, True]


def test_ple_internal_format_is_restored_on_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    updates: list[bool] = []
    monkeypatch.setattr(ple, "_get_allow_internal_format", lambda: True)
    monkeypatch.setattr(ple, "_set_allow_internal_format", updates.append)

    original = Mock(side_effect=RuntimeError("expected short-conv failure"))
    layer = object()
    inputs = torch.tensor([1.0])
    wrapped = ple._wrap_ple_short_conv(original)

    with pytest.raises(RuntimeError, match="expected short-conv failure"):
        wrapped(layer, inputs)
    original.assert_called_once_with(layer, inputs)
    assert updates == [False, True]


def test_installing_ple_patch_is_idempotent() -> None:
    from vllm.models.qwen4_exp.amd.ple_layer import Qwen4ExpPLELayer

    original_attr = "_ascend_original_short_conv"
    initial_method = Qwen4ExpPLELayer._short_conv
    had_original = hasattr(Qwen4ExpPLELayer, original_attr)
    initial_original = getattr(Qwen4ExpPLELayer, original_attr, None)

    def original(_layer: object, inputs: torch.Tensor) -> torch.Tensor:
        return inputs

    try:
        if had_original:
            delattr(Qwen4ExpPLELayer, original_attr)
        Qwen4ExpPLELayer._short_conv = original
        ple.patch_upstream_ple_short_conv()
        installed = Qwen4ExpPLELayer._short_conv
        ple.patch_upstream_ple_short_conv()

        assert Qwen4ExpPLELayer._short_conv is installed
        assert getattr(Qwen4ExpPLELayer, original_attr) is original
    finally:
        Qwen4ExpPLELayer._short_conv = initial_method
        if had_original:
            setattr(Qwen4ExpPLELayer, original_attr, initial_original)
        elif hasattr(Qwen4ExpPLELayer, original_attr):
            delattr(Qwen4ExpPLELayer, original_attr)
