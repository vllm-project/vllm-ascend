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

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm_ascend.models.qwen4_exp import ple


def test_set_internal_format_prefers_high_level_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = SimpleNamespace(allow_internal_format=True)
    low_level_calls: list[dict] = []
    monkeypatch.setattr(ple, "torch", SimpleNamespace(npu=SimpleNamespace(config=config)))
    monkeypatch.setattr(ple, "torch_npu", SimpleNamespace(_C=SimpleNamespace(_npu_setOption=low_level_calls.append)))

    ple._set_allow_internal_format(False)

    assert config.allow_internal_format is False
    assert low_level_calls == []


def test_set_internal_format_falls_back_to_low_level_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # torch_npu 2.10 (910B): torch.npu.config exists but carries no attribute.
    low_level_calls: list[dict] = []
    monkeypatch.setattr(ple, "torch", SimpleNamespace(npu=SimpleNamespace(config=SimpleNamespace())))
    monkeypatch.setattr(ple, "torch_npu", SimpleNamespace(_C=SimpleNamespace(_npu_setOption=low_level_calls.append)))

    ple._set_allow_internal_format(False)

    assert low_level_calls == [{"ALLOW_INTERNAL_FORMAT": "disable"}]


def test_set_internal_format_without_any_api_only_warns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ple, "torch", SimpleNamespace(npu=SimpleNamespace()))
    monkeypatch.setattr(ple, "torch_npu", SimpleNamespace(_C=SimpleNamespace()))

    ple._set_allow_internal_format(False)  # must not raise


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
