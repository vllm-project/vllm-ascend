# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");

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
    monkeypatch.setattr(
        ple, "torch", SimpleNamespace(npu=SimpleNamespace(config=config))
    )
    monkeypatch.setattr(
        ple,
        "torch_npu",
        SimpleNamespace(_C=SimpleNamespace(_npu_setOption=low_level_calls.append)),
    )

    ple._set_allow_internal_format(False)

    assert config.allow_internal_format is False
    assert low_level_calls == []


def test_set_internal_format_falls_back_to_low_level_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    low_level_calls: list[dict] = []
    monkeypatch.setattr(
        ple, "torch", SimpleNamespace(npu=SimpleNamespace(config=SimpleNamespace()))
    )
    monkeypatch.setattr(
        ple,
        "torch_npu",
        SimpleNamespace(_C=SimpleNamespace(_npu_setOption=low_level_calls.append)),
    )

    ple._set_allow_internal_format(False)

    assert low_level_calls == [{"ALLOW_INTERNAL_FORMAT": "disable"}]


def test_ple_internal_format_is_scoped(monkeypatch: pytest.MonkeyPatch) -> None:
    updates: list[bool] = []
    monkeypatch.setattr(ple, "_get_allow_internal_format", lambda: True)
    monkeypatch.setattr(ple, "_set_allow_internal_format", updates.append)
    output = torch.tensor([2.0])
    original = Mock(return_value=output)
    wrapped = ple._with_internal_format_disabled(original)

    assert wrapped(object(), torch.tensor([1.0])) is output
    assert updates == [False, True]


def test_ple_internal_format_is_restored_on_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    updates: list[bool] = []
    monkeypatch.setattr(ple, "_get_allow_internal_format", lambda: True)
    monkeypatch.setattr(ple, "_set_allow_internal_format", updates.append)
    wrapped = ple._with_internal_format_disabled(
        Mock(side_effect=RuntimeError("expected short-conv failure"))
    )

    with pytest.raises(RuntimeError, match="expected short-conv failure"):
        wrapped(object(), torch.tensor([1.0]))
    assert updates == [False, True]


def test_local_ple_layer_owns_internal_format_adaptation() -> None:
    assert hasattr(ple.Qwen4ExpPLELayer._short_conv, "__wrapped__")
    assert ple.Qwen4ExpPLELayer.__module__ == ple.__name__
