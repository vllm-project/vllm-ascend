# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Resource capability/fallback checks; no NPU or vLLM import required."""

import runpy
import sys
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

API = runpy.run_path(str(Path(__file__).resolve().parents[3] / "vllm_ascend/attention/flashmla_core_budget.py"))


@pytest.mark.parametrize(
    "cube,vector,expected",
    [
        (28, 56, (14, 28, 28)),
        (32, 64, (16, 32, 32)),
        (20, 40, (10, 20, 20)),
        (17, 35, (8, 16, 19)),
        (28, 20, (5, 10, 10)),
        (1, 3, (1, 2, 1)),
        (1, 2, None),
        (0, 56, None),
    ],
)
def test_resource_split(cube, vector, expected):
    assert API["split_core_budget"](cube, vector) == expected


@pytest.fixture
def npu(monkeypatch):
    parent, dedicated = object(), object()
    limits = {parent: dict(cube_core_num=18, vector_core_num=36)}

    def set_limit(stream, cube_num, vector_num):
        limits[stream] = dict(cube_core_num=cube_num, vector_core_num=vector_num)

    runtime = SimpleNamespace(
        device=lambda _: nullcontext(),
        current_device=lambda: 0,
        get_device_properties=Mock(return_value=SimpleNamespace(cube_core_num=28, vector_core_num=56)),
        get_device_limit=Mock(return_value=dict(cube_core_num=20, vector_core_num=40)),
        get_stream_limit=Mock(side_effect=lambda stream: limits[stream]),
        current_stream=lambda: parent,
        Stream=Mock(return_value=dedicated),
        set_stream_limit=Mock(side_effect=set_limit),
        reset_stream_limit=Mock(side_effect=lambda stream: limits.pop(stream, None)),
    )
    monkeypatch.setattr(torch, "npu", runtime, raising=False)
    utils = ModuleType("vllm_ascend.utils")
    utils.is_950 = lambda: True
    monkeypatch.setitem(sys.modules, "vllm_ascend.utils", utils)
    return runtime, parent, dedicated, limits, utils


def test_respects_application_limits_without_mutating_parent(npu):
    runtime, parent, dedicated, limits, _ = npu
    budget = API["create_flashmla_core_budget"](torch.device("cpu"))
    assert budget.cache_key == (9, 18, 18)
    assert budget.stream is dedicated
    assert limits[parent] == dict(cube_core_num=18, vector_core_num=36)
    runtime.set_stream_limit.assert_called_once_with(dedicated, cube_num=9, vector_num=18)


@pytest.mark.parametrize(
    "failure", ["unsupported", "missing_api", "query", "set", "readback", "invalid_count", "missing_count"]
)
def test_fallback_is_atomic_and_does_not_touch_shared_stream(npu, failure):
    runtime, parent, dedicated, limits, utils = npu
    if failure == "unsupported":
        utils.is_950 = lambda: False
    elif failure == "missing_api":
        del runtime.get_device_limit
    elif failure == "query":
        runtime.get_device_limit.side_effect = RuntimeError("unsupported runtime")
    elif failure == "set":
        runtime.set_stream_limit.side_effect = RuntimeError("unsupported limit")
    elif failure == "readback":
        runtime.get_stream_limit.side_effect = lambda _: limits[parent]
    elif failure == "missing_count":
        runtime.get_device_limit.return_value = {}
    else:
        runtime.get_device_limit.return_value["vector_core_num"] = 0
    assert API["create_flashmla_core_budget"](torch.device("cpu")) is None
    assert limits[parent] == dict(cube_core_num=18, vector_core_num=36)
    assert all(call.args[0] is dedicated for call in runtime.set_stream_limit.call_args_list)


@pytest.mark.parametrize("cube,vector", [(28, 56), (32, 64), (20, 40)])
def test_pack_work_gate_scales_with_exposed_resources(cube, vector):
    aic, aiv, pack = API["split_core_budget"](cube, vector)
    budget = API["FlashMLACoreBudget"](object(), aic, aiv, pack)
    assert budget.allows_pack(vector * 16)
    assert not budget.allows_pack(vector * 16 + 1)
