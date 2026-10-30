# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Private outputs must be owned before a launcher can partially submit.

Only the dispatch boundary is replaced here. Real device enqueue and event
completion require the corresponding NPU tests.
"""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


class DispatchFailure(RuntimeError):
    pass


class RetentionFailure(RuntimeError):
    pass


@pytest.fixture
def operators(monkeypatch):
    fake_triton = SimpleNamespace(
        jit=lambda function: function,
        cdiv=lambda numerator, divisor: (numerator + divisor - 1) // divisor,
        next_power_of_2=lambda value: 1 << (value - 1).bit_length(),
    )
    monkeypatch.setitem(
        sys.modules, "vllm.triton_utils", SimpleNamespace(triton=fake_triton, tl=SimpleNamespace(constexpr=int))
    )
    root = Path(__file__).resolve().parents[4]
    path = root / "vllm_ascend/ops/triton/v2/spec_decode/draft_window.py"
    spec = importlib.util.spec_from_file_location("draft_window_submission_contract", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def invoke(operators, operation, retain):
    if operation == "plan":
        return operators.plan_draft_window(
            torch.arange(8, dtype=torch.int32).reshape(1, 8),
            torch.tensor([0]),
            torch.tensor([17], dtype=torch.int32),
            None,
            block_span=4,
            anchor_span=4,
            window=8,
            capacity=12,
            query_width=3,
            causal=True,
            retain=retain,
        )
    if operation == "gather":
        return operators.gather_draft_kv_fp32(
            torch.tensor([[1, 2, 3]]), torch.empty(2, 4, 1, 8), torch.empty(2, 4, 1, 8), retain=retain
        )
    return operators.pack_draft_query_indices(torch.tensor([0, 2, 5]), 5, 3, retain=retain)


@pytest.mark.parametrize("operation,outputs", [("plan", 4), ("gather", 2), ("pack", 2)])
def test_outputs_are_retained_before_dispatch_can_raise(monkeypatch, operators, operation, outputs):
    held: list[torch.Tensor] = []

    class PartialDispatch:
        def __getitem__(self, grid):
            def launch(*args):
                assert len(held) >= outputs
                # The output arguments belong to this real call, rather than
                # a previously retained allocation with an equal shape.
                expected = args[7:11] if operation == "plan" else args[3:5] if operation == "gather" else args[1:3]
                assert all(any(value is output for value in held) for output in expected)
                raise DispatchFailure("submission can fail after enqueue")

            return launch

    kernel = {"plan": "_plan_draft_window", "gather": "_gather_draft_kv_fp32", "pack": "_pack_draft_query_indices"}
    monkeypatch.setattr(operators, kernel[operation], PartialDispatch())
    with pytest.raises(DispatchFailure):
        invoke(operators, operation, lambda *values: held.extend(values))
    assert len([value for value in held if isinstance(value, torch.Tensor)]) >= outputs


@pytest.mark.parametrize("operation", ["plan", "gather", "pack"])
def test_retention_failure_prevents_dispatch(monkeypatch, operators, operation):
    calls = []

    class Dispatch:
        def __getitem__(self, grid):
            calls.append(grid)
            return lambda *args: None

    def retain(*values):
        raise RetentionFailure("owner cannot accept new storage")

    kernel = {"plan": "_plan_draft_window", "gather": "_gather_draft_kv_fp32", "pack": "_pack_draft_query_indices"}
    monkeypatch.setattr(operators, kernel[operation], Dispatch())
    with pytest.raises(RetentionFailure):
        invoke(operators, operation, retain)
    assert calls == []
