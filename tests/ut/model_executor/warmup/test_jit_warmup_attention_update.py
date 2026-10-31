# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Warmup coverage for full and residual MLA prefill merges."""

from unittest.mock import MagicMock, Mock, patch

import pytest
import torch
from vllm.triton_utils import HAS_TRITON

from tests.ut.model_executor.warmup.helpers import make_registry
from vllm_ascend.ops.triton import attention_update as au

pytestmark = pytest.mark.skipif(not HAS_TRITON, reason="triton is not installed")


def test_keys_cover_production_merges():
    keys = au._PREPARE_ATTENTION_UPDATE_KERNEL.get_warmup_keys()
    assert len(keys) == 16
    assert {(key.count, key.has_anchor) for key in keys} == {(16, False)} | {(count, True) for count in range(2, 17)}


def test_registration_is_scoped_and_deduplicated():
    registry = make_registry()
    kernel = au._PREPARE_ATTENTION_UPDATE_KERNEL
    kernel.register_warmup()
    assert len(registry) == 0
    with registry.activate():
        kernel.register_warmup()
        kernel.register_warmup()
    assert len(registry) == 1
    with patch.object(kernel, "compile_many") as compile_many:
        registry.warmup()
    assert list(compile_many.call_args.args[0]) == kernel.get_warmup_keys()


@pytest.mark.parametrize("count,has_anchor", [(16, False), *[(count, True) for count in range(2, 17)]])
def test_warmup_matches_runtime_launch(monkeypatch, count, has_anchor):
    kernel = au.PrepareAttentionUpdateKernel()
    launch = Mock()
    monkeypatch.setattr(kernel, "launch", launch)
    key = kernel.dispatch(count=count, has_anchor=has_anchor)
    kernel(**kernel.warmup_inputs(key))
    warm_grid, warm_args = launch.call_args.args[0]

    rows = 129
    lses = [torch.zeros(rows, dtype=torch.float32) for _ in range(count)]
    previous_anchor = torch.zeros(rows) if has_anchor else None
    runtime_kernel = MagicMock()
    monkeypatch.setattr(au, "_prepare_attention_update_kernel", runtime_kernel)
    au.prepare_attention_update(lses, previous_anchor)
    grid = runtime_kernel.__getitem__.call_args.args[0]
    args = dict(zip(kernel.kernel.arg_names, runtime_kernel.__getitem__.return_value.call_args.args))
    assert warm_grid == (1,)
    assert grid == (2,)
    for name in ("COUNT", "HAS_ANCHOR", "Strides", "BLOCK"):
        assert warm_args[name] == args[name]
    assert args["COUNT"] == count
    assert args["HAS_ANCHOR"] == has_anchor
    assert args["Strides"] == (1,) * count
    assert args["ROWS"] == rows
    assert warm_args["ROWS"] == kernel.BLOCK
    assert args["PreviousAnchor"] is previous_anchor
    assert (warm_args["PreviousAnchor"] is None) == (previous_anchor is None)
    for i in range(16):
        name = f"L{i}"
        assert args[name] is (lses[i] if i < count else lses[0])
        assert warm_args[name].dtype == args[name].dtype == torch.float32


@pytest.mark.parametrize("has_anchor", [False, True])
def test_empty_rows_do_not_launch(monkeypatch, has_anchor):
    runtime_kernel = MagicMock()
    monkeypatch.setattr(au, "_prepare_attention_update_kernel", runtime_kernel)
    lses = [torch.empty(0) for _ in range(16)]
    relative, anchor = au.prepare_attention_update(lses, torch.empty(0) if has_anchor else None)
    assert len(relative) == 16
    assert all(value.shape == (0,) for value in (*relative, anchor))
    runtime_kernel.__getitem__.assert_not_called()
