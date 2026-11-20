# SPDX-License-Identifier: Apache-2.0
"""CPU allocation/launch contract checks; no device arithmetic is emulated."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch


@pytest.fixture
def preparation(monkeypatch):
    launch = MagicMock()
    runtime = ModuleType("vllm.triton_utils")
    runtime.__dict__.update(
        tl=SimpleNamespace(constexpr=object),
        triton=SimpleNamespace(jit=lambda fn: launch, cdiv=lambda a, b: (a + b - 1) // b),
    )
    monkeypatch.setitem(sys.modules, runtime.__name__, runtime)
    source = Path(__file__).resolve().parents[3] / "vllm_ascend/ops/triton/qwen4_exp/lightning_v2_prepare.py"
    spec = importlib.util.spec_from_file_location("qsa_v2_prepare_contract", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.prepare_qsa_lightning_v2_inputs, launch


@pytest.mark.parametrize("rows,requests", [(0, 0), (1, 1), (32, 5), (128, 33)])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_single_launch_preserves_metadata_dtype_and_strides(preparation, rows, requests, dtype):
    prepare, launch = preparation
    query = torch.empty(rows, 4, 128, dtype=torch.bfloat16)
    lengths = torch.zeros(2 * requests, dtype=dtype)[::2]
    starts = torch.zeros(2 * (requests + 1), dtype=dtype)[::2]
    outputs = prepare(query, lengths, starts, 4)
    assert [x.shape for x in outputs] == [(rows, 4), (requests + 1,), (requests,), (requests,)]
    assert [x.dtype for x in outputs] == [torch.float32, dtype, dtype, dtype]
    assert all(x.is_contiguous() for x in outputs)
    launch.__getitem__.assert_called_once_with(((max(requests + 1, rows * 4) + 255) // 256,))
    args, kwargs = launch.__getitem__.return_value.call_args
    assert args[0] is lengths and args[1] is starts
    assert args[6:] == (rows, requests, rows * 4, 2, 2)
    assert kwargs["SCORE_WEIGHT"] == torch.tensor(1 / 128**0.5, dtype=torch.bfloat16).float().item()


@pytest.mark.parametrize("case", ["width", "ratio", "dtype", "boundaries"])
def test_rejects_invalid_contract_before_launch(preparation, case):
    prepare, launch = preparation
    query = torch.empty(1, 4, 64 if case == "width" else 128, dtype=torch.bfloat16)
    lengths = torch.zeros(1, dtype=torch.float32 if case == "dtype" else torch.int32)
    starts = torch.zeros(1 if case == "boundaries" else 2, dtype=torch.int32)
    with pytest.raises(ValueError):
        prepare(query, lengths, starts, 0 if case == "ratio" else 4)
    launch.__getitem__.assert_not_called()
