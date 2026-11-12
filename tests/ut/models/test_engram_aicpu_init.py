# SPDX-License-Identifier: Apache-2.0
"""AICPU startup must complete before any foreign BM mapping is constructed."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

SOURCE = Path(__file__).resolve().parents[3] / "vllm_ascend/models/deepseek_v41/engram"


@pytest.fixture
def helper(monkeypatch):
    spec = importlib.util.spec_from_file_location("_aicpu_common_test", SOURCE / "common.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    backend = ModuleType("vllm_ascend.attention.mixed_quant_sparse_flash_mla")
    op = Mock()
    monkeypatch.setattr(backend, "_get_mixed_quant_sparse_flash_mla_ops", lambda: (None, op), raising=False)
    monkeypatch.setitem(sys.modules, backend.__name__, backend)
    tensor = torch.tensor
    monkeypatch.setattr(module.torch, "tensor", lambda values, **kwargs: tensor(values, dtype=kwargs["dtype"]))
    synchronize = Mock()
    monkeypatch.setattr(torch, "npu", SimpleNamespace(synchronize=synchronize), raising=False)
    yield module.warmup_shared_engram_aicpu, op, synchronize
    module.warmup_shared_engram_aicpu.cache_clear()


def test_scheduler_is_synchronized_once_before_reuse(helper):
    warm, op, synchronize = helper
    calls = []
    op.side_effect = lambda *args, **kwargs: calls.append("metadata")
    synchronize.side_effect = lambda: calls.append("synchronize")
    warm()
    warm()
    assert calls == ["metadata", "synchronize"]
    assert op.call_args.args == (64, 1, 512, 3)
    torch.testing.assert_close(op.call_args.kwargs["seqused_ori_kv"], torch.tensor([128], dtype=torch.int32))


def test_failed_startup_is_not_cached(helper):
    warm, op, synchronize = helper
    op.side_effect = RuntimeError("scheduler startup failed")
    with pytest.raises(RuntimeError, match="scheduler startup"):
        warm()
    synchronize.assert_not_called()
    op.side_effect = None
    warm()
    assert op.call_count == 2
    synchronize.assert_called_once()
