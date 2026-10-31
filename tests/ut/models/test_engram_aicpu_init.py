# SPDX-License-Identifier: Apache-2.0
"""AICPU startup must complete before any foreign BM mapping is constructed."""

import ast
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any
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


@pytest.mark.parametrize("failed", [False, True])
def test_real_allocator_orders_startup_before_client_creation(monkeypatch, failed):
    calls = []
    package = ModuleType("_allocator_order_test")
    package.__path__ = []
    monkeypatch.setitem(sys.modules, package.__name__, package)
    common, pool = ModuleType(package.__name__ + ".common"), ModuleType(package.__name__ + ".shared_pool")

    def warm():
        calls.append("aicpu")
        if failed:
            raise RuntimeError("scheduler startup failed")

    class Client:
        def __init__(self, *args, **kwargs):
            calls.append("client")

        def view(self, key, *args, **kwargs):
            calls.append(key)
            return SimpleNamespace(tensor=object())

        def close(self):
            calls.append("close")

    monkeypatch.setattr(common, "warmup_shared_engram_aicpu", warm, raising=False)
    monkeypatch.setattr(pool, "SharedPoolClient", Client, raising=False)
    monkeypatch.setitem(sys.modules, common.__name__, common)
    monkeypatch.setitem(sys.modules, pool.__name__, pool)
    owner = next(
        n
        for n in ast.parse((SOURCE / "embedding.py").read_text()).body
        if isinstance(n, ast.ClassDef) and n.name == "AscendParallelEngramEmbedding"
    )
    method = next(n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name == "_allocate_weights")
    tree = ast.Module(body=[ast.parse("from __future__ import annotations").body[0], method], type_ignores=[])
    fake_torch = SimpleNamespace(
        device=lambda *args: "cpu",
        npu=SimpleNamespace(current_device=lambda: 0),
        bfloat16=torch.bfloat16,
        int8=torch.int8,
        float32=torch.float32,
    )
    namespace: dict[str, Any] = dict(__package__=package.__name__, torch=fake_torch)
    exec(compile(ast.fix_missing_locations(tree), "embedding.py", "exec"), namespace)
    embedding = SimpleNamespace(
        part_num_embeddings=16,
        num_embeddings=16,
        vocab_start_idx=0,
        vocab_end_idx=16,
        dim=256,
        block_size=32,
        storage_dtype=torch.int8,
        _shared_pool_spec=("pool", "identity", "table"),
    )
    if failed:
        with pytest.raises(RuntimeError, match="scheduler startup"):
            namespace["_allocate_weights"](embedding)
        assert calls == ["aicpu"]
    else:
        namespace["_allocate_weights"](embedding)
        assert calls == ["aicpu", "client", "table.weight", "table.scale"]
