# SPDX-License-Identifier: Apache-2.0
"""Backend dispatch tests using the real model initialization statements."""

import ast
import sys
import types
from pathlib import Path
from typing import Any

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3] / "vllm_ascend/models/deepseek_v41"


@pytest.mark.parametrize("backend", [None, "a2", "a3", "a5"])
def test_model_selects_exactly_one_backend(monkeypatch, backend):
    module = ast.parse((ROOT / "model.py").read_text())
    cls = next(n for n in module.body if isinstance(n, ast.ClassDef) and n.name == "DeepseekV41Model")
    init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    start = next(
        i
        for i, n in enumerate(init.body)
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "shared_config" for t in n.targets)
    )
    end = next(i for i in range(start, len(init.body)) if isinstance(init.body[i], ast.With))
    tree = ast.fix_missing_locations(ast.Module(body=init.body[start:end], type_ignores=[]))
    calls = []
    package = types.ModuleType("_unified_test")
    package.__path__ = []
    engram = types.ModuleType("_unified_test.engram")
    engram.__path__ = []
    rdma = types.ModuleType("_unified_test.engram.shared_rdma")
    pool = types.ModuleType("_unified_test.engram.shared_pool")

    def create_reader(*args, **kwargs):
        calls.append("hixl")
        return "reader"

    def identity(*args):
        calls.append("gva")
        return "identity"

    monkeypatch.setattr(rdma, "RemoteReader", create_reader, raising=False)
    monkeypatch.setattr(rdma, "capacity_for_batch", lambda *args: 32, raising=False)
    monkeypatch.setattr(pool, "checkpoint_identity", identity, raising=False)
    for name, item in [
        (package.__name__, package),
        (engram.__name__, engram),
        (rdma.__name__, rdma),
        (pool.__name__, pool),
    ]:
        monkeypatch.setitem(sys.modules, name, item)
    config = (
        None
        if backend is None
        else types.SimpleNamespace(
            backend=backend,
            endpoint="owner:29500",
            capacity=32,
            library_path=None,
            listen_port=None,
            local_comm_res=None,
        )
    )
    model = types.SimpleNamespace(
        engram_layout=types.SimpleNamespace(primes=[[[1]]]), engram_dp_shared_memory=True, engram_weight_root="model"
    )
    vc = types.SimpleNamespace(
        load_config=types.SimpleNamespace(load_format="auto", safetensors_load_strategy="lazy"),
        scheduler_config=types.SimpleNamespace(max_num_batched_tokens=1),
        compilation_config=types.SimpleNamespace(max_cudagraph_capture_size=1),
    )
    ns = dict(
        __package__=package.__name__,
        self=model,
        vllm_config=vc,
        cpu_offload=True,
        get_ascend_config=lambda: types.SimpleNamespace(shared_engram=config),
        get_tensor_model_parallel_world_size=lambda: 1,
    )
    exec(compile(tree, str(ROOT / "model.py"), "exec"), ns)
    assert calls == ({None: [], "a2": ["hixl"], "a3": ["gva"], "a5": ["hixl"]}[backend])
    assert (model._engram_remote_reader is not None) == (backend in ("a2", "a5"))
    assert (ns["shared_pool_endpoint"] is not None) == (backend == "a3")


def test_embedding_rejects_simultaneous_backends():
    module = ast.parse((ROOT / "engram/embedding.py").read_text())
    cls = next(n for n in module.body if isinstance(n, ast.ClassDef) and n.name == "AscendParallelEngramEmbedding")
    init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    tree = ast.fix_missing_locations(
        ast.Module(body=[ast.parse("from __future__ import annotations").body[0], init], type_ignores=[])
    )
    ns: dict[str, Any] = {"torch": torch}
    exec(compile(tree, str(ROOT / "engram/embedding.py"), "exec"), ns)
    with pytest.raises(ValueError, match="mutually exclusive"):
        ns["__init__"](
            types.SimpleNamespace(), 32, 256, (32,), 0, remote_reader=object(), shared_pool_endpoint="owner:1"
        )


@pytest.mark.parametrize("backend", [None, "a2", "a3", "a5"])
def test_checkpoint_loading_routes_by_backend(backend):
    module = ast.parse((ROOT / "engram/embedding.py").read_text())
    cls = next(n for n in module.body if isinstance(n, ast.ClassDef) and n.name == "AscendParallelEngramEmbedding")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "load_checkpoint")
    tree = ast.fix_missing_locations(
        ast.Module(body=[ast.parse("from __future__ import annotations").body[0], method], type_ignores=[])
    )
    ns: dict[str, Any] = {}
    exec(compile(tree, str(ROOT / "engram/embedding.py"), "exec"), ns)
    calls = []
    embedding = types.SimpleNamespace(
        _remote_reader=object() if backend in ("a2", "a5") else None,
        _shared_pool_client=object() if backend == "a3" else None,
        _shared_pool_spec=(None, None, "layers.1.engram.embed"),
        _shared_group=None,
        _load_into_storage=lambda *args: calls.append(args),
    )
    ns["load_checkpoint"](embedding, "model", "layers.1.engram.embed.weight")
    assert bool(calls) == (backend is None)
    if backend == "a3":
        with pytest.raises(ValueError, match="checkpoint key mismatch"):
            ns["load_checkpoint"](embedding, "model", "layers.2.engram.embed.weight")


@pytest.mark.parametrize("backend", [None, "a2", "a3", "a5"])
def test_checkpoint_filter_only_enabled_for_hixl(backend):
    module = ast.parse((ROOT / "model.py").read_text())
    cls = next(n for n in module.body if isinstance(n, ast.ClassDef) and n.name == "AscendDeepseekV41LLMForCausalLM")
    init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    assignment = next(
        n
        for n in init.body
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Attribute) and t.attr == "filter_unused_checkpoint_weights" for t in n.targets)
    )
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "is_unused_checkpoint_weight")
    shared_assignment = next(
        n
        for n in init.body
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "shared_config" for t in n.targets)
    )
    tree = ast.fix_missing_locations(
        ast.Module(
            body=[ast.parse("from __future__ import annotations").body[0], shared_assignment, assignment, method],
            type_ignores=[],
        )
    )
    shared = None if backend is None else types.SimpleNamespace(backend=backend)
    model = types.SimpleNamespace(config=types.SimpleNamespace(engram_layer_ids=[1]))
    ns: dict[str, Any] = dict(self=model, get_ascend_config=lambda: types.SimpleNamespace(shared_engram=shared))
    exec(compile(tree, str(ROOT / "model.py"), "exec"), ns)
    assert ns["is_unused_checkpoint_weight"](model, "model.layers.1.engram.embed.weight") == (backend in ("a2", "a5"))
    assert not ns["is_unused_checkpoint_weight"](model, "model.layers.2.engram.embed.weight")
