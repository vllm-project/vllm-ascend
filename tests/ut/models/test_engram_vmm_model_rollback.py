# SPDX-License-Identifier: Apache-2.0
"""Exercise model-level rollback without requiring a vLLM runtime."""

import ast
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest


class SharedLoadError(RuntimeError):
    pass


class SharedInitError(RuntimeError):
    pass


class SharedReleaseError(RuntimeError):
    pass


def test_synchronized_allocation_failure_falls_back_to_registered_storage():
    source = Path(__file__).resolve().parents[3] / "vllm_ascend/models/deepseek_v41/engram/embedding.py"
    tree = ast.parse(source.read_text())
    cls = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "AscendParallelEngramEmbedding"
    )
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "_allocate_weights")
    method.returns = None
    wrapper = ast.parse("class Candidate:\n    pass\n").body[0]
    assert isinstance(wrapper, ast.ClassDef)
    wrapper.body = [method]
    allocations = []

    def fail(*args, **kwargs):
        raise SharedInitError("all ranks cleaned up")

    def registered(shape, dtype, device, group):
        buffer = SimpleNamespace(tensor=(shape, dtype))
        allocations.append(buffer)
        return buffer

    scope: dict[str, Any] = {
        "torch": SimpleNamespace(
            device=lambda *_: object(), npu=SimpleNamespace(current_device=lambda: 0), bfloat16=object()
        ),
        "LocalSharedVmmBuffer": fail,
        "EngramSharedInitError": SharedInitError,
        "SharedUvaBuffer": registered,
        "logger": SimpleNamespace(warning=lambda *args: None),
    }
    exec(compile(ast.fix_missing_locations(ast.Module(body=[wrapper], type_ignores=[])), str(source), "exec"), scope)
    instance = scope["Candidate"]()
    instance.part_num_embeddings, instance.dim, instance.block_size = 8, 256, 32
    instance.storage_dtype, instance.scale_dtype = object(), object()
    instance.cpu_offload = instance.local_vmm = True
    instance._shared_group, instance._shared_pool_spec = SimpleNamespace(), (None, None, None)
    codes, scales = instance._allocate_weights()
    assert instance._shared_group._engram_local_vmm_available is False
    assert not instance.local_vmm
    assert len(allocations) == 2 and (codes, scales) == (allocations[0].tensor, allocations[1].tensor)


class Buffer:
    pass


class Table:
    def __init__(self, name, calls, local_vmm=True, fail=False):
        self.name, self.calls, self.local_vmm, self.fail = name, calls, local_vmm, fail
        self._codes_uva: Any = None
        self._scales_uva: Any = None

    def close_host_offload(self):
        self.calls.append(self.name)
        if self.fail:
            raise RuntimeError("cleanup failure")


def candidate(tables, failure, retained=None):
    source = Path(__file__).resolve().parents[3] / "vllm_ascend/models/deepseek_v41/model.py"
    tree = ast.parse(source.read_text())
    helper = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_rollback_local_engram_tables"
    )
    retain_helper = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_retain_local_engram_tables"
    )
    model = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "AscendDeepseekV41LLMForCausalLM"
    )
    method = next(node for node in model.body if isinstance(node, ast.FunctionDef) and node.name == "load_weights")
    method.returns = None
    for arg in method.args.args:
        arg.annotation = None
    wrapper = ast.parse("class Candidate:\n    pass\n").body[0]
    assert isinstance(wrapper, ast.ClassDef)
    wrapper.body = [method]
    code = ast.fix_missing_locations(ast.Module(body=[retain_helper, helper, wrapper], type_ignores=[]))
    scope: dict[str, Any] = {
        "EngramSharedLoadError": SharedLoadError,
        "EngramSharedReleaseError": SharedReleaseError,
        "AscendParallelEngramEmbedding": Table,
        "LocalSharedVmmBuffer": Buffer,
        "retain_failed_initialization": (retained if retained is not None else []).append,
    }
    exec(compile(code, str(source), "exec"), scope)
    obj = scope["Candidate"]()
    obj.model = SimpleNamespace(has_engram=True)
    obj.modules = lambda: tables
    obj._is_milestone_weight = lambda _name: True

    def load(_weights):
        raise failure

    obj._load_model_weights = load
    return obj


def test_synchronized_release_failure_attempts_remaining_tables():
    calls: list[str] = []

    class FailingTable(Table):
        def close_host_offload(self):
            self.calls.append(self.name)
            raise SharedReleaseError("peer release failed")

    first, last = Table("first", calls), FailingTable("last", calls)
    with pytest.raises(RuntimeError, match="peer release failed"):
        candidate([first, last], SharedLoadError("load failed")).load_weights([])
    assert calls == ["last", "first"]
