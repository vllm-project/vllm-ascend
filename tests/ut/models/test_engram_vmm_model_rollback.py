# SPDX-License-Identifier: Apache-2.0
"""Exercise model-level rollback without requiring a vLLM runtime."""

import ast
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest


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


def test_model_rollback_stops_on_unsynchronized_failure():
    source = Path(__file__).resolve().parents[3] / "vllm_ascend/models/deepseek_v41/model.py"
    tree = ast.parse(source.read_text())
    helper = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_rollback_local_engram_tables"
    )
    calls = []

    class Table:
        local_vmm = True

        def __init__(self, name, error=None):
            self.name, self.error = name, error

        def close_host_offload(self):
            calls.append(self.name)
            if self.error:
                raise self.error

    retained = []
    scope = {
        "AscendParallelEngramEmbedding": Table,
        "EngramSharedReleaseError": SharedReleaseError,
        "_retain_local_engram_tables": retained.append,
    }
    exec(compile(ast.fix_missing_locations(ast.Module(body=[helper], type_ignores=[])), str(source), "exec"), scope)
    first, last = Table("first"), Table("last", RuntimeError("peer lost"))
    model = SimpleNamespace(modules=lambda: [first, last])
    with pytest.raises(RuntimeError, match="peer lost"):
        scope["_rollback_local_engram_tables"](model, RuntimeError("load failed"))
    assert calls == ["last"] and retained == [model]
    calls.clear()
    last.error = SharedReleaseError("synchronized")
    with pytest.raises(RuntimeError, match="synchronized"):
        scope["_rollback_local_engram_tables"](model, RuntimeError("load failed"))
    assert calls == ["last", "first"]
