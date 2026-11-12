# SPDX-License-Identifier: Apache-2.0
"""Exercise model-level rollback without requiring a vLLM runtime."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest


class SharedLoadError(RuntimeError):
    pass


class SharedInitError(RuntimeError):
    pass


class Table:
    def __init__(self, name, calls, local_vmm=True, fail=False):
        self.name, self.calls, self.local_vmm, self.fail = name, calls, local_vmm, fail

    def close_host_offload(self):
        self.calls.append(self.name)
        if self.fail:
            raise RuntimeError("cleanup failure")


def candidate(tables, failure):
    source = Path(__file__).resolve().parents[3] / "vllm_ascend/models/deepseek_v41/model.py"
    tree = ast.parse(source.read_text())
    helper = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_rollback_local_engram_tables"
    )
    model = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "AscendDeepseekV41LLMForCausalLM"
    )
    method = next(node for node in model.body if isinstance(node, ast.FunctionDef) and node.name == "load_weights")
    method.returns = None
    for arg in method.args.args:
        arg.annotation = None
    wrapper = ast.ClassDef(name="Candidate", bases=[], keywords=[], body=[method], decorator_list=[])
    code = ast.fix_missing_locations(ast.Module(body=[helper, wrapper], type_ignores=[]))
    scope = {"EngramSharedLoadError": SharedLoadError, "AscendParallelEngramEmbedding": Table}
    exec(compile(code, str(source), "exec"), scope)
    obj = scope["Candidate"]()
    obj.model = SimpleNamespace(has_engram=True)
    obj.modules = lambda: tables
    obj._is_milestone_weight = lambda _name: True

    def load(_weights):
        raise failure

    obj._load_model_weights = load
    return obj


def test_shared_load_failure_closes_all_vmm_tables_in_reverse_order():
    calls = []
    tables = [Table("first", calls), Table("default", calls, local_vmm=False), Table("last", calls)]
    error = SharedLoadError("rank 0 failed")
    with pytest.raises(SharedLoadError) as raised:
        candidate(tables, error).load_weights([])
    assert raised.value is error
    assert calls == ["last", "first"]


def test_unsynchronized_load_failure_does_not_enter_collective_cleanup():
    calls = []
    with pytest.raises(ValueError, match="unrelated"):
        candidate([Table("table", calls)], ValueError("unrelated")).load_weights([])
    assert calls == []


def test_failed_cleanup_preserves_original_failure_and_stops():
    calls = []
    tables = [Table("first", calls), Table("last", calls, fail=True)]
    with pytest.raises(RuntimeError, match="rank 0 failed; Engram VMM cleanup failed: cleanup failure"):
        candidate(tables, SharedLoadError("rank 0 failed")).load_weights([])
    assert calls == ["last"]


@pytest.mark.parametrize("synchronized", [True, False])
def test_only_synchronized_constructor_failure_rolls_back_previous_tables(synchronized):
    calls = []
    failure = SharedInitError("shared failure") if synchronized else ValueError("local failure")

    class ConstructorTable(Table):
        def __init__(self, *args, **kwargs):
            if "storage_dtype" in kwargs:
                raise failure
            super().__init__(*args, **kwargs)

    previous = ConstructorTable("previous", calls)
    source = Path(__file__).resolve().parents[3] / "vllm_ascend/models/deepseek_v41/model.py"
    tree = ast.parse(source.read_text())
    helper = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_rollback_local_engram_tables"
    )
    model = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "DeepseekV41Model")
    constructor = next(node for node in model.body if isinstance(node, ast.FunctionDef) and node.name == "__init__")
    guard = next(
        node
        for node in ast.walk(constructor)
        if isinstance(node, ast.Try)
        and any(
            isinstance(handler.type, ast.Name) and handler.type.id == "EngramSharedInitError"
            for handler in node.handlers
        )
    )
    wrapper = ast.parse("def construct(self):\n    pass\n").body[0]
    wrapper.body = [guard]
    code = ast.fix_missing_locations(ast.Module(body=[helper, wrapper], type_ignores=[]))
    scope = {
        "AscendParallelEngramEmbedding": ConstructorTable,
        "EngramSharedInitError": SharedInitError,
        "rows": 8,
        "config": SimpleNamespace(engram_head_dim=256),
        "head_sizes": (8,),
        "slot": 1,
        "storage_dtype": object(),
        "cpu_offload": True,
        "remote_reader": None,
        "shared_pool_dir": None,
        "shared_pool_identity": None,
        "layer_id": 1,
        "get_ascend_config": lambda: SimpleNamespace(engram_local_vmm=True),
    }
    exec(compile(code, str(source), "exec"), scope)
    instance = SimpleNamespace(engram_dp_shared_memory=True, modules=lambda: [previous])
    with pytest.raises(type(failure)) as raised:
        scope["construct"](instance)
    assert raised.value is failure
    assert calls == (["previous"] if synchronized else [])
