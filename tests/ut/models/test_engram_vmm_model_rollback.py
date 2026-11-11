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


class SharedCleanupError(RuntimeError):
    pass


@pytest.mark.parametrize(
    "failure_type,cleanup_fails",
    [(SharedInitError, False), (SharedInitError, True), (SharedCleanupError, False), (ValueError, False)],
)
def test_second_buffer_rollback_only_after_synchronized_failure(failure_type, cleanup_fails):
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
    code = ast.fix_missing_locations(ast.Module(body=[wrapper], type_ignores=[]))
    calls: list[str] = []
    failure = failure_type("injected failure")

    class Buffer:
        def __init__(self, *_args):
            if calls:
                raise failure
            calls.append("created")

        def close_collective(self):
            calls.append("close_collective")
            if cleanup_fails:
                raise RuntimeError("codes cleanup failed")

    torch = SimpleNamespace(
        device=lambda *_args: object(), npu=SimpleNamespace(current_device=lambda: 0), bfloat16=object()
    )
    retained: list[Any] = []
    scope: dict[str, Any] = {
        "torch": torch,
        "LocalSharedVmmBuffer": Buffer,
        "EngramSharedInitError": SharedInitError,
        "retain_failed_initialization": retained.append,
    }
    exec(compile(code, str(source), "exec"), scope)
    instance = scope["Candidate"]()
    instance.part_num_embeddings, instance.dim, instance.block_size = 8, 256, 32
    instance.storage_dtype, instance.scale_dtype = object(), object()
    instance.cpu_offload = instance.local_vmm = True
    instance._shared_group = object()
    expected = RuntimeError if cleanup_fails else type(failure)
    with pytest.raises(expected) as raised:
        instance._allocate_weights()
    if cleanup_fails:
        assert str(raised.value) == "codes cleanup failed"
    else:
        assert raised.value is failure
    assert calls == (["created", "close_collective"] if failure_type is SharedInitError else ["created"])
    assert retained == ([] if failure_type is SharedInitError and not cleanup_fails else [instance._codes_uva])


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


def test_shared_load_failure_closes_all_vmm_tables_in_reverse_order():
    calls: list[str] = []
    tables = [Table("first", calls), Table("default", calls, local_vmm=False), Table("last", calls)]
    error = SharedLoadError("rank 0 failed")
    with pytest.raises(SharedLoadError) as raised:
        candidate(tables, error).load_weights([])
    assert raised.value is error
    assert calls == ["last", "first"]


def test_unsynchronized_load_failure_does_not_enter_collective_cleanup():
    calls: list[str] = []
    with pytest.raises(ValueError, match="unrelated"):
        candidate([Table("table", calls)], ValueError("unrelated")).load_weights([])
    assert calls == []


def test_failed_cleanup_preserves_original_failure_and_stops():
    calls: list[str] = []
    tables = [Table("first", calls), Table("last", calls, fail=True)]
    with pytest.raises(RuntimeError, match="rank 0 failed; Engram VMM cleanup failed: cleanup failure"):
        candidate(tables, SharedLoadError("rank 0 failed")).load_weights([])
    assert calls == ["last"]


def test_failed_model_rollback_retains_all_remaining_vmm_buffers():
    calls: list[str] = []
    retained: list[Any] = []
    first, last = Table("first", calls), Table("last", calls, fail=True)
    first._codes_uva, first._scales_uva = Buffer(), Buffer()
    last._codes_uva, last._scales_uva = Buffer(), Buffer()
    with pytest.raises(RuntimeError, match="cleanup failure"):
        candidate([first, last], SharedLoadError("load failed"), retained).load_weights([])
    assert calls == ["last"]
    assert retained == [last._scales_uva, last._codes_uva, first._scales_uva, first._codes_uva]


@pytest.mark.parametrize("synchronized", [True, False])
def test_only_synchronized_constructor_failure_rolls_back_previous_tables(synchronized):
    calls: list[str] = []
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
    retain_helper = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_retain_local_engram_tables"
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
    assert isinstance(wrapper, ast.FunctionDef)
    wrapper.body = [guard]
    code = ast.fix_missing_locations(ast.Module(body=[retain_helper, helper, wrapper], type_ignores=[]))
    retained: list[Any] = []
    previous._codes_uva, previous._scales_uva = Buffer(), Buffer()
    scope: dict[str, Any] = {
        "AscendParallelEngramEmbedding": ConstructorTable,
        "LocalSharedVmmBuffer": Buffer,
        "retain_failed_initialization": retained.append,
        "EngramSharedInitError": SharedInitError,
        "rows": 8,
        "config": SimpleNamespace(engram_head_dim=256),
        "head_sizes": (8,),
        "slot": 1,
        "storage_dtype": object(),
        "cpu_offload": True,
        "get_ascend_config": lambda: SimpleNamespace(engram_local_vmm=True),
    }
    exec(compile(code, str(source), "exec"), scope)
    instance = SimpleNamespace(engram_dp_shared_memory=True, modules=lambda: [previous])
    with pytest.raises(type(failure)) as raised:
        scope["construct"](instance)
    assert raised.value is failure
    assert calls == (["previous"] if synchronized else [])
    assert retained == ([] if synchronized else [previous._scales_uva, previous._codes_uva])
