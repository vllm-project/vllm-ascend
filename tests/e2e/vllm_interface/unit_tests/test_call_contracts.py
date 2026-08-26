from __future__ import annotations

import ast
from pathlib import Path

from tests.e2e.vllm_interface.vllm_interface_contracts.call_contracts import (
    CallShape,
    DirectCallDetector,
    ReturnContract,
    ReturnShape,
    ReturnUse,
    bind_call_shape,
    call_shape,
    infer_return_contract,
    replacement_return_compatible,
    return_use_compatible,
)
from tests.e2e.vllm_interface.vllm_interface_contracts.generator import (
    InterfaceBoundaryGenerator,
    _jsonable_signature,
)


def _function(source: str) -> ast.FunctionDef | ast.AsyncFunctionDef:
    node = ast.parse(source).body[0]
    assert isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    return node


def _call(source: str) -> ast.Call:
    node = ast.parse(source, mode="eval").body
    assert isinstance(node, ast.Call)
    return node


def _direct_calls(
    tmp_path: Path,
    consumer: str,
    upstream: str = "def helper(value=None): return value\n",
):
    vllm_root = tmp_path / "vllm"
    ascend_root = tmp_path / "vllm_ascend"
    (vllm_root / "vllm").mkdir(parents=True)
    (ascend_root / "vllm_ascend").mkdir(parents=True)
    (vllm_root / "vllm" / "__init__.py").write_text("", encoding="utf-8")
    (vllm_root / "vllm" / "api.py").write_text(upstream, encoding="utf-8")
    (ascend_root / "vllm_ascend" / "__init__.py").write_text("", encoding="utf-8")
    (ascend_root / "vllm_ascend" / "consumer.py").write_text(consumer, encoding="utf-8")
    engine = InterfaceBoundaryGenerator(vllm_root, ascend_root)
    return DirectCallDetector(engine).discover()


def test_literal_argument_expansions_are_exact() -> None:
    shape = call_shape(_call("target(1, *[2, 3], named=4, **{'extra': 5})"))
    assert shape == CallShape(
        positional_count=3,
        keyword_names=("named", "extra"),
    )


def test_dynamic_argument_expansion_fails_closed() -> None:
    signature = _jsonable_signature(_function("def target(value): pass"))
    compatible, reason = bind_call_shape(
        signature,
        call_shape(_call("target(*values)")),
    )
    assert compatible is None
    assert "dynamic" in reason


def test_exact_call_binding_detects_missing_required_parameter() -> None:
    signature = _jsonable_signature(_function("def target(value, required): pass"))
    compatible, reason = bind_call_shape(signature, call_shape(_call("target(1)")))
    assert compatible is False
    assert "bind" in reason


def test_return_contract_distinguishes_none_tuple_and_bottom() -> None:
    none_contract = infer_return_contract(_function("def target():\n    pass\n"))
    tuple_contract = infer_return_contract(_function("def target():\n    return 1, 2\n"))
    bottom_contract = infer_return_contract(_function("def target():\n    raise RuntimeError\n"))
    assert none_contract is not None
    assert none_contract.variants == (ReturnShape("none"),)
    assert tuple_contract is not None
    assert tuple_contract.variants == (ReturnShape("tuple", arity=2),)
    assert bottom_contract is not None
    assert bottom_contract.status == "bottom"


def test_abstract_return_annotation_supplies_the_contract() -> None:
    contract = infer_return_contract(_function("def target() -> tuple[int, str]:\n    raise NotImplementedError\n"))
    assert contract is not None
    assert contract.status == "exact"
    assert contract.variants == (ReturnShape("tuple", arity=2),)


def test_annotation_body_conflict_fails_closed() -> None:
    contract = infer_return_contract(_function("def target() -> tuple[int, str]:\n    return 1\n"))
    assert contract is not None
    assert contract.status == "unknown"
    assert "annotation_body_conflict" in contract.provenance


def test_unknown_return_transform_fails_closed() -> None:
    contract = infer_return_contract(
        _function("@runtime_wrapper\ndef target() -> tuple[int, str]:\n    return 1, 'value'\n")
    )
    assert contract is not None
    assert contract.status == "unknown"
    assert "unknown_return_transform" in contract.provenance


def test_async_and_generator_protocols_are_distinct() -> None:
    async_contract = infer_return_contract(_function("async def target():\n    return 1\n"))
    generator_contract = infer_return_contract(_function("def target():\n    yield 1\n"))
    assert async_contract is not None and async_contract.protocol == "awaitable"
    assert generator_contract is not None and generator_contract.protocol == "iterator"


def test_replacement_return_is_covariant() -> None:
    upstream = ReturnContract(
        "value",
        (ReturnShape("none"), ReturnShape("tuple", arity=2)),
        "exact",
    )
    narrower = ReturnContract("value", (ReturnShape("tuple", arity=2),), "exact")
    incompatible = ReturnContract("value", (ReturnShape("tuple", arity=3),), "exact")
    assert replacement_return_compatible(upstream, narrower)[0] is True
    assert replacement_return_compatible(upstream, incompatible)[0] is False


def test_return_use_requires_every_possible_shape_to_match() -> None:
    contract = ReturnContract(
        "value",
        (ReturnShape("tuple", arity=2), ReturnShape("none")),
        "exact",
    )
    compatible, _ = return_use_compatible(contract, ReturnUse("unpack", arity=2))
    assert compatible is False


def test_return_annotations_map_observable_protocols() -> None:
    cases = {
        "Iterator[int]": "iterator",
        "Iterable[int]": "iterator",
        "ContextManager[int]": "context_manager",
        "AsyncIterator[int]": "async_iterator",
        "AsyncContextManager[int]": "async_context_manager",
    }
    for annotation, expected in cases.items():
        contract = infer_return_contract(_function(f"def target() -> {annotation}:\n    raise NotImplementedError\n"))
        assert contract is not None
        assert contract.protocol == expected
        assert contract.status == "exact"


def test_bare_return_protocol_annotations_are_supported() -> None:
    iterator = infer_return_contract(_function("def target() -> Iterator:\n    raise NotImplementedError\n"))
    context = infer_return_contract(_function("def target() -> ContextManager:\n    raise NotImplementedError\n"))
    assert iterator is not None and iterator.protocol == "iterator"
    assert context is not None and context.protocol == "context_manager"


def test_unimported_vllm_root_is_not_a_verified_call(tmp_path: Path) -> None:
    calls = _direct_calls(
        tmp_path,
        "def use():\n    return vllm.api.helper(1)\n",
    )
    assert calls == []


def test_triton_subscript_launch_uses_outer_call_arguments(tmp_path: Path) -> None:
    calls = _direct_calls(
        tmp_path,
        ("from vllm.api import kernel\n\ndef use():\n    kernel[(2,)](1, BLOCK_SIZE=16)\n"),
        upstream=(
            "from vllm.triton_utils import triton\n\n"
            "@triton.jit(do_not_specialize=['value'])\n"
            "def kernel(value, BLOCK_SIZE): pass\n"
        ),
    )
    assert len(calls) == 1
    assert calls[0].target == "vllm.api.kernel"
    assert calls[0].callee == "kernel[2,]"
    assert calls[0].invocation_kind == "triton_kernel_launch"
    assert calls[0].call_shape == CallShape(
        positional_count=1,
        keyword_names=("BLOCK_SIZE",),
    )


def test_ordinary_subscript_callable_is_not_assumed_to_be_triton(tmp_path: Path) -> None:
    calls = _direct_calls(
        tmp_path,
        "from vllm.api import kernel\n\ndef use():\n    kernel[(2,)](1)\n",
        upstream="def kernel(value): return value\n",
    )
    assert calls == []


def test_triton_launch_with_an_additional_decorator_fails_closed(tmp_path: Path) -> None:
    calls = _direct_calls(
        tmp_path,
        "from vllm.api import kernel\n\ndef use():\n    kernel[(2,)](1)\n",
        upstream=(
            "from vllm.triton_utils import triton\n\n"
            "def wrapper(function): return function\n\n"
            "@wrapper\n"
            "@triton.jit\n"
            "def kernel(value): pass\n"
        ),
    )
    assert calls == []


def test_module_binding_rebind_is_resolved_at_each_callsite(tmp_path: Path) -> None:
    calls = _direct_calls(
        tmp_path,
        ("from vllm.api import helper\n\nfirst = helper(1)\nhelper = lambda value: value\nsecond = helper(2)\n"),
    )
    assert [(item.line, item.target) for item in calls] == [(3, "vllm.api.helper")]


def test_function_local_import_then_rebind_is_fail_closed(tmp_path: Path) -> None:
    calls = _direct_calls(
        tmp_path,
        (
            "def use():\n"
            "    from vllm.api import helper\n"
            "    first = helper(1)\n"
            "    helper = lambda value: value\n"
            "    second = helper(2)\n"
        ),
    )
    assert [(item.line, item.target) for item in calls] == [(3, "vllm.api.helper")]


def test_later_function_local_binding_blocks_global_fallback(tmp_path: Path) -> None:
    calls = _direct_calls(
        tmp_path,
        ("from vllm.api import helper\n\ndef use():\n    first = helper(1)\n    helper = lambda value: value\n"),
    )
    assert calls == []


def test_enclosing_function_shadow_blocks_module_fallback(tmp_path: Path) -> None:
    calls = _direct_calls(
        tmp_path,
        (
            "from vllm.api import helper\n\n"
            "def outer():\n"
            "    helper = lambda value: value\n"
            "    def inner():\n"
            "        return helper(1)\n"
            "    return inner\n"
        ),
    )
    assert calls == []


def test_same_line_calls_keep_column_and_shape_identity(tmp_path: Path) -> None:
    calls = _direct_calls(
        tmp_path,
        "from vllm.api import helper\n\ndef use():\n    return helper(1), helper(value=2)\n",
    )
    assert len(calls) == 2
    assert len({item.column for item in calls}) == 2
    assert {item.call_shape for item in calls} == {
        CallShape(positional_count=1, keyword_names=()),
        CallShape(positional_count=0, keyword_names=("value",)),
    }


def test_direct_access_kind_does_not_depend_on_new_symbol_kind(tmp_path: Path) -> None:
    calls = _direct_calls(
        tmp_path,
        "from vllm.api import Config\n\ndef use():\n    return Config(1)\n",
        "class Config:\n    def __init__(self, value): self.value = value\n",
    )
    assert len(calls) == 1
    assert calls[0].access_kind == "direct"


def test_constructed_receiver_keeps_class_path_without_new_owner_lookup(tmp_path: Path) -> None:
    calls = _direct_calls(
        tmp_path,
        ("from vllm.api import Service\n\ndef use():\n    service = Service()\n    return service.run(1)\n"),
        "class Service:\n    pass\n",
    )
    method_calls = [item for item in calls if item.callee == "service.run"]
    assert len(method_calls) == 1
    assert method_calls[0].target == "vllm.api.Service.run"
    assert method_calls[0].access_kind == "instance"
    assert method_calls[0].receiver_type == "vllm.api.Service"


def test_conditional_constructed_receiver_is_not_verified(tmp_path: Path) -> None:
    calls = _direct_calls(
        tmp_path,
        (
            "from vllm.api import Service\n\n"
            "def use(enabled):\n"
            "    if enabled:\n"
            "        service = Service()\n"
            "    return service.run(1)\n"
        ),
        "class Service:\n    def run(self, value): return value\n",
    )
    assert not [item for item in calls if item.callee == "service.run"]


def test_reassigned_annotated_receiver_is_not_verified(tmp_path: Path) -> None:
    calls = _direct_calls(
        tmp_path,
        (
            "from vllm.api import Service\n\n"
            "def use(service: Service):\n"
            "    service = object()\n"
            "    return service.run(1)\n"
        ),
        "class Service:\n    def run(self, value): return value\n",
    )
    assert calls == []


def test_same_line_receiver_rebinding_is_not_verified(tmp_path: Path) -> None:
    calls = _direct_calls(
        tmp_path,
        (
            "from vllm.api import Base\n\n"
            "class Child(Base):\n"
            "    def use(self, other):\n"
            "        self = other; return self.run(1)\n"
        ),
        "class Base:\n    def run(self, value): return value\n",
    )
    assert not [item for item in calls if item.callee == "self.run"]


def test_non_callable_member_blocks_upstream_method_fallback(tmp_path: Path) -> None:
    calls = _direct_calls(
        tmp_path,
        (
            "from vllm.api import Base\n\n"
            "class Child(Base):\n"
            "    run = None\n"
            "    def use(self):\n"
            "        return self.run(1)\n"
        ),
        "class Base:\n    def run(self, value): return value\n",
    )
    assert not [item for item in calls if item.callee == "self.run"]


def test_same_line_constructed_receiver_is_verified(tmp_path: Path) -> None:
    calls = _direct_calls(
        tmp_path,
        ("from vllm.api import Service\n\ndef use():\n    service = Service(); return service.run(1)\n"),
        "class Service:\n    def run(self, value): return value\n",
    )
    method_calls = [item for item in calls if item.callee == "service.run"]
    assert len(method_calls) == 1
    assert method_calls[0].receiver_type == "vllm.api.Service"


def test_annotated_local_declared_after_call_is_not_verified(tmp_path: Path) -> None:
    calls = _direct_calls(
        tmp_path,
        ("from vllm.api import Service\n\ndef use():\n    service.run(1)\n    service: Service = Service()\n"),
        "class Service:\n    def run(self, value): return value\n",
    )
    assert not [item for item in calls if item.callee == "service.run"]


def test_shadowed_descriptor_decorator_is_not_treated_as_builtin() -> None:
    contract = infer_return_contract(
        _function("@classmethod\ndef target(cls):\n    return 1\n"),
        resolver=lambda name: f"vllm.api.{name}",
    )
    assert contract is not None
    assert contract.status == "unknown"
