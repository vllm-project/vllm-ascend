# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
"""Exact static contracts for downstream calls and callable return values.

This module intentionally lives in the shared engine.  The main2main skill is
only an orchestration layer and must not grow a second AST implementation.
The analysis here is conservative: a dependency is returned only when its
callee is uniquely resolved, and dynamic argument/return shapes stay unknown.
"""

from __future__ import annotations

import ast
import json
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from tools.vllm_interface_contracts.generator import (
    _TRITON_JIT_DECORATOR,
    _TRITON_KERNEL_PROTOCOL,
    InterfaceBoundaryGenerator,
    ModuleInfo,
    _expression_name,
    _function_local_names,
    _function_scope_nodes,
    _inspect_signature,
    _scope_reference_variants,
    _statements_must_terminate,
    _tag_guard_names,
)


@dataclass(frozen=True)
class CallShape:
    """One concrete Python call expression, before runtime expansion."""

    positional_count: int
    keyword_names: tuple[str, ...]
    dynamic_starargs: bool = False
    dynamic_kwargs: bool = False

    @property
    def exact(self) -> bool:
        return not self.dynamic_starargs and not self.dynamic_kwargs

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ReturnShape:
    """A structural shape that callers can observe without executing code."""

    kind: str
    arity: int | None = None
    keys: tuple[str, ...] = ()
    type_ref: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ReturnContract:
    """The observable call protocol and possible successful return shapes."""

    protocol: str
    variants: tuple[ReturnShape, ...]
    status: str
    provenance: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "protocol": self.protocol,
            "variants": [item.as_dict() for item in self.variants],
            "status": self.status,
            "provenance": list(self.provenance),
        }


@dataclass(frozen=True)
class ReturnUse:
    """How one downstream callsite immediately consumes a return value."""

    kind: str
    awaited: bool = False
    arity: int | None = None
    minimum_arity: int | None = None
    key: str | None = None
    index: int | None = None
    attribute: str | None = None
    status: str = "exact"

    @property
    def constrains_return(self) -> bool:
        return self.kind not in {"ignored", "passthrough"}

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DirectCallDependency:
    """One uniquely resolved vLLM call made by vllm-ascend."""

    target: str
    access_kind: str
    file: str
    line: int
    column: int
    owner: str | None
    scope: str | None
    callee: str
    call_shape: CallShape
    return_use: ReturnUse
    receiver_type: str | None = None
    member: str | None = None
    invocation_kind: str = "python_call"
    lookup_root: str | None = None
    resolution_basis: str = "new_exact"

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["call_shape"] = self.call_shape.as_dict()
        payload["return_use"] = self.return_use.as_dict()
        return payload


def call_shape(node: ast.Call) -> CallShape:
    positional_count = 0
    dynamic_starargs = False
    keyword_names: list[str] = []
    dynamic_kwargs = False
    for argument in node.args:
        if not isinstance(argument, ast.Starred):
            positional_count += 1
            continue
        if isinstance(argument.value, (ast.List, ast.Tuple)) and not any(
            isinstance(item, ast.Starred) for item in argument.value.elts
        ):
            positional_count += len(argument.value.elts)
        else:
            dynamic_starargs = True
    for keyword in node.keywords:
        if keyword.arg is not None:
            keyword_names.append(keyword.arg)
            continue
        if isinstance(keyword.value, ast.Dict) and all(
            isinstance(key, ast.Constant) and isinstance(key.value, str) for key in keyword.value.keys
        ):
            # A dict literal resolves duplicate keys before ``**`` expansion;
            # duplicates across separate expansions still remain visible.
            keyword_names.extend(dict.fromkeys(str(key.value) for key in keyword.value.keys))
        else:
            dynamic_kwargs = True
    return CallShape(
        positional_count=positional_count,
        keyword_names=tuple(keyword_names),
        dynamic_starargs=dynamic_starargs,
        dynamic_kwargs=dynamic_kwargs,
    )


def bind_call_shape(signature: list[object] | None, shape: CallShape) -> tuple[bool | None, str]:
    """Bind one actual call shape, not a replacement substitutability set."""

    if signature is None:
        return None, "callable signature could not be resolved"
    candidate = _inspect_signature(signature)
    if candidate is None:
        return None, "callable signature is not representable"
    if len(set(shape.keyword_names)) != len(shape.keyword_names):
        return False, "the call supplies the same keyword more than once"
    args = [object() for _ in range(shape.positional_count)]
    kwargs = {name: object() for name in shape.keyword_names}
    try:
        if shape.exact:
            candidate.bind(*args, **kwargs)
        else:
            candidate.bind_partial(*args, **kwargs)
    except TypeError as error:
        return False, f"call arguments do not bind: {error}"
    if not shape.exact:
        return None, "dynamic *args or **kwargs prevents exact binding"
    return True, "call arguments bind to the callable contract"


def _shape_key(shape: ReturnShape) -> str:
    return json.dumps(shape.as_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _ordered_shapes(shapes: Iterable[ReturnShape]) -> tuple[ReturnShape, ...]:
    unique = {_shape_key(item): item for item in shapes}
    return tuple(unique[key] for key in sorted(unique))


def _resolved_name(node: ast.AST | None, resolver: Callable[[str], str | None] | None) -> str | None:
    name = _expression_name(node)
    if name is None:
        return None
    return resolver(name) if resolver is not None else name


def _annotation_shapes(
    node: ast.AST | None,
    resolver: Callable[[str], str | None] | None,
) -> tuple[ReturnShape, ...] | None:
    if node is None:
        return None
    if isinstance(node, ast.Constant):
        if node.value is None:
            return (ReturnShape("none"),)
        if isinstance(node.value, str):
            # Forward-reference strings are deliberately not evaluated.
            return None
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        left = _annotation_shapes(node.left, resolver)
        right = _annotation_shapes(node.right, resolver)
        return _ordered_shapes((*left, *right)) if left is not None and right is not None else None
    if isinstance(node, ast.Subscript):
        base = (_expression_name(node.value) or "").rsplit(".", 1)[-1]
        elements = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
        if base in {"Optional"} and len(elements) == 1:
            nested = _annotation_shapes(elements[0], resolver)
            return _ordered_shapes((*nested, ReturnShape("none"))) if nested is not None else None
        if base in {"Union"}:
            variants: list[ReturnShape] = []
            for element in elements:
                nested = _annotation_shapes(element, resolver)
                if nested is None:
                    return None
                variants.extend(nested)
            return _ordered_shapes(variants)
        if base in {"tuple", "Tuple"}:
            if len(elements) == 2 and isinstance(elements[1], ast.Constant) and elements[1].value is Ellipsis:
                return (ReturnShape("tuple_variadic"),)
            return (ReturnShape("tuple", arity=len(elements)),)
        if base in {"list", "List", "Sequence"}:
            return (ReturnShape("sequence"),)
        if base in {"dict", "Dict", "Mapping"}:
            return (ReturnShape("mapping"),)
        if base in {"Iterator", "Iterable", "Generator"}:
            return (ReturnShape("opaque", type_ref="iterator_item"),)
        if base in {"AsyncIterator", "AsyncIterable", "AsyncGenerator"}:
            return (ReturnShape("opaque", type_ref="async_iterator_item"),)
        if base in {"ContextManager"}:
            return (ReturnShape("opaque", type_ref="context_value"),)
        if base in {"AsyncContextManager"}:
            return (ReturnShape("opaque", type_ref="async_context_value"),)
        return None
    name = _resolved_name(node, resolver)
    if name is None or name.rsplit(".", 1)[-1] in {"Any", "NoReturn", "Never"}:
        return None
    short = name.rsplit(".", 1)[-1]
    if short in {"Iterator", "Iterable", "Generator"}:
        return (ReturnShape("opaque", type_ref="iterator_item"),)
    if short in {"AsyncIterator", "AsyncIterable", "AsyncGenerator"}:
        return (ReturnShape("opaque", type_ref="async_iterator_item"),)
    if short == "ContextManager":
        return (ReturnShape("opaque", type_ref="context_value"),)
    if short == "AsyncContextManager":
        return (ReturnShape("opaque", type_ref="async_context_value"),)
    if short in {"None", "NoneType"}:
        return (ReturnShape("none"),)
    if short in {"bool", "bytes", "complex", "float", "int", "str"}:
        return (ReturnShape("scalar", type_ref=f"builtins.{short}"),)
    return (ReturnShape("object", type_ref=name),)


def _annotation_protocol(node: ast.AST | None) -> str | None:
    """Return an exact observable protocol declared by one annotation.

    A protocol hidden inside ``Optional`` or a heterogeneous union is not
    exact: callers cannot assume that every successful return supplies it.
    """

    if node is None:
        return None
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        left = _annotation_protocol(node.left)
        right = _annotation_protocol(node.right)
        return left if left is not None and left == right else None
    if isinstance(node, ast.Subscript):
        base = (_expression_name(node.value) or "").rsplit(".", 1)[-1]
        if base in {"Optional", "Union"}:
            return None
    else:
        base = (_expression_name(node) or "").rsplit(".", 1)[-1]
    return {
        "Iterator": "iterator",
        "Iterable": "iterator",
        "Generator": "iterator",
        "AsyncIterator": "async_iterator",
        "AsyncIterable": "async_iterator",
        "AsyncGenerator": "async_iterator",
        "ContextManager": "context_manager",
        "AsyncContextManager": "async_context_manager",
        "Awaitable": "awaitable",
        "Coroutine": "awaitable",
    }.get(base)


def _assignment_value(
    function: ast.AsyncFunctionDef | ast.FunctionDef,
    name: str,
    before_line: int,
) -> ast.AST | None:
    values: list[ast.AST] = []
    for node in _function_scope_nodes(function):
        if getattr(node, "lineno", before_line) >= before_line:
            continue
        if isinstance(node, ast.Assign):
            if any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
                values.append(node.value)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == name:
            if node.value is not None:
                values.append(node.value)
    return values[0] if len(values) == 1 else None


def _expression_shapes(
    node: ast.AST | None,
    *,
    function: ast.AsyncFunctionDef | ast.FunctionDef,
    resolver: Callable[[str], str | None] | None,
    forward_name: str | None,
    seen_names: frozenset[str] = frozenset(),
) -> tuple[ReturnShape, ...] | None:
    if node is None or (isinstance(node, ast.Constant) and node.value is None):
        return (ReturnShape("none"),)
    if isinstance(node, ast.IfExp):
        body = _expression_shapes(
            node.body,
            function=function,
            resolver=resolver,
            forward_name=forward_name,
            seen_names=seen_names,
        )
        other = _expression_shapes(
            node.orelse,
            function=function,
            resolver=resolver,
            forward_name=forward_name,
            seen_names=seen_names,
        )
        return _ordered_shapes((*body, *other)) if body is not None and other is not None else None
    if isinstance(node, ast.Tuple) and not any(isinstance(item, ast.Starred) for item in node.elts):
        return (ReturnShape("tuple", arity=len(node.elts)),)
    if isinstance(node, ast.List) and not any(isinstance(item, ast.Starred) for item in node.elts):
        return (ReturnShape("list", arity=len(node.elts)),)
    if isinstance(node, ast.Dict) and all(
        isinstance(key, ast.Constant) and isinstance(key.value, str) for key in node.keys
    ):
        return (ReturnShape("mapping", keys=tuple(sorted(str(key.value) for key in node.keys))),)
    if isinstance(node, ast.Constant):
        return (ReturnShape("scalar", type_ref=f"builtins.{type(node.value).__name__}"),)
    if isinstance(node, ast.Name) and node.id not in seen_names:
        value = _assignment_value(function, node.id, getattr(node, "lineno", 0))
        if value is not None:
            return _expression_shapes(
                value,
                function=function,
                resolver=resolver,
                forward_name=forward_name,
                seen_names=frozenset((*seen_names, node.id)),
            )
        return None
    if isinstance(node, ast.Call):
        callee = _expression_name(node.func)
        if (
            forward_name
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == forward_name
            and isinstance(node.func.value, ast.Call)
            and isinstance(node.func.value.func, ast.Name)
            and node.func.value.func.id == "super"
        ):
            return (ReturnShape("forward", type_ref=forward_name),)
        resolved = _resolved_name(node.func, resolver)
        short = (resolved or callee or "").rsplit(".", 1)[-1]
        if short == "tuple":
            return (ReturnShape("tuple_variadic"),)
        if short == "list":
            return (ReturnShape("sequence"),)
        if short == "dict":
            return (ReturnShape("mapping"),)
        if resolved and short[:1].isupper():
            return (ReturnShape("object", type_ref=resolved),)
        return None
    return None


def infer_return_contract(
    node: ast.AST | None,
    *,
    resolver: Callable[[str], str | None] | None = None,
    forward_name: str | None = None,
) -> ReturnContract | None:
    """Infer a conservative return contract for one Python function."""

    if not isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef)):
        return None
    scope_nodes = list(_function_scope_nodes(node))
    has_yield = any(isinstance(item, (ast.Yield, ast.YieldFrom)) for item in scope_nodes)
    decorators = [_expression_name(item.func if isinstance(item, ast.Call) else item) for item in node.decorator_list]
    resolved_decorators = [resolver(name) if resolver is not None and name is not None else name for name in decorators]
    known_origins = {
        "abc.abstractmethod",
        "contextlib.asynccontextmanager",
        "contextlib.contextmanager",
        "typing.override",
        "typing_extensions.override",
    }
    builtin_descriptors = {"builtins.classmethod", "builtins.property", "builtins.staticmethod"}
    fallback_known = {
        "abstractmethod",
        "asynccontextmanager",
        "contextmanager",
        "override",
    }
    unknown_decorator = any(
        raw is None
        or not (
            resolved in builtin_descriptors
            or (resolver is None and raw in {"classmethod", "property", "staticmethod"})
            or resolved in known_origins
            or (resolver is None and raw.rsplit(".", 1)[-1] in fallback_known)
        )
        for raw, resolved in zip(decorators, resolved_decorators, strict=True)
    )

    def exact_status(status: str) -> str:
        return "unknown" if status == "exact" and unknown_decorator else status

    annotation_protocol = _annotation_protocol(node.returns)
    if "contextlib.asynccontextmanager" in resolved_decorators or (
        resolver is None and any((name or "").endswith("asynccontextmanager") for name in decorators)
    ):
        protocol = "async_context_manager"
    elif "contextlib.contextmanager" in resolved_decorators or (
        resolver is None and any((name or "").endswith("contextmanager") for name in decorators)
    ):
        protocol = "context_manager"
    elif has_yield and isinstance(node, ast.AsyncFunctionDef):
        protocol = "async_iterator"
    elif has_yield:
        protocol = "iterator"
    elif isinstance(node, ast.AsyncFunctionDef):
        protocol = "awaitable"
    elif annotation_protocol is not None:
        protocol = annotation_protocol
    else:
        protocol = "value"

    declared = _annotation_shapes(node.returns, resolver)
    returns = [item for item in scope_nodes if isinstance(item, ast.Return)]
    observed: list[ReturnShape] = []
    observed_exact = True
    if not has_yield:
        for item in returns:
            shapes = _expression_shapes(
                item.value,
                function=node,
                resolver=resolver,
                forward_name=forward_name,
            )
            if shapes is None:
                observed_exact = False
            else:
                observed.extend(shapes)
        if not _statements_must_terminate(node.body):
            observed.append(ReturnShape("none"))

    provenance: list[str] = []
    if declared is not None:
        provenance.append("return_annotation")
    if observed:
        provenance.append("return_statements")
    if has_yield:
        provenance.append("yield_protocol")
    if unknown_decorator:
        provenance.append("unknown_return_transform")

    if has_yield:
        variants = declared or (ReturnShape("opaque", type_ref="yield_item"),)
        return ReturnContract(
            protocol,
            _ordered_shapes(variants),
            exact_status("exact" if declared else "unknown"),
            tuple(provenance),
        )
    if not returns and _statements_must_terminate(node.body):
        if declared is not None:
            return ReturnContract(
                protocol,
                _ordered_shapes(declared),
                exact_status("exact"),
                tuple(provenance),
            )
        return ReturnContract(protocol, (), "bottom", ("no_normal_return",))
    if observed_exact and observed:
        observed_variants = _ordered_shapes(observed)
        if all(candidate.kind == "forward" for candidate in observed_variants):
            # A transparent ``return super().same_method(...)`` override keeps
            # following the selected upstream endpoint at each snapshot.  A
            # stale annotation must not freeze the replacement to the old
            # return shape and create a runtime false positive.
            return ReturnContract(
                protocol,
                observed_variants,
                exact_status("exact"),
                (*provenance, "transparent_super_forward_precedes_annotation"),
            )
        if declared is not None:
            # A precise annotation is the public contract.  Keep the body as
            # corroborating evidence, but do not turn value-level differences
            # into interface breaks.
            declared_variants = _ordered_shapes(declared)
            conflicts = [
                candidate
                for candidate in observed_variants
                if all(_replacement_shape_accepted(expected, candidate) is False for expected in declared_variants)
            ]
            if conflicts:
                return ReturnContract(
                    protocol,
                    declared_variants,
                    "unknown",
                    (*provenance, "annotation_body_conflict"),
                )
            return ReturnContract(
                protocol,
                declared_variants,
                exact_status("exact"),
                tuple(provenance),
            )
        return ReturnContract(
            protocol,
            observed_variants,
            exact_status("exact"),
            tuple(provenance),
        )
    if declared is not None:
        return ReturnContract(
            protocol,
            _ordered_shapes(declared),
            exact_status("exact"),
            tuple(provenance),
        )
    return ReturnContract(protocol, _ordered_shapes(observed), "unknown", tuple(provenance or ["dynamic_return"]))


def return_contract_from_dict(payload: dict[str, Any] | None) -> ReturnContract | None:
    if payload is None:
        return None
    try:
        return ReturnContract(
            protocol=str(payload["protocol"]),
            variants=tuple(
                ReturnShape(
                    kind=str(item["kind"]),
                    arity=item.get("arity"),
                    keys=tuple(str(key) for key in item.get("keys", ())),
                    type_ref=item.get("type_ref"),
                )
                for item in payload.get("variants", ())
            ),
            status=str(payload["status"]),
            provenance=tuple(str(item) for item in payload.get("provenance", ())),
        )
    except (KeyError, TypeError, ValueError):
        return None


def _replacement_shape_accepted(expected: ReturnShape, candidate: ReturnShape) -> bool | None:
    if candidate.kind == "forward":
        return True
    if expected.kind == "object" and expected.type_ref in {"builtins.object", "object"}:
        return candidate.kind != "none"
    if expected.kind == "sequence":
        return candidate.kind in {"list", "sequence", "tuple", "tuple_variadic"}
    if expected.kind == "tuple_variadic":
        return candidate.kind in {"tuple", "tuple_variadic"}
    if candidate.kind == "tuple_variadic":
        # A variable-length tuple cannot promise any one fixed outer arity.
        return False if expected.kind == "tuple" else None
    if expected.kind != candidate.kind:
        return False
    if expected.kind in {"list", "tuple"}:
        return expected.arity == candidate.arity
    if expected.kind == "mapping":
        if not expected.keys:
            return True
        if not candidate.keys:
            return None
        return set(expected.keys).issubset(candidate.keys)
    if expected.kind in {"object", "scalar"}:
        if expected.type_ref is None or candidate.type_ref is None:
            return None
        return expected.type_ref == candidate.type_ref
    return True


def replacement_return_compatible(
    upstream: ReturnContract | None,
    downstream: ReturnContract | None,
) -> tuple[bool | None, str]:
    """Check covariant return substitutability for patch/override code."""

    if upstream is None or downstream is None:
        return None, "return contract could not be resolved"
    if upstream.status == "bottom":
        return None, "upstream has no observable normal return contract"
    if downstream.status == "bottom":
        return None, "replacement has no observable normal return contract"
    if upstream.status != "exact" or downstream.status != "exact":
        return None, "return contract contains a dynamic or ambiguous shape"
    if upstream.protocol != downstream.protocol:
        return False, f"return protocol changed from {upstream.protocol} to {downstream.protocol}"
    uncertain = False
    for candidate in downstream.variants:
        results = [_replacement_shape_accepted(expected, candidate) for expected in upstream.variants]
        if True in results:
            continue
        if None in results:
            uncertain = True
            continue
        return False, f"replacement return shape {candidate.kind} is outside the upstream contract"
    if uncertain:
        return None, "nominal return compatibility could not be proven"
    return True, "replacement return values satisfy the upstream return contract"


def _use_shape_compatible(shape: ReturnShape, use: ReturnUse) -> bool | None:
    if shape.kind == "forward":
        return None
    if use.kind == "unpack":
        if shape.kind not in {"list", "tuple"}:
            return None if shape.kind in {"sequence", "tuple_variadic"} else False
        if use.arity is not None:
            return shape.arity == use.arity
        return shape.arity is not None and shape.arity >= (use.minimum_arity or 0)
    if use.kind == "iterate":
        return shape.kind in {"list", "mapping", "sequence", "tuple", "tuple_variadic"}
    if use.kind == "subscript_index":
        if shape.kind not in {"list", "tuple"}:
            return None if shape.kind in {"sequence", "tuple_variadic"} else False
        if shape.arity is None or use.index is None:
            return None
        return -shape.arity <= use.index < shape.arity
    if use.kind == "subscript_key":
        if shape.kind != "mapping":
            return False
        if not shape.keys or use.key is None:
            return None
        return use.key in shape.keys
    if use.kind == "attribute":
        if shape.kind == "none":
            return False
        return None
    return True


def return_use_compatible(
    contract: ReturnContract | None,
    use: ReturnUse,
) -> tuple[bool | None, str]:
    if not use.constrains_return:
        return True, "the call result is not structurally consumed"
    if use.status != "exact":
        return None, "return value escapes or is consumed dynamically"
    if contract is None or contract.status != "exact":
        return None, "upstream return contract could not be proven"
    if use.awaited:
        if contract.protocol != "awaitable":
            return False, "the downstream awaits a non-awaitable return protocol"
    elif contract.protocol == "awaitable" and use.kind != "ignored":
        return False, "the downstream consumes an awaitable without awaiting it"
    if use.kind == "async_iterate":
        compatible = contract.protocol == "async_iterator"
        return compatible, "async iteration protocol matches" if compatible else "return is not an async iterator"
    if use.kind == "iterate" and contract.protocol == "iterator":
        return True, "iterator protocol matches"
    if use.kind == "context":
        compatible = contract.protocol == "context_manager"
        return compatible, "context-manager protocol matches" if compatible else "return is not a context manager"
    if use.kind == "async_context":
        compatible = contract.protocol == "async_context_manager"
        return compatible, (
            "async context-manager protocol matches" if compatible else "return is not an async context manager"
        )
    if use.kind == "await_only":
        return True, "awaitable protocol matches"
    if contract.protocol not in {"value", "awaitable"}:
        return False, f"{contract.protocol} does not provide the consumed value protocol"
    results = [_use_shape_compatible(shape, use) for shape in contract.variants]
    if False in results:
        return False, "at least one upstream return shape violates the downstream use"
    if results and all(result is True for result in results):
        return True, "all upstream return shapes satisfy the downstream use"
    return None, "the downstream return use could not be proven for every shape"


def _parents(tree: ast.AST) -> dict[int, ast.AST]:
    return {id(child): parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}


def _nearest(node: ast.AST, parents: dict[int, ast.AST], kinds: tuple[type[ast.AST], ...]) -> ast.AST | None:
    current = parents.get(id(node))
    while current is not None:
        if isinstance(current, kinds):
            return current
        current = parents.get(id(current))
    return None


def _unpack_use(target: ast.AST, *, awaited: bool) -> ReturnUse | None:
    if not isinstance(target, (ast.List, ast.Tuple)):
        return None
    starred = sum(isinstance(item, ast.Starred) for item in target.elts)
    if starred == 0:
        return ReturnUse("unpack", awaited=awaited, arity=len(target.elts))
    if starred == 1:
        return ReturnUse("unpack", awaited=awaited, minimum_arity=len(target.elts) - 1)
    return ReturnUse("unknown", awaited=awaited, status="unknown")


def _same_scope_nodes(scope: ast.AST) -> Iterable[ast.AST]:
    if isinstance(scope, (ast.AsyncFunctionDef, ast.FunctionDef)):
        yield from _function_scope_nodes(scope)
    else:
        for child in ast.walk(scope):
            if child is not scope:
                yield child


def infer_return_use(node: ast.Call, parents: dict[int, ast.AST], scope: ast.AST) -> ReturnUse:
    value: ast.AST = node
    awaited = False
    parent = parents.get(id(value))
    if isinstance(parent, ast.Await) and parent.value is value:
        awaited = True
        value = parent
        parent = parents.get(id(value))

    if isinstance(parent, ast.Assign) and parent.value is value and len(parent.targets) == 1:
        unpack = _unpack_use(parent.targets[0], awaited=awaited)
        if unpack is not None:
            return unpack
        if isinstance(parent.targets[0], ast.Name):
            alias = parent.targets[0].id
            loads = [
                child
                for child in _same_scope_nodes(scope)
                if isinstance(child, ast.Name)
                and isinstance(child.ctx, ast.Load)
                and child.id == alias
                and getattr(child, "lineno", 0) >= getattr(parent, "lineno", 0)
            ]
            stores = [
                child
                for child in _same_scope_nodes(scope)
                if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store) and child.id == alias
            ]
            if len(loads) == 1 and len(stores) == 1:
                value = loads[0]
                parent = parents.get(id(value))
            elif not loads:
                return ReturnUse("ignored", awaited=awaited)
            else:
                return ReturnUse("unknown", awaited=awaited, status="unknown")
    elif isinstance(parent, ast.NamedExpr) and parent.value is value:
        return ReturnUse("unknown", awaited=awaited, status="unknown")

    if isinstance(parent, ast.Assign) and parent.value is value:
        unpack = _unpack_use(parent.targets[0], awaited=awaited) if len(parent.targets) == 1 else None
        return unpack or ReturnUse("unknown", awaited=awaited, status="unknown")
    if isinstance(parent, ast.Attribute) and parent.value is value:
        return ReturnUse("attribute", awaited=awaited, attribute=parent.attr)
    if isinstance(parent, ast.Subscript) and parent.value is value:
        if isinstance(parent.slice, ast.Constant) and isinstance(parent.slice.value, int):
            return ReturnUse("subscript_index", awaited=awaited, index=parent.slice.value)
        if isinstance(parent.slice, ast.Constant) and isinstance(parent.slice.value, str):
            return ReturnUse("subscript_key", awaited=awaited, key=parent.slice.value)
        return ReturnUse("unknown", awaited=awaited, status="unknown")
    if isinstance(parent, (ast.For, ast.comprehension)) and parent.iter is value:
        return ReturnUse("iterate", awaited=awaited)
    if isinstance(parent, ast.AsyncFor) and parent.iter is value:
        return ReturnUse("async_iterate", awaited=awaited)
    if isinstance(parent, ast.withitem) and parent.context_expr is value:
        container = parents.get(id(parent))
        return ReturnUse("async_context" if isinstance(container, ast.AsyncWith) else "context", awaited=awaited)
    if awaited:
        return ReturnUse("await_only", awaited=True)
    if isinstance(parent, ast.Expr):
        return ReturnUse("ignored")
    if isinstance(parent, ast.Return):
        return ReturnUse("passthrough")
    if parent is None:
        return ReturnUse("ignored")
    return ReturnUse("unknown", status="unknown")


def _under_version_guard(node: ast.AST, parents: dict[int, ast.AST]) -> bool:
    current = parents.get(id(node))
    while current is not None:
        if isinstance(current, ast.If) and any(
            isinstance(item, ast.Call) and (_expression_name(item.func) or "").rsplit(".", 1)[-1] == "vllm_version_is"
            for item in ast.walk(current.test)
        ):
            return True
        current = parents.get(id(current))
    return False


def _annotation_reference(node: ast.AST | None) -> str | None:
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        candidates = {_annotation_reference(node.left), _annotation_reference(node.right)} - {None, "None"}
        return next(iter(candidates)) if len(candidates) == 1 else None
    if isinstance(node, ast.Subscript):
        base = (_expression_name(node.value) or "").rsplit(".", 1)[-1]
        if base == "Optional":
            return _annotation_reference(node.slice)
        return None
    return _expression_name(node)


class DirectCallDetector:
    """Discover exact downstream-to-upstream calls without changing golden relations."""

    def __init__(self, engine: InterfaceBoundaryGenerator):
        self.engine = engine
        self.historical_candidates: list[DirectCallDependency] = []
        self._candidate_roots: dict[tuple[str, int], frozenset[str]] = {}
        self._constructed_instances: dict[
            tuple[str, int],
            dict[str, ast.AnnAssign | ast.Call | None],
        ] = {}
        self._function_locals: dict[int, frozenset[str]] = {}
        self._scope_tag_guards: dict[int, set[str]] = {}

    def _local_names(
        self,
        function: ast.AsyncFunctionDef | ast.FunctionDef | None,
    ) -> frozenset[str]:
        if function is None:
            return frozenset()
        key = id(function)
        if key not in self._function_locals:
            self._function_locals[key] = frozenset(_function_local_names(function))
        return self._function_locals[key]

    def _tag_guards(self, statements: Sequence[ast.stmt]) -> set[str]:
        key = id(statements)
        if key not in self._scope_tag_guards:
            self._scope_tag_guards[key] = _tag_guard_names(statements)
        return self._scope_tag_guards[key]

    @staticmethod
    def _assignment_targets(node: ast.Assign | ast.AnnAssign) -> set[str]:
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        return {
            child.id
            for target in targets
            for child in ast.walk(target)
            if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store)
        }

    def _scope_candidate_roots(
        self,
        function: ast.AsyncFunctionDef | ast.FunctionDef | None,
        module_info: ModuleInfo,
    ) -> frozenset[str]:
        cache_key = (module_info.name, id(function) if function is not None else 0)
        if cache_key in self._candidate_roots:
            return self._candidate_roots[cache_key]
        roots = {
            local for local, target in module_info.imports.items() if target == "vllm" or target.startswith("vllm.")
        }
        nodes = list(_function_scope_nodes(function)) if function is not None else list(module_info.tree.body)
        assignments: list[tuple[set[str], str | None]] = []
        for node in nodes:
            if isinstance(node, ast.Import):
                roots.update(
                    alias.asname or alias.name.split(".", 1)[0]
                    for alias in node.names
                    if alias.name == "vllm" or alias.name.startswith("vllm.")
                )
            elif (
                isinstance(node, ast.ImportFrom)
                and node.module
                and (node.module == "vllm" or node.module.startswith("vllm."))
            ):
                roots.update(alias.asname or alias.name for alias in node.names if alias.name != "*")
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                value = node.value
                expression_node = value.func if isinstance(value, ast.Call) else value
                assignments.append((self._assignment_targets(node), _expression_name(expression_node)))
                if isinstance(node, ast.AnnAssign):
                    annotation = _annotation_reference(node.annotation)
                    annotation_root = annotation.split(".", 1)[0] if annotation else None
                    assignments.append((self._assignment_targets(node), annotation_root))
        changed = True
        while changed:
            changed = False
            for targets, expression in assignments:
                if expression is None or expression.split(".", 1)[0] not in roots:
                    continue
                additions = targets - roots
                if additions:
                    roots.update(additions)
                    changed = True
        result = frozenset(roots)
        self._candidate_roots[cache_key] = result
        return result

    def _constructed_instance_bindings(
        self,
        function: ast.AsyncFunctionDef | ast.FunctionDef,
        module_info: ModuleInfo,
    ) -> dict[str, ast.AnnAssign | ast.Call | None]:
        cache_key = (module_info.name, id(function))
        if cache_key in self._constructed_instances:
            return self._constructed_instances[cache_key]
        candidates: dict[str, list[ast.AnnAssign | ast.Call | None]] = {}
        instance_candidates: set[str] = set()
        for node in _function_scope_nodes(function):
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            targets = self._assignment_targets(node)
            if isinstance(node, ast.AnnAssign) and node.value is not None and node in function.body:
                reference = _annotation_reference(node.annotation)
                if reference is not None:
                    for target in targets:
                        candidates.setdefault(target, []).append(node)
                        instance_candidates.add(target)
                    continue
            if isinstance(node.value, ast.Call):
                for target in targets:
                    candidates.setdefault(target, []).append(node.value)
                    instance_candidates.add(target)
            else:
                for target in targets:
                    candidates.setdefault(target, []).append(None)
        bindings = {
            name: values[0] if len(values) == 1 else None
            for name, values in candidates.items()
            if name in instance_candidates
        }
        self._constructed_instances[cache_key] = bindings
        return bindings

    @staticmethod
    def _no_fallback(_node: ast.AST) -> set[str | None]:
        return {None}

    def _module_reference(
        self,
        expression: ast.AST,
        *,
        module_info: ModuleInfo,
        line: int,
    ) -> str | None:
        """Resolve a name from module flow at one exact program point.

        ``ModuleInfo.imports`` is an index of discovered imports, not a proof
        that the imported binding still owns the name at this callsite.  Use
        the shared normal-path scope interpreter so assignment, ``del`` and
        conditional rebinding remain fail-closed.
        """

        variants = _scope_reference_variants(
            expression,
            statements=module_info.tree.body,
            line=line,
            tag_guard_names=self._tag_guards(module_info.tree.body),
            module=module_info.name,
            is_package=module_info.is_package,
            fallback=self._no_fallback,
        )
        concrete = {item for item in variants if item is not None}
        if len(variants) != 1 or len(concrete) != 1:
            return None
        result = next(iter(concrete))
        return result if result.startswith("vllm.") else None

    def _module_fallback(
        self,
        module_info: ModuleInfo,
        blocked_names: set[str] | frozenset[str],
    ) -> Callable[[ast.AST], set[str | None]]:
        final_line = (
            max(
                (
                    getattr(statement, "end_lineno", getattr(statement, "lineno", 0))
                    for statement in module_info.tree.body
                ),
                default=0,
            )
            + 1
        )

        def resolve(node: ast.AST) -> set[str | None]:
            expression = _expression_name(node)
            if expression is None or expression.split(".", 1)[0] in blocked_names:
                return {None}
            resolved = self._module_reference(
                node,
                module_info=module_info,
                line=final_line,
            )
            return {resolved} if resolved is not None else {None}

        return resolve

    @staticmethod
    def _name_reassigned_before(
        function: ast.AsyncFunctionDef | ast.FunctionDef,
        name: str,
        point: ast.AST,
    ) -> bool:
        point_position = (
            getattr(point, "lineno", 0),
            getattr(point, "col_offset", 0),
        )
        for candidate in _function_scope_nodes(function):
            candidate_position = (
                getattr(candidate, "lineno", point_position[0]),
                getattr(candidate, "col_offset", 0),
            )
            if candidate_position >= point_position:
                continue
            if isinstance(candidate, ast.Name) and isinstance(candidate.ctx, (ast.Del, ast.Store)):
                if candidate.id == name:
                    return True
            if isinstance(candidate, (ast.Import, ast.ImportFrom)):
                bound = (
                    {alias.asname or alias.name.split(".", 1)[0] for alias in candidate.names}
                    if isinstance(candidate, ast.Import)
                    else {alias.asname or alias.name for alias in candidate.names if alias.name != "*"}
                )
                if name in bound:
                    return True
        return False

    def _outer_function_shadows(
        self,
        node: ast.AST,
        root: str,
        parents: dict[int, ast.AST],
        nearest_function: ast.AsyncFunctionDef | ast.FunctionDef | None,
    ) -> bool:
        current = parents.get(id(nearest_function)) if nearest_function is not None else parents.get(id(node))
        while current is not None:
            if isinstance(current, (ast.AsyncFunctionDef, ast.FunctionDef)) and root in self._local_names(current):
                return True
            current = parents.get(id(current))
        return False

    def _class_name(self, node: ast.AST, parents: dict[int, ast.AST], module: str) -> str | None:
        classes: list[str] = []
        current = parents.get(id(node))
        while current is not None:
            if isinstance(current, ast.ClassDef):
                classes.append(current.name)
            current = parents.get(id(current))
        return f"{module}.{'.'.join(reversed(classes))}" if classes else None

    def _self_or_super_target(
        self,
        node: ast.Call,
        parents: dict[int, ast.AST],
        module: str,
    ) -> tuple[str, str, str, str, str | None, str] | None:
        if not isinstance(node.func, ast.Attribute):
            return None
        is_super = (
            isinstance(node.func.value, ast.Call)
            and isinstance(node.func.value.func, ast.Name)
            and node.func.value.func.id == "super"
            and not node.func.value.args
            and not node.func.value.keywords
        )
        function = _nearest(node, parents, (ast.FunctionDef, ast.AsyncFunctionDef))
        receiver = None
        if isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
            positional = [*function.args.posonlyargs, *function.args.args]
            receiver = positional[0].arg if positional else None
        is_receiver = isinstance(node.func.value, ast.Name) and node.func.value.id == receiver
        if not is_super and not is_receiver:
            return None
        if (
            is_receiver
            and isinstance(function, (ast.AsyncFunctionDef, ast.FunctionDef))
            and receiver is not None
            and self._name_reassigned_before(function, receiver, node)
        ):
            return None
        class_name = self._class_name(node, parents, module)
        if class_name is None:
            return None
        mro = self.engine._linearized_mro(class_name)
        if not mro.complete:
            return None
        owners = mro.owners[1:] if is_super else mro.owners
        resolution = self.engine._effective_method_resolution(owners, node.func.attr)
        if (
            len(resolution.callable_owners) == 1
            and not resolution.may_be_missing
            and not resolution.may_be_non_callable
            and not resolution.has_unresolved_value
        ):
            target = f"{resolution.callable_owners[0]}.{node.func.attr}"
            return (
                (
                    target,
                    "instance",
                    class_name,
                    node.func.attr,
                    None,
                    "new_exact",
                )
                if target.startswith("vllm.")
                else None
            )
        definitely_missing = (
            not resolution.callable_owners
            and resolution.may_be_missing
            and not resolution.may_be_non_callable
            and not resolution.has_unresolved_value
            and not resolution.blocking_owners
            and not hasattr(object, node.func.attr)
        )
        if not definitely_missing:
            return None
        lookup_root = next(
            (owner for owner in owners if owner.startswith("vllm.")),
            None,
        )
        if lookup_root is None:
            return None
        return (
            f"{lookup_root}.{node.func.attr}",
            "instance",
            class_name,
            node.func.attr,
            lookup_root,
            "old_fallback_super" if is_super else "old_fallback_self",
        )

    def _annotated_instance_target(
        self,
        node: ast.Call,
        function: ast.AsyncFunctionDef | ast.FunctionDef | None,
        module_info: ModuleInfo,
    ) -> tuple[str, str, str, str] | None:
        if function is None or not isinstance(node.func, ast.Attribute) or not isinstance(node.func.value, ast.Name):
            return None
        root = node.func.value.id
        arguments = [*function.args.posonlyargs, *function.args.args, *function.args.kwonlyargs]
        argument = next((item for item in arguments if item.arg == root), None)
        reference = _annotation_reference(argument.annotation) if argument is not None else None
        if reference is None or self._name_reassigned_before(function, root, node):
            return None
        try:
            annotation_expression = ast.parse(reference, mode="eval").body
        except SyntaxError:
            return None
        resolved = self._module_reference(
            annotation_expression,
            module_info=module_info,
            line=getattr(function, "lineno", 0),
        )
        if resolved is None:
            return None
        target = f"{resolved}.{node.func.attr}"
        return (target, "instance", resolved, node.func.attr)

    def _resolve_in_scope(
        self,
        expression: ast.AST,
        *,
        function: ast.AsyncFunctionDef | ast.FunctionDef | None,
        module_info: ModuleInfo,
        line: int,
    ) -> str | None:
        expression_name = _expression_name(expression)
        if expression_name is None:
            return None
        local_names = self._local_names(function)
        if function is None:
            return self._module_reference(
                expression,
                module_info=module_info,
                line=line,
            )
        statements: Sequence[ast.stmt] = function.body
        variants = _scope_reference_variants(
            expression,
            statements=statements,
            line=line,
            tag_guard_names=self._tag_guards(statements),
            module=module_info.name,
            is_package=module_info.is_package,
            fallback=self._module_fallback(module_info, local_names),
        )
        concrete = {item for item in variants if item is not None}
        if len(variants) != 1 or len(concrete) != 1:
            return None
        result = next(iter(concrete))
        return result if result.startswith("vllm.") else None

    def _constructed_instance_target(
        self,
        node: ast.Call,
        function: ast.AsyncFunctionDef | ast.FunctionDef | None,
        module_info: ModuleInfo,
    ) -> tuple[str, str, str, str] | None:
        if not isinstance(node.func, ast.Attribute):
            return None
        annotation_reference: str | None = None
        if isinstance(node.func.value, ast.Call):
            reference = self._resolve_in_scope(
                node.func.value.func,
                function=function,
                module_info=module_info,
                line=getattr(node.func.value, "lineno", getattr(node, "lineno", 0)),
            )
            if reference is None:
                return None
            # This remains a candidate receiver path, not proof that the
            # symbol is a class.  Old/new endpoint resolution must each prove
            # the class and inherited member independently.
            return (f"{reference}.{node.func.attr}", "instance", reference, node.func.attr)
        if function is not None and isinstance(node.func.value, ast.Name):
            root = node.func.value.id
            binding = self._constructed_instance_bindings(function, module_info).get(root)
            if isinstance(binding, ast.AnnAssign):
                binding_end = (
                    getattr(binding, "end_lineno", getattr(binding, "lineno", 0)),
                    getattr(binding, "end_col_offset", getattr(binding, "col_offset", 0)),
                )
                call_start = (
                    getattr(node, "lineno", 0),
                    getattr(node, "col_offset", 0),
                )
                if binding_end > call_start:
                    return None
                annotation_reference = _annotation_reference(binding.annotation)
            elif isinstance(binding, ast.Call):
                binding_end = (
                    getattr(binding, "end_lineno", getattr(binding, "lineno", 0)),
                    getattr(binding, "end_col_offset", getattr(binding, "col_offset", 0)),
                )
                call_start = (
                    getattr(node, "lineno", 0),
                    getattr(node, "col_offset", 0),
                )
                if binding_end > call_start:
                    return None
                reference = self._resolve_in_scope(
                    node.func.value,
                    function=function,
                    module_info=module_info,
                    line=getattr(node, "lineno", 0),
                )
                if reference is None and binding_end[0] == call_start[0]:
                    reference = self._resolve_in_scope(
                        binding.func,
                        function=function,
                        module_info=module_info,
                        line=getattr(binding, "lineno", getattr(node, "lineno", 0)),
                    )
                if reference is None:
                    return None
                return (f"{reference}.{node.func.attr}", "instance", reference, node.func.attr)
        if annotation_reference is None:
            return None
        try:
            annotation_expression = ast.parse(annotation_reference, mode="eval").body
        except SyntaxError:
            return None
        reference = self._module_reference(
            annotation_expression,
            module_info=module_info,
            line=getattr(function, "lineno", 0),
        )
        if reference is None:
            return None
        target = f"{reference}.{node.func.attr}"
        return (target, "instance", reference, node.func.attr)

    def _resolved_access_kind(
        self,
        node: ast.Call,
        target: str,
        function: ast.AsyncFunctionDef | ast.FunctionDef | None,
        module_info: ModuleInfo,
    ) -> str | None:
        if function is None or not isinstance(node.func, ast.Attribute) or not isinstance(node.func.value, ast.Name):
            return self._access_kind(node, target)
        root = node.func.value.id
        bindings = self._constructed_instance_bindings(function, module_info)
        if root not in bindings:
            return self._access_kind(node, target)
        binding = bindings[root]
        if isinstance(binding, ast.AnnAssign):
            return "instance"
        return None

    @staticmethod
    def _access_kind(_node: ast.Call, _target: str) -> str:
        # A bare imported symbol can be a function or a class, and an
        # attribute can belong to either a module or a class.  Preserve that
        # syntax-neutral fact; each old/new snapshot derives its own binding.
        return "direct"

    def _triton_launch_target(self, target: str) -> bool:
        """Prove that one subscript launch resolves to a plain Triton JIT kernel."""

        callable_info = self.engine.upstream.find_callable(target)
        return callable_info is not None and callable_info.decorator_references == (_TRITON_JIT_DECORATOR,)

    def discover(self) -> list[DirectCallDependency]:
        dependencies: list[DirectCallDependency] = []
        self.historical_candidates = []
        for module_info in self.engine.downstream.modules.values():
            tree = module_info.tree
            parents = _parents(tree)
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call) or _under_version_guard(node, parents):
                    continue
                invocation_kind = "python_call"
                callable_node = node.func
                if isinstance(node.func, ast.Subscript):
                    invocation_kind = _TRITON_KERNEL_PROTOCOL
                    callable_node = node.func.value
                function_node = _nearest(node, parents, (ast.FunctionDef, ast.AsyncFunctionDef))
                function = function_node if isinstance(function_node, (ast.FunctionDef, ast.AsyncFunctionDef)) else None
                special = None
                if invocation_kind == "python_call":
                    special = self._self_or_super_target(node, parents, module_info.name)
                    special = special or self._annotated_instance_target(node, function, module_info)
                expression = _expression_name(callable_node)
                root = expression.split(".", 1)[0] if expression is not None else None
                candidate_roots = self._scope_candidate_roots(function, module_info)
                if root is not None and self._outer_function_shadows(
                    node,
                    root,
                    parents,
                    function,
                ):
                    continue
                may_be_constructed = invocation_kind == "python_call" and (
                    (
                        isinstance(callable_node, ast.Attribute)
                        and isinstance(callable_node.value, ast.Call)
                        and (_expression_name(callable_node.value.func) or "").split(".", 1)[0] in candidate_roots
                    )
                    or root in candidate_roots
                )
                if special is None and may_be_constructed:
                    special = self._constructed_instance_target(node, function, module_info)
                if special is not None:
                    targets = {special[0]}
                    access_kind = special[1]
                    receiver_type = special[2]
                    member = special[3]
                    lookup_root = special[4] if len(special) > 4 else None
                    resolution_basis = special[5] if len(special) > 5 else "new_exact"
                else:
                    if root not in candidate_roots:
                        continue
                    resolved = self._resolve_in_scope(
                        callable_node,
                        function=function,
                        module_info=module_info,
                        line=getattr(node, "lineno", 0),
                    )
                    if resolved is None or not resolved.startswith("vllm."):
                        continue
                    if invocation_kind == _TRITON_KERNEL_PROTOCOL and not self._triton_launch_target(resolved):
                        continue
                    targets = {resolved}
                    access_kind = self._resolved_access_kind(
                        node,
                        resolved,
                        function,
                        module_info,
                    )
                    if access_kind is None:
                        continue
                    receiver_type = None
                    member = callable_node.attr if isinstance(callable_node, ast.Attribute) else None
                    lookup_root = None
                    resolution_basis = "new_exact"
                target = next(iter(targets))
                if not target.startswith("vllm."):
                    continue
                scope_node = function or tree
                owner = self._class_name(node, parents, module_info.name)
                dependency = DirectCallDependency(
                    target=target,
                    access_kind=access_kind,
                    file=module_info.file,
                    line=getattr(node, "lineno", 0),
                    column=getattr(node, "col_offset", 0),
                    owner=owner.rsplit(".", 1)[-1] if owner else None,
                    scope=function.name if function is not None else None,
                    callee=ast.unparse(node.func),
                    call_shape=call_shape(node),
                    return_use=infer_return_use(node, parents, scope_node),
                    receiver_type=receiver_type,
                    member=member,
                    invocation_kind=invocation_kind,
                    lookup_root=lookup_root,
                    resolution_basis=resolution_basis,
                )
                if lookup_root is not None:
                    self.historical_candidates.append(dependency)
                else:
                    dependencies.append(dependency)
        unique = {
            (
                item.file,
                item.line,
                item.column,
                item.target,
                item.callee,
                item.access_kind,
                json.dumps(item.call_shape.as_dict(), sort_keys=True, separators=(",", ":")),
                json.dumps(item.return_use.as_dict(), sort_keys=True, separators=(",", ":")),
                item.receiver_type,
                item.member,
                item.invocation_kind,
            ): item
            for item in dependencies
        }
        return [unique[key] for key in sorted(unique)]


__all__ = [
    "CallShape",
    "DirectCallDependency",
    "DirectCallDetector",
    "ReturnContract",
    "ReturnShape",
    "ReturnUse",
    "bind_call_shape",
    "call_shape",
    "infer_return_contract",
    "replacement_return_compatible",
    "return_contract_from_dict",
    "return_use_compatible",
]
