# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Source-only parameter effects for uniquely bound ordinary helper functions.

An effect proof includes aliases returned inside fresh containers. Unknown
calls, escapes, descriptors and unsupported control flow are barriers; neither
function names nor a lack of explicit attribute stores alone prove read-only.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, replace

from .type_flow import (
    Lookup,
    Resolve,
    TypeShape,
    dictionary_get_safe,
    field_type,
    immutable_value,
    ordinary_instance_target,
    type_step,
)


@dataclass(frozen=True)
class EffectValue:
    origins: frozenset[str] = frozenset()
    shape: TypeShape | None = None
    owned: int | None = None
    path: tuple[str, ...] | None = None


@dataclass(frozen=True)
class HelperSummary:
    unsafe: frozenset[str]
    returned: frozenset[str]
    return_shape: TypeShape | None
    return_path: tuple[str, ...] | None


def _merge(left: EffectValue, right: EffectValue) -> EffectValue:
    shape = left.shape if left.shape == right.shape else None
    if left.owned is not None and left.owned == right.owned and left.shape is not None and right.shape is not None:
        if left.shape.reference == right.shape.reference:
            # An untouched fresh empty container and its populated loop branch
            # have the same element contract. Heterogeneous containers carry
            # explicit unknown arguments, not the empty marker.
            if not left.shape.arguments:
                shape = right.shape
            elif not right.shape.arguments:
                shape = left.shape
    return EffectValue(
        left.origins | right.origins,
        shape,
        left.owned if left.owned == right.owned else None,
        left.path if left.path == right.path else None,
    )


_EMPTY = EffectValue()


class HelperEffects:
    """Bounded alias/effect interpretation, parameterized by actual input types."""

    def __init__(
        self, function: ast.FunctionDef, shapes: dict[str, TypeShape | None], resolve: Resolve, lookup: Lookup
    ):
        self.resolve = resolve
        self.lookup = lookup
        self.unsafe: set[str] = set()
        self.returns: list[EffectValue] = []
        arguments = [*function.args.posonlyargs, *function.args.args, *function.args.kwonlyargs]
        self.parameters = {arg.arg for arg in arguments}
        self.globals = {name for node in ast.walk(function) if isinstance(node, ast.Global) for name in node.names}
        self.locals = self.parameters | {
            node.id for node in ast.walk(function) if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
        }
        env = {name: EffectValue(frozenset({name}), shapes.get(name), path=(name,)) for name in self.parameters}
        self.statements(function.body, env)
        returned = frozenset(origin for value in self.returns for origin in value.origins)
        return_shapes = {value.shape for value in self.returns}
        return_paths = {value.path for value in self.returns}
        self.summary = HelperSummary(
            frozenset(self.unsafe),
            returned,
            next(iter(return_shapes)) if len(return_shapes) == 1 else None,
            next(iter(return_paths)) if len(return_paths) == 1 else None,
        )

    @staticmethod
    def join(left: dict[str, EffectValue], right: dict[str, EffectValue]) -> dict[str, EffectValue]:
        return {name: _merge(left.get(name, _EMPTY), right.get(name, _EMPTY)) for name in left.keys() | right.keys()}

    def reference(self, node: ast.AST) -> str | None:
        expression = ast.unparse(node)
        return None if expression.split(".", 1)[0] in self.locals else self.resolve(expression)

    def derived(self, value: EffectValue, shape: TypeShape | None, step: str) -> EffectValue:
        if immutable_value(shape):
            return EffectValue(shape=shape)
        return EffectValue(value.origins, shape, path=(*value.path, step) if value.path is not None else None)

    def expression(self, node: ast.AST | None, env: dict[str, EffectValue]) -> EffectValue:
        if isinstance(node, ast.Name):
            return env.get(node.id, _EMPTY)
        if isinstance(node, ast.Constant):
            return EffectValue(shape=TypeShape(type(node.value).__name__))
        if isinstance(node, ast.Attribute):
            value = self.expression(node.value, env)
            shape = field_type(value.shape.reference, node.attr, self.lookup) if value.shape else None
            if shape is None:
                self.unsafe.update(value.origins)  # Unknown getters may have side effects.
            return self.derived(value, shape, f"field:{node.attr}")
        if isinstance(node, ast.Subscript):
            value = self.expression(node.value, env)
            self.expression(node.slice, env)
            step = "index:" + str(node.slice.value) if isinstance(node.slice, ast.Constant) else "index:dynamic"
            shape = type_step(value.shape, step, self.lookup) if value.shape else None
            if shape is None and value.owned is None:
                self.unsafe.update(value.origins)
            return self.derived(value, shape, step)
        if isinstance(node, (ast.List, ast.Tuple, ast.Set, ast.Dict)):
            if isinstance(node, ast.Dict):
                keys = [self.expression(key, env) for key in node.keys]
                values = [self.expression(value, env) for value in node.values]
                key_shapes, shapes = {v.shape for v in keys}, {v.shape for v in values}
                key_shape = next(iter(key_shapes)) if len(key_shapes) == 1 else None
                value_shape = next(iter(shapes)) if len(shapes) == 1 else None
                args: tuple[TypeShape, ...] = ()
                if key_shape is not None and value_shape is not None:
                    args = (key_shape, value_shape)
                elif keys or values:
                    args = (TypeShape("unknown"), TypeShape("unknown"))
                return EffectValue(
                    frozenset(o for v in (*keys, *values) for o in v.origins), TypeShape("dict", args), id(node)
                )
            values = [self.expression(value, env) for value in node.elts]
            shapes = {v.shape for v in values}
            element = next(iter(shapes)) if len(shapes) == 1 else None
            args = (element,) if element is not None else (TypeShape("unknown"),) if values else ()
            return EffectValue(
                frozenset(o for v in values for o in v.origins), TypeShape(type(node).__name__.lower(), args), id(node)
            )
        if isinstance(node, ast.Call):
            return self.call(node, env)
        if isinstance(node, (ast.ListComp, ast.SetComp, ast.GeneratorExp)):
            local = dict(env)
            for generator in node.generators:
                value = self.expression(generator.iter, local)
                shape = type_step(value.shape, "item", self.lookup) if value.shape and not generator.is_async else None
                if shape is None:
                    self.unsafe.update(value.origins)
                self.bind(generator.target, self.derived(value, shape, "item"), local)
                for condition in generator.ifs:
                    self.expression(condition, local)
            result = self.expression(node.elt, local)
            return EffectValue(
                result.origins,
                TypeShape("list", (result.shape,)) if result.shape else None,
                id(node),
                (*result.path, "collect") if result.path is not None else None,
            )
        if isinstance(node, ast.IfExp):
            self.expression(node.test, env)
            return _merge(self.expression(node.body, env), self.expression(node.orelse, env))
        if isinstance(node, (ast.Compare, ast.BoolOp, ast.UnaryOp, ast.BinOp)):
            values = [self.expression(child, env) for child in ast.iter_child_nodes(node)]
            # Scalar operations cannot mutate the parameter object; overloaded
            # operations on borrowed objects are not a read-only proof.
            self.unsafe.update(o for value in values for o in value.origins)
            return _EMPTY
        if isinstance(node, ast.NamedExpr):
            value = self.expression(node.value, env)
            self.bind(node.target, value, env)
            return value
        if node is not None:
            # Unmodeled expressions can execute or capture parameter objects.
            origins = {o for n in ast.walk(node) if isinstance(n, ast.Name) for o in env.get(n.id, _EMPTY).origins}
            self.unsafe.update(origins)
        return _EMPTY

    def call(self, node: ast.Call, env: dict[str, EffectValue]) -> EffectValue:
        values = [self.expression(value, env) for value in (*node.args, *(kw.value for kw in node.keywords))]
        reference = self.reference(node.func)
        if reference in {"len", "builtins.len"} and len(values) == 1 and values[0].shape is not None:
            if values[0].shape.reference.removeprefix("builtins.") in {"list", "tuple", "dict", "set", "str", "bytes"}:
                return EffectValue(shape=TypeShape("int"))
        if reference in {"isinstance", "builtins.isinstance"} and len(node.args) == 2:
            if ordinary_instance_target(
                node.args[1], lambda name: self.reference(ast.parse(name, mode="eval").body), self.lookup
            ):
                return EffectValue(shape=TypeShape("bool"))
        if isinstance(node.func, ast.Attribute):
            receiver = self.expression(node.func.value, env)
            query_shape = (
                TypeShape("empty_dict")
                if receiver.owned is not None and receiver.shape == TypeShape("dict")
                else receiver.shape
            )
            if (
                query_shape is not None
                and node.func.attr == "get"
                and not node.keywords
                and len(node.args) in {1, 2}
                and not any(isinstance(arg, ast.Starred) for arg in node.args)
                and dictionary_get_safe(query_shape, values[0].shape)
            ):
                default = values[1] if len(values) == 2 else _EMPTY
                return (
                    default
                    if query_shape.reference == "empty_dict"
                    else EffectValue(receiver.origins | default.origins)
                )
            shape = type_step(receiver.shape, node.func.attr, self.lookup) if receiver.shape else None
            if node.func.attr in {"items", "values", "keys"} and shape is not None and not values:
                return self.derived(receiver, shape, node.func.attr)
            self.unsafe.update(receiver.origins)
        else:
            self.unsafe.update(self.expression(node.func, env).origins)
        self.unsafe.update(origin for value in values for origin in value.origins)
        return _EMPTY

    def bind(self, target: ast.AST, value: EffectValue, env: dict[str, EffectValue]) -> None:
        if isinstance(target, ast.Name):
            if target.id in self.globals:
                self.unsafe.update(value.origins)
            env[target.id] = value
            return
        if isinstance(target, (ast.Tuple, ast.List)):
            for index, item in enumerate(target.elts):
                shape = type_step(value.shape, f"index:{index}", self.lookup) if value.shape else None
                self.bind(item, self.derived(value, shape, f"index:{index}"), env)
            return
        if isinstance(target, ast.Subscript):
            receiver = self.expression(target.value, env)
            key = self.expression(target.slice, env)
            if receiver.owned is not None:
                shape = receiver.shape
                if (
                    shape is not None
                    and shape.reference == "dict"
                    and key.shape is not None
                    and value.shape is not None
                ):
                    proposed = TypeShape("dict", (key.shape, value.shape))
                    shape = (
                        proposed
                        if not shape.arguments or shape == proposed
                        else TypeShape("dict", (TypeShape("unknown"), TypeShape("unknown")))
                    )
                updated = replace(receiver, origins=receiver.origins | value.origins, shape=shape)
                for name, candidate in list(env.items()):
                    if candidate.owned == receiver.owned:
                        env[name] = updated
                return
            self.unsafe.update(receiver.origins)
        elif isinstance(target, ast.Attribute):
            self.unsafe.update(self.expression(target.value, env).origins)
        self.unsafe.update(value.origins)  # Stored into an externally owned object.

    def statements(self, body: list[ast.stmt], env: dict[str, EffectValue]) -> None:
        for statement in body:
            if isinstance(statement, (ast.Assign, ast.AnnAssign)):
                if statement.value is None:
                    continue
                value = self.expression(statement.value, env)
                targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
                for target in targets:
                    self.bind(target, value, env)
            elif isinstance(statement, ast.Return):
                self.returns.append(self.expression(statement.value, env))
            elif isinstance(statement, (ast.Expr, ast.Assert)):
                self.expression(statement.value if isinstance(statement, ast.Expr) else statement.test, env)
            elif isinstance(statement, ast.If):
                self.expression(statement.test, env)
                left, right = dict(env), dict(env)
                test = statement.test
                if (
                    isinstance(test, ast.Call)
                    and self.reference(test.func) in {"isinstance", "builtins.isinstance"}
                    and len(test.args) == 2
                    and isinstance(test.args[0], ast.Name)
                ):
                    reference = self.reference(test.args[1])
                    if reference is not None and self.lookup(reference) is not None:
                        name = test.args[0].id
                        left[name] = replace(left.get(name, _EMPTY), shape=TypeShape(reference))
                self.statements(statement.body, left)
                self.statements(statement.orelse, right)
                env.update(self.join(left, right))
            elif isinstance(statement, ast.For):
                iterable = self.expression(statement.iter, env)
                shape = type_step(iterable.shape, "item", self.lookup) if iterable.shape else None
                if shape is None:
                    self.unsafe.update(iterable.origins)
                before = dict(env)
                for _ in range(16):
                    local = dict(env)
                    self.bind(statement.target, self.derived(iterable, shape, "item"), local)
                    self.statements(statement.body, local)
                    merged = self.join(before, local)
                    if merged == env:
                        break
                    env.update(merged)
                else:
                    self.unsafe.update(self.parameters)
                self.statements(statement.orelse, env)
            elif isinstance(statement, (ast.Pass, ast.Global)):
                continue
            else:
                self.unsafe.update(self.parameters)
