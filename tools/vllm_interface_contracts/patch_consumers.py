# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Trace scoped module patches whose constructor consumer moves to a factory.

Reuse the generator's verified installation and namespace bindings. A lexical
route change is review evidence unless the complete dispatch envelope is plain.
This does not infer patch effectiveness from callable signatures.
"""

from __future__ import annotations

import ast
import copy
from typing import TYPE_CHECKING

from .call_contracts import bind_call_shape, call_shape
from .construction_effects import construction_effects, has_construction_effect
from .generator import (
    InterfaceBoundaryGenerator,
    Relation,
    _expression_name,
    _function_local_names,
    _function_scope_nodes,
    _scope_final_bindings,
    _tag_guard_names,
)
from .models import CompatibilityState, RangeFinding, SourceEndpoint

if TYPE_CHECKING:
    from .range_analysis import GitSnapshot


def _unique_binding(body: list[ast.stmt], name: str) -> ast.AST | None:
    bindings = _scope_final_bindings(body, _tag_guard_names(body)).get(name, ())
    return bindings[0].node if len(bindings) == 1 and bindings[0].kind != "unbound" else None


def _single_assignment(node: ast.AST) -> tuple[ast.expr, ast.expr] | None:
    if isinstance(node, ast.Assign) and len(node.targets) == 1:
        return node.targets[0], node.value
    if isinstance(node, ast.AnnAssign) and node.value is not None:
        return node.target, node.value
    return None


def _scoped_super(function: ast.FunctionDef, line: int, expression: str) -> ast.Call | None:
    """Prove a patch/save/try-super/finally-restore interval without intervening calls."""
    for offset, statement in enumerate(function.body):
        if statement.lineno != line or offset < 1 or offset + 1 >= len(function.body):
            continue
        patch = _single_assignment(statement)
        saved = _single_assignment(function.body[offset - 1])
        protected = function.body[offset + 1]
        if (
            patch is None
            or saved is None
            or _expression_name(patch[0]) != expression
            or _expression_name(saved[1]) != expression
            or not isinstance(saved[0], ast.Name)
            or not isinstance(protected, ast.Try)
            or protected.handlers
            or protected.orelse
            or len(protected.body) != 1
            or len(protected.finalbody) != 1
        ):
            continue
        restored = _single_assignment(protected.finalbody[0])
        call_statement = protected.body[0]
        if (
            restored is None
            or _expression_name(restored[0]) != expression
            or not isinstance(restored[1], ast.Name)
            or restored[1].id != saved[0].id
            or not isinstance(call_statement, ast.Expr)
            or not isinstance(call_statement.value, ast.Call)
        ):
            continue
        call = call_statement.value
        member = call.func
        if (
            isinstance(member, ast.Attribute)
            and member.attr == "__init__"
            and isinstance(member.value, ast.Call)
            and isinstance(member.value.func, ast.Name)
            and member.value.func.id == "super"
            and not member.value.args
            and not member.value.keywords
        ):
            return call
    return None


def _call_shape(node: ast.Call) -> str:
    return ast.dump(ast.Call(func=ast.Name(id="callee", ctx=ast.Load()), args=node.args, keywords=node.keywords))


def _plain_constructor(node: ast.FunctionDef, call: ast.Call) -> bool:
    if node.decorator_list or len(node.body) != 1:
        return False
    assignment = _single_assignment(node.body[0])
    parameters = [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]
    names = {parameter.arg for parameter in parameters}

    def known(value: ast.expr) -> bool:
        return isinstance(value, ast.Constant) or isinstance(value, ast.Name) and value.id in names

    return (
        assignment is not None
        and assignment[1] is call
        and isinstance(assignment[0], ast.Attribute)
        and isinstance(assignment[0].value, ast.Name)
        and bool(parameters)
        and assignment[0].value.id == parameters[0].arg
        and all(known(value) for value in call.args)
        and all(keyword.arg is not None and known(keyword.value) for keyword in call.keywords)
    )


def _old_call_obligations(
    relation: Relation,
    engine: InterfaceBoundaryGenerator,
    old: GitSnapshot,
    parent: str,
    super_call: ast.Call,
    consumer_call: ast.Call,
) -> tuple[bool | None, str]:
    """Check old parent entry and installed replacement with shared binding rules."""
    # Delay the composite snapshot import until the range module is initialized.
    from .range_analysis import _ConstructorSnapshot

    parent_endpoint = old.call_endpoint(parent, "constructor")
    parent_ok, reason = bind_call_shape(parent_endpoint.signature, call_shape(super_call))
    if parent_ok is not True:
        return parent_ok, f"old parent constructor: {reason}"
    installed = relation.installed_signature_contract
    signature = installed.bound_call_signature if installed is not None and installed.status == "exact" else None
    if signature is None:
        module = relation.downstream_file.removesuffix(".py").replace("/", ".")
        reference = ".".join(part for part in (module, relation.downstream_owner, relation.downstream_name) if part)
        bindings = engine.downstream.find_final_bindings(reference)
        if len(bindings) == 1 and isinstance(bindings[0].node, ast.ClassDef):
            view = _ConstructorSnapshot(old, engine.downstream)
            signature = view.call_endpoint(reference, "constructor").signature
    replacement_ok, reason = bind_call_shape(signature, call_shape(consumer_call))
    return replacement_ok, f"old installed replacement: {reason}"


def _dispatch_effects(
    engine: InterfaceBoundaryGenerator,
    file: str,
    function: ast.FunctionDef,
    patch_line: int,
) -> frozenset[str]:
    """Exclude only the already-proven temporary patch and its matching restore."""
    index = next(index for index, node in enumerate(function.body) if node.lineno == patch_line)
    protected = function.body[index + 1]
    assert isinstance(protected, ast.Try)
    excluded = {patch_line, protected.finalbody[0].lineno}

    class RemoveProvenWrites(ast.NodeTransformer):
        def visit_Assign(self, node: ast.Assign) -> ast.AST:
            return ast.copy_location(ast.Pass(), node) if node.lineno in excluded else node

        def visit_AnnAssign(self, node: ast.AnnAssign) -> ast.AST:
            return ast.copy_location(ast.Pass(), node) if node.lineno in excluded else node

    effects: set[str] = set()
    for module in engine.downstream.modules.values():
        tree = module.tree
        if module.file == file:
            transformed = RemoveProvenWrites().visit(copy.deepcopy(tree))
            assert isinstance(transformed, ast.Module)
            tree = transformed
        effects.update(construction_effects(tree, module.name, module.is_package))
    return frozenset(effects)


def patch_consumer_findings(
    relation: Relation,
    engine: InterfaceBoundaryGenerator,
    old: GitSnapshot,
    new: GitSnapshot,
) -> list[RangeFinding]:
    """Compare proven scoped installations, keeping uncertain effects review-only."""
    if relation.relation != "monkey_patch" or relation.upstream_owner is not None:
        return []
    target_module = relation.upstream_file.removesuffix(".py").replace("/", ".")
    target = f"{target_module}.{relation.upstream_name}"
    results: list[RangeFinding] = []
    for evidence in relation.evidence:
        if not evidence.scope or not evidence.scope.endswith(".__init__") or evidence.guards:
            continue
        module_name = evidence.file.removesuffix(".py").replace("/", ".")
        module = engine.downstream.modules.get(module_name)
        installer = engine.downstream.find_callable(f"{module_name}.{evidence.scope}")
        child_name = f"{module_name}.{evidence.scope.rsplit('.', 1)[0]}"
        child_bindings = engine.downstream.find_final_bindings(child_name)
        child = child_bindings[0].node if len(child_bindings) == 1 else None
        if (
            module is None
            or installer is None
            or not isinstance(child, ast.ClassDef)
            or not isinstance(installer.node, ast.FunctionDef)
        ):
            continue
        function = installer.node
        super_call = _scoped_super(function, evidence.line, evidence.target_expression or "")
        if (
            super_call is None
            or "super" in _function_local_names(function)
            or _scope_final_bindings(module.tree.body, _tag_guard_names(module.tree.body)).get("super")
        ):
            continue
        bases, missing = engine._class_bases(child_name)
        if missing or len(bases) != 1 or not bases[0].startswith("vllm."):
            continue
        parent = bases[0]
        old_source, new_source = old._type_source(parent), new._type_source(parent)
        if old_source is None or new_source is None:
            continue
        old_init = _unique_binding(old_source.node.body, "__init__")
        new_init = _unique_binding(new_source.node.body, "__init__")
        if not isinstance(old_init, ast.FunctionDef) or not isinstance(new_init, ast.FunctionDef):
            continue
        parameters = [*new_init.args.posonlyargs, *new_init.args.args]
        if not parameters:
            continue
        receiver = parameters[0].arg
        old_locals = _function_local_names(old_init)
        old_calls = [
            node
            for node in _function_scope_nodes(old_init)
            if isinstance(node, ast.Call)
            and _expression_name(node.func) is not None
            and (_expression_name(node.func) or "").split(".")[0] not in old_locals
            and old_source.resolve(_expression_name(node.func) or "") == target
        ]
        for call in _function_scope_nodes(new_init):
            if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Attribute):
                continue
            if not isinstance(call.func.value, ast.Name) or call.func.value.id != receiver:
                continue
            factory = call.func.attr
            assignment_node = _unique_binding(new_source.node.body, factory)
            assignment = _single_assignment(assignment_node) if assignment_node is not None else None
            if assignment is None or new_source.resolve(_expression_name(assignment[1]) or "") != target:
                continue
            # A local override changes this dispatch independently of the module patch.
            if _scope_final_bindings(child.body, _tag_guard_names(child.body)).get(factory):
                continue
            matching = [previous for previous in old_calls if _call_shape(previous) == _call_shape(call)]
            if len(matching) != 1:
                continue
            previous = matching[0]
            old_root = (_expression_name(previous.func) or "").split(".")[0]
            installer_parameters = {
                parameter.arg
                for parameter in [*function.args.posonlyargs, *function.args.args, *function.args.kwonlyargs]
            }
            patch_index = next(
                index for index, statement in enumerate(function.body) if statement.lineno == evidence.line
            )
            plain = (
                not old_source.node.bases
                and not new_source.node.bases
                and not old_source.node.decorator_list
                and not new_source.node.decorator_list
                and not child.decorator_list
                and not child.keywords
                and not old_source.node.keywords
                and not new_source.node.keywords
                and not function.decorator_list
                and old_root not in old_locals
                and all(item is old_init for item in old_source.node.body)
                and all(item is new_init or item is assignment_node for item in new_source.node.body)
                and all(item is function for item in child.body)
                and all(isinstance(item, (ast.Import, ast.ImportFrom)) for item in function.body[: patch_index - 1])
                and _plain_constructor(old_init, previous)
                and _plain_constructor(new_init, call)
                and all(
                    isinstance(value, ast.Constant) or isinstance(value, ast.Name) and value.id in installer_parameters
                    for value in [*super_call.args, *(keyword.value for keyword in super_call.keywords)]
                )
                and not any(
                    isinstance(item, (ast.Call, ast.Await, ast.Yield))
                    for value in [*super_call.args, *(keyword.value for keyword in super_call.keywords)]
                    for item in ast.walk(value)
                )
            )
            if plain:
                effects = _dispatch_effects(engine, evidence.file, function, evidence.line)
                plain = (
                    not old_source.storage_effects
                    and not new_source.storage_effects
                    and not has_construction_effect(parent, effects)
                    and not has_construction_effect(child_name, effects)
                )
            old_call_ok: bool | None = None
            old_call_reason = "complex dispatch prevents an exact old-call proof"
            if plain:
                old_call_ok, old_call_reason = _old_call_obligations(
                    relation, engine, old, parent, super_call, previous
                )
                plain = old_call_ok is True
            # Complex constructors can mutate dispatch or execute descriptors.
            # Preserve the exact lexical evidence without inventing a runtime result.
            reason = "constructor moved from a live module binding to a captured class factory"
            results.append(
                RangeFinding(
                    finding_id=f"patch-consumer:{evidence.file}:{evidence.line}:{factory}:{call.lineno}",
                    classification="introduced_break" if plain else "analysis_unresolved",
                    relation="monkey_patch",
                    priority="P0" if plain else "P2",
                    action="modify" if plain else "review",
                    confidence="high" if plain else "medium",
                    upstream_old=SourceEndpoint(
                        old_source.file, parent.rsplit(".", 1)[-1], "__init__", previous.lineno
                    ),
                    upstream_new=SourceEndpoint(new_source.file, parent.rsplit(".", 1)[-1], "__init__", call.lineno),
                    downstream=SourceEndpoint(
                        evidence.file, evidence.scope.rsplit(".", 1)[0], "__init__", evidence.line
                    ),
                    old_state=CompatibilityState(
                        True, True if plain else None, "old constructor reads the patched module binding"
                    ),
                    new_state=CompatibilityState(True, False if plain else None, reason),
                    change=reason,
                    evidence=[evidence.as_dict()],
                    gates={
                        "relationship_verified": True,
                        "contract_changed": True,
                        "runtime_reachable": plain,
                        "version_lane_matches": True,
                    },
                    suggestion=(
                        f"Review and adapt {factory}; changing the module binding "
                        "no longer changes the captured factory."
                    ),
                    contract_kind="patch_consumer_route",
                    direction="patch_installation_to_upstream_consumer",
                    details={
                        "patch_target": target,
                        "factory_attribute": factory,
                        "super_call_line": super_call.lineno,
                        "old_call_line": previous.lineno,
                        "new_call_line": call.lineno,
                        "dispatch_proven": plain,
                        "old_call_compatible": old_call_ok,
                        "old_call_reason": old_call_reason,
                        "unresolved_reason": None if plain else old_call_reason,
                    },
                )
            )
    return results
