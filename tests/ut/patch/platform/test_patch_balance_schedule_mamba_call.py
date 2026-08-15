# SPDX-License-Identifier: Apache-2.0
"""Regression guard for the balance scheduler's Mamba split call."""

import ast
import inspect
import sys
from pathlib import Path
from types import SimpleNamespace


def _find_method(path: Path, class_name: str, method_name: str) -> ast.FunctionDef:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    class_node = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    return next(node for node in class_node.body if isinstance(node, ast.FunctionDef) and node.name == method_name)


def _find_upstream_scheduler_path() -> Path:
    relative_path = Path("vllm/v1/core/sched/scheduler.py")
    for entry in sys.path:
        candidate = Path(entry or ".").resolve() / relative_path
        if candidate.is_file():
            return candidate
    raise AssertionError(f"could not find {relative_path.as_posix()} on sys.path")


def _find_guarded_mamba_call(schedule: ast.FunctionDef) -> tuple[ast.If, ast.Call]:
    matches: list[tuple[ast.If, ast.Call]] = []
    for node in ast.walk(schedule):
        if not isinstance(node, ast.If):
            continue
        test_nodes = list(ast.walk(node.test))
        has_mamba_guard = any(
            isinstance(part, ast.Attribute) and part.attr == "need_mamba_block_aligned_split" for part in test_nodes
        )
        has_non_async_guard = any(
            isinstance(part, ast.UnaryOp)
            and isinstance(part.op, ast.Not)
            and isinstance(part.operand, ast.Name)
            and part.operand.id == "load_kv_async"
            for part in test_nodes
        )
        if not (has_mamba_guard and has_non_async_guard):
            continue
        matches.extend(
            (node, part)
            for statement in node.body
            for part in ast.walk(statement)
            if isinstance(part, ast.Call)
            and isinstance(part.func, ast.Attribute)
            and part.func.attr == "_mamba_block_aligned_split"
        )
    assert len(matches) == 1
    return matches[0]


def _signature_from_method(method: ast.FunctionDef) -> inspect.Signature:
    assert not method.args.posonlyargs
    assert method.args.vararg is None
    assert not method.args.kwonlyargs
    assert method.args.kwarg is None
    first_default = len(method.args.args) - len(method.args.defaults)
    parameters = [
        inspect.Parameter(
            arg.arg,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            default=inspect.Parameter.empty if index < first_default else object(),
        )
        for index, arg in enumerate(method.args.args)
    ]
    return inspect.Signature(parameters)


def test_mamba_split_waiting_call_matches_upstream_signature():
    repo_root = Path(__file__).resolve().parents[4]
    balance_path = repo_root / "vllm_ascend/patch/platform/patch_balance_schedule.py"
    schedule = _find_method(balance_path, "BalanceScheduler", "schedule")
    guard, call = _find_guarded_mamba_call(schedule)

    guard_code = compile(ast.fix_missing_locations(ast.Expression(guard.test)), str(balance_path), "eval")
    scheduler = SimpleNamespace(need_mamba_block_aligned_split=True)
    assert eval(guard_code, {}, {"self": scheduler, "load_kv_async": False})

    assert not call.keywords
    assert [ast.unparse(arg) for arg in call.args] == [
        "request",
        "num_new_tokens",
        "num_new_local_computed_tokens",
        "num_external_computed_tokens",
    ]

    upstream_method = _find_method(
        _find_upstream_scheduler_path(),
        "Scheduler",
        "_mamba_block_aligned_split",
    )
    assert [arg.arg for arg in upstream_method.args.args] == [
        "self",
        "request",
        "num_new_tokens",
        "num_new_local_computed_tokens",
        "num_external_computed_tokens",
    ]
    # bind() applies Python's positional-argument count check without entering
    # the NPU-dependent scheduler implementation.
    upstream_signature = _signature_from_method(upstream_method)
    upstream_signature.bind(scheduler, *(object() for _ in call.args))

    for path in (repo_root / "vllm_ascend").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "_mamba_block_aligned_split"
            ):
                continue
            keyword_names = [keyword.arg for keyword in node.keywords]
            assert all(name is not None for name in keyword_names)
            try:
                upstream_signature.bind(
                    scheduler,
                    *(object() for _ in node.args),
                    **{name: object() for name in keyword_names if name is not None},
                )
            except TypeError as error:
                location = f"{path.relative_to(repo_root)}:{node.lineno}"
                raise AssertionError(f"{location} cannot bind upstream Mamba split signature") from error
