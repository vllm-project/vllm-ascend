# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import __future__

import ast
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest


@pytest.mark.parametrize("is_profiling", [False, True])
def test_v2_kv_cache_initialization_forwards_profiling(is_profiling):
    # The native xlite extension is optional in CPU CI. Execute the actual
    # override with a fake parent to exercise the profiling/cache lifecycle.
    source = Path(__file__).resolve().parents[3] / "vllm_ascend/xlite/xlite_model_runner.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    runner_class = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "XliteModelRunnerV2"
    )
    method = next(
        node for node in runner_class.body if isinstance(node, ast.FunctionDef) and node.name == "initialize_kv_cache"
    )

    class Parent:
        initialize_kv_cache = MagicMock()

    runner_class.bases = [ast.Name(id="Parent", ctx=ast.Load())]
    runner_class.body = [method]
    module = ast.Module(body=[runner_class], type_ignores=[])
    namespace: dict[str, Any] = {"Parent": Parent, "logger": MagicMock(), "_TensorRepr": MagicMock()}
    # Production annotations depend on the optional native runtime imports.
    exec(
        compile(ast.fix_missing_locations(module), str(source), "exec", flags=__future__.annotations.compiler_flag),
        namespace,
    )
    runner = namespace["XliteModelRunnerV2"]()
    metadata_type = object()
    runner.kv_caches = [object()]
    runner._get_flattened_kv_caches = MagicMock(return_value=runner.kv_caches)
    runner.runner_model = MagicMock()
    runner.runner_model.adapter_xlite_model._attn_metadata_type = metadata_type
    attention_group = MagicMock()
    attention_group.get_metadata_builder.return_value = SimpleNamespace(metadata_cls=metadata_type)
    runner.attn_groups = [[attention_group]]
    config, allocation_context = object(), object()

    runner.initialize_kv_cache(config, is_profiling=is_profiling, kv_cache_allocation_context=allocation_context)

    Parent.initialize_kv_cache.assert_called_once_with(
        config, is_profiling=is_profiling, kv_cache_allocation_context=allocation_context
    )
    if is_profiling:
        runner.runner_model.register_kv_caches.assert_not_called()
        runner._get_flattened_kv_caches.assert_not_called()
    else:
        runner.runner_model.register_kv_caches.assert_called_once_with(runner.kv_caches)
