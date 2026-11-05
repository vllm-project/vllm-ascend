# SPDX-License-Identifier: Apache-2.0
"""310P MRV2 startup must not import modules removed with Model Runner V1."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def _tree(relative: str) -> ast.AST:
    return ast.parse((ROOT / relative).read_text())


def _direct_imports(body: list[ast.stmt]) -> set[str]:
    modules: set[str] = set()
    for node in body:
        if isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
        elif isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
    return modules


def _imported_modules(tree: ast.AST) -> set[str]:
    body = tree.body if isinstance(tree, (ast.Module, ast.If)) else []
    modules = _direct_imports(body)
    for node in body:
        if isinstance(node, ast.If):
            modules |= _imported_modules(node)
    return modules


def test_310p_worker_constructs_v2_runner_only() -> None:
    source = (ROOT / "vllm_ascend/_310p/worker_310p.py").read_text()
    assert "NPUModelRunner310V2" in source
    assert "worker.v1._310p.model_runner_310p" not in source
    assert "self.use_v2_model_runner = True" in source


def test_310p_graph_resolution_tolerates_removed_v2_flag() -> None:
    source = (ROOT / "vllm_ascend/_310p/worker/v2/model_runner.py").read_text()
    assert "inspect.signature(resolve_cudagraph)" in source
    assert 'resolve_kwargs["use_v2_model_runner"] = True' in source
    assert "ModelAclGraphManager310" in source


def test_310p_startup_skips_v1_patch_package() -> None:
    tree = _tree("vllm_ascend/patch/worker/__init__.py")
    top_level = _direct_imports(tree.body)
    assert "vllm_ascend.patch.worker.patch_v1" not in top_level
    guarded = False
    for node in tree.body:
        if not isinstance(node, ast.If):
            continue
        if "vllm_ascend.patch.worker.patch_v1" in _imported_modules(node):
            guarded = True
    assert guarded


def test_310p_mtp_shift_does_not_import_v1_proposer() -> None:
    source = (ROOT / "vllm_ascend/_310p/worker/v2/spec_utils.py").read_text()
    modules = _imported_modules(_tree("vllm_ascend/_310p/worker/v2/spec_utils.py"))
    assert "llm_base_proposer" not in source
    assert not any("worker.v1" in name or name.startswith("vllm.v1.spec_decode") for name in modules)
    assert "tail_save" in source


def test_310p_does_not_import_v1_mamba_kernel_at_module_scope() -> None:
    tree = _tree("vllm_ascend/patch/worker/patch_mamba_utils.py")
    top_level = _direct_imports(tree.body)
    assert "vllm_ascend.ops.triton.v1.mamba.postprocess" not in top_level
    nested = False
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and "vllm_ascend.ops.triton.v1.mamba.postprocess" in _imported_modules(node):
            nested = True
    assert nested
