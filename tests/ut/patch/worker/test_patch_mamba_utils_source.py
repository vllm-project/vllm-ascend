# SPDX-License-Identifier: Apache-2.0
"""Source-level checks for the Ascend Mamba precision-kernel overridess."""

from __future__ import annotations

import ast
from pathlib import Path

from vllm_ascend.utils import vllm_version_is

ROOT = Path(__file__).resolve().parents[4]
POSTPROCESS = ROOT / "vllm_ascend" / "ops" / "triton" / "mamba" / "postprocess.py"


def _top_level_functions(path: Path) -> dict[str, ast.FunctionDef]:
    return {node.name: node for node in ast.parse(path.read_text()).body if isinstance(node, ast.FunctionDef)}


def _postprocess_kernels(path: Path) -> list[ast.FunctionDef]:
    """Collect both postprocess_mamba_fused_kernel definitions nested under
    the vllm_version_is('0.27.1') if/else gate. Index 0 = v0.27.1 branch,
    index 1 = main (else) branch."""
    kernels = []

    def _walk(node: ast.AST) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.FunctionDef) and child.name == "postprocess_mamba_fused_kernel":
                kernels.append(child)
            else:
                _walk(child)

    _walk(ast.parse(path.read_text()))
    return kernels


def _selected_kernel_source(path: Path) -> str:
    """Return the kernel source that is active for the current vllm version."""
    kernels = _postprocess_kernels(path)
    assert len(kernels) == 2, "expected one kernel per vllm_version_is branch"
    idx = 0 if vllm_version_is("0.27.1") else 1
    return ast.unparse(kernels[idx])


def test_postprocess_keeps_only_existing_ascend_precision_kernel() -> None:
    functions = _top_level_functions(POSTPROCESS)

    assert set(functions) == set()
    postprocess_source = _selected_kernel_source(POSTPROCESS)
    assert "src_ptr = src_addr.to(tl.pointer_type(tl.uint8))" in postprocess_source
    assert "dst_ptr = dst_addr.to(tl.pointer_type(tl.uint8))" in postprocess_source
    assert "PRECOMPUTED_NEW_COMPUTED" in postprocess_source
    assert "tl.store(num_accepted_tokens_ptr + req_idx, 1)" in postprocess_source

    if vllm_version_is("0.27.1"):
        assert "TEMPORAL_TILES" not in postprocess_source
        assert "tile_idx" not in postprocess_source
    else:
        assert "TEMPORAL_TILES" in postprocess_source
        assert "tile_idx" in postprocess_source
        assert "if tile_idx == 0:" in postprocess_source
        assert "and state_idx == 0 and tile_idx == 0" not in postprocess_source
