# SPDX-License-Identifier: Apache-2.0
"""Source-level checks for the Ascend Mamba precision-kernel override."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
POSTPROCESS = (
    ROOT / "vllm_ascend" / "ops" / "triton" / "mamba" / "postprocess.py"
)
PATCH_MAMBA_UTILS = (
    ROOT / "vllm_ascend" / "patch" / "worker" / "patch_mamba_utils.py"
)


def _top_level_functions(path: Path) -> dict[str, ast.FunctionDef]:
    return {
        node.name: node
        for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.FunctionDef)
    }


def test_postprocess_keeps_ascend_precision_kernels() -> None:
    functions = _top_level_functions(POSTPROCESS)

    assert set(functions) == {
        "_copy_mamba_state_block",
        "postprocess_mamba_fused_kernel",
        "precopy_mamba_align_fused_kernel",
    }
    helper_source = ast.unparse(functions["_copy_mamba_state_block"])
    assert "src_ptr = src_addr.to(tl.pointer_type(tl.uint8))" in helper_source
    assert "dst_ptr = dst_addr.to(tl.pointer_type(tl.uint8))" in helper_source
    postprocess_source = ast.unparse(
        functions["postprocess_mamba_fused_kernel"]
    )
    assert "PRECOMPUTED_NEW_COMPUTED" in postprocess_source
    assert "tl.store(num_accepted_tokens_ptr + req_idx, 1)" in postprocess_source


def test_patch_installs_ascend_precision_kernels() -> None:
    patch_source = PATCH_MAMBA_UTILS.read_text()

    assert (
        "mamba_utils._copy_mamba_state_block = _copy_mamba_state_block"
        in patch_source
    )
    assert "mamba_utils.postprocess_mamba_fused_kernel = (" in patch_source
    assert "mamba_utils.precopy_mamba_align_fused_kernel = (" in patch_source
