# SPDX-License-Identifier: Apache-2.0
"""Source-level regressions for the 310P Mamba align fallback.

The fallback is only active on 310P and depends on runtime NPU/vLLM state.
Keep these checks import-free so they can run in lightweight DT environments
while still guarding the important upstream semantic contract.
"""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PATCH_MAMBA_UTILS = ROOT / "vllm_ascend" / "patch" / "worker" / "patch_mamba_utils.py"


def _func(path: Path, name: str) -> ast.FunctionDef:
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"function {name} not found in {path}")


def _src(node: ast.AST) -> str:
    return ast.unparse(node)


def test_310p_postprocess_fallback_preserves_upstream_metadata_semantics() -> None:
    src = _src(_func(PATCH_MAMBA_UTILS, "_postprocess_mamba_align_gpu_cpu_fallback"))

    assert "num_accepted_tokens_gpu" in src
    assert "num_accepted_tokens_cpu_tensor[:num_reqs].copy_(num_accepted_tokens_gpu[:num_reqs])" in src
    assert "num_tokens_running_state = num_computed_tokens[i] + num_scheduled_tokens[i] - num_draft_tokens[i]" in src
    assert "new_num_computed_tokens = num_tokens_running_state + num_accepted_tokens[i] - 1" in src
    assert "aligned_new_computed_tokens = new_num_computed_tokens // block_size * block_size" in src
    assert "if aligned_new_computed_tokens < num_tokens_running_state:" in src
    assert "if src_block_idx == dest_block_idx:" in src
    assert "num_accepted_tokens_cpu_tensor[i] = 1" in src


def test_310p_postprocess_fallback_mirrors_state_copy_without_triton() -> None:
    src = _src(_func(PATCH_MAMBA_UTILS, "_postprocess_mamba_align_gpu_cpu_fallback"))

    assert "run_fused_postprocess" not in src
    assert "postprocess_mamba_fused_kernel" not in src
    assert "accept_token_bias = aligned_new_computed_tokens - num_tokens_running_state" in src
    assert "if accept_token_bias == 0:" in src
    assert "continue" in src
    assert "for mamba_group_id in ctx.mamba_group_ids:" in src
    assert "get_numpy_array()" in src
    assert "copy_spec = state_copy_func(state, block_ids, src_block_idx, accept_token_bias + 1)" in src
    assert "_tensor_view_from_data_ptr(state, copy_spec.start_addr, copy_spec.num_elements)" in src
    assert "dst_state.copy_(src_state.clone())" in src
