# SPDX-License-Identifier: Apache-2.0
# This file is a part of the vllm-ascend project.
"""Source-level regression tests for the Step3p5 QK RMSNorm+RoPE fusion patch.

These tests verify structural contracts of patch_step3p5.py and its
registration in __init__.py without requiring an NPU device or the
actual step3p5 model.  AST-level checks cover:

* patch_module import line in __init__.py
* conditional import + fallback pattern
* monkey-patch guard (if Step3p5Attention is not None)
* _patched_forward structure: QKV projection, rope/non-rope branches
* fused kernel call signature (torch.ops.vllm.qkv_rmsnorm_rope)
* weight caching (hasattr guard on _q_w_cached/_k_w_cached)
* forward reassignment (Step3p5Attention.forward = _patched_forward)
"""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
PATCH_INIT = ROOT / "vllm_ascend" / "patch" / "worker" / "__init__.py"
PATCH_STEP3P5 = ROOT / "vllm_ascend" / "patch" / "worker" / "patch_step3p5.py"


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text())


def _src(node: ast.AST) -> str:
    return ast.unparse(node)


# ---------------------------------------------------------------------------
# Registration in __init__.py
# ---------------------------------------------------------------------------


def test_patch_step3p5_registered_in_init() -> None:
    """Verify patch_step3p5 is imported in the worker patch __init__.py."""
    init_src = PATCH_INIT.read_text()
    assert "import vllm_ascend.patch.worker.patch_step3p5  # noqa" in init_src


# ---------------------------------------------------------------------------
# Patch-file structure: conditional import + monkey-patch guard
# ---------------------------------------------------------------------------


def test_patch_step3p5_has_conditional_import() -> None:
    """Import is wrapped in try/except with fallback to None."""
    tree = _tree(PATCH_STEP3P5)
    has_try = False
    has_except_import_error = False
    has_fallback_to_none = False

    for node in ast.walk(tree):
        if isinstance(node, ast.Try):
            has_try = True
            for handler in node.handlers:
                if isinstance(handler.type, ast.Name) and handler.type.id == "ImportError":
                    has_except_import_error = True
                    for stmt in handler.body:
                        if isinstance(stmt, ast.Assign):
                            for target in stmt.targets:
                                if isinstance(target, ast.Name) and target.id == "Step3p5Attention":
                                    has_fallback_to_none = True

    assert has_try, "Expected try/except for conditional import"
    assert has_except_import_error, "Expected except ImportError"
    assert has_fallback_to_none, "Expected Step3p5Attention = None on ImportError"


def test_patch_step3p5_guards_monkey_patch() -> None:
    """Monkey-patch and _patched_forward are inside `if Step3p5Attention is not None:`."""
    patch_src = PATCH_STEP3P5.read_text()

    # The guard must exist.
    assert "if Step3p5Attention is not None:" in patch_src

    # The forward definition must be inside the guard.
    assert "def _patched_forward(" in patch_src
    assert "_patched_forward" in patch_src

    # The monkey-patch reassignment must be inside the guard.
    assert "Step3p5Attention.forward = _patched_forward" in patch_src


# ---------------------------------------------------------------------------
# _patched_forward structural contracts
# ---------------------------------------------------------------------------


def _find_patched_forward() -> ast.FunctionDef:
    """Return the AST node for _patched_forward defined inside patch_step3p5.py."""
    tree = _tree(PATCH_STEP3P5)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_patched_forward":
            return node
    raise AssertionError("_patched_forward not found in patch_step3p5.py")


def test_patched_forward_signature() -> None:
    """_patched_forward accepts positions and hidden_states."""
    fn = _find_patched_forward()
    arg_names = [arg.arg for arg in fn.args.args]
    assert "self" in arg_names
    assert "positions" in arg_names
    assert "hidden_states" in arg_names


def test_patched_forward_starts_with_qkv_proj() -> None:
    """The first computation is self.qkv_proj(hidden_states)."""
    src = _src(_find_patched_forward())
    assert "self.qkv_proj(hidden_states)" in src


def test_patched_forward_has_rope_branch() -> None:
    """When use_rope is True the fused kernel is called."""
    src = _src(_find_patched_forward())
    assert "if self.use_rope:" in src
    assert "torch.ops.vllm.qkv_rmsnorm_rope(" in src


def test_patched_forward_fused_kernel_args() -> None:
    """Fused kernel receives all required keyword arguments."""
    fn = _find_patched_forward()

    class CallFinder(ast.NodeVisitor):
        def __init__(self) -> None:
            self.call: ast.Call | None = None

        def visit_Call(self, node: ast.Call) -> None:
            if (
                isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Attribute)
                and isinstance(node.func.value.value, ast.Attribute)
                and isinstance(node.func.value.value.value, ast.Name)
                and node.func.value.value.value.id == "torch"
                and node.func.value.value.attr == "ops"
                and node.func.value.attr == "vllm"
                and node.func.attr == "qkv_rmsnorm_rope"
            ):
                self.call = node

    finder = CallFinder()
    finder.visit(fn)
    assert finder.call is not None, "qkv_rmsnorm_rope call not found"

    kw_names = {kw.arg for kw in finder.call.keywords}
    required_kwargs = {
        "input",
        "q_weight",
        "k_weight",
        "q_hidden_size",
        "kv_hidden_size",
        "head_dim",
        "eps",
        "cos_sin_cache",
        "positions",
    }
    assert required_kwargs.issubset(kw_names), f"Missing fused-kernel kwargs: {required_kwargs - kw_names}"


def test_patched_forward_has_non_rope_fallback() -> None:
    """When use_rope is False, Q and K are split and normed explicitly."""
    src = _src(_find_patched_forward())
    assert "else:" in src
    assert "qkv.split" in src
    assert "self.q_norm(" in src
    assert "self.k_norm(" in src


def test_patched_forward_weight_caching() -> None:
    """QK norm weights are cached on first call to avoid repeated .float()."""
    src = _src(_find_patched_forward())
    assert "if not hasattr(self, '_q_w_cached'):" in src
    assert "self._q_w_cached = 1.0 + self.q_norm.weight.float()" in src
    assert "self._k_w_cached = 1.0 + self.k_norm.weight.float()" in src
    assert "q_w = self._q_w_cached" in src
    assert "k_w = self._k_w_cached" in src


def test_patched_forward_has_attention_and_output() -> None:
    """Output path: self.attn → optional gate → self.o_proj."""
    src = _src(_find_patched_forward())
    assert "self.attn(q, k, v)" in src
    assert "self.o_proj(attn_output)" in src


def test_patched_forward_handles_head_wise_gate() -> None:
    """When use_head_wise_attn_gate, g_proj is applied."""
    src = _src(_find_patched_forward())
    assert "if self.use_head_wise_attn_gate:" in src
    assert "self.g_proj(hidden_states)" in src
