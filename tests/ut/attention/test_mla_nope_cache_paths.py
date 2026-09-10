# SPDX-License-Identifier: Apache-2.0
"""MLA-NoPE layers must keep their own KV cache write path.

Two notions of "no RoPE" meet in ``AscendMLAImpl``. ``use_mla_rope`` is false
whenever the layer has no rotary embedding, which for GLM-5.3-Flash follows
``mla_nope``; ``qk_rope_head_dim == 0`` additionally says the rope *cache* is
empty. Kimi K3 is the first case without the second, and its ``_exec_kv_no_rope``
writes both caches through ``reshape_and_cache``. GLM-5.3-Flash is both, so it
needs ``_exec_kv_mla_nope``, which scatters only the nope cache and tolerates a
padded (non-contiguous) one.

Guarding the K3 path on ``use_mla_rope`` alone shadowed the NoPE branches, and
letting the fused decode prolog run on a NoPE layer fed the operator an empty
rope cache -- whose torch strides it rejects, since contiguous strides are
computed over ``max(size, 1)`` and so report 1 where the tiling wants 0:

    krCache dim2 must be contiguous, actual stride is 1, expected stride is 0

Both live inside methods that need a device to construct, hence the
source-level checks.
"""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
MLA_V1 = ROOT / "vllm_ascend" / "attention" / "mla_v1.py"

ROPE_CACHE_EXISTS = "self.qk_rope_head_dim > 0"
K3_GUARD = f"not self.use_mla_rope and {ROPE_CACHE_EXISTS}"


def _function(name: str) -> ast.FunctionDef:
    tree = ast.parse(MLA_V1.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} not found in {MLA_V1}")


def _calls(node: ast.AST) -> set[str]:
    return {ast.unparse(inner.func) for inner in ast.walk(node) if isinstance(inner, ast.Call)}


def _decode_prolog_gate() -> str:
    for node in ast.walk(_function("forward")):
        if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == "can_use_decode_prolog":
            return ast.unparse(node.value)
    raise AssertionError("can_use_decode_prolog is no longer assigned in forward")


def test_decode_prolog_needs_a_rope_cache() -> None:
    gate = _decode_prolog_gate()

    assert ROPE_CACHE_EXISTS in gate, (
        "the fused decode prolog is enabled without checking that a rope cache "
        f"exists; NoPE layers will hit the tiling error. Gate: {gate}"
    )
    # Hardware support is necessary but not sufficient.
    assert "MLA_DECODE_PROLOG_WITHOUT_ROPE" in gate
    assert "self.use_mla_rope" in gate


def test_decode_prolog_still_allowed_with_rope() -> None:
    """Layers that do have RoPE must keep the fused path."""
    gate = _decode_prolog_gate()

    assert gate.startswith("self.use_mla_rope or")


def _k3_guards(function_name: str) -> list[str]:
    return [
        ast.unparse(node.test)
        for node in ast.walk(_function(function_name))
        if isinstance(node, ast.If) and "use_mla_rope" in ast.unparse(node.test)
    ]


def test_kimi_k3_path_is_gated_on_a_rope_cache() -> None:
    for function_name in ("exec_kv_decode", "exec_kv_prefill"):
        guards = _k3_guards(function_name)

        assert K3_GUARD in guards, (
            f"{function_name} routes every rope-less layer to _exec_kv_no_rope; "
            f"NoPE layers never reach _exec_kv_mla_nope. Guards: {guards}"
        )


def test_nope_branch_is_reachable_from_both_phases() -> None:
    """The NoPE scatter is the point of the guard above; keep both call sites."""
    for function_name in ("exec_kv_decode", "exec_kv_prefill"):
        assert "self._exec_kv_mla_nope" in _calls(_function(function_name)), (
            f"{function_name} no longer calls _exec_kv_mla_nope"
        )
