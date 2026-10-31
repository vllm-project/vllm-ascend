# SPDX-License-Identifier: Apache-2.0
"""Unit-test the decomposed-MoE multistream configuration boundary."""

import ast
from pathlib import Path
from unittest import mock


def _load_resolver():
    source = Path(__file__).resolve().parents[2] / "vllm_ascend/ops/fused_moe/fused_moe_0_23_0.py"
    tree = ast.parse(source.read_text())
    resolver = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_resolve_moe_multistream_overlap"
    )
    resolver.decorator_list = []
    logger = mock.Mock()
    namespace = {"logger": logger}
    exec(compile(ast.Module(body=[resolver], type_ignores=[]), str(source), "exec"), namespace)
    return namespace["_resolve_moe_multistream_overlap"], logger


def test_decomposed_moe_intercepts_requested_multistream_overlap():
    resolve, logger = _load_resolver()

    actual = resolve(
        fxrt_prefill_decompose=True,
        requested_shared_expert=True,
        requested_gate=True,
        has_shared_experts=True,
    )

    assert actual == (False, False, False)
    message = logger.warning_once.call_args.args[0]
    assert "[DSV4_PREFILL_MOE_OVERLAP]" in message
    assert "intercepted unsupported overlap" in message
    assert "effective shared_expert=0 shared_gate=0 gate=0" in message


def test_eager_moe_preserves_requested_multistream_overlap():
    resolve, logger = _load_resolver()

    assert resolve(
        fxrt_prefill_decompose=False,
        requested_shared_expert=True,
        requested_gate=True,
        has_shared_experts=True,
    ) == (True, True, True)
    assert resolve(
        fxrt_prefill_decompose=False,
        requested_shared_expert=True,
        requested_gate=True,
        has_shared_experts=False,
    ) == (False, False, True)
    logger.warning_once.assert_not_called()


if __name__ == "__main__":
    test_decomposed_moe_intercepts_requested_multistream_overlap()
    test_eager_moe_preserves_requested_multistream_overlap()
    print("PASS: decomposed interception and opaque overlap settings")
