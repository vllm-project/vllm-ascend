# SPDX-License-Identifier: Apache-2.0
"""Exercise the runner's DSpark residual predicate without NPU imports."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize(
    "model_type,architecture,dspark,expected",
    [
        ("qwen3", "Qwen3DSparkModel", True, True),
        ("qwen3", "DSparkDraftModel", True, True),
        ("deepseek_v4", "DSparkDraftModel", True, False),
        ("kimi_k3_dspark", "KimiK3DSparkForCausalLM", True, False),
        ("qwen3", "Qwen3ForCausalLM", True, False),
        ("qwen3", "DSparkDraftModel", False, False),
        ("qwen3", "Qwen3DSparkModel", False, False),
    ],
)
def test_gqa_draft_residual_predicate(model_type, architecture, dspark, expected):
    source = Path(__file__).resolve().parents[3] / "vllm_ascend/worker/model_runner_v1.py"
    tree = ast.parse(source.read_text())
    method = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_draft_uses_qwen3_gqa_dspark"
    )
    scope = {}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), scope)
    config = SimpleNamespace(
        use_dspark=lambda: dspark,
        draft_model_config=SimpleNamespace(
            hf_config=SimpleNamespace(model_type=model_type, architectures=[architecture])
        ),
    )
    runner = SimpleNamespace(speculative_config=config)
    assert scope[method.name](runner) is expected
    config.draft_model_config = None
    assert not scope[method.name](runner)
    runner.speculative_config = None
    assert not scope[method.name](runner)
