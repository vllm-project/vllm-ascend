# SPDX-License-Identifier: Apache-2.0

import ast
from pathlib import Path

MODEL_RUNNER_PATH = Path(__file__).parents[3] / "vllm_ascend" / "worker" / "model_runner_v1.py"


def _npu_input_batch_calls() -> list[ast.Call]:
    tree = ast.parse(MODEL_RUNNER_PATH.read_text())
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "NPUInputBatch"
    ]


def test_npu_input_batches_keep_reasoning_phase_output_ids():
    calls = _npu_input_batch_calls()

    assert len(calls) == 2
    for call in calls:
        keywords = {keyword.arg: keyword.value for keyword in call.keywords}
        assert "logitsprocs_need_output_token_ids" in keywords

    initial_value = ast.unparse(
        {keyword.arg: keyword.value for keyword in calls[0].keywords}["logitsprocs_need_output_token_ids"]
    )
    assert "reasoning_eos_policy_enabled" in initial_value

    reinitialized_value = ast.unparse(
        {keyword.arg: keyword.value for keyword in calls[1].keywords}["logitsprocs_need_output_token_ids"]
    )
    assert reinitialized_value == ("self.input_batch.logitsprocs_need_output_token_ids")
