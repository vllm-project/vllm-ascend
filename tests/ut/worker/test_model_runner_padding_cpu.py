# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Real batches must not inherit dummy/profile MoE padding markers."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


@pytest.mark.parametrize("flag", [True, False, None])
def test_prepare_inputs_refreshes_moe_padding_after_dummy_and_changing_batches(flag):
    source = Path(__file__).resolve().parents[3] / "vllm_ascend/worker/v2/model_runner.py"
    method = next(
        n
        for n in ast.walk(ast.parse(source.read_text()))
        if isinstance(n, ast.FunctionDef) and n.name == "_prepare_inputs_impl"
    )
    statements = [
        n
        for n in method.body
        if (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "is_padding" for t in n.targets))
        or (isinstance(n, ast.If) and "VLLM_MOE_SKIP_PADDING" in ast.unparse(n.test))
    ]
    assert len(statements) == 2
    code = compile(ast.fix_missing_locations(ast.Module(body=statements, type_ignores=[])), str(source), "exec")
    # Core InputBatch.make_dummy marks profile/capture rows True.
    backing = torch.ones(32, dtype=torch.bool)
    config = SimpleNamespace() if flag is None else SimpleNamespace(VLLM_MOE_SKIP_PADDING=flag)
    instance = SimpleNamespace(input_buffers=SimpleNamespace(is_padding=backing))
    for active, padded in [(20, 24), (4, 16), (9, 12), (1, 1)]:
        previous = backing.clone()
        namespace = dict(self=instance, vllm_envs=config, num_tokens=active, num_tokens_after_padding=padded)
        exec(code, namespace)
        view = namespace["is_padding"]
        assert view.data_ptr() == backing.data_ptr() and view.numel() == padded
        if flag:
            assert not view[:active].any()
            assert view[active:].all()
            assert torch.equal(backing[padded:], previous[padded:])
        else:
            assert torch.equal(backing, previous)
