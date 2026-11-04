# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Exercise the real weight-hook wrapper with both loader call signatures."""

import ast
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("explicit_dtype", [False, True])
def test_post_load_hook_preserves_activation_dtype(monkeypatch, sparse, explicit_dtype):
    source = Path(__file__).resolve().parents[3] / "vllm_ascend/ops/mla.py"
    tree = ast.parse(source.read_bytes())
    wrapper = next(
        node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "wrapped_process_weights"
    )
    calls = []

    class SFA:
        def process_weights_after_loading(self, dtype):
            calls.append(("SFA", dtype))

    def old_hook(dtype):
        calls.append(("original", dtype))

    def current_hook():
        calls.append(("original", torch.bfloat16))

    monkeypatch.setitem(sys.modules, "vllm_ascend.attention.sfa_v1", SimpleNamespace(AscendSFAImpl=SFA))
    namespace = {
        "torch": torch,
        "self": SimpleNamespace(mla_attn=SimpleNamespace(impl=SFA() if sparse else object(), act_dtype=torch.bfloat16)),
        "original_process_weights": old_hook if explicit_dtype else current_hook,
    }
    exec(compile(ast.Module(body=[wrapper], type_ignores=[]), str(source), "exec"), namespace)
    if explicit_dtype:
        namespace["wrapped_process_weights"](torch.float16)
    else:
        namespace["wrapped_process_weights"]()
    expected = torch.float16 if explicit_dtype else torch.bfloat16
    assert calls == [("SFA" if sparse else "original", expected)]
