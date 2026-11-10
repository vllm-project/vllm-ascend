# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import ast
import builtins
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize("model", ["deepseek_v4", "deepseek_v41"])
@pytest.mark.parametrize("triton_supported", [False, True])
@pytest.mark.parametrize("scale", [0.0, -2.5, 0.125])
def test_moe_scaled_sum_preserves_operand_order(model, triton_supported, scale):
    # Isolate the model helper from model registration and device initialization.
    # This also lets the no-Triton case reject imports before a package can load.
    source = REPO_ROOT / f"vllm_ascend/models/{model}/model.py"
    tree = ast.parse(source.read_text())
    helper = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_muls_add")
    kernel = Mock(side_effect=lambda x, y, scale: x * scale + y)
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.startswith("vllm_ascend.ops.triton"):
            assert triton_supported, f"310P fallback imported {name}"
            return SimpleNamespace(muls_add_triton=kernel)
        return original_import(name, *args, **kwargs)

    namespace = {
        "torch": torch,
        "supports_triton": lambda: triton_supported,
        "__builtins__": {**vars(builtins), "__import__": guarded_import},
    }
    exec(compile(ast.Module(body=[helper], type_ignores=[]), str(source), "exec"), namespace)
    x = torch.tensor([[2.0, -4.0], [8.0, 16.0]])
    y = torch.tensor([[3.0, 5.0], [-7.0, 11.0]])
    x_before, y_before = x.clone(), y.clone()
    result = namespace["_muls_add"](x, y, scale)
    torch.testing.assert_close(result, torch.add(y, x, alpha=scale))
    torch.testing.assert_close(x, x_before)
    torch.testing.assert_close(y, y_before)
    if triton_supported:
        kernel.assert_called_once_with(x, y, scale)
    else:
        kernel.assert_not_called()
