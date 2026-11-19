# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run the constructor's HC allocation without constructing decoder weights."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from vllm.config import SpeculativeConfig


@pytest.mark.parametrize("method", [None, "dspark", "mtp", "eagle", "eagle3", "dflash", "draft_model"])
@pytest.mark.parametrize("last_pp_rank", [False, True])
def test_target_hc_buffer_allocation(method, last_pp_rank):
    # Execute the production allocation block with real CPU tensors. Importing
    # the full model would require unrelated native operators and model weights.
    source = Path(__file__).resolve().parents[3] / "vllm_ascend/models/deepseek_v41/model.py"
    module = ast.parse(source.read_text(encoding="utf-8"))
    model = next(node for node in module.body if isinstance(node, ast.ClassDef) and node.name == "DeepseekV41Model")
    initializer = next(node for node in model.body if isinstance(node, ast.FunctionDef) and node.name == "__init__")
    names = {"spec_config", "needs_mtp_hidden_states", "self._mtp_hidden_buffer"}
    allocation: list[ast.stmt] = [
        node
        for node in initializer.body
        if isinstance(node, ast.Assign) and any(ast.unparse(target) in names for target in node.targets)
    ]
    spec = None
    if method is not None:
        spec = SpeculativeConfig.__new__(SpeculativeConfig)
        object.__setattr__(spec, "method", method)
    config = SimpleNamespace(hidden_size=8)
    state = SimpleNamespace(hc_mult=4, device="cpu")
    namespace = {
        "self": state,
        "config": config,
        "torch": torch,
        "get_pp_group": lambda: SimpleNamespace(is_last_rank=last_pp_rank),
        "vllm_config": SimpleNamespace(
            speculative_config=spec,
            scheduler_config=SimpleNamespace(max_num_batched_tokens=16),
            model_config=SimpleNamespace(dtype=torch.bfloat16),
        ),
    }
    exec(compile(ast.Module(body=allocation, type_ignores=[]), str(source), "exec"), namespace)
    buffer = state._mtp_hidden_buffer
    if method in (None, "dspark") or not last_pp_rank:
        assert buffer is None
    else:
        assert buffer.shape == (16, 32)
        assert buffer.dtype == torch.bfloat16
        assert buffer.numel() * buffer.element_size() == 1024
