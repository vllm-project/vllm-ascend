# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import builtins
import runpy
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock

import pytest
import torch

SOURCE = Path(__file__).resolve().parents[3] / "vllm_ascend/device/flash_attention_output.py"


@pytest.mark.parametrize("supported", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("tokens", [0, 1, 8])
@pytest.mark.parametrize("gate_prefix", [0, 3])
def test_output_gate_dispatch_and_warmup(monkeypatch, supported, dtype, tokens, gate_prefix):
    utils = ModuleType("vllm_ascend.utils")
    utils.__dict__["supports_triton"] = lambda: supported
    monkeypatch.setitem(sys.modules, utils.__name__, utils)
    gate_kernel = Mock(side_effect=lambda x, gate: x.mul_(gate.sigmoid()))
    register = Mock()
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.startswith("vllm_ascend.ops.triton"):
            assert supported, f"Unsupported output gate imported {name}"
            module = ModuleType(name)
            module.__dict__["flash_attention_gate"] = gate_kernel
            module.__dict__["register_flash_attention_gate_warmup"] = register
            return module
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    functions = runpy.run_path(str(SOURCE))
    projected = torch.arange(tokens * 32, dtype=dtype).reshape(tokens, 32)[:, :16]
    gate = torch.linspace(-4, 4, max(tokens * (16 + gate_prefix), 1), dtype=dtype)
    gate = gate[: tokens * (16 + gate_prefix)].reshape(tokens, 16 + gate_prefix)[:, gate_prefix:]
    expected = projected.clone() * gate.sigmoid()
    gate_before = gate.clone()
    assert functions["flash_attention_gate"](projected, gate) is projected
    torch.testing.assert_close(projected, expected, rtol=0, atol=0)
    torch.testing.assert_close(gate, gate_before, rtol=0, atol=0)
    functions["register_flash_attention_gate_warmup"](dtype, 16, gate_prefix=gate_prefix)
    if supported:
        gate_kernel.assert_called_once_with(projected, gate)
        register.assert_called_once_with(dtype, 16, gate_prefix=gate_prefix)
    else:
        gate_kernel.assert_not_called()
        register.assert_not_called()
