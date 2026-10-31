# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import importlib
import importlib.util
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
import torch._inductor.pattern_matcher as pm
from torch._subclasses.fake_tensor import FakeTensorMode


@pytest.fixture(scope="module")
def pass_module():
    stubbed = importlib.util.find_spec("npugraph_ex") is None and importlib.util.find_spec("torchair") is None
    if stubbed:
        sys.modules["npugraph_ex"] = MagicMock()
    try:
        return importlib.import_module("vllm_ascend.compilation.passes.muls_add_pass")
    finally:
        if stubbed:
            del sys.modules["npugraph_ex"]


@pytest.fixture
def fused_op(monkeypatch):
    # Execute rewritten CPU graphs without importing or launching NPU kernels.
    lib = torch.library.Library("ascend_test_muls_add", "DEF")
    try:
        lib.define("fused(Tensor x, Tensor y, float scale) -> Tensor")
        lib.impl("fused", lambda x, y, scale: x * scale + y, "CPU")
        torch.library.register_fake("ascend_test_muls_add::fused", lambda x, y, scale: torch.empty_like(x), lib=lib)
        op = torch.ops.ascend_test_muls_add.fused.default
        monkeypatch.setattr(torch.ops.vllm, "muls_add", op, raising=False)
        yield op
    finally:
        lib._destroy()


def _config(dtype):
    return SimpleNamespace(model_config=SimpleNamespace(dtype=dtype))


def _register_patterns(pass_module, inputs, scale=2.5):
    matcher = pm.PatternMatcherPass()
    for cls in (
        pass_module.MulsAddPattern,
        pass_module.MulsAddSharedFirstPattern,
        pass_module.MulsForeachAddPattern,
        pass_module.MulsForeachAddSharedFirstPattern,
    ):
        pattern = cls(_config(inputs[0].dtype), scale=scale)
        pm.register_replacement(pattern.get_pattern(), pattern.get_replacement(), inputs, pm.fwd_only, matcher)
    return matcher


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("shared_first", [False, True])
@pytest.mark.parametrize("foreach", [False, True])
def test_fuses_both_operand_orders_and_add_forms(pass_module, fused_op, dtype, shared_first, foreach):
    def forward(x, y):
        scaled = x * 2.5
        left, right = (y, scaled) if shared_first else (scaled, y)
        return torch._foreach_add([left], [right])[0] if foreach else left + right

    with FakeTensorMode():
        inputs = [torch.randn(2, 16, dtype=dtype) for _ in range(2)]
        matcher = _register_patterns(pass_module, inputs)
        graph = pm.fwd_only(forward, inputs)
        assert matcher.apply(graph.graph) == 1
    graph.recompile()
    assert sum(node.target == fused_op for node in graph.graph.nodes) == 1
    real_inputs = [torch.randn(2, 16, dtype=dtype) for _ in range(2)]
    assert torch.equal(graph(*real_inputs), forward(*real_inputs))


def test_pass_registers_all_four_patterns(pass_module, monkeypatch):
    registered = []
    monkeypatch.setattr(pass_module.MulsAddPattern, "register", lambda self, matcher: registered.append(type(self)))
    config = MagicMock()
    config.model_config.dtype = torch.bfloat16
    config.model_config.hf_text_config.routed_scaling_factor = 2.5
    pass_module.MulsAddFusionPass(config)
    assert registered == [
        pass_module.MulsAddPattern,
        pass_module.MulsAddSharedFirstPattern,
        pass_module.MulsForeachAddPattern,
        pass_module.MulsForeachAddSharedFirstPattern,
    ]
