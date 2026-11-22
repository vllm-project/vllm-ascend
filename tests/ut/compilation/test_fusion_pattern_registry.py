#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#
"""The fusion-pattern registry must feed every graph that vLLM compiles.

``GraphFusionPassManager`` owns one ``PatternMatcherPass`` and a new manager is
built for every compiled graph (prefill range, decode range, speculator). The
registry used to de-duplicate pattern registration globally, so every manager
after the first one ended up with an empty matcher and silently lost all FX
fusion passes. These tests pin the per-manager behaviour with aten-only ops so
they do not need a device.
"""

import importlib
import importlib.util
import sys
from unittest.mock import MagicMock

import pytest
import torch
import torch._inductor.pattern_matcher as pm
import torch.fx as fx
from pytest import MonkeyPatch
from torch.fx.experimental.proxy_tensor import make_fx

from tests.ut.base import TestBase


@pytest.fixture(scope="module", autouse=True)
def _base_pattern_module():
    # base_pattern imports npugraph_ex (or torchair) at module level.
    stubbed = importlib.util.find_spec("npugraph_ex") is None and importlib.util.find_spec("torchair") is None
    if stubbed:
        sys.modules["npugraph_ex"] = MagicMock()
    try:
        yield importlib.import_module("vllm_ascend.compilation.passes.base_pattern")
    finally:
        if stubbed:
            sys.modules.pop("npugraph_ex", None)


def _make_dummy_pattern(base_pattern_mod, name: str):
    class _Dummy(base_pattern_mod.BasePattern):
        def get_inputs(self):
            return [torch.randn(2, 4)]

        def get_pattern(self):
            def pattern(x: torch.Tensor):
                return torch.relu(torch.neg(x))

            return pattern

        def get_replacement(self):
            def replacement(x: torch.Tensor):
                return torch.clamp(x, min=0.0)

            return replacement

    _Dummy.__name__ = name
    return _Dummy


def _build_neg_relu_graph() -> fx.GraphModule:
    """make_fx so every node carries ``meta["example_value"]``; the Inductor
    matcher requires it."""

    def f(x: torch.Tensor) -> torch.Tensor:
        return torch.relu(-x)

    return make_fx(f)(torch.randn(2, 4))


class TestPatternRegistrationIsPerMatcher(TestBase):
    def test_second_matcher_still_receives_patterns(self):
        """A second pass manager must still receive the patterns."""
        base = sys.modules["vllm_ascend.compilation.passes.base_pattern"]
        pattern_cls = _make_dummy_pattern(base, "DummyPatternPerMatcher")
        cfg = MagicMock()
        cfg.model_config.dtype = torch.bfloat16

        first = pm.PatternMatcherPass(pass_name="first")
        second = pm.PatternMatcherPass(pass_name="second")
        pattern_cls(cfg).register(first)
        pattern_cls(cfg).register(second)

        assert first.apply(_build_neg_relu_graph().graph) == 1
        assert second.apply(_build_neg_relu_graph().graph) == 1

    def test_repeated_registration_into_same_matcher_is_idempotent(self):
        base = sys.modules["vllm_ascend.compilation.passes.base_pattern"]
        pattern_cls = _make_dummy_pattern(base, "DummyPatternIdempotent")
        cfg = MagicMock()
        cfg.model_config.dtype = torch.bfloat16

        matcher = pm.PatternMatcherPass(pass_name="repeated")
        pattern_cls(cfg).register(matcher)
        pattern_cls(cfg).register(matcher)

        # Registering twice into the same matcher must stay a no-op.
        assert matcher.apply(_build_neg_relu_graph().graph) == 1

    def test_failed_registration_stays_retryable(self):
        """A pattern that fails to register must not be marked as registered.

        Otherwise a single failure would be cached for the lifetime of the
        matcher and the retry would be silently skipped.
        """
        base = sys.modules["vllm_ascend.compilation.passes.base_pattern"]
        pattern_cls = _make_dummy_pattern(base, "DummyPatternRetry")
        cfg = MagicMock()
        cfg.model_config.dtype = torch.bfloat16

        matcher = pm.PatternMatcherPass(pass_name="retry")
        real_register_replacement = pm.register_replacement
        attempts = []

        def flaky_register_replacement(*args, **kwargs):
            attempts.append(args)
            if len(attempts) == 1:
                raise RuntimeError("simulated registration failure")
            return real_register_replacement(*args, **kwargs)

        with MonkeyPatch.context() as patched:
            patched.setattr(pm, "register_replacement", flaky_register_replacement)

            with pytest.raises(RuntimeError):
                pattern_cls(cfg).register(matcher)

            # The retry must actually reach the matcher instead of being skipped.
            pattern_cls(cfg).register(matcher)

        assert len(attempts) == 2
        assert matcher.apply(_build_neg_relu_graph().graph) == 1
