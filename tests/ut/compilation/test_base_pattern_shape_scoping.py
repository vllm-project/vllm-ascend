#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
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
"""Unit tests for per-model-shape scoping of fusion patterns (BasePattern).

Regression tests for the FULL-mode (npugraph_ex) crash where a target-sized
fusion pattern was applied to the spec-decode draft model's graph:
"ValueError: Split sizes add up to 6144 but got the tensor's size of 4096".

The tests mirror that failure mode with a toy ``cat(split(...))`` pattern
(whose split sizes are baked from the example-input width, exactly like the
qkv split in QKNormRope) and drive it through the real torch
``PatternMatcherPass`` path:

- a width-mismatched graph must be gracefully rejected (applied == 0, no
  exception escaping ``PatternMatcherPass.apply``), which only works when
  the search_fn guard raises ``RuntimeError`` (the exception type that
  torch's check_fn catches);
- a width-matched graph must still fuse (applied == 1);
- target- and draft-sized variants of the same pattern class must coexist
  in one process-global pass.

The npugraph_ex/torchair registration is mocked out to keep the test
hermetic (the process-global nge registry must not accumulate toy
patterns) and to let the test run on machines without the NPU backend.
"""

from __future__ import annotations

import sys
import types
from collections.abc import Callable
from unittest import mock

import pytest
import torch
import torch._inductor.pattern_matcher as pm
from torch.fx.experimental.proxy_tensor import make_fx


def _import_base_pattern():
    """Import base_pattern, stubbing the nge backend when unavailable.

    On CI the npugraph_ex (or torchair) backend is importable; on machines
    without it the stub keeps these logic tests runnable. The real nge
    registration is mocked in every test below regardless.
    """
    try:
        import vllm_ascend.compilation.passes.base_pattern as bp  # noqa: F401

        return bp
    except ImportError:
        stub = types.ModuleType("npugraph_ex")
        stub.register_replacement = lambda **kwargs: None  # type: ignore[attr-defined]
        sys.modules.setdefault("npugraph_ex", stub)
        import vllm_ascend.compilation.passes.base_pattern as bp

        return bp


base_pattern = _import_base_pattern()
BasePattern = base_pattern.BasePattern


@pytest.fixture(autouse=True)
def _isolate_global_pattern_registry():
    # The dedup registry is process-global by design (that is exactly what
    # used to drop the draft model's registration); snapshot/restore it so
    # tests stay independent.
    saved = set(base_pattern._registered_patterns)
    yield
    base_pattern._registered_patterns.clear()
    base_pattern._registered_patterns.update(saved)


def _fake_vllm_config(dtype=torch.float32):
    return types.SimpleNamespace(model_config=types.SimpleNamespace(dtype=dtype))


class ToySplitCatPattern(BasePattern):  # type: ignore[misc, valid-type]
    # (mypy cannot statically resolve BasePattern, which is imported from a
    # function-returned module to stub the NPU backend when absent)
    """cat(split(x, [w/2, w/2])) with sizes baked from the input width.

    Structurally matches graphs of any width (torch's initial match ignores
    int constants), but the re-trace verification in check_fn executes the
    split with the real tensor width -- narrower inputs raise ValueError,
    mirroring the QKNormRope qkv-split crash.
    """

    def __init__(self, vllm_config, width: int, num_tokens: int = 4):
        super().__init__(vllm_config)
        self.width = width
        self.num_tokens = num_tokens

    def get_inputs(self):
        return [torch.randn(self.num_tokens, self.width, dtype=self.dtype)]

    def get_pattern(self) -> Callable:
        width = self.width

        def pattern(x):
            parts = torch.split(x, [width // 2, width - width // 2], dim=-1)
            return torch.cat(parts, dim=-1)

        return pattern

    def get_replacement(self) -> Callable:
        def replacement(x):
            return x * 1.0

        return replacement


def _build_graph(width: int, num_tokens: int = 4) -> torch.fx.GraphModule:
    def fn(x):
        parts = torch.split(x, [width // 2, width - width // 2], dim=-1)
        return torch.cat(parts, dim=-1)

    x = torch.randn(num_tokens, width)
    return make_fx(fn, tracing_mode="real")(x)


def _register(pattern: ToySplitCatPattern, pm_pass: pm.PatternMatcherPass) -> None:
    # Mock the nge registration: the toy pattern must not leak into the
    # process-global npugraph_ex/torchair registry on NPU runners.
    with mock.patch.object(base_pattern.nge, "register_replacement"):
        pattern.register(pm_pass)


def _apply(pm_pass: pm.PatternMatcherPass, graph: torch.fx.GraphModule) -> int:
    applied = pm_pass.apply(graph)
    graph.recompile()
    return applied


def test_width_mismatch_is_gracefully_rejected():
    # Target-sized variant (width 64) applied to a draft-sized graph
    # (width 40): the old code let ValueError escape from check_fn's
    # re-trace and crash the whole compilation.
    pm_pass = pm.PatternMatcherPass()
    _register(ToySplitCatPattern(_fake_vllm_config(), width=64), pm_pass)

    graph = _build_graph(width=40)
    try:
        applied = _apply(pm_pass, graph)
    except ValueError as e:
        pytest.fail(f"ValueError escaped the graceful rejection path: {e}")
    assert applied == 0


def test_width_match_still_fuses():
    pm_pass = pm.PatternMatcherPass()
    _register(ToySplitCatPattern(_fake_vllm_config(), width=64), pm_pass)

    graph = _build_graph(width=64)
    assert _apply(pm_pass, graph) == 1
    # The split is gone; only the replacement op remains.
    target_ops = {n.target.__name__ if callable(n.target) else str(n.target) for n in graph.graph.nodes}
    assert not any("split" in op for op in target_ops), target_ops


def test_target_and_draft_variants_coexist_in_one_global_pass():
    # FULL mode compiles both models through one process-global pass; both
    # width variants must be registered (the class+eps-only dedup key used
    # to silently drop the draft-sized registration) and both graphs fuse.
    pm_pass = pm.PatternMatcherPass()
    _register(ToySplitCatPattern(_fake_vllm_config(), width=64), pm_pass)
    _register(ToySplitCatPattern(_fake_vllm_config(), width=40), pm_pass)

    target_graph = _build_graph(width=64)
    draft_graph = _build_graph(width=40)
    assert _apply(pm_pass, target_graph) == 1
    assert _apply(pm_pass, draft_graph) == 1


def test_same_shape_reregistration_is_deduplicated():
    pm_pass = pm.PatternMatcherPass()
    _register(ToySplitCatPattern(_fake_vllm_config(), width=64), pm_pass)
    with mock.patch.object(pm, "register_replacement") as torch_register:
        _register(ToySplitCatPattern(_fake_vllm_config(), width=64), pm_pass)
    torch_register.assert_not_called()


def test_unguarded_pattern_crashes_on_mismatch_anchors_the_regression():
    # Anchor: with a plain (unguarded) search_fn, the very same setup lets
    # ValueError escape PatternMatcherPass.apply -- this is the original
    # "Split sizes add up to 6144 but got the tensor's size of 4096"
    # crash. It proves the mismatched graph does structurally match, so the
    # graceful rejection above is the guard working, not a missed match.
    pm_pass = pm.PatternMatcherPass()
    unguarded = ToySplitCatPattern(_fake_vllm_config(), width=64)
    with (
        mock.patch.object(base_pattern.nge, "register_replacement"),
        mock.patch.object(base_pattern, "_wrap_search_fn_with_width_guard", side_effect=lambda f, *a, **k: f),
    ):
        unguarded.register(pm_pass)

    graph = _build_graph(width=40)
    with pytest.raises(ValueError, match="[Ss]plit sizes"):
        pm_pass.apply(graph)


def test_search_fn_wrapper_preserves_signature():
    def search_fn(x, bias):
        return x

    wrapped = base_pattern._wrap_search_fn_with_width_guard(search_fn, "x", 64, "Toy")
    import inspect

    assert list(inspect.signature(wrapped).parameters) == ["x", "bias"]


class _UnresolvableBool:
    def __bool__(self):
        raise RuntimeError("GuardOnDataDependentSymNode: data-dependent symbol")


def test_matched_width_unresolvable_symbolic_returns_none():
    # Symbolic shapes: val.shape[-1] is a SymInt whose comparison yields a
    # SymBool that cannot be statically resolved. The guard must treat that
    # as unverifiable (None -> reject-on-unknown) instead of letting bool()
    # raise inside the caller's conditional and crash the compilation.
    sym_width = mock.MagicMock()
    sym_width.__eq__.return_value = _UnresolvableBool()  # type: ignore[attr-defined]

    fake_val = mock.MagicMock(spec=torch.Tensor)  # passes isinstance check
    fake_val.dim.return_value = 2
    fake_val.shape.__getitem__.return_value = sym_width

    fake_node = mock.MagicMock()
    fake_node.meta = {"val": fake_val}
    fake_match = mock.MagicMock()
    fake_match.kwargs = {"x": fake_node}

    assert base_pattern._matched_main_input_has_expected_width(fake_match, "x", 64) is None


def test_search_fn_guard_raises_runtime_error_on_provable_mismatch():
    def search_fn(x):
        return x

    wrapped = base_pattern._wrap_search_fn_with_width_guard(search_fn, "x", 64, "Toy")
    with pytest.raises(RuntimeError, match="declining to apply"):
        wrapped(torch.randn(2, 40))
    # Width match and non-tensor args fall through to the original fn.
    assert wrapped(torch.randn(2, 64)).shape == (2, 64)
