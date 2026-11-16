# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the MoE tail combine.

The combine only changes *which* op performs the shared+routed sum, so the
tests pin down all three things that matter: the two branches produce
bit-identical results, the plain add is chosen only when no graph mode is
requested, and the graph modes (and an uninitialised Ascend config) fall back
to the original ``torch._foreach_add``.
"""

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from vllm_ascend.ops.fused_moe import fused_moe
from vllm_ascend.ops.fused_moe.fused_moe import AscendMoERunner


def _make_runner(**attrs: Any) -> AscendMoERunner:
    """Build a runner without running the heavy MoE construction path."""
    runner = AscendMoERunner.__new__(AscendMoERunner)
    for name, value in attrs.items():
        setattr(runner, name, value)
    return runner


def _spy_foreach_add(monkeypatch: pytest.MonkeyPatch) -> list[list[torch.Tensor]]:
    """Record the tensor lists passed to ``torch._foreach_add``."""
    calls: list[list[torch.Tensor]] = []
    original = torch._foreach_add

    def spy(tensors, others, *args, **kwargs):  # type: ignore[no-untyped-def]
        calls.append(list(tensors))
        return original(tensors, others, *args, **kwargs)

    monkeypatch.setattr(torch, "_foreach_add", spy)
    return calls


def _patch_compilation_config(
    monkeypatch: pytest.MonkeyPatch,
    *,
    enable_super_kernel: bool,
    enable_static_kernel: bool,
) -> None:
    compilation_config = SimpleNamespace(
        enable_super_kernel=enable_super_kernel,
        enable_static_kernel=enable_static_kernel,
    )
    monkeypatch.setattr(
        fused_moe,
        "get_ascend_config",
        lambda: SimpleNamespace(ascend_compilation_config=compilation_config),
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("plain_tail_add", [False, True])
def test_both_branches_are_bit_identical(dtype: torch.dtype, plain_tail_add: bool):
    shared = torch.randn(4, 8, dtype=dtype)
    routed = torch.randn(4, 8, dtype=dtype)
    expected = torch._foreach_add([shared], [routed])[0]

    runner = _make_runner(_plain_tail_add=plain_tail_add)
    torch.testing.assert_close(runner._combine_shared_and_routed_output(shared, routed), expected, rtol=0, atol=0)


def test_both_branches_are_bit_identical_on_empty_input():
    shared = torch.zeros(0, 8, dtype=torch.bfloat16)
    routed = torch.zeros(0, 8, dtype=torch.bfloat16)
    for plain_tail_add in (False, True):
        runner = _make_runner(_plain_tail_add=plain_tail_add)
        out = runner._combine_shared_and_routed_output(shared, routed)
        assert out.shape == (0, 8)
        assert out.dtype == torch.bfloat16


def test_foreach_branch_dispatches_foreach_add_once(monkeypatch: pytest.MonkeyPatch):
    shared = torch.ones(2, 4)
    routed = torch.ones(2, 4)
    calls = _spy_foreach_add(monkeypatch)

    out = _make_runner(_plain_tail_add=False)._combine_shared_and_routed_output(shared, routed)

    # Exactly one dispatch, on a one-element list: that single-element list is
    # the shape the plain add removes.
    assert len(calls) == 1
    assert len(calls[0]) == 1
    assert calls[0][0] is shared
    torch.testing.assert_close(out, shared + routed)


def test_plain_branch_never_dispatches_foreach_add(monkeypatch: pytest.MonkeyPatch):
    def fail(*args: Any, **kwargs: Any):
        raise AssertionError("the plain-add branch must not use torch._foreach_add")

    monkeypatch.setattr(torch, "_foreach_add", fail)
    shared = torch.ones(2, 4)
    routed = torch.ones(2, 4)

    out = _make_runner(_plain_tail_add=True)._combine_shared_and_routed_output(shared, routed)

    torch.testing.assert_close(out, shared + routed)


def test_runner_built_without_init_keeps_foreach_add(monkeypatch: pytest.MonkeyPatch):
    # Some tests build runners via `__new__` + `nn.Module.__init__`, which skips
    # our `__init__`; the class-level default must keep the graph-safe branch.
    runner = _make_runner()
    assert runner._plain_tail_add is False
    calls = _spy_foreach_add(monkeypatch)

    runner._combine_shared_and_routed_output(torch.ones(1, 4), torch.ones(1, 4))

    assert len(calls) == 1


@pytest.mark.parametrize(
    ("enable_super_kernel", "enable_static_kernel", "expected"),
    [
        (False, False, True),
        (True, False, False),
        (False, True, False),
        (True, True, False),
    ],
)
def test_plain_add_only_outside_graph_modes(
    monkeypatch: pytest.MonkeyPatch,
    enable_super_kernel: bool,
    enable_static_kernel: bool,
    expected: bool,
):
    _patch_compilation_config(
        monkeypatch,
        enable_super_kernel=enable_super_kernel,
        enable_static_kernel=enable_static_kernel,
    )

    assert fused_moe._plain_tail_add_enabled() is expected


def test_uninitialised_ascend_config_keeps_foreach_add(monkeypatch: pytest.MonkeyPatch):
    def raise_not_initialised():
        raise RuntimeError("Ascend config is not initialized")

    monkeypatch.setattr(fused_moe, "get_ascend_config", raise_not_initialised)

    assert fused_moe._plain_tail_add_enabled() is False
