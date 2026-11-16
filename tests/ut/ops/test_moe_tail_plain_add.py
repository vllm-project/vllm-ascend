# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the MoE tail combine switch.

``additional_config.enable_moe_tail_plain_add`` only changes *which* op performs
the shared+routed sum, so the tests pin down both branches: the default keeps
the single ``torch._foreach_add`` dispatch, the switch replaces it with a plain
add, and both produce bit-identical results. The switch must also fall back to
the original branch when the flag was never populated (runners built without an
initialised Ascend config).
"""

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from vllm_ascend.ascend_config import AscendConfig
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


def test_default_branch_dispatches_foreach_add_once(monkeypatch: pytest.MonkeyPatch):
    shared = torch.ones(2, 4)
    routed = torch.ones(2, 4)
    calls = _spy_foreach_add(monkeypatch)

    out = _make_runner(_plain_tail_add=False)._combine_shared_and_routed_output(shared, routed)

    # Exactly one dispatch, on a one-element list: that single-element list is
    # the shape the plain-add switch removes.
    assert len(calls) == 1
    assert len(calls[0]) == 1
    assert calls[0][0] is shared
    torch.testing.assert_close(out, shared + routed)


def test_plain_branch_never_dispatches_foreach_add(monkeypatch: pytest.MonkeyPatch):
    def fail(*args: Any, **kwargs: Any):
        raise AssertionError("enable_moe_tail_plain_add must not use torch._foreach_add")

    monkeypatch.setattr(torch, "_foreach_add", fail)
    shared = torch.ones(2, 4)
    routed = torch.ones(2, 4)

    out = _make_runner(_plain_tail_add=True)._combine_shared_and_routed_output(shared, routed)

    torch.testing.assert_close(out, shared + routed)


def test_missing_switch_falls_back_to_foreach_add(monkeypatch: pytest.MonkeyPatch):
    runner = _make_runner()
    assert not hasattr(runner, "_plain_tail_add")
    calls = _spy_foreach_add(monkeypatch)

    runner._combine_shared_and_routed_output(torch.ones(1, 4), torch.ones(1, 4))

    assert len(calls) == 1


def test_switch_defaults_to_off():
    config = AscendConfig(sparse_kv_offload_config=SimpleNamespace(enabled=False))

    assert config.enable_moe_tail_plain_add is False


def test_switch_is_accepted_from_additional_config():
    # init_ascend_config forwards user keys from additional_config as kwargs,
    # so this mirrors --additional-config '{"enable_moe_tail_plain_add": true}'.
    config = AscendConfig(
        sparse_kv_offload_config=SimpleNamespace(enabled=False),
        enable_moe_tail_plain_add=True,
    )

    assert config.enable_moe_tail_plain_add is True
