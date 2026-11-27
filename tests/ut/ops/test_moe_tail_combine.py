# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm_ascend.ops.fused_moe import fused_moe as fused_moe_module


def _make_runner(plain_tail_add: bool):
    """Build a runner shell that only carries the resolved combine mode."""
    runner = fused_moe_module.AscendMoERunner.__new__(fused_moe_module.AscendMoERunner)
    runner._plain_tail_add = plain_tail_add
    return runner


def _config(enable_super_kernel: bool, enable_static_kernel: bool):
    return SimpleNamespace(
        ascend_compilation_config=SimpleNamespace(
            enable_super_kernel=enable_super_kernel,
            enable_static_kernel=enable_static_kernel,
        )
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_tail_combine_branches_are_bit_identical(dtype):
    torch.manual_seed(0)
    shared = torch.randn(5, 8, dtype=dtype)
    routed = torch.randn(5, 8, dtype=dtype)

    combined_plain = _make_runner(True)._combine_shared_and_routed_output(shared, routed)
    combined_foreach = _make_runner(False)._combine_shared_and_routed_output(shared, routed)

    assert combined_plain.dtype == dtype
    torch.testing.assert_close(combined_plain, combined_foreach, rtol=0, atol=0)


def test_tail_combine_preserves_empty_input_shape():
    shared = torch.empty(0, 8, dtype=torch.bfloat16)
    routed = torch.empty(0, 8, dtype=torch.bfloat16)

    for plain in (True, False):
        out = _make_runner(plain)._combine_shared_and_routed_output(shared, routed)
        assert out.shape == (0, 8)
        assert out.dtype == torch.bfloat16


def test_default_branch_uses_single_element_foreach_add():
    shared = torch.randn(3, 8, dtype=torch.bfloat16)
    routed = torch.randn(3, 8, dtype=torch.bfloat16)
    original = torch._foreach_add
    list_lengths = []

    def spy(tensors_a, tensors_b, *args, **kwargs):
        list_lengths.append((len(tensors_a), len(tensors_b)))
        return original(tensors_a, tensors_b, *args, **kwargs)

    with patch.object(torch, "_foreach_add", side_effect=spy) as mocked:
        out = _make_runner(False)._combine_shared_and_routed_output(shared, routed)

    assert mocked.call_count == 1
    assert list_lengths == [(1, 1)]
    torch.testing.assert_close(out, shared + routed, rtol=0, atol=0)


def test_plain_branch_never_dispatches_foreach_add():
    shared = torch.randn(3, 8, dtype=torch.bfloat16)
    routed = torch.randn(3, 8, dtype=torch.bfloat16)

    def fail(*args, **kwargs):
        raise AssertionError("torch._foreach_add must not be dispatched when the plain add is used")

    with patch.object(torch, "_foreach_add", side_effect=fail):
        out = _make_runner(True)._combine_shared_and_routed_output(shared, routed)

    torch.testing.assert_close(out, shared + routed, rtol=0, atol=0)


@pytest.mark.parametrize(
    "enable_super_kernel,enable_static_kernel,expected",
    [
        (False, False, True),
        # Static kernel does not need the multi-tensor combine, so it must not
        # disable the plain add.
        (False, True, True),
        (True, False, False),
        (True, True, False),
    ],
)
def test_plain_tail_add_follows_super_kernel_only(monkeypatch, enable_super_kernel, enable_static_kernel, expected):
    monkeypatch.setattr(
        fused_moe_module,
        "get_ascend_config",
        lambda: _config(enable_super_kernel, enable_static_kernel),
    )

    assert fused_moe_module._plain_tail_add_enabled() is expected


def test_plain_tail_add_requires_initialized_ascend_config(monkeypatch):
    def raise_runtime_error():
        raise RuntimeError("Ascend config is not initialized")

    monkeypatch.setattr(fused_moe_module, "get_ascend_config", raise_runtime_error)

    assert fused_moe_module._plain_tail_add_enabled() is False


def test_runner_built_without_init_keeps_foreach_path():
    """Runners assembled via ``__new__`` (as existing tests do) must not crash."""
    runner = fused_moe_module.AscendMoERunner.__new__(fused_moe_module.AscendMoERunner)
    shared = torch.randn(3, 8, dtype=torch.bfloat16)
    routed = torch.randn(3, 8, dtype=torch.bfloat16)

    with patch.object(torch, "_foreach_add", wraps=torch._foreach_add) as mocked:
        out = runner._combine_shared_and_routed_output(shared, routed)

    assert mocked.call_count == 1
    torch.testing.assert_close(out, shared + routed, rtol=0, atol=0)
