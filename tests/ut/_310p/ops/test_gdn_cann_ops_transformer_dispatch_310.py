from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from vllm_ascend._310p.ops.fla import (
    cann_ops,
    chunk_gated_delta_rule,
    fused_gdn_gating,
)


def test_fused_gdn_gating_dispatches_to_cann_ops_transformer():
    custom_op = Mock(return_value=(torch.empty(1), torch.empty(1)))
    inputs = [torch.randn(2, 4) for _ in range(2)]
    A_log = torch.randn(4)
    dt_bias = torch.randn(4)
    # Simulate NPU dispatch while keeping all tensor operations on CPU.
    a = Mock(wraps=inputs[0])
    a.device = SimpleNamespace(type="npu")

    with patch.object(fused_gdn_gating, "get_gdn_op", return_value=custom_op):
        fused_gdn_gating.fused_gdn_gating(A_log, a, inputs[1], dt_bias)

    custom_op.assert_called_once()


def test_fused_gdn_gating_falls_back_for_cpu_inputs():
    custom_op = Mock()
    A_log = torch.randn(4)
    a = torch.randn(2, 4)
    b = torch.randn(2, 4)
    dt_bias = torch.randn(4)

    with patch.object(fused_gdn_gating, "get_gdn_op", return_value=custom_op):
        actual = fused_gdn_gating.fused_gdn_gating(A_log, a, b, dt_bias)

    expected = fused_gdn_gating.fused_gdn_gating_pytorch(A_log, a, b, dt_bias)
    custom_op.assert_not_called()
    torch.testing.assert_close(actual[0], expected[0])
    torch.testing.assert_close(actual[1], expected[1])


def test_chunk_compute_wy_dispatches_to_cann_ops_transformer():
    expected = tuple(torch.empty(1) for _ in range(5))
    custom_op = Mock(return_value=expected)
    q = torch.randn(1, 64, 2, 16, dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn(1, 64, 4, 16, dtype=torch.float16)
    g = torch.randn(1, 64, 4, dtype=torch.float32)
    beta = torch.randn(1, 64, 4, dtype=torch.float16)

    with (
        patch.object(
            chunk_gated_delta_rule,
            "get_gdn_op",
            return_value=custom_op,
        ),
        patch.object(
            chunk_gated_delta_rule,
            "_can_use_chunk_gated_delta_rule_compute_wy_op",
            return_value=True,
        ),
    ):
        actual = chunk_gated_delta_rule._compute_kernel_inputs(q, k, v, g, beta, 64)

    assert actual is expected
    custom_op.assert_called_once()


def test_chunk_compute_wy_falls_back_for_unsupported_inputs():
    expected = tuple(torch.empty(1) for _ in range(5))
    custom_op = Mock()
    fallback = Mock(return_value=expected)
    q = torch.randn(1, 64, 2, 16, dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn(1, 64, 4, 16, dtype=torch.float16)
    g = torch.randn(1, 64, 4, dtype=torch.float32)
    beta = torch.randn(1, 64, 4, dtype=torch.float16)

    with (
        patch.object(
            chunk_gated_delta_rule,
            "get_gdn_op",
            return_value=custom_op,
        ),
        patch.object(
            chunk_gated_delta_rule,
            "_can_use_chunk_gated_delta_rule_compute_wy_op",
            return_value=False,
        ),
        patch.object(
            chunk_gated_delta_rule,
            "_compute_kernel_inputs_from_torch_wy",
            fallback,
        ),
    ):
        actual = chunk_gated_delta_rule._compute_kernel_inputs(q, k, v, g, beta, 64)

    assert actual is expected
    custom_op.assert_not_called()
    fallback.assert_called_once()


@pytest.fixture
def clear_gdn_op_cache():
    cann_ops.get_gdn_op.cache_clear()
    cann_ops._load_cann_ops_transformer.cache_clear()
    yield
    cann_ops.get_gdn_op.cache_clear()
    cann_ops._load_cann_ops_transformer.cache_clear()


def test_gdn_ops_share_one_import_and_cache_lookups(clear_gdn_op_cache):
    gating_op = Mock()
    wy_op = Mock()
    ops = SimpleNamespace(
        fused_gdn_gating=gating_op,
        chunk_gated_delta_rule_compute_wy=wy_op,
    )
    with patch.object(cann_ops, "import_module", return_value=ops) as load:
        assert cann_ops.get_gdn_op("fused_gdn_gating") is gating_op
        assert cann_ops.get_gdn_op("chunk_gated_delta_rule_compute_wy") is wy_op
        # Changing the module must not trigger another attribute lookup.
        ops.fused_gdn_gating = Mock()
        assert cann_ops.get_gdn_op("fused_gdn_gating") is gating_op
        assert cann_ops.get_gdn_op("chunk_gated_delta_rule_compute_wy") is wy_op
        load.assert_called_once_with("cann_ops_transformer.ops")


@pytest.mark.parametrize("error", [ImportError, OSError, RuntimeError])
def test_unavailable_extension_is_cached(clear_gdn_op_cache, error):
    with patch.object(cann_ops, "import_module", side_effect=error("unavailable")) as load:
        for _ in range(2):
            assert cann_ops.get_gdn_op("fused_gdn_gating") is None
            assert cann_ops.get_gdn_op("chunk_gated_delta_rule_compute_wy") is None
        load.assert_called_once_with("cann_ops_transformer.ops")


def test_missing_gdn_op_does_not_disable_other_op(clear_gdn_op_cache):
    wy_op = Mock()
    ops = SimpleNamespace(chunk_gated_delta_rule_compute_wy=wy_op)
    with patch.object(cann_ops, "import_module", return_value=ops):
        assert cann_ops.get_gdn_op("fused_gdn_gating") is None
        assert cann_ops.get_gdn_op("chunk_gated_delta_rule_compute_wy") is wy_op
        ops.fused_gdn_gating = Mock()
        assert cann_ops.get_gdn_op("fused_gdn_gating") is None


def test_gating_falls_back_when_extension_is_unavailable():
    inputs = [torch.randn(4), torch.randn(2, 4), torch.randn(2, 4), torch.randn(4)]
    expected = fused_gdn_gating.fused_gdn_gating_pytorch(*inputs)
    with patch.object(fused_gdn_gating, "get_gdn_op", return_value=None):
        actual = fused_gdn_gating.fused_gdn_gating(*inputs)
    torch.testing.assert_close(actual[0], expected[0])
    torch.testing.assert_close(actual[1], expected[1])


def test_wy_falls_back_when_extension_is_unavailable():
    expected = tuple(torch.empty(1) for _ in range(5))
    inputs = [torch.empty(1) for _ in range(5)]
    with (
        patch.object(chunk_gated_delta_rule, "get_gdn_op", return_value=None),
        patch.object(chunk_gated_delta_rule, "_compute_kernel_inputs_from_torch_wy", return_value=expected) as fallback,
    ):
        assert chunk_gated_delta_rule._compute_kernel_inputs(*inputs, 64) is expected
    fallback.assert_called_once_with(*inputs, 64)
