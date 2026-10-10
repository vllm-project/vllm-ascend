from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from vllm_ascend._310p.ops.fla import (
    chunk_gated_delta_rule,
    fused_gdn_gating,
)


def test_fused_gdn_gating_dispatches_to_cann_ops_transformer():
    custom_op = Mock(return_value=(torch.empty(1), torch.empty(1)))
    inputs = [torch.randn(2, 4) for _ in range(2)]
    A_log = torch.randn(4)
    dt_bias = torch.randn(4)

    with (
        patch.object(fused_gdn_gating, "_get_fused_gdn_gating_op", return_value=custom_op),
        patch.object(fused_gdn_gating, "_can_use_fused_gdn_gating_op", return_value=True),
    ):
        fused_gdn_gating.fused_gdn_gating(A_log, *inputs, dt_bias)

    custom_op.assert_called_once()


def test_fused_gdn_gating_falls_back_for_unsupported_inputs():
    custom_op = Mock()
    A_log = torch.randn(4)
    a = torch.randn(2, 4)
    b = torch.randn(2, 4)
    dt_bias = torch.randn(4)

    with (
        patch.object(fused_gdn_gating, "_get_fused_gdn_gating_op", return_value=custom_op),
        patch.object(fused_gdn_gating, "_can_use_fused_gdn_gating_op", return_value=False),
    ):
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
            "_get_chunk_gated_delta_rule_compute_wy_op",
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
            "_get_chunk_gated_delta_rule_compute_wy_op",
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


def test_getters_return_none_when_extension_is_unavailable():
    with (
        patch.object(fused_gdn_gating.torch, "ops", SimpleNamespace()),
        patch.object(fused_gdn_gating, "_load_cann_ops_transformer", return_value=False),
    ):
        assert fused_gdn_gating._get_fused_gdn_gating_op() is None

    with (
        patch.object(chunk_gated_delta_rule.torch, "ops", SimpleNamespace()),
        patch.object(chunk_gated_delta_rule, "_load_cann_ops_transformer", return_value=False),
    ):
        assert chunk_gated_delta_rule._get_chunk_gated_delta_rule_compute_wy_op() is None
