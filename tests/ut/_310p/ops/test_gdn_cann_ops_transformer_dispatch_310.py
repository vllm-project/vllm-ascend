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

    fused_gdn_gating.fused_gdn_gating(A_log, a, inputs[1], dt_bias, gating_op=custom_op)

    custom_op.assert_called_once()


def test_fused_gdn_gating_falls_back_for_cpu_inputs():
    custom_op = Mock()
    A_log = torch.randn(4)
    a = torch.randn(2, 4)
    b = torch.randn(2, 4)
    dt_bias = torch.randn(4)

    actual = fused_gdn_gating.fused_gdn_gating(A_log, a, b, dt_bias, gating_op=custom_op)

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
            "_can_use_chunk_gated_delta_rule_compute_wy_op",
            return_value=True,
        ),
    ):
        actual = chunk_gated_delta_rule._compute_kernel_inputs(q, k, v, g, beta, 64, compute_wy_op=custom_op)

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
            "_can_use_chunk_gated_delta_rule_compute_wy_op",
            return_value=False,
        ),
        patch.object(
            chunk_gated_delta_rule,
            "_compute_kernel_inputs_from_torch_wy",
            fallback,
        ),
    ):
        actual = chunk_gated_delta_rule._compute_kernel_inputs(q, k, v, g, beta, 64, compute_wy_op=custom_op)

    assert actual is expected
    custom_op.assert_not_called()
    fallback.assert_called_once()


def test_load_gdn_ops():
    gating_op, wy_op = Mock(), Mock()
    ops = SimpleNamespace(fused_gdn_gating=gating_op, chunk_gated_delta_rule_compute_wy=wy_op)
    with patch.object(cann_ops, "import_module", return_value=ops) as load:
        assert cann_ops.load_cann_gdn_ops() == (gating_op, wy_op)
    load.assert_called_once_with("cann_ops_transformer.ops")


@pytest.mark.parametrize("error", [ImportError, OSError, RuntimeError])
def test_load_gdn_ops_when_extension_is_unavailable(error):
    with patch.object(cann_ops, "import_module", side_effect=error("unavailable")):
        assert cann_ops.load_cann_gdn_ops() == (None, None)


@pytest.mark.parametrize("missing", ["fused_gdn_gating", "chunk_gated_delta_rule_compute_wy"])
def test_missing_gdn_op_does_not_disable_other_op(missing):
    gating_op, wy_op = Mock(), Mock()
    ops = SimpleNamespace(fused_gdn_gating=gating_op, chunk_gated_delta_rule_compute_wy=wy_op)
    delattr(ops, missing)
    with patch.object(cann_ops, "import_module", return_value=ops):
        expected = (None, wy_op) if missing == "fused_gdn_gating" else (gating_op, None)
        assert cann_ops.load_cann_gdn_ops() == expected


def test_gating_falls_back_when_extension_is_unavailable():
    inputs = [torch.randn(4), torch.randn(2, 4), torch.randn(2, 4), torch.randn(4)]
    expected = fused_gdn_gating.fused_gdn_gating_pytorch(*inputs)
    actual = fused_gdn_gating.fused_gdn_gating(*inputs)
    torch.testing.assert_close(actual[0], expected[0])
    torch.testing.assert_close(actual[1], expected[1])


def test_wy_falls_back_when_extension_is_unavailable():
    expected = tuple(torch.empty(1) for _ in range(5))
    inputs = [torch.empty(1) for _ in range(5)]
    with (
        patch.object(chunk_gated_delta_rule, "_compute_kernel_inputs_from_torch_wy", return_value=expected) as fallback,
    ):
        assert chunk_gated_delta_rule._compute_kernel_inputs(*inputs, 64) is expected
    fallback.assert_called_once_with(*inputs, 64)


@pytest.mark.parametrize("ops", [(Mock(), Mock()), (None, None)])
def test_init_wrapper_binds_ops_once(ops):
    class QwenAttention:
        def __init__(self, config, *, prefix):
            self.config = config
            self.prefix = prefix

    original_init = QwenAttention.__init__
    # Both patch entry points may target the same upstream class.
    QwenAttention.__init__ = cann_ops.bind_cann_gdn_ops_on_init(original_init)
    wrapped_init = QwenAttention.__init__
    QwenAttention.__init__ = cann_ops.bind_cann_gdn_ops_on_init(wrapped_init)
    assert QwenAttention.__init__ is wrapped_init
    assert wrapped_init.__wrapped__ is original_init
    with patch.object(cann_ops, "load_cann_gdn_ops", return_value=ops) as load:
        attention = QwenAttention("config", prefix="gdn")
        assert attention.config == "config"
        assert attention.prefix == "gdn"
        assert attention.fused_gdn_gating_op is ops[0]
        assert attention.chunk_gated_delta_rule_compute_wy_op is ops[1]
        load.assert_called_once_with()


def test_wrapped_gdn_parent_and_child_share_one_binding():
    class Parent:
        @cann_ops.bind_cann_gdn_ops_on_init
        def __init__(self):
            pass

    class Child(Parent):
        @cann_ops.bind_cann_gdn_ops_on_init
        def __init__(self):
            super().__init__()

    ops = (Mock(), Mock())
    with patch.object(cann_ops, "load_cann_gdn_ops", return_value=ops) as load:
        attention = Child()
        assert attention.fused_gdn_gating_op is ops[0]
        assert attention.chunk_gated_delta_rule_compute_wy_op is ops[1]
        load.assert_called_once_with()
