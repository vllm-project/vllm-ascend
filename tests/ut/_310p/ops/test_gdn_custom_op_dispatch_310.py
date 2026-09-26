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

import importlib
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
import torch.nn.functional as F

chunk_gated_delta_rule = importlib.import_module("vllm_ascend._310p.ops.fla.chunk_gated_delta_rule")
fused_gdn_gating = importlib.import_module("vllm_ascend._310p.ops.fla.fused_gdn_gating")


def _make_chunk_inputs():
    batch_size = 1
    sequence_length = 64
    num_key_heads = 2
    num_value_heads = 4
    key_dim = 16
    value_dim = 16
    q = torch.randn(
        batch_size,
        sequence_length,
        num_key_heads,
        key_dim,
        dtype=torch.float16,
    )
    k = torch.randn_like(q)
    v = torch.randn(
        batch_size,
        sequence_length,
        num_value_heads,
        value_dim,
        dtype=torch.float16,
    )
    g = torch.randn(
        batch_size,
        sequence_length,
        num_value_heads,
        dtype=torch.float32,
    )
    beta = torch.randn(
        batch_size,
        sequence_length,
        num_value_heads,
        dtype=torch.float16,
    )
    return q, k, v, g, beta


def test_fused_gdn_gating_getter_rejects_unavailable_custom_op():
    custom_op = Mock()
    ascend_ops = SimpleNamespace(
        fused_gdn_gating=custom_op,
        is_fused_gdn_gating_available=Mock(return_value=False),
    )

    with patch.object(fused_gdn_gating.torch, "ops", SimpleNamespace(_C_ascend=ascend_ops)):
        actual = fused_gdn_gating._get_fused_gdn_gating_op()

    assert actual is None
    ascend_ops.is_fused_gdn_gating_available.assert_called_once_with()


def test_chunk_compute_wy_getter_rejects_unavailable_custom_op():
    custom_op = Mock()
    ascend_ops = SimpleNamespace(
        chunk_gated_delta_rule_compute_wy=custom_op,
        is_chunk_gated_delta_rule_compute_wy_available=Mock(return_value=False),
    )

    with patch.object(chunk_gated_delta_rule.torch, "ops", SimpleNamespace(_C_ascend=ascend_ops)):
        actual = chunk_gated_delta_rule._get_chunk_gated_delta_rule_compute_wy_op()

    assert actual is None
    ascend_ops.is_chunk_gated_delta_rule_compute_wy_available.assert_called_once_with()


def test_fused_gdn_gating_dispatches_to_custom_op():
    num_tokens = 3
    num_heads = 4
    A_log = torch.randn(num_heads, dtype=torch.float32)
    a = torch.randn(num_tokens, num_heads, dtype=torch.bfloat16)
    b = torch.randn(num_tokens, num_heads, dtype=torch.float32)
    dt_bias = torch.randn(num_heads, dtype=torch.float32)
    expected = (
        torch.randn(1, num_tokens, num_heads, dtype=torch.float32),
        torch.randn(1, num_tokens, num_heads, dtype=torch.float16),
    )
    custom_op = Mock(return_value=expected)

    with (
        patch.object(fused_gdn_gating, "_get_fused_gdn_gating_op", return_value=custom_op),
        patch.object(fused_gdn_gating, "_can_use_fused_gdn_gating_op", return_value=True),
    ):
        actual = fused_gdn_gating.fused_gdn_gating_pytorch(
            A_log,
            a,
            b,
            dt_bias,
            beta=1.5,
            threshold=18.0,
        )

    assert actual is expected
    custom_op.assert_called_once()
    op_args = custom_op.call_args.args
    assert all(tensor.dtype == torch.float16 for tensor in op_args[:4])
    assert all(tensor.is_contiguous() for tensor in op_args[:4])
    assert op_args[4:] == (1.5, 18.0)


def test_fused_gdn_gating_falls_back_for_unsupported_inputs():
    A_log = torch.tensor([0.1, -0.2], dtype=torch.float32)
    a = torch.tensor([[0.2, -0.4], [0.7, 0.3]], dtype=torch.float32)
    b = torch.tensor([[0.4, -0.1], [-0.2, 0.8]], dtype=torch.float32)
    dt_bias = torch.tensor([0.05, -0.15], dtype=torch.float32)
    custom_op = Mock()

    with (
        patch.object(fused_gdn_gating, "_get_fused_gdn_gating_op", return_value=custom_op),
        patch.object(fused_gdn_gating, "_can_use_fused_gdn_gating_op", return_value=False),
    ):
        actual_g, actual_beta = fused_gdn_gating.fused_gdn_gating_pytorch(
            A_log,
            a,
            b,
            dt_bias,
        )

    custom_op.assert_not_called()
    expected_g = -torch.exp(A_log).unsqueeze(0) * F.softplus(a + dt_bias)
    expected_beta = torch.sigmoid(b)
    torch.testing.assert_close(actual_g, expected_g.unsqueeze(0))
    torch.testing.assert_close(actual_beta, expected_beta.unsqueeze(0))


def test_chunk_compute_wy_dispatches_to_custom_op():
    inputs = _make_chunk_inputs()
    expected = tuple(torch.randn(1) for _ in range(5))
    custom_op = Mock(return_value=expected)

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
        actual = chunk_gated_delta_rule._compute_kernel_inputs(*inputs, chunk_size=64)

    assert actual is expected
    custom_op.assert_called_once()
    op_args = custom_op.call_args.args
    assert all(actual_arg.data_ptr() == input_arg.data_ptr() for actual_arg, input_arg in zip(op_args[:5], inputs))
    assert all(tensor.is_contiguous() for tensor in op_args[:5])
    assert op_args[5] == 64


def test_chunk_compute_wy_falls_back_for_unsupported_inputs():
    inputs = _make_chunk_inputs()
    expected = tuple(torch.randn(1) for _ in range(5))
    custom_op = Mock()

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
            return_value=expected,
        ) as torch_fallback,
    ):
        actual = chunk_gated_delta_rule._compute_kernel_inputs(*inputs, chunk_size=64)

    assert actual is expected
    custom_op.assert_not_called()
    torch_fallback.assert_called_once()
    fallback_args = torch_fallback.call_args.args
    assert all(actual_arg is input_arg for actual_arg, input_arg in zip(fallback_args[:5], inputs))
    assert fallback_args[5] == 64
