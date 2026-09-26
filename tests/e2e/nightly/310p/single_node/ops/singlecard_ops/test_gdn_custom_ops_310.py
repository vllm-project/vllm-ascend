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

import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401

from vllm_ascend._310p.ops.fla.chunk_gated_delta_rule import (
    _compute_kernel_inputs_from_torch_wy,
)
from vllm_ascend.utils import enable_custom_op

CHUNK_SIZE = 64


def _get_required_custom_op(op_name: str, availability_op_name: str):
    enable_custom_op()
    ascend_ops = torch.ops._C_ascend
    availability_op = getattr(ascend_ops, availability_op_name)
    assert availability_op(), (
        f"{op_name} is unavailable. Install the ops-transformer package "
        "containing the required ACLNN operator before running this test."
    )
    return getattr(ascend_ops, op_name)


def test_fused_gdn_gating_custom_op_matches_torch_reference():
    torch.manual_seed(0)
    num_tokens = 37
    num_heads = 8
    beta = 1.0
    threshold = 20.0

    A_log = torch.randn(num_heads, dtype=torch.float16) * 0.1
    a = torch.randn(num_tokens, num_heads, dtype=torch.float16) * 0.1
    b = torch.randn(num_tokens, num_heads, dtype=torch.float16) * 0.1
    dt_bias = torch.randn(num_heads, dtype=torch.float16) * 0.1

    expected_g = -torch.exp(A_log.float()).unsqueeze(0) * F.softplus(
        a.float() + dt_bias.float(),
        beta=beta,
        threshold=threshold,
    )
    expected_g = expected_g.unsqueeze(0)
    expected_beta = torch.sigmoid(b.float()).to(b.dtype).unsqueeze(0)

    custom_op = _get_required_custom_op(
        "fused_gdn_gating",
        "is_fused_gdn_gating_available",
    )
    actual_g, actual_beta = custom_op(
        A_log.npu(),
        a.npu(),
        b.npu(),
        dt_bias.npu(),
        beta,
        threshold,
    )

    torch.testing.assert_close(
        actual_g.cpu(),
        expected_g,
        rtol=1e-3,
        atol=1e-3,
    )
    torch.testing.assert_close(
        actual_beta.cpu(),
        expected_beta,
        rtol=1e-3,
        atol=1e-3,
    )


def test_chunk_compute_wy_custom_op_matches_torch_reference():
    torch.manual_seed(0)
    batch_size = 1
    sequence_length = CHUNK_SIZE
    num_key_heads = 2
    num_value_heads = 4
    key_dim = 16
    value_dim = 16

    q = (
        torch.randn(
            batch_size,
            sequence_length,
            num_key_heads,
            key_dim,
            dtype=torch.float16,
        )
        * 0.1
    )
    k = torch.randn_like(q) * 0.1
    v = (
        torch.randn(
            batch_size,
            sequence_length,
            num_value_heads,
            value_dim,
            dtype=torch.float16,
        )
        * 0.1
    )
    g = (
        -torch.rand(
            batch_size,
            sequence_length,
            num_value_heads,
            dtype=torch.float32,
        )
        * 0.1
    )
    beta = torch.sigmoid(
        torch.randn(
            batch_size,
            sequence_length,
            num_value_heads,
            dtype=torch.float16,
        )
    )

    expected = _compute_kernel_inputs_from_torch_wy(
        q,
        k,
        v,
        g,
        beta,
        CHUNK_SIZE,
    )
    custom_op = _get_required_custom_op(
        "chunk_gated_delta_rule_compute_wy",
        "is_chunk_gated_delta_rule_compute_wy_available",
    )
    actual = custom_op(
        q.npu(),
        k.npu(),
        v.npu(),
        g.npu(),
        beta.npu(),
        CHUNK_SIZE,
    )

    for actual_tensor, expected_tensor in zip(actual, expected):
        torch.testing.assert_close(
            actual_tensor.cpu(),
            expected_tensor,
            rtol=2e-2,
            atol=2e-2,
        )
