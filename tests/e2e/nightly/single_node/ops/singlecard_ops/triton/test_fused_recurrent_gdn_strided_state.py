# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
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

"""Padded-state regressions for the Ascend Triton GDN decode path."""

import pytest
import torch
import torch_npu
from vllm.third_party.flash_linear_attention.ops.fused_recurrent import (
    fused_recurrent_gated_delta_rule_packed_decode,
)

from tests.e2e.nightly.single_node.ops.singlecard_ops.test_recurrent_gated_delta_rule_strided import (
    _assert_padding_canary,
    _golden_recurrent_gated_delta_rule,
    _make_padded_state,
)
from vllm_ascend.ops.triton.fla.fused_recurrent import fused_recurrent_gated_delta_rule

_SEED = 73
_NUM_KEY_HEADS = 4
_NUM_VALUE_HEADS = 8
_KEY_DIM = 128
_VALUE_DIM = 128


def _l2norm_like_kernel(tensor: torch.Tensor) -> torch.Tensor:
    tensor_f32 = tensor.float()
    return tensor_f32 * torch.rsqrt(torch.sum(tensor_f32 * tensor_f32, dim=-1, keepdim=True) + 1e-6)


def _assert_numerically_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    name: str,
) -> None:
    actual_f32 = actual.detach().float().cpu()
    expected_f32 = expected.detach().float().cpu()
    max_abs_error = torch.max(torch.abs(actual_f32 - expected_f32)).item()
    rmse = torch.mean(torch.square(actual_f32 - expected_f32)).sqrt().item()
    reference_rmse = torch.mean(torch.square(expected_f32)).sqrt().item()
    relative_rmse = rmse / (reference_rmse + 1e-8)
    assert max_abs_error < 3e-2, f"{name}: max abs error {max_abs_error:.6f}"
    assert relative_rmse < 1e-2, f"{name}: relative RMSE {relative_rmse:.6f}"


def _assert_state_and_guards(
    state_npu: torch.Tensor,
    expected_state: torch.Tensor,
    state_before: torch.Tensor,
    backing_npu: torch.Tensor,
    num_states: int,
    leading_stride: int,
    selected_state_ids: set[int],
) -> None:
    for state_id in range(num_states):
        if state_id in selected_state_ids:
            _assert_numerically_close(
                state_npu[state_id],
                expected_state[state_id],
                name=f"state[{state_id}]",
            )
        else:
            torch.testing.assert_close(
                state_npu[state_id].cpu(),
                state_before[state_id],
                rtol=0,
                atol=0,
            )
    _assert_padding_canary(
        backing_npu,
        num_states,
        _NUM_VALUE_HEADS * _VALUE_DIM * _KEY_DIM,
        leading_stride,
    )


@pytest.mark.skip_global_cleanup
@torch.inference_mode()
def test_packed_decode_honors_padded_state_stride_and_storage_offset():
    """Ordinary one-token Decode must update non-zero and last-page slots."""
    torch.manual_seed(_SEED)
    num_states = 5
    num_active_tokens = 2
    batch_size = 3
    state_cpu, state_npu, backing_npu, leading_stride = _make_padded_state(
        num_states,
        _NUM_VALUE_HEADS,
        _VALUE_DIM,
        _KEY_DIM,
    )
    state_before = state_cpu.clone()
    # The final row models full-graph padding. The packed kernel defines state
    # index zero as NULL and must zero its output without reading or writing a
    # recurrent state.
    state_indices = torch.tensor([1, num_states - 1, 0], dtype=torch.int32)

    mixed_qkv_width = 2 * _NUM_KEY_HEADS * _KEY_DIM + _NUM_VALUE_HEADS * _VALUE_DIM
    mixed_qkv = (torch.randn(batch_size, mixed_qkv_width, dtype=torch.float32) * 0.1).to(torch.bfloat16)
    a = (torch.randn(batch_size, _NUM_VALUE_HEADS) * 0.2).to(torch.bfloat16)
    b = (torch.randn(batch_size, _NUM_VALUE_HEADS) * 0.2).to(torch.bfloat16)
    A_log = torch.randn(_NUM_VALUE_HEADS, dtype=torch.float32) * 0.1
    dt_bias = torch.randn(_NUM_VALUE_HEADS, dtype=torch.float32) * 0.1

    query = mixed_qkv[:, : _NUM_KEY_HEADS * _KEY_DIM].reshape(
        batch_size,
        _NUM_KEY_HEADS,
        _KEY_DIM,
    )
    key_start = _NUM_KEY_HEADS * _KEY_DIM
    value_start = 2 * key_start
    key = mixed_qkv[:, key_start:value_start].reshape(
        batch_size,
        _NUM_KEY_HEADS,
        _KEY_DIM,
    )
    value = mixed_qkv[:, value_start:].reshape(
        batch_size,
        _NUM_VALUE_HEADS,
        _VALUE_DIM,
    )
    gate_input = (a + dt_bias.unsqueeze(0)).float()
    g = -torch.exp(A_log.unsqueeze(0)) * torch.nn.functional.softplus(gate_input)
    # The packed kernel rounds sigmoid(beta) to b's element type before the
    # recurrent fp32 math; mirror that conversion in the CPU reference.
    beta = torch.sigmoid(b.float()).to(b.dtype).float()
    expected_output, expected_state = _golden_recurrent_gated_delta_rule(
        _l2norm_like_kernel(query[:num_active_tokens]),
        _l2norm_like_kernel(key[:num_active_tokens]),
        value[:num_active_tokens],
        g[:num_active_tokens],
        beta[:num_active_tokens],
        state_cpu,
        [1, 1],
        state_indices[:num_active_tokens],
        None,
    )

    out = torch.empty(
        batch_size,
        1,
        _NUM_VALUE_HEADS,
        _VALUE_DIM,
        dtype=torch.bfloat16,
        device="npu",
    )
    packed_output, _ = fused_recurrent_gated_delta_rule_packed_decode(
        mixed_qkv=mixed_qkv.npu(),
        a=a.npu(),
        b=b.npu(),
        A_log=A_log.npu(),
        dt_bias=dt_bias.npu(),
        scale=_KEY_DIM**-0.5,
        initial_state=state_npu,
        out=out,
        ssm_state_indices=state_indices.npu(),
        use_qk_l2norm_in_kernel=True,
    )
    torch_npu.npu.synchronize()

    assert state_npu.stride(0) == leading_stride
    assert state_npu.storage_offset() != 0
    _assert_numerically_close(
        packed_output[:num_active_tokens, 0],
        expected_output,
        name="packed output",
    )
    torch.testing.assert_close(
        packed_output[num_active_tokens],
        torch.zeros_like(packed_output[num_active_tokens]),
        rtol=0,
        atol=0,
    )
    _assert_state_and_guards(
        state_npu,
        expected_state,
        state_before,
        backing_npu,
        num_states,
        leading_stride,
        selected_state_ids={1, num_states - 1},
    )


@pytest.mark.skip_global_cleanup
@pytest.mark.parametrize("sequence_width", [1, 3, 16], ids=["single-token", "three-tokens", "max-mtp-width"])
@torch.inference_mode()
def test_generic_spec_decode_honors_padded_state_and_2d_state_table(sequence_width: int):
    """MTP/spec Decode must preserve its request-by-token state table."""
    torch.manual_seed(_SEED + 1)
    num_states = 2 * sequence_width + 1
    sequence_lengths = [sequence_width, sequence_width, 0]
    total_tokens = sum(sequence_lengths)
    state_cpu, state_npu, backing_npu, leading_stride = _make_padded_state(
        num_states,
        _NUM_VALUE_HEADS,
        _VALUE_DIM,
        _KEY_DIM,
    )
    state_before = state_cpu.clone()
    # The third request is graph padding: repeated cu_seqlens make it a
    # zero-token sequence and its state-table row must remain untouched.
    state_table = torch.tensor(
        [
            list(range(1, sequence_width + 1)),
            list(range(sequence_width + 1, num_states)),
            [0] * sequence_width,
        ],
        dtype=torch.int32,
    )
    accepted_tokens = torch.tensor([min(2, sequence_width), sequence_width, 1], dtype=torch.int32)
    cu_seqlens = torch.tensor([0, sequence_width, total_tokens, total_tokens], dtype=torch.int32)

    query = (torch.randn(1, total_tokens, _NUM_KEY_HEADS, _KEY_DIM) * 0.1).to(torch.bfloat16)
    key = (torch.randn(1, total_tokens, _NUM_KEY_HEADS, _KEY_DIM) * 0.1).to(torch.bfloat16)
    value = (torch.randn(1, total_tokens, _NUM_VALUE_HEADS, _VALUE_DIM) * 0.1).to(torch.bfloat16)
    g = -torch.rand(
        1,
        total_tokens,
        _NUM_VALUE_HEADS,
        dtype=torch.float32,
    )
    beta = torch.rand(
        1,
        total_tokens,
        _NUM_VALUE_HEADS,
        dtype=torch.float32,
    ).to(torch.bfloat16)

    expected_output, expected_state = _golden_recurrent_gated_delta_rule(
        _l2norm_like_kernel(query.squeeze(0)),
        _l2norm_like_kernel(key.squeeze(0)),
        value.squeeze(0),
        g.squeeze(0),
        beta.squeeze(0),
        state_cpu,
        sequence_lengths,
        state_table.flatten(),
        accepted_tokens,
    )

    generic_output, _ = fused_recurrent_gated_delta_rule(
        q=query.npu(),
        k=key.npu(),
        v=value.npu(),
        g=g.npu(),
        beta=beta.npu(),
        scale=_KEY_DIM**-0.5,
        initial_state=state_npu,
        inplace_final_state=True,
        cu_seqlens=cu_seqlens.npu(),
        ssm_state_indices=state_table.npu(),
        num_accepted_tokens=accepted_tokens.npu(),
        use_qk_l2norm_in_kernel=True,
    )
    torch_npu.npu.synchronize()

    assert state_npu.stride(0) == leading_stride
    assert state_npu.storage_offset() != 0
    _assert_numerically_close(
        generic_output.squeeze(0),
        expected_output,
        name="generic output",
    )
    _assert_state_and_guards(
        state_npu,
        expected_state,
        state_before,
        backing_npu,
        num_states,
        leading_stride,
        selected_state_ids=set(range(1, num_states)),
    )
