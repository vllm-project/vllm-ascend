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

"""Regression tests for recurrent GDN state views with a padded stride(0).

The typed KV-cache allocator places the recurrent state behind another cache
payload in each physical page.  Consequently, adjacent logical GDN states are
separated by a leading stride larger than ``Nv * Dv * Dk``.  These tests keep a
canary in that gap so an implementation using a dense state-size multiplier is
guaranteed to be detected, even if its output happens to be numerically close.
"""

from collections.abc import Sequence

import pytest
import torch
import torch.nn.functional as F
import torch_npu

_CANARY = -12345.0
_PAGE_PADDING_ELEMENTS = 15_360
_STORAGE_OFFSET_ELEMENTS = 256
_TAIL_CANARY_ELEMENTS = 128
_SEED = 42


def _make_padded_state(
    num_states: int,
    num_value_heads: int,
    value_dim: int,
    key_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Build matching CPU/NPU state views over canary-filled storage."""
    dense_state_elements = num_value_heads * value_dim * key_dim
    leading_stride = dense_state_elements + _PAGE_PADDING_ELEMENTS
    backing_cpu = torch.full(
        (_STORAGE_OFFSET_ELEMENTS + num_states * leading_stride + _TAIL_CANARY_ELEMENTS,),
        _CANARY,
        dtype=torch.float32,
    )
    state_shape = (num_states, num_value_heads, value_dim, key_dim)
    state_stride = (
        leading_stride,
        value_dim * key_dim,
        key_dim,
        1,
    )
    state_cpu = torch.as_strided(
        backing_cpu,
        state_shape,
        state_stride,
        storage_offset=_STORAGE_OFFSET_ELEMENTS,
    )

    generator = torch.Generator().manual_seed(_SEED)
    state_cpu.copy_(torch.randn(state_shape, generator=generator, dtype=torch.float32) * 0.01)

    backing_npu = backing_cpu.npu()
    state_npu = torch.as_strided(
        backing_npu,
        state_shape,
        state_stride,
        storage_offset=_STORAGE_OFFSET_ELEMENTS,
    )
    assert state_npu.stride(0) == leading_stride
    assert state_npu.storage_offset() == _STORAGE_OFFSET_ELEMENTS
    assert not state_npu.is_contiguous()
    return state_cpu, state_npu, backing_npu, leading_stride


def _golden_recurrent_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    state: torch.Tensor,
    actual_seq_lengths: Sequence[int],
    state_indices: torch.Tensor,
    num_accepted_tokens: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Small CPU reference matching the custom operator's state-index rules."""
    query_f32 = query.float()
    key_f32 = key.float()
    value_f32 = value.float()
    g_f32 = g.float().exp()
    beta_f32 = beta.float()
    updated_state = state.clone()
    output = torch.empty_like(value_f32)

    num_value_heads = value.shape[1]
    num_key_heads = key.shape[1]
    value_heads_per_key_head = num_value_heads // num_key_heads
    scale = key.shape[-1] ** -0.5
    seq_start = 0

    for seq_id, seq_length in enumerate(actual_seq_lengths):
        initial_token_offset = 0
        if num_accepted_tokens is not None:
            initial_token_offset = int(num_accepted_tokens[seq_id]) - 1
        initial_state_id = int(state_indices[seq_start + initial_token_offset])
        recurrent_state = updated_state[initial_state_id].float()

        for token_id in range(seq_start, seq_start + seq_length):
            for value_head in range(num_value_heads):
                key_head = value_head // value_heads_per_key_head
                q_t = query_f32[token_id, key_head] * scale
                k_t = key_f32[token_id, key_head]
                v_t = value_f32[token_id, value_head]

                head_state = recurrent_state[value_head] * g_f32[token_id, value_head]
                state_projection = torch.sum(head_state * k_t.unsqueeze(0), dim=-1)
                delta = (v_t - state_projection) * beta_f32[token_id, value_head]
                head_state = head_state + delta.unsqueeze(-1) * k_t.unsqueeze(0)
                recurrent_state[value_head] = head_state
                output[token_id, value_head] = torch.sum(head_state * q_t.unsqueeze(0), dim=-1)

            updated_state[int(state_indices[token_id])] = recurrent_state
        seq_start += seq_length

    return output.to(value.dtype), updated_state


def _assert_padding_canary(
    backing: torch.Tensor,
    num_states: int,
    dense_state_elements: int,
    leading_stride: int,
) -> None:
    backing_cpu = backing.cpu()

    def assert_canary(start: int, end: int) -> None:
        torch.testing.assert_close(
            backing_cpu[start:end],
            torch.full(
                (end - start,),
                _CANARY,
                dtype=backing_cpu.dtype,
            ),
            rtol=0,
            atol=0,
        )

    # A non-zero storage offset catches kernels that use the base storage
    # pointer rather than the Tensor's logical data pointer.
    assert_canary(0, _STORAGE_OFFSET_ELEMENTS)
    for state_id in range(num_states):
        padding_start = _STORAGE_OFFSET_ELEMENTS + state_id * leading_stride + dense_state_elements
        padding_end = _STORAGE_OFFSET_ELEMENTS + (state_id + 1) * leading_stride
        assert_canary(padding_start, padding_end)

    # Keep a suffix guard as well so an oversized last-page write cannot pass.
    tail_start = _STORAGE_OFFSET_ELEMENTS + num_states * leading_stride
    assert_canary(tail_start, tail_start + _TAIL_CANARY_ELEMENTS)


@pytest.mark.parametrize(
    ("actual_seq_lengths", "state_indices", "accepted_tokens", "num_states"),
    [
        pytest.param([1, 1], [1, 4], None, 5, id="decode-nonzero-and-last-page"),
        pytest.param(
            [3, 3],
            [1, 2, 3, 4, 5, 6],
            [2, 3],
            7,
            id="mtp-spec-state-table",
        ),
    ],
)
def test_recurrent_gated_delta_rule_honors_padded_leading_stride(
    actual_seq_lengths: list[int],
    state_indices: list[int],
    accepted_tokens: list[int] | None,
    num_states: int,
):
    torch.manual_seed(_SEED)
    num_key_heads = 4
    num_value_heads = 8
    key_dim = 128
    value_dim = 128
    total_tokens = sum(actual_seq_lengths)

    state_cpu, state_npu, backing_npu, leading_stride = _make_padded_state(
        num_states,
        num_value_heads,
        value_dim,
        key_dim,
    )
    state_before = state_cpu.clone()

    query = F.normalize(
        torch.randn(total_tokens, num_key_heads, key_dim),
        p=2,
        dim=-1,
    ).to(torch.bfloat16)
    key = F.normalize(
        torch.randn(total_tokens, num_key_heads, key_dim),
        p=2,
        dim=-1,
    ).to(torch.bfloat16)
    value = (torch.randn(total_tokens, num_value_heads, value_dim) * 0.1).to(torch.bfloat16)
    g = -torch.rand(total_tokens, num_value_heads, dtype=torch.float32)
    beta = torch.rand(total_tokens, num_value_heads).to(torch.bfloat16)
    state_indices_cpu = torch.tensor(state_indices, dtype=torch.int32)
    accepted_tokens_cpu = None if accepted_tokens is None else torch.tensor(accepted_tokens, dtype=torch.int32)

    expected_output, expected_state = _golden_recurrent_gated_delta_rule(
        query,
        key,
        value,
        g,
        beta,
        state_cpu,
        actual_seq_lengths,
        state_indices_cpu,
        accepted_tokens_cpu,
    )
    op_actual_seq_lengths = torch.tensor(
        [0, *actual_seq_lengths],
        dtype=torch.int32,
        device="npu",
    )

    output_npu = torch.ops._C_ascend.npu_recurrent_gated_delta_rule(
        query=query.npu(),
        key=key.npu(),
        value=value.npu(),
        g=g.npu(),
        beta=beta.npu(),
        state=state_npu,
        scale=key_dim**-0.5,
        actual_seq_lengths=op_actual_seq_lengths,
        ssm_state_indices=state_indices_cpu.npu(),
        num_accepted_tokens=(None if accepted_tokens_cpu is None else accepted_tokens_cpu.npu()),
    )
    torch_npu.npu.synchronize()

    torch.testing.assert_close(
        output_npu.float().cpu(),
        expected_output.float(),
        rtol=3e-3,
        atol=1e-2,
        equal_nan=True,
    )
    torch.testing.assert_close(
        state_npu.cpu(),
        expected_state,
        rtol=3e-3,
        atol=1e-2,
        equal_nan=True,
    )

    selected_state_ids = set(state_indices)
    for state_id in range(num_states):
        if state_id not in selected_state_ids:
            torch.testing.assert_close(
                state_npu[state_id].cpu(),
                state_before[state_id],
                rtol=0,
                atol=0,
            )

    _assert_padding_canary(
        backing_npu,
        num_states,
        num_value_heads * value_dim * key_dim,
        leading_stride,
    )
