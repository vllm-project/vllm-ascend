#
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
#

import inspect

import torch

import vllm_ascend.ops.triton.mamba.causal_conv1d as causal_conv

causal_conv1d_fn = causal_conv.causal_conv1d_fn


def test_causal_conv1d_fn_accepts_vllm_block_cache_kwargs():
    signature = inspect.signature(causal_conv1d_fn)

    signature.bind(
        torch.empty(4, 8),
        torch.empty(4, 3),
        None,
        activation="silu",
        conv_states=torch.empty(1, 4, 2),
        has_initial_state=torch.empty(1, dtype=torch.bool),
        cache_indices=torch.empty(1, dtype=torch.int32),
        query_start_loc=torch.tensor([0, 8], dtype=torch.int32),
        block_idx_first_scheduled_token=torch.empty(1, dtype=torch.int32),
        block_idx_last_scheduled_token=torch.empty(1, dtype=torch.int32),
        initial_state_idx=torch.empty(1, dtype=torch.int32),
        num_computed_tokens=torch.empty(1, dtype=torch.int32),
        block_size_to_align=8,
        metadata=None,
    )


def test_causal_conv1d_fn_handles_decode_only_batch(monkeypatch):
    class _FakePCPGroup:
        world_size = 1

    monkeypatch.setattr(causal_conv, "get_pcp_group", _FakePCPGroup)

    x = torch.tensor([[2.0, 3.0], [5.0, 7.0]])
    weight = torch.tensor([[0.5, 1.0, 1.5], [1.0, -0.5, 0.25]])
    bias = torch.tensor([0.25, -1.0])
    conv_states = torch.tensor(
        [
            [[10.0, 11.0], [12.0, 13.0]],
            [[20.0, 21.0], [22.0, 23.0]],
        ]
    )
    initial_states = conv_states.clone()

    output = causal_conv1d_fn(
        x,
        weight,
        bias,
        activation=None,
        conv_states=conv_states,
        cache_indices=torch.tensor([0, 1], dtype=torch.int32),
        query_start_loc=torch.tensor([0, 1, 2], dtype=torch.int32),
    )

    expected = torch.empty_like(x)
    for batch_index in range(2):
        window = torch.cat(
            [initial_states[batch_index], x[:, batch_index : batch_index + 1]],
            dim=-1,
        )
        expected[:, batch_index] = (window * weight).sum(dim=-1) + bias

    torch.testing.assert_close(output, expected)
    torch.testing.assert_close(
        conv_states[:, :, 0],
        initial_states[:, :, 1],
    )
    torch.testing.assert_close(conv_states[0, :, 1], x[:, 0])
    torch.testing.assert_close(conv_states[1, :, 1], x[:, 1])


def test_causal_conv1d_fn_ignores_stale_state_without_initial_state(monkeypatch):
    class _FakePCPGroup:
        world_size = 1

    monkeypatch.setattr(causal_conv, "get_pcp_group", _FakePCPGroup)

    x = torch.tensor([[2.0, 3.0], [5.0, 7.0]])
    weight = torch.tensor([[0.5, 1.0, 1.5], [1.0, -0.5, 0.25]])
    bias = torch.tensor([0.25, -1.0])
    conv_states = torch.full((1, 2, 2), 1000.0)

    output = causal_conv1d_fn(
        x,
        weight,
        bias,
        activation=None,
        conv_states=conv_states,
        has_initial_state=torch.tensor([False]),
        cache_indices=torch.tensor([0], dtype=torch.int32),
        query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
    )

    expected, expected_state = causal_conv.causal_conv1d_ref(
        x.unsqueeze(0),
        weight,
        bias,
        activation=None,
        return_final_states=True,
    )
    torch.testing.assert_close(output, expected.squeeze(0))
    torch.testing.assert_close(conv_states, expected_state)


def test_causal_conv1d_fn_handles_all_padding(monkeypatch):
    class _FakePCPGroup:
        world_size = 1

    monkeypatch.setattr(causal_conv, "get_pcp_group", _FakePCPGroup)

    x = torch.empty(2, 0)
    output = causal_conv1d_fn(
        x,
        torch.empty(2, 3),
        conv_states=torch.empty(1, 2, 2),
        cache_indices=torch.tensor([causal_conv.PAD_SLOT_ID], dtype=torch.int32),
        query_start_loc=torch.tensor([0, 0], dtype=torch.int32),
    )

    assert output.shape == x.shape
    assert output.dtype == x.dtype
    assert output.device == x.device


def test_causal_conv1d_fn_delegates_block_cache_to_vllm(monkeypatch):
    class _FakePCPGroup:
        world_size = 1

    sentinel = torch.tensor([123.0])
    captured_kwargs = {}

    def upstream_causal_conv1d_fn(**kwargs):
        captured_kwargs.update(kwargs)
        return sentinel

    monkeypatch.setattr(causal_conv, "get_pcp_group", _FakePCPGroup)
    monkeypatch.setattr(
        causal_conv,
        "_ORIGINAL_CAUSAL_CONV1D_FN",
        upstream_causal_conv1d_fn,
    )

    output = causal_conv1d_fn(
        torch.empty(2, 1),
        torch.empty(2, 3),
        None,
        conv_states=torch.empty(2, 2, 2),
        has_initial_state=torch.tensor([True]),
        cache_indices=torch.tensor([[0, 1]], dtype=torch.int32),
        query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
        block_size_to_align=8,
    )

    assert output is sentinel
    assert captured_kwargs["cache_indices"].shape == (1, 2)
    assert captured_kwargs["block_size_to_align"] == 8
