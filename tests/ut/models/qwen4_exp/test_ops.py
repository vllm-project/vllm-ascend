# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch

from vllm_ascend.models.qwen4_exp.ops import (
    grouped_gemma_rmsnorm,
    hc_combine,
    hc_gate_mix,
    qsa_compress_groups_with_ratio,
    qsa_select_paged_tokens,
    qsa_sparse_paged_attention,
    qsa_store_cache_rows,
)


def test_hyperconnection_torch_fallbacks() -> None:
    hidden_states = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    weight = torch.zeros(2)
    normalized = grouped_gemma_rmsnorm(hidden_states, weight, 1e-6, 2)
    expected = hidden_states.unflatten(-1, (2, 2))
    expected = expected / torch.sqrt(expected.square().mean(-1, keepdim=True) + 1e-6)
    torch.testing.assert_close(normalized, expected.flatten(-2))

    gate = torch.zeros_like(hidden_states)
    mixed = hc_gate_mix(hidden_states, gate, 2)
    torch.testing.assert_close(mixed, hidden_states.unflatten(-1, (2, 2)).mean(-2) / 2)

    block_output = torch.tensor([[2.0, 4.0]])
    injection = torch.zeros((1, 2))
    combined = hc_combine(hidden_states, block_output, injection, 2)
    torch.testing.assert_close(
        combined,
        (hidden_states.unflatten(-1, (2, 2)) + block_output.unsqueeze(-2)).flatten(-2),
    )


def test_qsa_store_cache_rows_ignores_padding_slots() -> None:
    cache = torch.zeros((2, 2, 1, 2))
    rows = torch.tensor([[[1.0, 2.0]], [[3.0, 4.0]], [[5.0, 6.0]]])
    qsa_store_cache_rows(cache, torch.tensor([2, -1, 0]), rows)
    torch.testing.assert_close(cache.reshape(-1, 2)[0], rows[2, 0])
    torch.testing.assert_close(cache.reshape(-1, 2)[2], rows[0, 0])
    assert torch.count_nonzero(cache).item() == 4


def test_qsa_store_cache_rows_updates_non_contiguous_layout() -> None:
    storage = torch.zeros((2, 2, 2, 3))
    cache = storage.transpose(1, 2)
    assert not cache.is_contiguous()
    rows = torch.tensor(
        [
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
        ]
    )
    qsa_store_cache_rows(cache, torch.tensor([1, 2]), rows)
    torch.testing.assert_close(cache[0, 1], rows[0])
    torch.testing.assert_close(cache[1, 0], rows[1])
    torch.testing.assert_close(storage[0, :, 1], rows[0])
    torch.testing.assert_close(storage[1, :, 0], rows[1])


def test_qsa_compresses_current_group() -> None:
    raw_keys = torch.arange(8, dtype=torch.float32).reshape(4, 1, 2)
    positions = torch.arange(4).reshape(4, 1, 1).expand(-1, 1, 3)
    state_cache = torch.zeros((1, 4, 1, 2))
    pooled, first_positions = qsa_compress_groups_with_ratio(
        raw_keys,
        positions,
        state_cache,
        torch.tensor([[0]]),
        torch.zeros(4, dtype=torch.int32),
        torch.tensor([0, 4]),
        torch.arange(4),
        torch.tensor([-1, -1, -1, 0]),
        4,
        None,
    )
    torch.testing.assert_close(pooled[3], raw_keys.float().mean(0))
    torch.testing.assert_close(first_positions[3], torch.zeros(3, dtype=torch.int64))
    assert torch.count_nonzero(pooled[:3]).item() == 0


def test_qsa_selection_and_sparse_attention() -> None:
    compressed_cache = torch.zeros((1, 4, 1, 2))
    compressed_cache[0, :, 0] = torch.tensor([[1.0, 0.0], [0.0, 1.0], [2.0, 0.0], [0.0, 2.0]])
    query = torch.tensor([[[1.0, 0.0]]])
    selected = qsa_select_paged_tokens(
        query,
        compressed_cache,
        torch.tensor([[0]]),
        torch.tensor([0]),
        torch.tensor([15]),
        torch.tensor([16]),
        token_topk=4,
        compress_ratio=2,
    )
    assert selected.shape == (1, 5)
    assert set(selected[0, :4].tolist()) == {0, 1, 4, 5}

    key_cache = torch.zeros((1, 8, 1, 2))
    value_cache = torch.zeros_like(key_cache)
    key_cache[0, 0, 0] = torch.tensor([1.0, 0.0])
    key_cache[0, 1, 0] = torch.tensor([0.0, 1.0])
    value_cache[0, 0, 0] = torch.tensor([2.0, 0.0])
    value_cache[0, 1, 0] = torch.tensor([0.0, 4.0])
    output = qsa_sparse_paged_attention(
        query,
        key_cache,
        value_cache,
        torch.tensor([[0, 1]], dtype=torch.int32),
        torch.tensor([[0]]),
        torch.tensor([0]),
    )
    probabilities = torch.softmax(torch.tensor([1.0, 0.0]) / 2**0.5, dim=0)
    expected = probabilities[0] * value_cache[0, 0, 0] + probabilities[1] * value_cache[0, 1, 0]
    torch.testing.assert_close(output[0, 0], expected)
