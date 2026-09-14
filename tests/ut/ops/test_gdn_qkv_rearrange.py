# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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

import torch

from vllm_ascend.ops.gdn import try_rearrange_single_token_mixed_qkv


def test_single_token_qkv_rearrange_uses_input_storage() -> None:
    q_dim, k_dim, v_dim = 8, 8, 16
    mixed_qkv = torch.arange(q_dim + k_dim + v_dim).view(1, -1)

    result = try_rearrange_single_token_mixed_qkv(
        mixed_qkv,
        q_dim=q_dim,
        k_dim=k_dim,
        v_dim=v_dim,
        head_k_dim=4,
        head_v_dim=8,
    )

    assert result is not None
    query, key, value = result
    expected_query, expected_key, expected_value = mixed_qkv.split([q_dim, k_dim, v_dim], dim=-1)
    torch.testing.assert_close(query.flatten(), expected_query.flatten())
    torch.testing.assert_close(key.flatten(), expected_key.flatten())
    torch.testing.assert_close(value.flatten(), expected_value.flatten())
    assert query.shape == (1, 1, 2, 4)
    assert key.shape == (1, 1, 2, 4)
    assert value.shape == (1, 1, 2, 8)

    input_storage = mixed_qkv.untyped_storage().data_ptr()
    assert query.untyped_storage().data_ptr() == input_storage
    assert key.untyped_storage().data_ptr() == input_storage
    assert value.untyped_storage().data_ptr() == input_storage


def test_multi_token_qkv_rearrange_uses_original_path() -> None:
    mixed_qkv = torch.arange(64).view(2, 32)

    result = try_rearrange_single_token_mixed_qkv(
        mixed_qkv,
        q_dim=8,
        k_dim=8,
        v_dim=16,
        head_k_dim=4,
        head_v_dim=8,
    )

    assert result is None


def test_non_contiguous_qkv_rearrange_uses_original_path() -> None:
    mixed_qkv = torch.arange(64).view(1, 64)[:, ::2]
    assert not mixed_qkv.is_contiguous()

    result = try_rearrange_single_token_mixed_qkv(
        mixed_qkv,
        q_dim=8,
        k_dim=8,
        v_dim=16,
        head_k_dim=4,
        head_v_dim=8,
    )

    assert result is None


def test_invalid_qkv_dimensions_fall_back() -> None:
    mixed_qkv = torch.arange(32).view(1, 32)

    result = try_rearrange_single_token_mixed_qkv(
        mixed_qkv,
        q_dim=6,
        k_dim=8,
        v_dim=18,
        head_k_dim=4,
        head_v_dim=8,
    )

    assert result is None
