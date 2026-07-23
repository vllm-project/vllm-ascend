# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# All Rights Reserved.
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
# This file is a part of the vllm-ascend project.

import pytest
import torch
import torch_npu  # noqa: F401

from tests.ut.conftest import RunnerDeviceType, npu_test
from vllm_ascend.ops.triton.rope import rope_forward_triton
from vllm_ascend.ops.triton.triton_utils import (
    init_device_properties_triton,
)

NUM_TOKENS = 2
NUM_HEADS = 64
MAX_POSITION = 128
POSITIONS = (89, 90)


def _make_input(
    num_tokens: int,
    num_heads: int,
    head_dim: int,
    offset: int,
) -> torch.Tensor:
    numel = num_tokens * num_heads * head_dim
    values = torch.arange(numel, dtype=torch.float32)
    values = ((values + offset) % 251 - 125) / 32
    return values.reshape(num_tokens, num_heads, head_dim).to(
        torch.bfloat16
    )


def _make_cos_sin_cache(head_dim: int) -> torch.Tensor:
    half_dim = head_dim // 2
    angles = torch.arange(
        MAX_POSITION * half_dim,
        dtype=torch.float32,
    ).reshape(MAX_POSITION, half_dim)
    angles = angles / 10000.0

    return torch.cat(
        (torch.cos(angles), torch.sin(angles)),
        dim=-1,
    ).to(torch.bfloat16)


def _reference_neox(
    value: torch.Tensor,
    selected_cache: torch.Tensor,
) -> torch.Tensor:
    half_dim = value.shape[-1] // 2

    cos = selected_cache[:, :half_dim].float().unsqueeze(1)
    sin = selected_cache[:, half_dim:].float().unsqueeze(1)

    first = value[..., :half_dim].float()
    second = value[..., half_dim:].float()

    output = torch.cat(
        (
            first * cos - second * sin,
            second * cos + first * sin,
        ),
        dim=-1,
    )
    return output.to(value.dtype)


@pytest.mark.parametrize("head_dim", [128, 192])
@npu_test(num_npus=1, npu_type=RunnerDeviceType.A3)
@torch.inference_mode()
def test_rope_forward_triton_neox_large_head_dim(
    head_dim: int,
) -> None:
    torch.npu.set_device(0)
    init_device_properties_triton()

    positions_cpu = torch.tensor(POSITIONS, dtype=torch.long)
    cache_cpu = _make_cos_sin_cache(head_dim)

    query_cpu = _make_input(
        NUM_TOKENS,
        NUM_HEADS,
        head_dim,
        offset=0,
    )
    key_cpu = _make_input(
        NUM_TOKENS,
        NUM_HEADS,
        head_dim,
        offset=37,
    )

    selected_cache = cache_cpu.index_select(0, positions_cpu)
    query_reference = _reference_neox(query_cpu, selected_cache)
    key_reference = _reference_neox(key_cpu, selected_cache)

    device = torch.device("npu:0")

    query = query_cpu.to(device)
    key = key_cpu.to(device)
    cache = cache_cpu.to(device)
    positions = positions_cpu.to(device)

    query_output, key_output = rope_forward_triton(
        query,
        key,
        cos_sin_cache=cache,
        positions=positions,
        rope_dim=head_dim,
        is_neox_style=True,
    )

    torch.npu.synchronize()

    assert query_output.shape == query_cpu.shape
    assert key_output.shape == key_cpu.shape
    assert query_output.dtype == torch.bfloat16
    assert key_output.dtype == torch.bfloat16

    torch.testing.assert_close(
        query_output.cpu(),
        query_reference,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        key_output.cpu(),
        key_reference,
        rtol=0,
        atol=0,
    )