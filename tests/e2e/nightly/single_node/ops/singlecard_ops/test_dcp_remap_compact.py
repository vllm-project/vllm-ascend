# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
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

import itertools

import pytest
import torch
import torch_npu  # noqa: F401

pytestmark = pytest.mark.skipif(
    not hasattr(torch.ops._C_ascend, "dcp_remap_compact"),
    reason="dcp_remap_compact direct kernel is unavailable on this platform",
)


def reference(indices: torch.Tensor, rank: int, size: int, interleave: int) -> torch.Tensor:
    rows = indices.reshape(-1, indices.shape[-1])
    output = torch.full_like(rows, -1)
    for row_id, row in enumerate(rows):
        valid = []
        for value in row.tolist():
            if value < 0:
                continue
            block = value // interleave
            if block % size == rank:
                valid.append((block // size) * interleave + value % interleave)
        if valid:
            output[row_id, : len(valid)] = torch.tensor(valid, dtype=indices.dtype)
    return output.reshape_as(indices)


def make_case(rows: int, width: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    values = torch.randint(0, 1024 * 1024, (rows, 1, width), generator=generator, dtype=torch.int32)
    holes = torch.rand((rows, 1, width), generator=generator) < 0.08
    return torch.where(holes, torch.full_like(values, -1), values)


@pytest.mark.parametrize("rows,width", [(1, 1), (3, 17), (32, 2048), (512, 2048)])
@pytest.mark.parametrize("size,interleave", [(1, 1), (2, 128), (16, 128)])
def test_random(rows: int, width: int, size: int, interleave: int) -> None:
    cpu = make_case(rows, width, rows * 10000 + width + size)
    npu = cpu.npu()
    for rank in sorted({0, size - 1}):
        actual = torch.ops._C_ascend.dcp_remap_compact(npu, rank, size, interleave).cpu()
        expected = reference(cpu, rank, size, interleave)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("mask", itertools.product([False, True], repeat=6))
def test_all_small_masks(mask: tuple[bool, ...]) -> None:
    values = torch.tensor([0, 128, 256, 384, 512, 640], dtype=torch.int32)
    cpu = torch.where(torch.tensor(mask), values, torch.full_like(values, -1)).reshape(1, 1, -1)
    actual = torch.ops._C_ascend.dcp_remap_compact(cpu.npu(), 1, 2, 128).cpu()
    expected = reference(cpu, 1, 2, 128)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
