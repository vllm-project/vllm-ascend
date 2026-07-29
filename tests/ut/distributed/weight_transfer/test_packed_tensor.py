# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm_ascend.distributed.weight_transfer.packed_tensor import (
    dtype_name,
    packed_tensor_size,
    tensor_nbytes,
    unpack_packed_tensor,
)


def test_dtype_name_returns_torch_dtype_suffix():
    assert dtype_name(torch.bfloat16) == "bfloat16"
    assert dtype_name(torch.float32) == "float32"


def test_tensor_size_helpers():
    tensor = torch.zeros((2, 3), dtype=torch.float32)

    assert tensor_nbytes(tensor) == 24
    assert packed_tensor_size([2, 3], torch.float32) == 24


def test_unpack_packed_tensor_views_original_storage_by_default():
    first = torch.tensor([1, 2], dtype=torch.float32).view(torch.uint8)
    second = torch.tensor([3], dtype=torch.float32).view(torch.uint8)
    packed = torch.cat([first, second])

    weights = unpack_packed_tensor(
        packed,
        ["first", "second"],
        [[2], [1]],
        [torch.float32, torch.float32],
        [first.numel(), second.numel()],
    )

    assert weights[0][0] == "first"
    assert weights[0][1].tolist() == [1, 2]
    assert weights[1][0] == "second"
    assert weights[1][1].tolist() == [3]


def test_unpack_packed_tensor_can_clone_storage():
    raw = torch.tensor([1, 2], dtype=torch.float32).view(torch.uint8)
    packed = raw.clone()

    weight = unpack_packed_tensor(
        packed,
        ["weight"],
        [[2]],
        [torch.float32],
        [raw.numel()],
        clone=True,
    )[0][1]

    packed.zero_()
    assert weight.tolist() == [1, 2]
