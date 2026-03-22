# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang
#
# This file contains code copied from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# ruff: noqa: E501
import contextlib
import functools
from collections.abc import Callable

import torch
from vllm.triton_utils import tl, triton


def prepare_lens(cu_seqlens: torch.LongTensor) -> torch.LongTensor:
    return cu_seqlens[1:] - cu_seqlens[:-1]


def prepare_chunk_indices(cu_seqlens: torch.LongTensor, chunk_size: int) -> torch.LongTensor:
    cu = cu_seqlens.tolist()
    seq_ids = []
    block_ids = []
    for i in range(len(cu) - 1):
        n = (cu[i + 1] - cu[i] + chunk_size - 1) // chunk_size
        for j in range(n):
            seq_ids.append(i)
            block_ids.append(j)
    if not seq_ids:
        return cu_seqlens.new_empty(0, 2)
    result = [[seq_ids[i], block_ids[i]] for i in range(len(seq_ids))]
    return torch.tensor(result, dtype=cu_seqlens.dtype, device=cu_seqlens.device)


def prepare_chunk_offsets(cu_seqlens: torch.LongTensor, chunk_size: int) -> torch.LongTensor:
    cu = cu_seqlens.tolist()
    offsets = [0]
    for i in range(len(cu) - 1):
        n = (cu[i + 1] - cu[i] + chunk_size - 1) // chunk_size
        offsets.append(offsets[-1] + n)
    return torch.tensor(offsets, dtype=cu_seqlens.dtype, device=cu_seqlens.device)


def input_guard(fn: Callable[..., torch.Tensor]) -> Callable[..., torch.Tensor]:
    """
    A decorator to make sure all input tensors are contiguous and set the device based on input tensors.
    """

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        contiguous_args = (i if not isinstance(i, torch.Tensor) else i.contiguous() for i in args)
        contiguous_kwargs = {k: (v if not isinstance(v, torch.Tensor) else v.contiguous()) for k, v in kwargs.items()}

        tensor = None
        for arg in args:
            if isinstance(arg, torch.Tensor):
                tensor = arg
                break
        if tensor is None:
            for value in kwargs.values():
                if isinstance(value, torch.Tensor):
                    tensor = value
                    break

        if tensor is not None:
            ctx = torch.npu.device(tensor.device.index)
        else:
            ctx = contextlib.nullcontext()

        with ctx:
            return fn(*contiguous_args, **contiguous_kwargs)

    return wrapper


@triton.jit
def safe_exp(x):
    return tl.exp(tl.where(x <= 0, x, float("-inf")))