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
import logging
import os
from collections.abc import Callable
from enum import Enum
from typing import Any, Literal

import torch
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

logger = logging.getLogger(__name__)

COMPILER_MODE = os.getenv("FLA_COMPILER_MODE") == "1"
FLA_CI_ENV = os.getenv("FLA_CI_ENV") == "1"
SUPPRESS_LEVEL = int(os.getenv("GDN_RECOMPUTE_SUPPRESS_LEVEL", "0"))

# Default chunk size used across FLA triton kernels (kda, chunk, chunk_o, etc.)
FLA_CHUNK_SIZE = 64


def tensor_cache(fn: Callable[..., torch.Tensor]) -> Callable[..., torch.Tensor]:
    """
    Cache recent tensor-derived helper results by input object identity.
    """

    cache_entries: list[tuple[tuple[Any, ...], dict[str, Any], Any]] = []
    cache_size = 8

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        nonlocal cache_entries
        for i, entry in enumerate(cache_entries):
            last_args, last_kwargs, last_result = entry
            if (
                len(args) == len(last_args)
                and len(kwargs) == len(last_kwargs)
                and all(a is b for a, b in zip(args, last_args))
                and all(k in last_kwargs and v is last_kwargs[k] for k, v in kwargs.items())
            ):
                cache_entries = cache_entries[:i] + cache_entries[i + 1:] + [(args, kwargs, last_result)]
                return last_result

        result = fn(*args, **kwargs)
        if len(cache_entries) >= cache_size:
            cache_entries = cache_entries[1:]
        cache_entries.append((args, kwargs, result))
        return result

    return wrapper


@tensor_cache
def prepare_lens(cu_seqlens: torch.LongTensor) -> torch.LongTensor:
    return cu_seqlens[1:] - cu_seqlens[:-1]


@tensor_cache
def prepare_chunk_indices(cu_seqlens: torch.LongTensor, chunk_size: int) -> torch.LongTensor:
    indices = torch.cat([torch.arange(n) for n in triton.cdiv(prepare_lens(cu_seqlens), chunk_size).tolist()])
    return torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1).to(cu_seqlens)


def prepare_final_chunk_indices(cu_seqlens: torch.LongTensor, chunk_size: int) -> torch.LongTensor:
    indices = triton.cdiv(prepare_lens(cu_seqlens), chunk_size) + 1
    return torch.cumsum(indices, 0) - 1


@tensor_cache
def prepare_chunk_offsets(cu_seqlens: torch.LongTensor, chunk_size: int) -> torch.LongTensor:
    return torch.cat([cu_seqlens.new_tensor([0]), triton.cdiv(prepare_lens(cu_seqlens), chunk_size)]).cumsum(-1)


def prepare_update_chunk_offsets(cu_seqlens: torch.LongTensor, chunk_size: int) -> torch.LongTensor:
    return torch.cat([cu_seqlens.new_tensor([0]), triton.cdiv(prepare_lens(cu_seqlens), chunk_size) + 1]).cumsum(-1)


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


@triton.jit(do_not_specialize=["inner_size", "row_stride"])
def _clear_ssm_states_kernel(
    states_ptr,
    has_initial_state_ptr,
    inner_size,
    row_stride,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(axis=0)
    col_block_idx = tl.program_id(axis=1)

    has_state = tl.load(has_initial_state_ptr + row_idx).to(tl.int1)
    if has_state:
        return

    cols = col_block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = cols < inner_size
    row_ptr = states_ptr + row_idx * row_stride + cols
    tl.store(row_ptr, tl.zeros((BLOCK_SIZE,), dtype=states_ptr.dtype.element_ty), mask=mask)


def clear_ssm_states(ssm_states: torch.Tensor, has_initial_state: torch.Tensor) -> None:
    """Zero out specific rows for the SSM states."""
    if ssm_states.numel() == 0:
        return

    if has_initial_state.device != ssm_states.device:
        has_initial_state = has_initial_state.to(ssm_states.device, non_blocking=True)
    if has_initial_state.dtype != torch.bool:
        has_initial_state = has_initial_state.to(torch.bool)

    has_initial_state = has_initial_state.reshape(-1).contiguous()
    num_rows = ssm_states.shape[0]
    if num_rows == 0:
        return
    if has_initial_state.numel() != num_rows:
        raise ValueError(f"has_initial_state size mismatch: expected {num_rows}, got {has_initial_state.numel()}")

    inner_size = ssm_states.numel() // num_rows
    if inner_size == 0:
        return

    block_size = 4096
    grid = (num_rows, triton.cdiv(inner_size, block_size))
    _clear_ssm_states_kernel[grid](
        ssm_states,
        has_initial_state,
        inner_size,
        ssm_states.stride(0),
        BLOCK_SIZE=block_size,
    )


@functools.cache
def get_available_device() -> str:
    try:
        return triton.runtime.driver.active.get_current_target().backend
    except (RuntimeError, AttributeError):
        return "cpu"


@functools.cache
def _check_platform() -> Literal["nvidia", "amd", "intel", "musa"]:
    device_name = get_available_device()
    mapping = {
        "cuda": "nvidia",
        "hip": "amd",
        "xpu": "intel",
    }
    return mapping.get(device_name, device_name)


# For AMD GPUs, the triton backend is 'hip', while the torch backend is
# 'cuda'. Use the triton backend to identify the actual vendor.
device = "cuda" if current_platform.is_cuda_alike() else get_available_device()
device_torch_lib = getattr(torch, device, None)
device_platform = _check_platform()

is_amd = device_platform == "amd"
is_intel = device_platform == "intel"
is_nvidia = device_platform == "nvidia"
is_intel_alchemist = is_intel and "Intel(R) Arc(TM) A" in torch.xpu.get_device_name(0)
is_nvidia_hopper = is_nvidia and (
    "NVIDIA H" in torch.cuda.get_device_name(0) or torch.cuda.get_device_capability()[0] >= 9
)
use_cuda_graph = is_nvidia and os.environ.get("FLA_USE_CUDA_GRAPH", "0") == "1"
is_gather_supported = hasattr(triton.language, "gather")
is_tma_supported = (
    is_nvidia_hopper
    and os.getenv("FLA_USE_TMA", "0") == "1"
    and (
        hasattr(triton.language, "_experimental_make_tensor_descriptor")
        or hasattr(triton.language, "make_tensor_descriptor")
    )
)


def get_all_max_shared_mem():
    try:
        return [
            triton.runtime.driver.active.utils.get_device_properties(i)["max_shared_mem"]
            for i in range(device_torch_lib.device_count())
        ]
    except BaseException:
        return [-1]


class Backend(Enum):
    ADA = 101376  # RTX 4090
    AMPERE = 166912  # A100
    HOPPER = 232448  # H100
    DEFAULT = 102400

    @classmethod
    def get_shared_memory(cls, arch: str) -> int:
        try:
            return cls[arch.upper()].value
        except KeyError:
            return cls.DEFAULT.value


@functools.cache
def check_shared_mem(arch: str = "none", tensor_idx: int = 0) -> bool:
    try:
        device_shared_mem_list = get_all_max_shared_mem()
        max_shared_memory = device_shared_mem_list[tensor_idx]
        return max_shared_memory >= Backend.get_shared_memory(arch)
    except Exception:
        return False
