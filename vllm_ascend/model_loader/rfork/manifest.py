# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Parse and validate weight metadata without changing tensor storage or layout."""

from collections.abc import Mapping
from typing import Any

import torch
from vllm.logger import logger

from vllm_ascend.model_loader.rfork.types import SeedTransferInfo


def normalize_dtype_name(dtype: Any) -> str | None:
    if dtype is None:
        return None
    if isinstance(dtype, torch.dtype):
        name = str(dtype)
    elif isinstance(dtype, str):
        name = dtype.strip()
    else:
        return None
    if name.startswith("torch."):
        name = name[6:]
    return name.lower() if name else None


def read_npu_format(tensor: torch.Tensor) -> int | None:
    """Return the NPU storage format when it is available."""
    if tensor.device.type != "npu":
        return None
    try:
        import torch_npu
    except Exception:  # pragma: no cover - CPU-only inspection and tests
        return None
    try:
        return int(torch_npu.get_npu_format(tensor))
    except Exception:
        return None


def is_positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def normalize_weight_shape(shape: Any) -> tuple[int, ...] | None:
    if not isinstance(shape, (list, tuple)):
        return None
    if not all(isinstance(dim, int) and not isinstance(dim, bool) and dim >= 0 for dim in shape):
        return None
    return tuple(shape)


def numel_from_shape(shape: tuple[int, ...]) -> int:
    numel = 1
    for dim in shape:
        numel *= dim
    return numel


def parse_weight_info(weight_info: Any):
    """Parse the only supported RFork manifest entry format."""
    if not isinstance(weight_info, (list, tuple)) or len(weight_info) != 5:
        return None
    seed_ptr, seed_len, seed_size, seed_shape, seed_dtype = weight_info

    if not all(is_positive_int(value) for value in (seed_ptr, seed_len, seed_size)):
        return None
    seed_shape = normalize_weight_shape(seed_shape)
    seed_dtype = normalize_dtype_name(seed_dtype)
    if seed_shape is None or numel_from_shape(seed_shape) != seed_len or seed_dtype is None:
        return None
    return seed_ptr, seed_len, seed_size, seed_shape, seed_dtype


def update_registered_weight_info(info: dict[str, Any] | None, name: str, tensor: torch.Tensor) -> None:
    if not isinstance(info, dict) or name not in info:
        return
    current = info[name]
    if isinstance(current, tuple) and len(current) == 5:
        info[name] = (*current[:3], tuple(tensor.shape), normalize_dtype_name(tensor.dtype))


def validate_weight_manifest(
    seed_info: SeedTransferInfo,
    transferable_tensors: list[tuple[str, torch.Tensor]],
    skipped_shared_names: set[str],
) -> dict[str, tuple[int, int, int, tuple[int, ...], str]] | None:
    """Validate seed metadata against local tensors before any layout change or read."""
    remote_name_set = set(seed_info.weights)
    remote_formats = seed_info.formats
    if (
        not isinstance(remote_formats, Mapping)
        or set(remote_formats) != remote_name_set
        or any(not isinstance(value, int) or isinstance(value, bool) for value in remote_formats.values())
    ):
        logger.error("RFork remote format manifest is missing, malformed, or differs from the weight manifest.")
        return None
    parsed_remote: dict[str, tuple[int, int, int, tuple[int, ...], str]] = {}
    remote_total_bytes = 0
    for name, weight_info in seed_info.weights.items():
        if name in skipped_shared_names:
            continue
        parsed_weight_info = parse_weight_info(weight_info)
        if parsed_weight_info is None:
            logger.error("Invalid weight info for %s: %s", name, weight_info)
            return None
        seed_ptr, seed_len, seed_size, seed_shape, seed_dtype = parsed_weight_info
        parsed_remote[name] = (seed_ptr, seed_len, seed_size, seed_shape, seed_dtype)
        remote_total_bytes += seed_len * seed_size

    local_total_bytes = 0
    local_by_name: dict[str, torch.Tensor] = {}
    for name, tensor in transferable_tensors:
        if name in skipped_shared_names:
            continue
        if (
            not isinstance(name, str)
            or not name
            or not isinstance(tensor, torch.Tensor)
            or not is_positive_int(tensor.data_ptr())
            or not is_positive_int(tensor.numel())
            or not is_positive_int(tensor.element_size())
        ):
            logger.error("Invalid local tensor manifest entry for %s", name)
            return None
        local_by_name[name] = tensor
        local_total_bytes += tensor.numel() * tensor.element_size()
        _, seed_len, seed_size, _, seed_dtype = parsed_remote[name]
        if seed_len != tensor.numel() or seed_size != tensor.element_size():
            logger.error(
                "Weight info mismatch for %s, expected (%s, %s), got (%s, %s)",
                name,
                seed_len,
                seed_size,
                tensor.numel(),
                tensor.element_size(),
            )
            return None
        if seed_dtype != normalize_dtype_name(tensor.dtype):
            logger.error(
                "Weight dtype mismatch for %s, expected %s, got %s",
                name,
                seed_dtype,
                normalize_dtype_name(tensor.dtype),
            )
            return None
        seed_format = remote_formats[name]
        local_format = read_npu_format(tensor)
        if local_format is None:
            logger.error("RFork could not read the local NPU storage format for %s.", name)
            return None
        if seed_format != local_format:
            logger.error(
                "Weight storage format mismatch for %s: seed=%s local=%s. "
                "A post-load layout conversion diverged between seed and receiver.",
                name,
                seed_format,
                local_format,
            )
            return None

    if len(local_by_name) != len(parsed_remote) or local_total_bytes != remote_total_bytes:
        logger.error(
            "RFork manifest count/bytes differ: local=(%d, %d), remote=(%d, %d)",
            len(local_by_name),
            local_total_bytes,
            len(parsed_remote),
            remote_total_bytes,
        )
        return None

    return parsed_remote
