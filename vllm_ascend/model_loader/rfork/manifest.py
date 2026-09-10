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
    has_full_metadata = False
    seed_shape = None
    seed_dtype = None
    if isinstance(weight_info, (list, tuple)):
        if len(weight_info) not in (3, 4, 5):
            return None
        seed_ptr, seed_len, seed_size = weight_info[:3]
        if len(weight_info) >= 4:
            shape_or_metadata = weight_info[3]
            if isinstance(shape_or_metadata, Mapping):
                has_full_metadata = True
                seed_shape = shape_or_metadata.get("shape")
                seed_dtype = shape_or_metadata.get("dtype")
            else:
                if shape_or_metadata is None:
                    return None
                seed_shape = shape_or_metadata
            if len(weight_info) == 5:
                has_full_metadata = True
                seed_dtype = weight_info[4]
    elif isinstance(weight_info, Mapping):
        has_full_metadata = True

        def first(*keys: str):
            return next((weight_info[key] for key in keys if key in weight_info), None)

        seed_ptr = first("ptr", "pointer", "seed_ptr", "address")
        seed_len = first("numel", "count", "seed_len", "length")
        seed_size = first("element_size", "itemsize", "seed_size", "size")
        seed_shape = weight_info.get("shape")
        seed_dtype = weight_info.get("dtype")
    else:
        return None

    if not all(is_positive_int(value) for value in (seed_ptr, seed_len, seed_size)):
        return None
    if seed_shape is not None:
        seed_shape = normalize_weight_shape(seed_shape)
        if seed_shape is None or numel_from_shape(seed_shape) != seed_len:
            return None
    if seed_dtype is not None:
        seed_dtype = normalize_dtype_name(seed_dtype)
        if seed_dtype is None:
            return None
    if has_full_metadata:
        return seed_ptr, seed_len, seed_size, seed_shape, seed_dtype
    return seed_ptr, seed_len, seed_size, seed_shape


def unpack_weight_info(parsed: tuple[Any, ...]) -> tuple[int, int, int, Any, str | None]:
    if len(parsed) == 4:
        seed_ptr, seed_len, seed_size, seed_shape = parsed
        return seed_ptr, seed_len, seed_size, seed_shape, None
    return parsed


def update_registered_weight_shape(info: dict[str, Any] | None, name: str, tensor: torch.Tensor) -> None:
    if isinstance(info, dict):
        info[name] = tuple(tensor.shape)


def update_registered_weight_info(info: dict[str, Any] | None, name: str, tensor: torch.Tensor) -> None:
    if not isinstance(info, dict) or name not in info:
        return
    current = info[name]
    if isinstance(current, tuple) and len(current) >= 5:
        info[name] = (*current[:3], tuple(tensor.shape), normalize_dtype_name(tensor.dtype))
    elif isinstance(current, tuple) and len(current) == 4:
        info[name] = (*current[:3], tuple(tensor.shape))


def _extract_manifest_entries(manifest_metadata: Any) -> Mapping[str, Any] | None:
    if not isinstance(manifest_metadata, Mapping):
        return None
    for key in ("weights", "tensors", "manifest", "entries", "weight_info"):
        entries = manifest_metadata.get(key)
        if isinstance(entries, Mapping):
            return entries

    metadata_keys = {
        "tensor_count",
        "num_tensors",
        "count",
        "total_bytes",
        "byte_count",
        "bytes",
    }
    entries = {key: value for key, value in manifest_metadata.items() if key not in metadata_keys}
    return entries if entries and all(isinstance(key, str) for key in entries) else None


def _get_manifest_value(metadata: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in metadata:
            return metadata[key]
    return None


def _parse_manifest_scalar(value: Any, *, allow_zero: bool = False) -> int | None:
    if not isinstance(value, int) or isinstance(value, bool):
        return None
    if value < 0 or (value == 0 and not allow_zero):
        return None
    return value


def validate_weight_manifest(
    seed_info: SeedTransferInfo,
    transferable_tensors: list[tuple[str, torch.Tensor]],
    skipped_shared_names: set[str],
    manifest_metadata: Any | None = None,
) -> dict[str, tuple[int, int, int, tuple[int, ...], str]] | None:
    """Validate seed metadata against local tensors before any layout change or read."""
    remote_name_set = set(seed_info.weights)
    remote_shapes = seed_info.shapes
    if remote_shapes is not None:
        if not isinstance(remote_shapes, Mapping) or set(remote_shapes) != remote_name_set:
            logger.error("RFork remote shape manifest names differ from weight manifest.")
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
        seed_ptr, seed_len, seed_size, seed_shape, seed_dtype = unpack_weight_info(parsed_weight_info)
        if remote_shapes is not None:
            manifest_shape = normalize_weight_shape(remote_shapes[name])
            if manifest_shape is None:
                logger.error("RFork invalid remote shape metadata for %s", name)
                return None
            if seed_shape is None:
                seed_shape = manifest_shape
            elif seed_shape != manifest_shape:
                logger.error("RFork conflicting shape metadata for %s", name)
                return None
        if seed_shape is None or seed_dtype is None:
            logger.error("RFork manifest entry for %s must include shape and dtype", name)
            return None
        if numel_from_shape(seed_shape) != seed_len:
            logger.error("RFork shape metadata does not match numel for %s", name)
            return None
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

    if len(local_by_name) != len(parsed_remote) or local_total_bytes != remote_total_bytes:
        logger.error(
            "RFork manifest count/bytes differ: local=(%d, %d), remote=(%d, %d)",
            len(local_by_name),
            local_total_bytes,
            len(parsed_remote),
            remote_total_bytes,
        )
        return None

    if manifest_metadata is not None:
        if not isinstance(manifest_metadata, Mapping):
            logger.error("RFork manifest metadata is malformed.")
            return None
        count_value = _get_manifest_value(manifest_metadata, "tensor_count", "num_tensors", "count")
        if count_value is not None and _parse_manifest_scalar(count_value) != len(local_by_name):
            logger.error("RFork manifest tensor count mismatch: %s", count_value)
            return None
        bytes_value = _get_manifest_value(manifest_metadata, "total_bytes", "byte_count", "bytes")
        if bytes_value is not None and _parse_manifest_scalar(bytes_value) != local_total_bytes:
            logger.error("RFork manifest byte count mismatch: %s", bytes_value)
            return None

        metadata_entries = _extract_manifest_entries(manifest_metadata)
        if metadata_entries is not None:
            if set(metadata_entries) != set(local_by_name):
                logger.error("RFork optional manifest names differ from local manifest.")
                return None
            for name, metadata in metadata_entries.items():
                if not isinstance(metadata, Mapping):
                    logger.error("RFork optional manifest entry for %s is malformed", name)
                    return None
                tensor = local_by_name[name]
                remote_entry = parsed_remote[name]
                expected_numel = _get_manifest_value(metadata, "numel", "count", "seed_len")
                if expected_numel is not None and (
                    _parse_manifest_scalar(expected_numel) != tensor.numel()
                    or _parse_manifest_scalar(expected_numel) != remote_entry[1]
                ):
                    logger.error("RFork optional manifest numel mismatch for %s", name)
                    return None
                expected_size = _get_manifest_value(metadata, "element_size", "itemsize", "seed_size")
                if expected_size is not None and (
                    _parse_manifest_scalar(expected_size) != tensor.element_size()
                    or _parse_manifest_scalar(expected_size) != remote_entry[2]
                ):
                    logger.error("RFork optional manifest element size mismatch for %s", name)
                    return None
                expected_shape = metadata.get("shape")
                if expected_shape is not None:
                    expected_shape = normalize_weight_shape(expected_shape)
                    if (
                        expected_shape is None
                        or numel_from_shape(expected_shape) != tensor.numel()
                        or expected_shape != remote_entry[3]
                    ):
                        logger.error("RFork optional manifest shape mismatch for %s", name)
                        return None
                expected_dtype = metadata.get("dtype")
                if expected_dtype is not None:
                    expected_dtype = normalize_dtype_name(expected_dtype)
                    if expected_dtype != normalize_dtype_name(tensor.dtype) or expected_dtype != remote_entry[4]:
                        logger.error("RFork optional manifest dtype mismatch for %s", name)
                        return None

    return parsed_remote
