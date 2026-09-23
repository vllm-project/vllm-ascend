# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Collect live model tensors and adapt their layout for RFork transfer."""

import hashlib
import inspect
import json
import logging
from collections.abc import Iterable, Iterator
from typing import Any

import torch
from torch import nn
from vllm.logger import logger

from vllm_ascend.model_loader.rfork.manifest import (
    normalize_dtype_name,
    numel_from_shape,
    read_npu_format,
)

TENSOR_LAYOUT_SAMPLE_LIMIT = 3

# Runtime scratch tensors are local execution state, not checkpoint-derived
# model state.  They must keep the capacity selected by the receiving instance
# instead of being copied from a seed that may use different scheduler limits.
_RUNTIME_ONLY_TENSOR_NAMES = frozenset({"topk_indices_buffer"})


def _is_runtime_only_tensor(name: str) -> bool:
    return name.rsplit(".", 1)[-1] in _RUNTIME_ONLY_TENSOR_NAMES


def _layout_digest(records: list[dict[str, Any]]) -> str:
    payload = json.dumps(records, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def build_structural_digest(tensors: list[tuple[str, torch.Tensor]]) -> str:
    """Digest the transferable tensor set's names, shapes, dtypes, and NPU formats.

    This summarizes exactly what ``validate_weight_manifest`` compares between a
    seed and a receiver, derived from the built model rather than from
    configuration fields.  Pass the output of ``collect_transferable_tensors``
    so runtime-only scratch buffers stay excluded.  ``read_npu_format`` yields
    ``None`` off-device, which keeps the digest well defined during CPU-only
    inspection.
    """
    records = [
        {
            "name": name,
            "shape": tuple(int(dim) for dim in tensor.shape),
            "dtype": normalize_dtype_name(tensor.dtype),
            "npu_format": read_npu_format(tensor),
        }
        for name, tensor in sorted(tensors, key=lambda item: item[0])
    ]
    return _layout_digest(records)


def log_tensor_layout_summary(
    tensors: list[tuple[str, torch.Tensor]],
    *,
    stage: str,
    session_id: str | None,
    processed_layout: bool,
    peer_session_id: str | None = None,
    known_formats: dict[str, int] | None = None,
) -> None:
    """Log a bounded summary of logical and physical tensor layouts at INFO."""
    if not logger.isEnabledFor(logging.INFO):
        return

    try:
        import torch_npu
    except Exception:
        torch_npu = None

    semantic_records: list[dict[str, Any]] = []
    physical_records: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []
    fallback_samples: list[dict[str, Any]] = []
    format_counts: dict[str, int] = {}
    error_counts: dict[str, int] = {}
    logical_bytes_total = 0
    unique_storage_bytes = 0
    unique_storages: set[tuple[str, int]] = set()
    storage_view_tensors = 0
    physical_nonlogical_tensors = 0

    def capture(read, field: str):
        try:
            return read()
        except Exception as exc:
            error_name = f"{field}:{type(exc).__name__}"
            error_counts[error_name] = error_counts.get(error_name, 0) + 1
            return "unavailable"

    for name, tensor in sorted(tensors, key=lambda item: item[0]):
        device = capture(lambda tensor=tensor: str(tensor.device), "device")
        dtype = capture(lambda tensor=tensor: str(tensor.dtype), "dtype")
        shape = capture(lambda tensor=tensor: tuple(int(value) for value in tensor.shape), "shape")
        stride = capture(lambda tensor=tensor: tuple(int(value) for value in tensor.stride()), "stride")
        numel = capture(lambda tensor=tensor: int(tensor.numel()), "numel")
        element_size = capture(lambda tensor=tensor: int(tensor.element_size()), "element_size")
        logical_bytes = (
            numel * element_size if isinstance(numel, int) and isinstance(element_size, int) else "unavailable"
        )
        storage_offset = capture(lambda tensor=tensor: int(tensor.storage_offset()), "storage_offset")
        storage_bytes = capture(lambda tensor=tensor: int(tensor.untyped_storage().nbytes()), "storage_bytes")
        storage_ptr = capture(lambda tensor=tensor: int(tensor.untyped_storage().data_ptr()), "storage_ptr")
        if known_formats is not None and name in known_formats:
            npu_format: Any = known_formats[name]
        elif getattr(getattr(tensor, "device", None), "type", None) == "npu" and torch_npu is not None:
            npu_format = capture(
                lambda tensor=tensor: int(torch_npu.get_npu_format(tensor)),
                "npu_format",
            )
        else:
            npu_format = "unavailable"
        if getattr(getattr(tensor, "device", None), "type", None) == "npu" and torch_npu is not None:
            npu_storage_numel: Any = capture(
                lambda tensor=tensor: int(torch_npu.get_storage_size(tensor)),
                "npu_storage_numel",
            )
        else:
            npu_storage_numel = "unavailable"

        if isinstance(logical_bytes, int):
            logical_bytes_total += logical_bytes
        if isinstance(storage_ptr, int) and isinstance(storage_bytes, int):
            storage_key = (str(device), storage_ptr)
            if storage_key not in unique_storages:
                unique_storages.add(storage_key)
                unique_storage_bytes += storage_bytes
        is_storage_view = (
            isinstance(storage_offset, int)
            and isinstance(storage_bytes, int)
            and isinstance(logical_bytes, int)
            and (storage_offset != 0 or storage_bytes != logical_bytes)
        )
        is_physical_nonlogical = (
            isinstance(npu_storage_numel, int) and isinstance(numel, int) and npu_storage_numel != numel
        )
        storage_view_tensors += int(is_storage_view)
        physical_nonlogical_tensors += int(is_physical_nonlogical)

        semantic = {
            "name": name,
            "dtype": dtype,
            "shape": shape,
            "stride": stride,
            "logical_bytes": logical_bytes,
            "npu_format": npu_format,
        }
        physical = {
            "name": name,
            "storage_offset": storage_offset,
            "storage_bytes": storage_bytes,
            "npu_storage_numel": npu_storage_numel,
        }
        semantic_records.append(semantic)
        physical_records.append(physical)
        sample = {**semantic, **physical, "device": device}
        if len(fallback_samples) < TENSOR_LAYOUT_SAMPLE_LIMIT:
            fallback_samples.append(sample)
        if (is_storage_view or is_physical_nonlogical) and len(samples) < TENSOR_LAYOUT_SAMPLE_LIMIT:
            samples.append(sample)

        format_key = str(npu_format)
        format_counts[format_key] = format_counts.get(format_key, 0) + 1

    if not samples:
        samples = fallback_samples
    logger.info(
        "RFork tensor layout summary: stage=%s session=%s peer_session=%s layout=%s tensors=%d "
        "logical_bytes=%d unique_storage_bytes=%d storage_view_tensors=%d physical_nonlogical_tensors=%d "
        "formats=%s semantic_digest=%s physical_digest=%s samples=%s errors=%s",
        stage,
        session_id,
        peer_session_id,
        "processed" if processed_layout else "checkpoint",
        len(semantic_records),
        logical_bytes_total,
        unique_storage_bytes,
        storage_view_tensors,
        physical_nonlogical_tensors,
        format_counts,
        _layout_digest(semantic_records),
        _layout_digest(physical_records),
        samples,
        error_counts,
    )


def reshape_tensor_to_seed_shape(
    name: str,
    tensor: torch.Tensor,
    seed_shape: tuple[int, ...] | None,
    reshape_events: list[tuple[str, tuple[int, ...], tuple[int, ...]]] | None = None,
) -> bool:
    if seed_shape is None or tuple(tensor.shape) == seed_shape:
        return True
    if tensor.numel() != numel_from_shape(seed_shape):
        logger.error("Weight shape mismatch for %s: local=%s, seed=%s", name, tuple(tensor.shape), seed_shape)
        return False
    local_shape = tuple(tensor.shape)
    try:
        tensor.data = tensor.data.view(seed_shape)
    except Exception as exc:
        logger.error("Failed to reshape RFork tensor %s from %s to %s: %s", name, local_shape, seed_shape, exc)
        return False
    if reshape_events is not None:
        reshape_events.append((name, local_shape, seed_shape))
    return True


def is_tensor_on_transfer_device(tensor: torch.Tensor) -> bool:
    return tensor.device.type == "npu"


def is_transferable_tensor(tensor: torch.Tensor) -> bool:
    return not tensor.is_meta and tensor.numel() > 0 and is_tensor_on_transfer_device(tensor)


def is_non_overlapping_dense_tensor(tensor: torch.Tensor) -> bool:
    """Return whether logical elements occupy one contiguous byte range."""
    if tensor.numel() <= 1:
        return True

    dense_stride = 1
    for stride, size in sorted(
        (int(stride), int(size)) for size, stride in zip(tensor.shape, tensor.stride(), strict=True) if size > 1
    ):
        if stride != dense_stride:
            return False
        dense_stride *= size
    return True


def validate_transferable_tensor_layout(name: str, tensor: torch.Tensor) -> None:
    """Reject tensor views that cannot be represented by RFork byte ranges."""
    if is_non_overlapping_dense_tensor(tensor):
        return
    raise ValueError(
        "RFork cannot transfer a tensor with gapped or overlapping storage: "
        f"{name!r}; shape={tuple(tensor.shape)}, stride={tuple(tensor.stride())}."
    )


def _iter_tensors_in_value(
    prefix: str,
    value: Any,
    visited_object_ids: set[int],
    scan_objects: bool = False,
) -> Iterator[tuple[str, torch.Tensor]]:
    if isinstance(value, torch.Tensor):
        yield prefix, value
        return
    if isinstance(value, (nn.Module, str, bytes)):
        return
    # Scan callable instances for tensors, but skip executable function, method, and class objects.
    if inspect.isfunction(value) or inspect.ismethod(value) or inspect.isclass(value):
        return
    if isinstance(value, (list, tuple)):
        value_id = id(value)
        if value_id in visited_object_ids:
            return
        visited_object_ids.add(value_id)
        try:
            for index, item in enumerate(value):
                yield from _iter_tensors_in_value(f"{prefix}.{index}", item, visited_object_ids, scan_objects)
        finally:
            visited_object_ids.remove(value_id)
        return
    if isinstance(value, dict):
        value_id = id(value)
        if value_id in visited_object_ids:
            return
        visited_object_ids.add(value_id)
        try:
            for key, item in value.items():
                yield from _iter_tensors_in_value(f"{prefix}.{key}", item, visited_object_ids, scan_objects)
        finally:
            visited_object_ids.remove(value_id)
        return
    if callable(value) and (not scan_objects or not hasattr(value, "__dict__")):
        return
    if not scan_objects or not hasattr(value, "__dict__"):
        return
    value_id = id(value)
    if value_id in visited_object_ids:
        return
    visited_object_ids.add(value_id)
    try:
        for attr_name, attr_value in vars(value).items():
            if not attr_name.startswith("_"):
                yield from _iter_tensors_in_value(
                    f"{prefix}.{attr_name}",
                    attr_value,
                    visited_object_ids,
                    scan_objects,
                )
    finally:
        visited_object_ids.remove(value_id)


def _try_collect(
    name: str,
    tensor: torch.Tensor,
    seen_names: dict[str, int],
    seen_tensors: dict[tuple[Any, ...], int],
    collected: list[tuple[str, torch.Tensor]],
) -> None:
    if _is_runtime_only_tensor(name) or not is_transferable_tensor(tensor):
        return
    validate_transferable_tensor_layout(name, tensor)
    data_ptr = tensor.data_ptr()
    tensor_signature = (
        data_ptr,
        tensor.numel(),
        tuple(tensor.shape),
        tensor.dtype,
        tensor.device,
        tuple(tensor.stride()),
    )
    existing_index = seen_names.get(name)
    if existing_index is not None:
        existing_tensor = collected[existing_index][1]
        if existing_tensor is tensor or tensor_signature == (
            existing_tensor.data_ptr(),
            existing_tensor.numel(),
            tuple(existing_tensor.shape),
            existing_tensor.dtype,
            existing_tensor.device,
            tuple(existing_tensor.stride()),
        ):
            return
        raise ValueError(
            "RFork encountered conflicting tensor entries for logical name "
            f"{name!r}; shape, dtype, stride, or storage differs."
        )

    # Parameters and buffers are canonical. An implementation object may expose
    # the exact same tensor under another public name; transferring that range
    # twice only bloats the manifest. Distinct views retain separate entries.
    existing_index = seen_tensors.get(tensor_signature)
    if existing_index is not None:
        seen_names[name] = existing_index
        return

    seen_names[name] = len(collected)
    seen_tensors[tensor_signature] = len(collected)
    collected.append((name, tensor))


def collect_transferable_tensors(
    model: nn.Module, processed_layout: bool, exclude_prefixes: frozenset[str] | None = None
) -> list[tuple[str, torch.Tensor]]:
    """Collect tensors eligible for RFork transfer, optionally excluding by name prefix.

    Args:
        model: The model to inspect.
        processed_layout: Whether to scan all public module attributes (True) or
            only the 'impl' attribute (False).
        exclude_prefixes: If provided, skip tensors whose fully-qualified name
            starts with any of these prefixes. Used to exclude draft modules that
            will be shared with or rebuilt from the target model.
    """
    seen: dict[str, int] = {}
    seen_tensors: dict[tuple[Any, ...], int] = {}
    collected: list[tuple[str, torch.Tensor]] = []

    def _should_exclude(name: str) -> bool:
        return exclude_prefixes is not None and any(name.startswith(prefix) for prefix in exclude_prefixes)

    for name, tensor in model.named_parameters():
        if not _should_exclude(name):
            _try_collect(name, tensor, seen, seen_tensors, collected)
    for name, tensor in model.named_buffers():
        if not _should_exclude(name):
            _try_collect(name, tensor, seen, seen_tensors, collected)
    for module_prefix, module in model.named_modules():
        attributes: Iterable[tuple[str, Any, bool]]
        if processed_layout:
            attributes = (
                (name, value, name == "impl")
                for name, value in vars(module).items()
                if not name.startswith("_") and not isinstance(value, nn.Module)
            )
        else:
            impl = getattr(module, "impl", None)
            attributes = () if impl is None or isinstance(impl, nn.Module) else (("impl", impl, True),)

        for attr_name, attr_value, scan_objects in attributes:
            for tensor_name, tensor in _iter_tensors_in_value(
                attr_name,
                attr_value,
                set(),
                scan_objects,
            ):
                full_name = f"{module_prefix}.{tensor_name}" if module_prefix else tensor_name
                if not _should_exclude(full_name):
                    _try_collect(full_name, tensor, seen, seen_tensors, collected)
    return collected


def find_non_npu_state_tensors(model: Any) -> list[str]:
    if not isinstance(model, nn.Module):
        return []
    return [
        name
        for iterator in (model.named_parameters(), model.named_buffers())
        for name, tensor in iterator
        if not tensor.is_meta and tensor.numel() > 0 and not is_tensor_on_transfer_device(tensor)
    ]
