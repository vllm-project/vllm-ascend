# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Collect live model tensors and adapt their layout for RFork transfer."""

import enum
import hashlib
import json
import logging
from collections.abc import Iterator
from types import FunctionType, MethodType
from typing import Any

import regex as re
import torch
from torch import nn
from vllm.logger import logger

from vllm_ascend.model_loader.rfork.manifest import (
    normalize_dtype_name,
    numel_from_shape,
    read_npu_format,
)

TENSOR_LAYOUT_SAMPLE_LIMIT = 3

# Exact types only: numeric subclasses can carry tensor attributes.
_TENSOR_ATTRIBUTE_LEAF_TYPES = frozenset({str, bytes, int, float, bool, complex, type(None)})

# Runtime scratch tensors are local execution state, not checkpoint-derived
# model state.  They must keep the capacity selected by the receiving instance
# instead of being copied from a seed that may use different scheduler limits.
_RUNTIME_ONLY_TENSOR_NAMES = frozenset({"topk_indices_buffer"})

_PLAIN_PATH_SEGMENT = re.compile(r"[A-Za-z0-9_]+")


def _is_runtime_only_tensor(name: str) -> bool:
    return name.rsplit(".", 1)[-1] in _RUNTIME_ONLY_TENSOR_NAMES


def _layout_digest(records: list[dict[str, Any]]) -> str:
    payload = json.dumps(records, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def build_structural_digest(
    tensors: list[tuple[str, torch.Tensor]], known_formats: dict[str, int] | None = None
) -> str:
    """Hash the manifest's tensor IDs, shapes, dtypes, and NPU formats.

    Pass ``collect_transferable_tensors`` output to exclude runtime buffers.
    ``known_formats`` reuses NPU formats read during registration.
    """
    records = [
        {
            "name": name,
            "shape": tuple(int(dim) for dim in tensor.shape),
            "dtype": normalize_dtype_name(tensor.dtype),
            "npu_format": (
                known_formats[name] if known_formats is not None and name in known_formats else read_npu_format(tensor)
            ),
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


def _format_tensor_key(key: Any) -> str | None:
    """Render a container key as a typed label that is identical across processes."""
    # Order matters: bool and Enum members subclass int/str.
    if key is None or isinstance(key, bool):
        return json.dumps(key)
    if isinstance(key, enum.Enum):
        key_type = type(key)
        enum_type = f"{key_type.__module__}.{key_type.__qualname__}"
        if isinstance(key, enum.Flag):
            # Composite Flag members can be unnamed, but their integer values are unique.
            return f"enum({enum_type}:{int(key.value)})" if isinstance(key.value, int) else None
        # Plain members are unique by canonical name (aliases resolve to it); values such as NaN may collide.
        return f"enum({enum_type}.{key.name})" if key.name is not None else None
    if isinstance(key, str):
        return json.dumps(str(key), ensure_ascii=True)
    if isinstance(key, int):
        return str(int(key))
    if isinstance(key, float):
        return f"float({float(key).hex()})"
    if isinstance(key, torch.dtype):
        return f"dtype({key})"
    if type(key) is tuple:
        items: list[str] = []
        for item in key:
            label = _format_tensor_key(item)
            if label is None:
                return None
            items.append(label)
        return f"({items[0]},)" if len(items) == 1 else f"({','.join(items)})"
    # Default reprs embed addresses, so unknown key types cannot yield stable IDs.
    return None


def _tensor_child_path(parent_path: str, kind: str, name: Any) -> str:
    if kind in ("index", "key"):
        # Typed container labels keep dict keys 1 / "1" and "a.b" / a -> b distinct.
        label = _format_tensor_key(name)
        if label is None:
            name_type = type(name)
            raise ValueError(
                "RFork cannot derive a stable tensor ID from container key type "
                f"{name_type.__module__}.{name_type.__qualname__}"
            )
        return f"{parent_path}[{label}]"
    name = str(name)
    if not _PLAIN_PATH_SEGMENT.fullmatch(name):
        return f"{parent_path}[{json.dumps(name, ensure_ascii=True)}]"
    return f"{parent_path}.{name}" if parent_path else name


def _iter_tensor_children(
    value: Any, scan_objects: bool, processed_layout: bool, registered_only: bool
) -> Iterator[tuple[str, Any, Any, bool]]:
    if isinstance(value, nn.Module):
        # Read local registries: named_parameters/named_modules discard aliases before we can canonicalize them.
        for kind, members in (("module", value._modules), ("parameter", value._parameters), ("buffer", value._buffers)):
            for name, item in members.items():
                if type(item) not in _TENSOR_ATTRIBUTE_LEAF_TYPES:
                    yield kind, name, item, False
        if registered_only:
            return
        if processed_layout:
            for name, item in vars(value).items():
                if not name.startswith("_") and type(item) not in _TENSOR_ATTRIBUTE_LEAF_TYPES:
                    yield "attribute", name, item, name == "impl"
        else:
            impl = getattr(value, "impl", None)
            if type(impl) not in _TENSOR_ATTRIBUTE_LEAF_TYPES:
                yield "attribute", "impl", impl, True
    elif registered_only:
        return
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            if type(item) not in _TENSOR_ATTRIBUTE_LEAF_TYPES:
                yield "index", index, item, scan_objects
    elif isinstance(value, dict):
        for name, item in value.items():
            if type(item) not in _TENSOR_ATTRIBUTE_LEAF_TYPES:
                yield "key", name, item, scan_objects
    else:
        for name, item in vars(value).items():
            if not name.startswith("_") and type(item) not in _TENSOR_ATTRIBUTE_LEAF_TYPES:
                yield "attribute", name, item, scan_objects


def _tensor_child_key(
    kind: str,
    name: Any,
    value: Any,
    scan_objects: bool,
    tensor_metadata: dict[int, tuple[torch.Tensor, tuple[Any, ...] | None]],
    parent_path: str = "",
) -> tuple[Any, ...] | None:
    if isinstance(value, torch.Tensor):
        # Reject this alias edge only; another alias may still collect the tensor.
        if _is_runtime_only_tensor(f"{name}"):
            return None
        metadata = tensor_metadata.get(id(value))
        if metadata is not None:
            return metadata[1]
        signature = None
        if is_transferable_tensor(value):
            validate_transferable_tensor_layout(_tensor_child_path(parent_path, kind, name), value)
            signature = (
                value.data_ptr(),
                value.numel(),
                tuple(value.shape),
                value.dtype,
                value.device,
                tuple(value.stride()),
            )
        tensor_metadata[id(value)] = (value, signature)
        return signature
    if isinstance(value, nn.Module):
        # Follow modules only through their registration edges.
        return (id(value), False) if kind == "module" else None
    if isinstance(value, (str, bytes)):
        return None
    if isinstance(value, (list, tuple, dict)):
        return (id(value), scan_objects)
    if not scan_objects or isinstance(value, (FunctionType, MethodType, type)):
        return None
    return (id(value), scan_objects) if hasattr(value, "__dict__") else None


def _collect_shortest_path_tensors(
    model: nn.Module,
    processed_layout: bool,
    registered_only: bool,
    claimed: set[tuple[Any, ...]],
    tensor_metadata: dict[int, tuple[torch.Tensor, tuple[Any, ...] | None]],
    collected: list[tuple[str, torch.Tensor]],
) -> None:
    root_key = (id(model), False)
    frontier = {root_key: ""}
    # Keep strong references throughout the scan so object IDs cannot be reused.
    objects: dict[tuple[Any, ...], tuple[Any, bool]] = {root_key: (model, False)}
    visited = set(claimed)

    while frontier:
        # IDs in this layer are final. Ignore back edges, cycles, and longer paths.
        visited.update(frontier)
        next_frontier: dict[tuple[Any, ...], str] = {}
        for key, path in frontier.items():
            value, scan_objects = objects[key]
            if isinstance(value, torch.Tensor):
                claimed.add(key)
                collected.append((path, value))
                continue

            children = _iter_tensor_children(value, scan_objects, processed_layout, registered_only)
            child_paths: set[str] = set()
            for kind, name, item, child_scan_objects in children:
                child_key = _tensor_child_key(kind, name, item, child_scan_objects, tensor_metadata, path)
                if child_key is None:
                    continue
                candidate_path = _tensor_child_path(path, kind, name)
                # Distinct keys with one label (e.g. two NaN keys) would let swapped subtrees
                # share IDs and digests; leaf-level ID checks cannot see that collision.
                if candidate_path in child_paths:
                    raise ValueError(
                        f"RFork found distinct container entries with one tensor ID label: {candidate_path!r}"
                    )
                child_paths.add(candidate_path)
                if child_key in visited:
                    continue
                previous_path = next_frontier.get(child_key)
                if previous_path is None or candidate_path < previous_path:
                    # Same-signature aliases can be distinct objects; keep the one the ID names.
                    objects[child_key] = (item, child_scan_objects)
                    next_frontier[child_key] = candidate_path
        frontier = next_frontier


def collect_transferable_tensors(model: nn.Module, processed_layout: bool) -> list[tuple[str, torch.Tensor]]:
    """Collect each tensor range once under its canonical shortest-path ID.

    Registered parameters and buffers take priority over attribute aliases so
    in-place layout changes reach the registered tensor.
    """
    claimed: set[tuple[Any, ...]] = set()
    tensor_metadata: dict[int, tuple[torch.Tensor, tuple[Any, ...] | None]] = {}
    collected: list[tuple[str, torch.Tensor]] = []
    for registered_only in (True, False):
        _collect_shortest_path_tensors(model, processed_layout, registered_only, claimed, tensor_metadata, collected)
    tensor_ids = [tensor_id for tensor_id, _ in collected]
    if len(tensor_ids) != len(set(tensor_ids)):
        raise ValueError("RFork encountered conflicting tensor IDs in one model.")
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
