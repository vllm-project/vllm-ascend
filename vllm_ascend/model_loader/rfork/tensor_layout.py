# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Collect and describe live model tensors for RFork transfer."""

import dataclasses
import enum
import hashlib
import json
import logging
import weakref
from bisect import bisect_right
from collections import Counter
from collections.abc import Iterator
from types import FunctionType, MethodType, ModuleType
from typing import Any

import regex as re
import torch
from torch import nn
from vllm.logger import logger

from vllm_ascend.model_loader.rfork.local_weights import get_local_tensor_ids

# Exact types only: numeric subclasses can carry tensor attributes.
_TENSOR_ATTRIBUTE_LEAF_TYPES = frozenset({str, bytes, int, float, bool, complex, type(None)})

# Runtime scratch and locally derived caches keep receiver-owned storage.
_RUNTIME_ONLY_TENSOR_NAMES = frozenset({"topk_indices_buffer", "_rope_derived_cos_cache", "_rope_derived_sin_cache"})

_PLAIN_PATH_SEGMENT = re.compile(r"[A-Za-z0-9_]+")

# Workspaces that post-load processing allocates without reading weight values.
POST_LOAD_RUNTIME_ONLY_ATTRS = _RUNTIME_ONLY_TENSOR_NAMES | frozenset(
    {
        "_flashmla_writer_buffers",
        "_strided_kv_workspace",
        "_dcp_current_kv_buffers",
        "_dcp_prefill_current_kv_buffers",
    }
)

_MODULE_INTERNAL_ATTRS = frozenset(vars(nn.Module()))
_MODEL_STATE_SKIPPED_TYPES = (nn.Module, str, bytes, type, FunctionType, MethodType, ModuleType, enum.Enum)
# Bounds the walk through configs and helpers reachable from module attributes.
_MAX_MODEL_STATE_DEPTH = 8


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


def _is_runtime_only_tensor(name: str) -> bool:
    return name.rsplit(".", 1)[-1] in _RUNTIME_ONLY_TENSOR_NAMES


def _layout_digest(records: list[dict[str, Any]]) -> str:
    payload = json.dumps(records, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def build_structural_digest(
    tensors: list[tuple[str, torch.Tensor]], known_formats: dict[str, int] | None = None
) -> str:
    """Hash the manifest's tensor IDs, shapes, strides, dtypes, and NPU formats.

    Pass ``collect_transferable_tensors`` output to exclude runtime buffers.
    ``known_formats`` reuses NPU formats read during registration.
    """
    records = [
        {
            "name": name,
            "shape": tuple(int(dim) for dim in tensor.shape),
            "stride": tuple(int(stride) for stride in tensor.stride()),
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
    peer_session_id: str | None = None,
    known_formats: dict[str, int] | None = None,
) -> None:
    """Log a compact summary without affecting model loading."""
    if not logger.isEnabledFor(logging.INFO):
        return
    try:
        formats = Counter(
            str(known_formats[name] if known_formats is not None and name in known_formats else read_npu_format(tensor))
            for name, tensor in tensors
        )
        total_bytes = sum(tensor.numel() * tensor.element_size() for _, tensor in tensors)
        logger.info(
            "RFork tensor layout summary: stage=%s session=%s peer_session=%s tensors=%d bytes=%d formats=%s",
            stage,
            session_id,
            peer_session_id,
            len(tensors),
            total_bytes,
            dict(sorted(formats.items())),
        )
    except Exception as exc:
        logger.info(
            "RFork tensor layout summary: stage=%s session=%s peer_session=%s unavailable=%s",
            stage,
            session_id,
            peer_session_id,
            type(exc).__name__,
        )


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
            return f"enum({enum_type}:{int(key.value)})" if isinstance(key.value, int) else None
        return f"enum({enum_type}.{key.name})" if key.name is not None else None
    if isinstance(key, str):
        return json.dumps(key, ensure_ascii=True)
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
    value: Any, scan_objects: bool, registered_only: bool
) -> Iterator[tuple[str, Any, Any, bool]]:
    if isinstance(value, nn.Module):
        # Read local registries: named_parameters/named_modules discard aliases before we can canonicalize them.
        for kind, members in (("module", value._modules), ("parameter", value._parameters), ("buffer", value._buffers)):
            for name, item in members.items():
                if type(item) not in _TENSOR_ATTRIBUTE_LEAF_TYPES:
                    yield kind, name, item, False
        if registered_only:
            return
        for name, item in vars(value).items():
            if not name.startswith("_") and type(item) not in _TENSOR_ATTRIBUTE_LEAF_TYPES:
                yield "attribute", name, item, name == "impl"
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
    parent_path: str,
) -> tuple[Any, ...] | None:
    if isinstance(value, torch.Tensor):
        # Reject this alias edge only; another alias may still collect the tensor.
        if _is_runtime_only_tensor(str(name)):
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

            children = _iter_tensor_children(value, scan_objects, registered_only)
            child_paths: set[str] = set()
            for kind, name, item, child_scan_objects in children:
                child_key = _tensor_child_key(kind, name, item, child_scan_objects, tensor_metadata, path)
                if child_key is None:
                    continue
                candidate_path = _tensor_child_path(path, kind, name)
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


def collect_transferable_tensors(model: nn.Module) -> list[tuple[str, torch.Tensor]]:
    """Collect each tensor range once under its canonical shortest-path ID.

    Registered parameters and buffers take priority over attribute aliases so
    in-place layout changes reach the registered tensor.
    """
    claimed: set[tuple[Any, ...]] = set()
    tensor_metadata: dict[int, tuple[torch.Tensor, tuple[Any, ...] | None]] = {}
    collected: list[tuple[str, torch.Tensor]] = []
    for registered_only in (True, False):
        _collect_shortest_path_tensors(model, registered_only, claimed, tensor_metadata, collected)
    tensor_ids = [tensor_id for tensor_id, _ in collected]
    if len(tensor_ids) != len(set(tensor_ids)):
        raise ValueError("RFork encountered conflicting tensor IDs in one model.")
    return collected


def _iter_model_state_tensors(model: nn.Module, exempt_attrs: frozenset[str]) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield tensors held by module attributes, including private attributes and nested helpers."""
    if not isinstance(model, nn.Module):
        return
    stack: list[tuple[str, Any, int]] = []
    for module_name, module in model.named_modules():
        for name, value in vars(module).items():
            if name not in _MODULE_INTERNAL_ATTRS and name not in exempt_attrs:
                stack.append((f"{module_name}.{name}" if module_name else name, value, 0))
    visited: set[int] = set()
    while stack:
        path, value, depth = stack.pop()
        if isinstance(value, torch.Tensor):
            yield path, value
            continue
        if (
            type(value) in _TENSOR_ATTRIBUTE_LEAF_TYPES
            or isinstance(value, _MODEL_STATE_SKIPPED_TYPES)
            or depth >= _MAX_MODEL_STATE_DEPTH
            or id(value) in visited
        ):
            continue
        visited.add(id(value))
        if isinstance(value, dict):
            children = ((f"{path}[{key!r}]", item) for key, item in value.items())
        elif isinstance(value, (list, tuple, set, frozenset)):
            children = ((f"{path}[{index}]", item) for index, item in enumerate(value))
        elif dataclasses.is_dataclass(value):
            children = (
                (f"{path}.{field.name}", getattr(value, field.name, None))
                for field in dataclasses.fields(value)
                if field.name not in exempt_attrs
            )
        elif hasattr(value, "__dict__"):
            children = ((f"{path}.{name}", item) for name, item in vars(value).items() if name not in exempt_attrs)
        else:
            continue
        stack.extend((child_path, item, depth + 1) for child_path, item in children)


def snapshot_model_state_tensors(
    model: nn.Module, exempt_attrs: frozenset[str] = POST_LOAD_RUNTIME_ONLY_ATTRS
) -> dict[int, weakref.ref[torch.Tensor]]:
    """Record tensors that exist before post-load processing.

    References are weak so the snapshot does not pin weights that processing replaces.
    """
    return {id(tensor): weakref.ref(tensor) for _, tensor in _iter_model_state_tensors(model, exempt_attrs)}


def _tensor_byte_span(tensor: torch.Tensor) -> int:
    extent = 1 + sum((int(size) - 1) * int(stride) for size, stride in zip(tensor.shape, tensor.stride(), strict=True))
    return extent * tensor.element_size()


def find_untransferred_post_load_tensors(
    model: nn.Module,
    baseline: dict[int, weakref.ref[torch.Tensor]],
    transferable_tensors: list[tuple[str, torch.Tensor]],
    exempt_attrs: frozenset[str] = POST_LOAD_RUNTIME_ONLY_ATTRS,
) -> list[str]:
    """Return tensors created by post-load processing outside the transferred byte ranges.

    A receiver processes uninitialized weights before the transfer, so such tensors would keep
    values derived from them. This misses values written in place into tensors that existed
    before processing, and Python scalars derived from weights.
    """
    ranges: list[tuple[int, int]] = []
    for start, end in sorted(
        (tensor.data_ptr(), tensor.data_ptr() + tensor.numel() * tensor.element_size())
        for _, tensor in transferable_tensors
    ):
        if ranges and start <= ranges[-1][1]:
            ranges[-1] = (ranges[-1][0], max(ranges[-1][1], end))
        else:
            ranges.append((start, end))
    starts = [start for start, _ in ranges]

    untransferred: list[str] = []
    seen: set[int] = set()
    for path, tensor in _iter_model_state_tensors(model, exempt_attrs):
        if id(tensor) in seen:
            continue
        seen.add(id(tensor))
        previous = baseline.get(id(tensor))
        if (previous is not None and previous() is tensor) or tensor.is_meta or tensor.numel() == 0:
            continue
        if is_tensor_on_transfer_device(tensor):
            start = tensor.data_ptr()
            index = bisect_right(starts, start) - 1
            if index >= 0 and start + _tensor_byte_span(tensor) <= ranges[index][1]:
                continue
        untransferred.append(path)
    return sorted(untransferred)


def find_non_npu_state_tensors(model: Any) -> list[str]:
    """Reject host state unless RFork completed its supported local loading."""
    if not isinstance(model, nn.Module):
        return []
    local_tensor_ids = get_local_tensor_ids(model)
    return [
        name
        for iterator in (model.named_parameters(), model.named_buffers())
        for name, tensor in iterator
        if not tensor.is_meta
        and tensor.numel() > 0
        and not is_tensor_on_transfer_device(tensor)
        and id(tensor) not in local_tensor_ids
    ]
