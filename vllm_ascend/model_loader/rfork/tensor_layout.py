# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""RFork layout digests, summaries, and shape adaptation.

Tensor enumeration itself lives in the shared ``weight_state`` module
(strategy B: canonical shortest-path BFS, registered-first collection);
this module keeps RFork's protocol-side layout concerns -- structural
digests, bounded layout summaries, seed-shape adaptation, and the
non-NPU state check -- and re-exports the enumeration API it is built on
for existing RFork consumers (session, transfer_backend, tests).
"""

import hashlib
import json
import logging
from typing import Any

import torch
from torch import nn
from vllm.logger import logger

# Module handle: find_non_npu_state_tensors resolves the transferability
# predicate late, through weight_state (the predicate's home), so monkeypatches
# on weight_state are visible to it. A from-import binding would snapshot the
# original function at import time and miss them.
from vllm_ascend.model_loader import weight_state
from vllm_ascend.model_loader.rfork.local_weights import get_local_tensor_ids
from vllm_ascend.model_loader.rfork.manifest import (
    normalize_dtype_name,
    numel_from_shape,
    read_npu_format,
)
from vllm_ascend.model_loader.weight_state import (
    collect_transferable_tensors as collect_transferable_tensors,  # noqa: F401 (re-export)
)
from vllm_ascend.model_loader.weight_state import (
    is_tensor_on_transfer_device as is_tensor_on_transfer_device,  # noqa: F401 (re-export)
)
from vllm_ascend.model_loader.weight_state import (
    is_transferable_tensor as is_transferable_tensor,  # noqa: F401 (re-export)
)
from vllm_ascend.model_loader.weight_state import (
    validate_transferable_tensor_layout as validate_transferable_tensor_layout,  # noqa: F401 (re-export)
)

TENSOR_LAYOUT_SAMPLE_LIMIT = 3


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
        logger.error("Weight shape mismatch for %s, local=%s, seed=%s", name, tuple(tensor.shape), seed_shape)
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
        and not weight_state.is_tensor_on_transfer_device(tensor)
        and id(tensor) not in local_tensor_ids
    ]
