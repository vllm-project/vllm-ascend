#
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
# This file is a part of the vllm-ascend project.
#
"""Host-only tensor contracts for 310P DFlash Piecewise graph inputs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from typing import Any

import torch


class GraphInputContractError(AssertionError):
    """Raised before replay when a retained graph input contract changed."""


@dataclass(frozen=True)
class GraphInputTensorContract:
    path: str
    data_ptr: int
    base_ptr: int
    storage_offset: int
    storage_nbytes: int
    view_start_byte: int
    view_end_byte: int
    dtype: str
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    contiguous: bool
    device: str
    required_alignment: int
    alignment_ok: bool


def _mapping_path(parent: str, key: Any) -> str:
    if isinstance(key, str) and key.isidentifier():
        return f"{parent}.{key}"
    return f"{parent}[{key!r}]"


def _walk_tensors(
    value: Any,
    path: str,
    seen_containers: set[int],
):
    if isinstance(value, torch.Tensor):
        yield path, value
        return

    if isinstance(value, Mapping):
        container_id = id(value)
        if container_id in seen_containers:
            return
        seen_containers.add(container_id)
        for key, nested in value.items():
            yield from _walk_tensors(
                nested,
                _mapping_path(path, key),
                seen_containers,
            )
        return

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        container_id = id(value)
        if container_id in seen_containers:
            return
        seen_containers.add(container_id)
        for index, nested in enumerate(value):
            yield from _walk_tensors(
                nested,
                f"{path}[{index}]",
                seen_containers,
            )


def _capture_tensor(path: str, tensor: torch.Tensor) -> GraphInputTensorContract:
    storage = tensor.untyped_storage()
    element_size = tensor.element_size()
    min_element = tensor.storage_offset()
    max_element = tensor.storage_offset()
    if tensor.numel() == 0:
        view_end_byte = min_element * element_size
    else:
        for size, stride in zip(tensor.shape, tensor.stride()):
            delta = (size - 1) * stride
            min_element += min(0, delta)
            max_element += max(0, delta)
        view_end_byte = (max_element + 1) * element_size

    view_start_byte = min_element * element_size
    storage_nbytes = storage.nbytes()
    if view_start_byte < 0 or view_end_byte > storage_nbytes:
        raise GraphInputContractError(
            f"{path}: tensor view byte range [{view_start_byte}, {view_end_byte}) exceeds storage size {storage_nbytes}"
        )

    data_ptr = tensor.data_ptr()
    return GraphInputTensorContract(
        path=path,
        data_ptr=data_ptr,
        base_ptr=storage.data_ptr(),
        storage_offset=tensor.storage_offset(),
        storage_nbytes=storage_nbytes,
        view_start_byte=view_start_byte,
        view_end_byte=view_end_byte,
        dtype=str(tensor.dtype),
        shape=tuple(tensor.shape),
        stride=tuple(tensor.stride()),
        contiguous=tensor.is_contiguous(),
        device=str(tensor.device),
        required_alignment=element_size,
        alignment_ok=data_ptr % element_size == 0,
    )


def capture_graph_input_contracts(
    args: tuple[Any, ...],
    kwargs: Mapping[str, Any],
) -> tuple[GraphInputTensorContract, ...]:
    """Capture host-visible metadata without reading or synchronizing tensor data."""
    seen_containers: set[int] = set()
    contracts = [_capture_tensor(path, tensor) for path, tensor in _walk_tensors(args, "args", seen_containers)]
    contracts.extend(_capture_tensor(path, tensor) for path, tensor in _walk_tensors(kwargs, "kwargs", seen_containers))
    return tuple(contracts)


def validate_graph_input_contracts(
    expected: tuple[GraphInputTensorContract, ...],
    actual: tuple[GraphInputTensorContract, ...],
) -> None:
    """Reject any changed tensor identity, view, layout, device, or alignment."""
    if len(expected) != len(actual):
        raise GraphInputContractError(f"graph input tensor count changed: expected {len(expected)}, got {len(actual)}")

    comparable_fields = tuple(field.name for field in fields(GraphInputTensorContract))
    for expected_contract, actual_contract in zip(expected, actual):
        if expected_contract.path != actual_contract.path:
            raise GraphInputContractError(
                f"graph input tensor path changed: expected {expected_contract.path}, got {actual_contract.path}"
            )
        for field_name in comparable_fields:
            expected_value = getattr(expected_contract, field_name)
            actual_value = getattr(actual_contract, field_name)
            if expected_value != actual_value:
                raise GraphInputContractError(
                    f"{expected_contract.path}: graph input {field_name} changed "
                    f"from {expected_value!r} to {actual_value!r}"
                )
        if not actual_contract.alignment_ok:
            raise GraphInputContractError(
                f"{actual_contract.path}: data_ptr {actual_contract.data_ptr} is not "
                f"aligned to {actual_contract.required_alignment} bytes"
            )
