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
#

from dataclasses import dataclass

import torch
from torch import nn

REGISTER_BASE_ALIGNMENT = 2 * 1024 * 1024
ACL_FORMAT_ND = 2


class AlignedStorageError(RuntimeError):
    pass


@dataclass(frozen=True)
class AlignedMemoryRegistration:
    logical_addr: int
    logical_length: int
    backing_addr: int
    backing_length: int


@dataclass
class AlignedStorage:
    raw_owner: torch.Tensor
    backing_view: torch.Tensor
    owned_views: list[torch.Tensor]
    registrations: list[AlignedMemoryRegistration]


@dataclass
class _StorageGroup:
    storage: torch.UntypedStorage
    tensors: list[torch.Tensor]
    byte_length: int
    arena_offset: int = 0


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def _named_model_tensors(model: nn.Module) -> list[torch.Tensor]:
    tensors: list[torch.Tensor] = []
    seen_ids: set[int] = set()

    try:
        named_parameters = model.named_parameters(remove_duplicate=False)
    except TypeError:
        named_parameters = model.named_parameters()
    try:
        named_buffers = model.named_buffers(remove_duplicate=False)
    except TypeError:
        named_buffers = model.named_buffers()

    for _, tensor in (*named_parameters, *named_buffers):
        if id(tensor) not in seen_ids:
            seen_ids.add(id(tensor))
            tensors.append(tensor)
    return tensors


def _validate_tensor(tensor: torch.Tensor) -> None:
    if tensor.is_meta or tensor.numel() == 0:
        raise AlignedStorageError("aligned storage cannot contain an empty or meta tensor")
    if not tensor.is_contiguous():
        raise AlignedStorageError("RFork aligned storage currently requires contiguous tensors")
    if tensor.device.type == "npu":
        try:
            import torch_npu

            npu_format = int(torch_npu.get_npu_format(tensor))
        except Exception as error:
            raise AlignedStorageError("cannot determine the NPU format for an RFork tensor") from error
        if npu_format != ACL_FORMAT_ND:
            raise AlignedStorageError(
                f"RFork aligned storage currently supports only ND tensors, got NPU format {npu_format}"
            )


def _storage_key(tensor: torch.Tensor) -> tuple[int, int]:
    storage = tensor.untyped_storage()
    return storage.data_ptr(), storage.nbytes()


def _build_storage_groups(
    model: nn.Module,
    transferable_tensors: list[tuple[str, torch.Tensor]],
) -> list[_StorageGroup]:
    named_tensors = _named_model_tensors(model)
    named_tensor_ids = {id(tensor) for tensor in named_tensors}
    for name, tensor in transferable_tensors:
        if id(tensor) not in named_tensor_ids:
            raise AlignedStorageError(
                f"RFork tensor {name} is stored outside model parameters/buffers and cannot be safely rebound"
            )

    selected_storage_keys = {_storage_key(tensor) for _, tensor in transferable_tensors}
    groups: dict[tuple[int, int], _StorageGroup] = {}
    for tensor in named_tensors:
        key = _storage_key(tensor)
        if key not in selected_storage_keys:
            continue
        _validate_tensor(tensor)
        group = groups.get(key)
        if group is None:
            storage = tensor.untyped_storage()
            group = _StorageGroup(storage=storage, tensors=[], byte_length=storage.nbytes())
            groups[key] = group
        group.tensors.append(tensor)

    if not groups:
        raise AlignedStorageError("RFork did not find any transferable tensor storage")
    return list(groups.values())


def _merge_logical_ranges(tensors: list[tuple[str, torch.Tensor]]) -> list[tuple[int, int]]:
    ranges = sorted((tensor.data_ptr(), tensor.numel() * tensor.element_size()) for _, tensor in tensors)
    merged: list[tuple[int, int]] = []
    for address, length in ranges:
        if not merged or merged[-1][0] + merged[-1][1] < address:
            merged.append((address, length))
            continue
        previous_address, previous_length = merged[-1]
        merged[-1] = (previous_address, max(previous_address + previous_length, address + length) - previous_address)
    return merged


def materialize_aligned_storage(
    model: nn.Module,
    transferable_tensors: list[tuple[str, torch.Tensor]],
    *,
    copy_values: bool,
    alignment: int = REGISTER_BASE_ALIGNMENT,
) -> AlignedStorage:
    if alignment <= 0 or alignment & (alignment - 1):
        raise AlignedStorageError("registration alignment must be a positive power of two")
    if not transferable_tensors:
        raise AlignedStorageError("RFork did not find any transferable tensors")

    device = transferable_tensors[0][1].device
    if any(tensor.device != device for _, tensor in transferable_tensors):
        raise AlignedStorageError("all RFork tensors in one aligned arena must use the same device")

    groups = _build_storage_groups(model, transferable_tensors)
    payload_bytes = 0
    for group in groups:
        element_alignment = max(tensor.element_size() for tensor in group.tensors)
        payload_bytes = _align_up(payload_bytes, element_alignment)
        group.arena_offset = payload_bytes
        payload_bytes += group.byte_length

    raw_owner = torch.empty(payload_bytes + alignment - 1, dtype=torch.uint8, device=device)
    aligned_offset = (-raw_owner.data_ptr()) % alignment
    backing_view = raw_owner.narrow(0, aligned_offset, payload_bytes)
    if backing_view.data_ptr() % alignment != 0:
        raise AlignedStorageError("failed to create a 2 MiB-aligned RFork backing view")

    raw_storage = raw_owner.untyped_storage()
    owned_views: list[torch.Tensor] = []
    replacements: list[tuple[torch.Tensor, torch.Tensor]] = []
    with torch.no_grad():
        for group in groups:
            destination_start = aligned_offset + group.arena_offset
            if copy_values:
                source_bytes = torch.empty(0, dtype=torch.uint8, device=device).set_(
                    group.storage,
                    0,
                    (group.byte_length,),
                )
                raw_owner.narrow(0, destination_start, group.byte_length).copy_(source_bytes)

            for tensor in group.tensors:
                element_size = tensor.element_size()
                tensor_offset_bytes = destination_start + tensor.storage_offset() * element_size
                if tensor_offset_bytes % element_size != 0:
                    raise AlignedStorageError("aligned arena produced an invalid typed storage offset")
                view = torch.empty(0, dtype=tensor.dtype, device=device).set_(
                    raw_storage,
                    tensor_offset_bytes // element_size,
                    tensor.size(),
                    tensor.stride(),
                )
                owned_views.append(view)
                replacements.append((tensor, view))

        for tensor, view in replacements:
            tensor.data = view

    backing_addr = backing_view.data_ptr()
    registrations = [
        AlignedMemoryRegistration(
            logical_addr=logical_addr,
            logical_length=logical_length,
            backing_addr=backing_addr,
            backing_length=payload_bytes,
        )
        for logical_addr, logical_length in _merge_logical_ranges(transferable_tensors)
    ]
    return AlignedStorage(
        raw_owner=raw_owner,
        backing_view=backing_view,
        owned_views=owned_views,
        registrations=registrations,
    )
