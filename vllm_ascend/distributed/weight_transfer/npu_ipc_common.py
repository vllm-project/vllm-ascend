# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared transport primitives for Ascend NPU IPC weight transfer."""

import os
import socket
from functools import lru_cache

import torch


@lru_cache(maxsize=1)
def _get_ip() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            return sock.getsockname()[0]
    except Exception:  # noqa: BLE001
        return socket.gethostbyname(socket.gethostname())


@lru_cache(maxsize=1)
def npu_generate_uuid(logical_device: int | None = None) -> str:
    """Return a stable host-and-physical-NPU identifier for IPC matching."""
    if logical_device is None:
        logical_device = torch.accelerator.current_device_index()
    visible_devices = os.environ.get("ASCEND_RT_VISIBLE_DEVICES")
    if visible_devices:
        physical_device = int(visible_devices.split(",")[logical_device].strip())
    else:
        physical_device = logical_device
    return f"{_get_ip()}-{physical_device}"


def is_rank_zero() -> bool:
    """Return whether this rank owns the trainer-side transport send."""
    return not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0


def all_gather_and_merge_handles(
    handles: list[dict[str, tuple]],
    *,
    is_sender: bool | None = None,
) -> list[dict[str, tuple]]:
    """Collect per-rank IPC handles and merge them on the sending rank."""
    if not torch.distributed.is_initialized() or torch.distributed.get_world_size() == 1:
        return handles

    world_size = torch.distributed.get_world_size()
    gathered: list[list[dict[str, tuple]] | None] = [None] * world_size
    torch.distributed.all_gather_object(gathered, handles)
    torch.distributed.barrier()
    torch.npu.synchronize()

    should_merge = is_rank_zero() if is_sender is None else is_sender
    if should_merge:
        merged: list[dict[str, tuple]] = []
        for handle_index in range(len(handles)):
            merged_handle: dict[str, tuple] = {}
            for rank_handles in gathered:
                if rank_handles is not None:
                    merged_handle.update(rank_handles[handle_index])
            merged.append(merged_handle)
        return merged
    return [{} for _ in handles]


def post_send_sync() -> None:
    """Synchronize all trainer ranks after the receiver consumes IPC handles."""
    if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
        torch.distributed.barrier()
    torch.npu.synchronize()
