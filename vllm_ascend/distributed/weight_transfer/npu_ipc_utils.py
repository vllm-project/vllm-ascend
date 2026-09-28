# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Small NPU-specific helpers for weight-transfer IPC.

The generic weight-transfer orchestration stays in vLLM/Ascend engines.  This
module owns the parts that depend on torch_npu's device mapping and IPC
rebuild tuple so those assumptions are explicit and easy to audit.
"""

import os
import socket
from functools import cache, lru_cache
from typing import Any


@lru_cache(maxsize=1)
def get_ip() -> str:
    """Return the host address used in the legacy IPC handle identity."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            return sock.getsockname()[0]
    except Exception:  # noqa: BLE001 - hostname is the documented fallback
        return socket.gethostbyname(socket.gethostname())


def _visible_device_ids() -> list[int] | None:
    value = os.environ.get("ASCEND_RT_VISIBLE_DEVICES")
    if not value:
        return None
    ids: list[int] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            ids.append(int(item))
        except ValueError as exc:
            raise ValueError(
                f"ASCEND_RT_VISIBLE_DEVICES must contain comma-separated integer device ids, got {value!r}"
            ) from exc
    if not ids:
        raise ValueError("ASCEND_RT_VISIBLE_DEVICES does not name any device")
    return ids


@cache
def npu_generate_uuid(logical_device: int | None = None) -> str:
    """Return the stable host/physical-device identity used by NPU IPC.

    ``logical_device`` is part of the cache key.  Callers must pass the worker
    device explicitly when the current device is not guaranteed to match it.
    """
    if logical_device is None:
        import torch

        logical_device = torch.accelerator.current_device_index()
    if logical_device < 0:
        raise ValueError(f"logical NPU device must be non-negative, got {logical_device}")

    visible_ids = _visible_device_ids()
    if visible_ids is None:
        physical_device = logical_device
    else:
        if logical_device >= len(visible_ids):
            raise ValueError(f"logical NPU device {logical_device} is outside the visible device list {visible_ids}")
        physical_device = visible_ids[logical_device]
    return f"{get_ip()}-{physical_device}"


def rewrite_rebuild_device(args: tuple[Any, ...] | list[Any], device_index: int) -> list[Any]:
    """Copy a torch_npu rebuild tuple and replace its device index.

    The current torch_npu tuple ABI stores the device at index 6.  Keep the
    check local to this helper so a future ABI change fails before collective
    work instead of silently rebuilding on the wrong device.
    """
    if len(args) <= 6:
        raise ValueError(
            f"Unexpected torch_npu IPC rebuild tuple: expected a device field at index 6, got {len(args)} fields"
        )
    rewritten = list(args)
    rewritten[6] = device_index
    return rewritten


def _freeze(value: Any) -> Any:
    """Make nested export arguments safe to use as a cache key."""
    if isinstance(value, dict):
        return tuple(sorted((_freeze(k), _freeze(v)) for k, v in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    try:
        hash(value)
    except TypeError:
        return repr(value)
    return value


class NpuPackedBufferImporter:
    """Cache one packed NPU IPC import for the lifetime of an update.

    A producer exports one reusable buffer and sends the same rebuild arguments
    for every chunk.  Rebuilding once per chunk over-releases torch_npu's IPC
    reference counter.  The cache is replaced for a new export and explicitly
    closed after FINISH/shutdown.
    """

    def __init__(self) -> None:
        self._entry: tuple[Any, Any] | None = None

    def rebuild(self, args: tuple[Any, ...] | list[Any], device_index: int) -> Any:
        from torch_npu.multiprocessing.reductions import rebuild_npu_tensor

        rewritten = rewrite_rebuild_device(args, device_index)
        key = _freeze(rewritten)
        if self._entry is not None and self._entry[0] == key:
            return self._entry[1]

        tensor = rebuild_npu_tensor(*rewritten)
        # Dropping the previous tensor releases its one importer-side reference
        # only after all users have finished the previous update.
        self._entry = (key, tensor)
        return tensor

    def close(self) -> None:
        """Release the cached imported tensor reference."""
        self._entry = None

