# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Small NPU-specific helpers for weight-transfer IPC.

The generic weight-transfer orchestration stays in vLLM/Ascend engines.  This
module owns the parts that depend on torch_npu's device mapping and IPC
rebuild tuple so those assumptions are explicit and easy to audit.
"""

import os
import socket
from functools import lru_cache
from typing import Any

import torch


@lru_cache(maxsize=1)
def get_ip() -> str:
    """Return the host address used in the legacy IPC handle identity."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            return sock.getsockname()[0]
    except Exception:  # noqa: BLE001 - hostname is the documented fallback
        return socket.gethostbyname(socket.gethostname())


def npu_generate_uuid(logical_device: int | None = None) -> str:
    """Return the stable host/physical-device identity used by NPU IPC.

    The identity is intentionally host-local and follows the visible-device
    mapping.  Callers on worker paths pass the logical device explicitly;
    omitted values are resolved at call time rather than cached as ``None``.
    """
    if logical_device is None:
        logical_device = torch.accelerator.current_device_index()
    if logical_device is None:
        raise ValueError("logical NPU device is unavailable; pass an explicit device index")
    if logical_device < 0:
        raise ValueError(f"logical NPU device must be non-negative, got {logical_device}")

    value = os.environ.get("ASCEND_RT_VISIBLE_DEVICES")
    if not value:
        physical_device = logical_device
    else:
        visible_ids: list[int] = []
        for item in value.split(","):
            item = item.strip()
            if not item:
                continue
            try:
                visible_ids.append(int(item))
            except ValueError as exc:
                raise ValueError(
                    f"ASCEND_RT_VISIBLE_DEVICES must contain comma-separated integer device ids, got {value!r}"
                ) from exc
        if not visible_ids:
            raise ValueError("ASCEND_RT_VISIBLE_DEVICES does not name any device")
        if logical_device >= len(visible_ids):
            raise ValueError(f"logical NPU device {logical_device} is outside the visible device list {visible_ids}")
        physical_device = visible_ids[logical_device]
    return f"{get_ip()}-{physical_device}"


NPU_IPC_DEVICE_INDEX = 6


class NpuPackedBufferImporter:
    """Cache one packed NPU IPC import for the lifetime of an update.

    A producer exports one reusable buffer and sends the same rebuild arguments
    for every chunk.  Rebuilding once per chunk over-releases torch_npu's IPC
    reference counter.  The cache is replaced for a new export and explicitly
    closed after FINISH/shutdown.
    """

    def __init__(self) -> None:
        self._entry: tuple[tuple[Any, ...], torch.Tensor] | None = None

    def rebuild(self, list_args: list[Any]) -> torch.Tensor:
        """Rebuild one packed buffer, reusing the import for each chunk."""
        from torch_npu.multiprocessing.reductions import rebuild_npu_tensor

        key = tuple(list_args)
        if self._entry is not None and self._entry[0] == key:
            return self._entry[1]

        tensor = rebuild_npu_tensor(*list_args)
        # Dropping the previous tensor releases its one importer-side reference
        # only after all users have finished the previous update.
        self._entry = (key, tensor)
        return tensor

    def close(self) -> None:
        """Release the cached imported tensor reference."""
        self._entry = None
