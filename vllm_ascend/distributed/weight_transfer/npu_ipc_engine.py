# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NPU IPC-based weight transfer engine using Ascend IPC for communication."""

import os
import socket
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Any, ClassVar

import torch
from vllm.config import VllmConfig
from vllm.config.weight_transfer import WeightTransferConfig
from vllm.distributed.weight_transfer.base import (
    TrainerWeightTransferEngine,
)
from vllm.distributed.weight_transfer.ipc_engine import (
    IPCTrainerInitInfo,
    IPCTrainerWeightTransferEngine,
    IPCWeightTransferEngine,
    IPCWeightTransferInitInfo,
    IPCWeightTransferUpdateInfo,
)
from vllm.distributed.weight_transfer.packed_tensor import PackedBufferImporter

from vllm_ascend.distributed.weight_transfer.packed_tensor import (
    DEFAULT_PACKED_BUFFER_SIZE_BYTES,
    packed_ipc_producer,
    packed_npu_ipc_consumer,
)

if TYPE_CHECKING:
    from vllm.distributed.weight_transfer.base import (
        VLLMWeightSyncClient,
        WeightSource,
    )


@dataclass
class NPUIPCWeightTransferInitInfo(IPCWeightTransferInitInfo):
    """Initialization info for NPU IPC weight transfer backend.

    No initialization needed for NPU IPC.
    """


@lru_cache(maxsize=1)
def get_ip() -> str:
    try:
        # try to get ip from network interface
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception:  # noqa: BLE001
        # fallback to get ip from hostname
        return socket.gethostbyname(socket.gethostname())


@lru_cache(maxsize=1)
def npu_generate_uuid(logical_device: int | None = None) -> str:
    """Generate a unique identifier for the current process's physical NPU chip.

    Returns ``{host_ip}-{physical_chip_id}`` where ``host_ip`` is the local
    machine's IP address and ``physical_chip_id`` is derived from the current
    logical device index mapped through ``ASCEND_RT_VISIBLE_DEVICES``. The
    logical index is read from the current device when it is not provided.

    On Ascend NPU, ``torch.accelerator.current_device_index()`` returns the
    *logical* device index. When ``ASCEND_RT_VISIBLE_DEVICES`` is set, it
    maps logical indices to physical chip IDs (e.g., ``ASCEND_RT_VISIBLE_DEVICES=2,3``
    means logical device 0 → physical chip 2, logical device 1 → physical chip 3).
    If the env var is not set, the logical index is used directly as the
    physical chip ID (identity mapping).

    The result is cached because it is constant for the lifetime of the
    process. Both the trainer and inference worker processes co-located
    on the same physical NPU chip will produce the same UUID, which is
    required for NPU IPC handle matching.
    """
    if logical_device is None:
        logical_device = torch.accelerator.current_device_index()
    visible_devices = os.environ.get("ASCEND_RT_VISIBLE_DEVICES", None)
    if visible_devices:
        physical_device = int(visible_devices.split(",")[logical_device].strip())
    else:
        physical_device = logical_device
    return f"{get_ip()}-{physical_device}"


@dataclass
class NPUIPCTrainerInitInfo(IPCTrainerInitInfo):
    """NPU IPC trainer init info — overrides the backend key only.

    ``IPCTrainerInitInfo`` already provides ``packed`` and
    ``packed_buffer_size_bytes``; this subclass only rebinds the
    factory ``backend`` from ``"ipc"`` to ``"npu_ipc"``.
    """

    backend: ClassVar[str] = "npu_ipc"


@dataclass
class NPUIPCWeightTransferUpdateInfo(IPCWeightTransferUpdateInfo):  # type: ignore[no-redef]
    """NPU IPC variant — inherits all fields and validation from the CUDA IPC
    base class.  No overrides needed; the field types and ``__post_init__`` are
    identical."""


class PackedNpuBufferImporter(PackedBufferImporter):
    """Reuse upstream's IPC mapping lifetime with NPU reconstruction."""

    def rebuild(self, list_args: list) -> torch.Tensor:
        from torch_npu.multiprocessing.reductions import rebuild_npu_tensor

        key = tuple(list_args)
        if self._entry is not None and self._entry[0] == key:
            return self._entry[1]
        packed = rebuild_npu_tensor(*list_args)
        self._entry = (key, packed)
        return packed

    def rebuild_for_device(self, ipc_args: tuple[Any, ...], device_index: int) -> torch.Tensor:
        list_args = list(ipc_args)
        if len(list_args) <= 6:
            raise ValueError(f"NPU IPC rebuild arguments do not contain a device index: got {len(list_args)} values")
        list_args[6] = device_index
        return self.rebuild(list_args)


class NPUIPCWeightTransferEngine(IPCWeightTransferEngine):
    """
    Weight transfer engine using NPU IPC for communication between
    trainer and workers.

    This implementation uses Ascend NPU IPC to transfer weights from the
    trainer (rank 0) to all inference workers. IPC handles are used to
    share memory between processes on the same node.

    Requires ``torch_npu`` to be imported (which patches
    ``torch.multiprocessing.reductions.reduce_tensor`` to support
    NPU tensors via ``_share_npu_()`` / ``rebuild_npu_tensor``).
    """

    init_info_cls = NPUIPCWeightTransferInitInfo
    update_info_cls = NPUIPCWeightTransferUpdateInfo

    @staticmethod
    def trainer_send_weights(*args: Any, **kwargs: Any) -> None:
        raise NotImplementedError(
            "The static NPU IPC trainer path has been replaced by "
            "NPUIPCTrainerWeightTransferEngine. Build it via "
            "WeightTransferTrainerFactory.trainer_init("
            "NPUIPCTrainerInitInfo(...), client=..., "
            "source=...) and drive it with send_weights()."
        )

    def __init__(  # type: ignore[misc]
        self,
        config: WeightTransferConfig,
        vllm_config: VllmConfig,
        device: torch.device,
        model: torch.nn.Module,
    ) -> None:
        super().__init__(config, vllm_config, device, model)
        self._packed_importer = PackedNpuBufferImporter()

    def receive_weights(self, update_info: NPUIPCWeightTransferUpdateInfo) -> None:
        """Receive weights from the trainer via NPU IPC handles.

        Args:
            update_info: NPU IPC update info containing parameter names,
                dtypes, shapes, and IPC handles.
        """
        # Use the worker's assigned device rather than the ambient current
        # device: the receive path is no longer wrapped in
        # ``with torch.device(self.device)`` by the caller, so the current
        # device is not guaranteed to match ``self.device``. The IPC tensors
        # must be rebuilt on the device the model lives on.
        device_index = self.device.index
        physical_npu_id = npu_generate_uuid(device_index)

        if self.packed:
            if update_info.tensor_sizes is None:
                raise ValueError("`tensor_sizes` is required when packed=True")
            assert isinstance(update_info.ipc_handles, dict)
            weights = packed_npu_ipc_consumer(
                ipc_handle=update_info.ipc_handles,
                physical_npu_id=physical_npu_id,
                names=update_info.names,
                shapes=update_info.shapes,
                dtype_names=update_info.dtype_names,
                tensor_sizes=update_info.tensor_sizes,
                device_index=device_index,
                rebuild_func=self._packed_importer.rebuild_for_device,
                device=self.device,
            )
        else:
            # Lazy import: ``rebuild_npu_tensor`` lives in ``torch_npu`` and
            # must not be imported at module load time on non-NPU hosts.
            from torch_npu.multiprocessing.reductions import rebuild_npu_tensor

            assert isinstance(update_info.ipc_handles, list)
            weights = []
            for name, ipc_handle in zip(
                update_info.names,
                update_info.ipc_handles,
            ):
                if physical_npu_id not in ipc_handle:
                    raise ValueError(
                        f"IPC handle not found for NPU UUID {physical_npu_id}. "
                        f"Available UUIDs: {list(ipc_handle.keys())}. "
                        f"This may indicate that the trainer and worker are "
                        f"not co-located on the same physical NPU (node)."
                    )

                args = ipc_handle[physical_npu_id]
                list_args = list(args)
                # Index 6 is the device_index parameter in torch's
                # IPC handle tuple (rebuild_npu_tensor). Update it
                # to the current device since the logical index can
                # differ between sender and receiver.
                list_args[6] = device_index
                weight = rebuild_npu_tensor(*list_args)
                weights.append((name, weight))

        self.model.load_weights(weights)


class NPUIPCTrainerWeightTransferEngine(IPCTrainerWeightTransferEngine):
    """Trainer-side NPU IPC weight transfer engine.

    Mirrors upstream ``IPCTrainerWeightTransferEngine`` but swaps the
    GPU-side primitives for their NPU counterparts (``torch.npu`` instead
    of ``torch.cuda``, ``npu_generate_uuid`` instead of the GPU UUID, the
    NPU packed producer/consumer). HTTP/JSON transport is delegated to
    ``HTTPVLLMWeightSyncClient`` so the handles are serialized there
    rather than in this engine.
    """

    init_info_cls = NPUIPCTrainerInitInfo

    def __init__(  # type: ignore[misc]
        self,
        *,
        client: "VLLMWeightSyncClient",
        source: "WeightSource",
        is_sender: bool = True,
        packed: bool = False,
        packed_buffer_size_bytes: int = DEFAULT_PACKED_BUFFER_SIZE_BYTES,
    ) -> None:
        TrainerWeightTransferEngine.__init__(
            self,
            client=client,
            source=source,
            is_sender=is_sender,
        )
        self.packed = packed
        self.packed_buffer_size_bytes = packed_buffer_size_bytes
        self.device_index = torch.accelerator.current_device_index()
        self.npu_uuid = npu_generate_uuid(self.device_index)

    def _all_gather_and_merge_handles(
        self,
        handles: list[dict[str, tuple]],
    ) -> list[dict[str, tuple]]:
        """All-gather and merge IPC handle dicts across ranks in one call.

        Each rank contributes a list of ``{npu_uuid: ipc_args}`` dicts.
        A single all_gather_object collects every rank's full list, then
        the sender merges per-index so each dict maps every NPU UUID to
        its args. No-op (returns handles unchanged) when no distributed
        group exists.
        """
        if not torch.distributed.is_initialized() or torch.distributed.get_world_size() == 1:
            return handles

        world_size = torch.distributed.get_world_size()
        gathered: list[list[dict[str, tuple]] | None] = [None] * world_size
        torch.distributed.all_gather_object(gathered, handles)
        torch.distributed.barrier()
        torch.npu.synchronize()

        if self.is_sender:
            merged: list[dict[str, tuple]] = []
            for param_idx in range(len(handles)):
                m: dict[str, tuple] = {}
                for rank_handles in gathered:
                    if rank_handles is not None:
                        m.update(rank_handles[param_idx])
                merged.append(m)
            return merged
        return [{} for _ in handles]

    @staticmethod
    def _post_send_sync() -> None:
        """Barrier + synchronize after a send; no-op if single-NPU."""
        if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
            torch.distributed.barrier()
        torch.npu.synchronize()

    @property
    def gpu_uuid(self) -> str:
        """Supply the NPU UUID to the inherited device-independent sender."""
        return self.npu_uuid

    def _send_packed(self, source: "WeightSource") -> None:
        """Send weights in bounded-memory chunks (packed mode)."""
        for chunk in packed_ipc_producer(
            iterator=iter(source),
            npu_uuid=self.npu_uuid,
            post_iter_func=lambda item: item[1],
            buffer_size_bytes=self.packed_buffer_size_bytes,
        ):
            ipc_handle = self._all_gather_and_merge_handles([chunk.ipc_handle])[0]
            self._do_send(
                names=chunk.names,
                dtype_names=chunk.dtype_names,
                shapes=chunk.shapes,
                ipc_handles=ipc_handle,
                tensor_sizes=chunk.tensor_sizes,
            )
            # Per-chunk barrier: the producer reuses a single IPC buffer
            # across chunks. Without syncing every rank here, non-sender
            # ranks race ahead and overwrite their buffer while their
            # colocated worker is still reading the current chunk, silently
            # corrupting the transfer.
            self._post_send_sync()
