# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NPU IPC-based weight transfer engine using Ascend IPC for communication."""

import os
import socket
from contextlib import nullcontext
from dataclasses import dataclass
from functools import lru_cache
from importlib import import_module
from typing import TYPE_CHECKING, Any, ClassVar

import torch
from torch.multiprocessing.reductions import reduce_tensor
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

from vllm_ascend.distributed.weight_transfer.packed_tensor import (
    DEFAULT_PACKED_BUFFER_SIZE_BYTES,
    NPUPackedBufferImporter,
    packed_npu_ipc_consumer,
    packed_npu_ipc_producer,
)
from vllm_ascend.utils import vllm_version_is

# Unpacked tensors keep independent IPC exports, but share a control-plane RPC.
# A single larger parameter is sent alone. No full checkpoint is retained.
UNPACKED_BATCH_SIZE_BYTES = 256 * 2**20

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.config.weight_transfer import WeightTransferConfig
    from vllm.distributed.weight_transfer.base import (
        VLLMWeightSyncClient,
        WeightSource,
    )


@dataclass
class NPUIPCWeightTransferInitInfo(IPCWeightTransferInitInfo):
    """Worker-side NPU IPC wire parameters, shared with upstream IPC."""


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
class NPUIPCWeightTransferUpdateInfo(IPCWeightTransferUpdateInfo):
    """NPU IPC variant — inherits all fields and validation from the CUDA IPC
    base class.  No overrides needed; the field types and ``__post_init__`` are
    identical."""


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

    def __init__(
        self,
        config: "WeightTransferConfig",
        vllm_config: "VllmConfig",
        device: torch.device,
        model: torch.nn.Module,
    ) -> None:
        super().__init__(config, vllm_config, device, model)
        self._packed_importer = NPUPackedBufferImporter()

    def start_weight_update(self) -> None:
        """Discard an interrupted packed import before starting a full update."""
        self._packed_importer.close()
        super().start_weight_update()

    def shutdown(self) -> None:
        self._packed_importer.close()

    def update_weights(self, update_info: dict[str, Any]) -> None:
        """Bind the assigned device for rebuilding, loading and synchronization."""
        with torch.npu.device(self.device):
            super().update_weights(update_info)

    def finish_weight_update(self) -> None:
        """Release packed mappings even when reload finalization fails."""
        try:
            if vllm_version_is("0.33.0"):
                super().finish_weight_update()
            else:
                from vllm.model_executor.model_loader.reload import finish_reload

                with torch.npu.device(self.device):
                    finish_reload(self.model, self.model_config, self.device)
        finally:
            self._packed_importer.close()

    @staticmethod
    def trainer_send_weights(*args: Any, **kwargs: Any) -> None:
        raise NotImplementedError(
            "The static NPU IPC trainer path has been replaced by "
            "NPUIPCTrainerWeightTransferEngine. Build it via "
            "WeightTransferTrainerFactory.trainer_init("
            "NPUIPCTrainerInitInfo(...), client=..., "
            "source=...) and drive it with send_weights()."
        )

    def receive_weights(self, update_info: IPCWeightTransferUpdateInfo) -> None:
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
                importer=self._packed_importer,
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
                if not vllm_version_is("0.33.0"):
                    # The main sender releases each export after update_weights
                    # replies. A layerwise loader may retain its input until
                    # later parameters arrive, so give it owned local storage.
                    # The inherited update wrapper synchronizes these copies
                    # before replying and releasing the producer's allocation.
                    weight = weight.clone(memory_format=torch.contiguous_format)
                weights.append((name, weight))

        mtp_check = nullcontext()
        if vllm_version_is("0.33.0"):
            # The pinned main tree used by mypy no longer contains this
            # release-only module. Resolve it only in the release runtime.
            mtp_validation = import_module("vllm.model_executor.model_loader.mtp_validation")
            mtp_check = mtp_validation.disable_mtp_completeness_check()

        with mtp_check:
            self.model.load_weights(weights)


class NPUIPCTrainerWeightTransferEngine(IPCTrainerWeightTransferEngine):
    """Trainer-side NPU IPC weight transfer engine.

    Inherits the handshake, send lifecycle and payload construction. NPU
    initialization, collectives, packing and cleanup are local. On main the
    unpacked path sends bounded batches of owned exports to limit trainer HBM.
    """

    init_info_cls = NPUIPCTrainerInitInfo

    def __init__(
        self,
        *,
        client: "VLLMWeightSyncClient",
        source: "WeightSource",
        is_sender: bool = True,
        packed: bool = False,
        packed_buffer_size_bytes: int = DEFAULT_PACKED_BUFFER_SIZE_BYTES,
    ) -> None:
        # IPCTrainerWeightTransferEngine.__init__ queries CUDA properties.
        # Initialize its device-independent base, then supply NPU identity.
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
        # Upstream's unpacked sender treats gpu_uuid as an opaque wire key.
        self.gpu_uuid = self.npu_uuid

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
        """Synchronize each send and keep all trainer ranks in step."""
        if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
            torch.distributed.barrier()
        torch.npu.synchronize()

    def _send_packed(self, source: "WeightSource") -> None:
        """Send weights in bounded-memory chunks (packed mode)."""
        # Both supported versions drive the WeightSource iter_full protocol.
        iterator = source.iter_full()
        for chunk in packed_npu_ipc_producer(
            iterator=iterator,
            npu_uuid=self.npu_uuid,
            post_iter_func=lambda item: item[1],
            buffer_size_bytes=self.packed_buffer_size_bytes,
        ):
            ipc_handle = self._all_gather_and_merge_handles([chunk["ipc_handle"]])[0]
            self._do_send(
                names=chunk["names"],
                dtype_names=chunk["dtype_names"],
                shapes=chunk["shapes"],
                ipc_handles=ipc_handle,
                tensor_sizes=chunk["tensor_sizes"],
            )
            # Per-chunk barrier: the producer reuses a single IPC buffer
            # across chunks. Without syncing every rank here, non-sender
            # ranks race ahead and overwrite their buffer while their
            # colocated worker is still reading the current chunk, silently
            # corrupting the transfer.
            self._post_send_sync()

    def _send_unpacked(self, source: "WeightSource") -> list[torch.Tensor]:
        """Keep a bounded batch of exports until its update is consumed.

        Holding an entire full checkpoint on each trainer NPU can exceed HBM
        when it shares a chip with a TP worker. The worker synchronizes each
        update before replying; all trainer ranks wait before releasing a
        batch. The receiver copies imports to owned storage for loaders that
        retain their inputs across update calls.
        """
        if vllm_version_is("0.33.0"):
            return super()._send_unpacked(source)
        pending: list[tuple[str, torch.Tensor]] = []
        pending_bytes = 0
        for name, tensor in source.iter_full():
            size = tensor.numel() * tensor.element_size()
            if pending and pending_bytes + size > UNPACKED_BATCH_SIZE_BYTES:
                self._send_unpacked_batch(pending)
                pending.clear()
                pending_bytes = 0
            # A source may reuse its buffer on the next iteration. Keep an
            # independent export alive until every worker has consumed it.
            pending.append((name, tensor.detach().clone(memory_format=torch.contiguous_format)))
            pending_bytes += size
            if pending_bytes >= UNPACKED_BATCH_SIZE_BYTES:
                self._send_unpacked_batch(pending)
                pending.clear()
                pending_bytes = 0
        if pending:
            self._send_unpacked_batch(pending)
        return []

    def _send_unpacked_batch(self, weights: list[tuple[str, torch.Tensor]]) -> None:
        """Acknowledge a batch before its producer-owned storage is released."""
        torch.npu.current_stream().synchronize()
        handles = self._all_gather_and_merge_handles(
            [{self.npu_uuid: reduce_tensor(weight)[1]} for _, weight in weights]
        )
        self._do_send(
            names=[name for name, _ in weights],
            dtype_names=[str(weight.dtype).removeprefix("torch.") for _, weight in weights],
            shapes=[list(weight.shape) for _, weight in weights],
            ipc_handles=handles,
        )
        self._post_send_sync()
