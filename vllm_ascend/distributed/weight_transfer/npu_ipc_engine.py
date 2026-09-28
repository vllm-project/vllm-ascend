# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NPU IPC-based weight transfer engine using Ascend IPC for communication."""

import warnings
from collections.abc import Callable
from dataclasses import asdict, dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, ClassVar

import torch
from torch.multiprocessing.reductions import reduce_tensor
from vllm.config import VllmConfig
from vllm.config.weight_transfer import WeightTransferConfig
from vllm.distributed.weight_transfer.base import (
    TrainerWeightTransferEngine,
    WeightTransferEngine,
    WeightTransferInitInfo,
)
from vllm.distributed.weight_transfer.ipc_engine import (
    IPCTrainerInitInfo,
    IPCTrainerWeightTransferEngine,
    IPCWeightTransferUpdateInfo,
)

from vllm_ascend.distributed.weight_transfer.npu_ipc_utils import (
    NpuPackedBufferImporter,
    get_ip,  # noqa: F401 - preserved as a compatibility import
    npu_generate_uuid,
    rewrite_rebuild_device,
)
from vllm_ascend.distributed.weight_transfer.packed_tensor import (
    DEFAULT_PACKED_BUFFER_SIZE_BYTES,
    packed_npu_ipc_consumer,
    packed_npu_ipc_producer,
)

if TYPE_CHECKING:
    from vllm.distributed.weight_transfer.base import (
        VLLMWeightSyncClient,
        WeightSource,
    )


class _NPUIPCTrainerState(Enum):
    READY = auto()
    FAILED = auto()
    CLOSED = auto()


@dataclass
class NPUIPCWeightTransferInitInfo(WeightTransferInitInfo):
    """Initialization info for NPU IPC weight transfer backend.

    No initialization needed for NPU IPC.
    """

    packed: bool = False


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


class NPUIPCWeightTransferEngine(  # type: ignore[no-redef]
    WeightTransferEngine[NPUIPCWeightTransferInitInfo, NPUIPCWeightTransferUpdateInfo],
):
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
        # Set from the trainer-supplied init info at the handshake; defaults
        # are only for the (unreachable) receive-before-init case.
        self.packed = False
        self._packed_importer = NpuPackedBufferImporter()

    def init_transfer_engine(self, init_info: NPUIPCWeightTransferInitInfo) -> None:
        """Record the trainer-supplied wire params so the worker decodes
        exactly as the trainer encoded."""
        self.packed = init_info.packed
        importer = getattr(self, "_packed_importer", None)
        if importer is not None:
            importer.close()

    def start_weight_update(self) -> None:
        from vllm.model_executor.model_loader.reload import (
            initialize_layerwise_reload,
        )

        initialize_layerwise_reload(self.model)

    def finish_weight_update(self) -> None:
        from vllm.model_executor.model_loader.reload import (
            finalize_layerwise_reload,
        )

        finalize_layerwise_reload(self.model, self.model_config)
        importer = getattr(self, "_packed_importer", None)
        if importer is not None:
            importer.close()

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
        if device_index is None:
            raise ValueError(f"NPU worker device must have an explicit index, got {self.device}")
        # Fully initialized workers use the explicit logical device.  Keeping
        # the no-argument fallback preserves compatibility with old CPU-only
        # unit doubles that construct the engine via ``object.__new__``.
        importer = getattr(self, "_packed_importer", None)
        physical_npu_id = npu_generate_uuid(device_index) if importer is not None else npu_generate_uuid()

        with torch.npu.device(self.device):
            if self.packed:
                if update_info.tensor_sizes is None:
                    raise ValueError("`tensor_sizes` is required when packed=True")
                if not isinstance(update_info.ipc_handles, dict):
                    raise ValueError("packed NPU IPC update requires one handle dictionary")
                consumer_kwargs = dict(
                    ipc_handle=update_info.ipc_handles,
                    physical_npu_id=physical_npu_id,
                    names=update_info.names,
                    shapes=update_info.shapes,
                    dtype_names=update_info.dtype_names,
                    tensor_sizes=update_info.tensor_sizes,
                    device_index=device_index,
                )
                if importer is not None:
                    consumer_kwargs["importer"] = importer
                    consumer_kwargs["device"] = self.device
                weights = packed_npu_ipc_consumer(**consumer_kwargs)
            else:
                # Lazy import: ``rebuild_npu_tensor`` lives in ``torch_npu`` and
                # must not be imported at module load time on non-NPU hosts.
                from torch_npu.multiprocessing.reductions import rebuild_npu_tensor

                if not isinstance(update_info.ipc_handles, list):
                    raise ValueError("unpacked NPU IPC update requires one handle per parameter")
                weights = []
                for name, ipc_handle in zip(update_info.names, update_info.ipc_handles):
                    if physical_npu_id not in ipc_handle:
                        raise ValueError(
                            f"IPC handle not found for NPU UUID {physical_npu_id}. "
                            f"Available UUIDs: {list(ipc_handle.keys())}. "
                            f"This may indicate that the trainer and worker are "
                            f"not co-located on the same physical NPU (node)."
                        )

                    args = rewrite_rebuild_device(ipc_handle[physical_npu_id], device_index)
                    weight = rebuild_npu_tensor(*args)
                    weights.append((name, weight))

            from vllm.model_executor.model_loader.mtp_validation import (
                disable_mtp_completeness_check,
            )

            with disable_mtp_completeness_check():
                self.model.load_weights(weights)

    def update_weights(self, update_info: dict[str, Any]) -> None:
        """Keep the worker device current through the base synchronize call."""
        with torch.npu.device(self.device):
            super().update_weights(update_info)

    def shutdown(self) -> None:
        importer = getattr(self, "_packed_importer", None)
        if importer is not None:
            importer.close()


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
        self.device = torch.device("npu", self.device_index)
        self.npu_uuid = npu_generate_uuid(self.device_index)
        self._state = _NPUIPCTrainerState.READY

    @classmethod
    def trainer_init(
        cls,
        init_info: NPUIPCTrainerInitInfo,
        *,
        client: "VLLMWeightSyncClient",
        source: "WeightSource | None" = None,
    ) -> "NPUIPCTrainerWeightTransferEngine":
        if source is None:
            raise ValueError("NPU IPC trainer weight transfer requires a WeightSource.")
        engine = cls(
            client=client,
            source=source,
            is_sender=init_info.is_sender,
            packed=init_info.packed,
            packed_buffer_size_bytes=init_info.packed_buffer_size_bytes,
        )
        # IPC needs no data-plane rendezvous. The sender ships the must-agree
        # ``packed`` flag so the worker decodes exactly as this trainer encodes.
        engine._run_sender_rpc(
            "initialize",
            lambda: engine.client.init_weight_transfer_engine({"packed": init_info.packed}),
        )
        return engine

    def send_weights(self) -> None:
        self._ensure_ready()
        weight_refs: list[torch.Tensor] | None = None
        try:
            self._run_sender_rpc("start", self.client.start_weight_update)
            weight_refs = self._send(self.source)
            self._run_sender_rpc("finish", self.client.finish_weight_update)
            self._post_send_sync()
        except Exception:
            self._state = _NPUIPCTrainerState.FAILED
            raise
        finally:
            del weight_refs

    def _ensure_ready(self) -> None:
        state = self._state
        if state is _NPUIPCTrainerState.FAILED:
            raise RuntimeError(
                "NPU IPC trainer engine is failed and cannot be reused; "
                "reinitialize the worker and trainer engine before retrying."
            )
        if state is _NPUIPCTrainerState.CLOSED:
            raise RuntimeError("NPU IPC trainer engine is closed.")

    def _run_sender_rpc(self, operation: str, call: Callable[[], Any]) -> None:
        """Run one sender RPC and propagate its result to every trainer rank."""
        sender_error: Exception | None = None
        status: dict[str, Any] = {
            "is_sender": self.is_sender,
            "operation": operation,
            "ok": True,
            "error_type": None,
            "error_message": None,
        }
        if self.is_sender:
            try:
                call()
            except Exception as exc:
                sender_error = exc
                status.update(
                    ok=False,
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                )

        if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
            gathered: list[dict[str, Any] | None] = [None] * torch.distributed.get_world_size()
            torch.distributed.all_gather_object(gathered, status)
            sender_statuses = [item for item in gathered if isinstance(item, dict) and item.get("is_sender") is True]
            if len(sender_statuses) != 1:
                self._state = _NPUIPCTrainerState.FAILED
                raise RuntimeError(
                    "NPU IPC trainer ranks must contain exactly one sender; "
                    f"found {len(sender_statuses)} during {operation}."
                )
            status = sender_statuses[0]

        if not status["ok"]:
            self._state = _NPUIPCTrainerState.FAILED
            error = RuntimeError(
                f"NPU IPC {operation} RPC failed on the sender: {status['error_type']}: {status['error_message']}"
            )
            if sender_error is not None:
                raise error from sender_error
            raise error

    def _run_rank_preparation(
        self,
        operation: str,
        call: Callable[[], Any],
    ) -> tuple[bool, Any | None]:
        """Run local preparation and exchange its outcome before data collectives.

        Every rank calls this at the same chunk boundary.  A local source,
        materialization, or IPC-export error is therefore reported before any
        peer enters the handle/schema collective for that chunk.
        """
        local_error: Exception | None = None
        value: Any | None = None
        outcome = "ready"
        try:
            value = call()
        except StopIteration:
            outcome = "done"
        except Exception as exc:
            local_error = exc
            outcome = "error"

        status = {
            "operation": operation,
            "outcome": outcome,
            "error_type": type(local_error).__name__ if local_error else None,
            "error_message": str(local_error) if local_error else None,
        }
        gathered: list[dict[str, Any] | None] = [status]
        if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
            gathered = [None] * torch.distributed.get_world_size()
            torch.distributed.all_gather_object(gathered, status)

        for rank, rank_status in enumerate(gathered):
            if not isinstance(rank_status, dict):
                self._state = _NPUIPCTrainerState.FAILED
                raise RuntimeError(
                    f"NPU IPC {operation} received an invalid preparation status from trainer rank {rank}."
                )
            if rank_status.get("operation") != operation:
                self._state = _NPUIPCTrainerState.FAILED
                raise RuntimeError(
                    f"NPU IPC preparation order mismatch: local operation "
                    f"{operation!r}, trainer rank {rank} reported "
                    f"{rank_status.get('operation')!r}."
                )
            if rank_status.get("outcome") == "error":
                self._state = _NPUIPCTrainerState.FAILED
                error = RuntimeError(
                    f"NPU IPC {operation} failed on trainer rank {rank}: "
                    f"{rank_status.get('error_type')}: "
                    f"{rank_status.get('error_message')}"
                )
                if local_error is not None and rank_status is status:
                    raise error from local_error
                raise error

        outcomes = {rank_status.get("outcome") for rank_status in gathered if rank_status}
        if len(outcomes) != 1:
            self._state = _NPUIPCTrainerState.FAILED
            raise RuntimeError(f"NPU IPC {operation} completed inconsistently across trainer ranks: {sorted(outcomes)}")
        return outcome == "done", value

    def shutdown(self) -> None:
        """Make this trainer instance permanently unavailable for new rounds."""
        self._state = _NPUIPCTrainerState.CLOSED

    def _send(self, source: "WeightSource") -> list[torch.Tensor] | None:
        if self.packed:
            self._send_packed(source)
            return None
        return self._send_unpacked(source)

    def _all_gather_and_merge_handles(
        self,
        handles: list[dict[str, tuple]],
        *,
        schema: list[tuple[str, str, tuple[int, ...], int]],
        chunk_index: int,
        done: bool = False,
    ) -> list[dict[str, tuple]]:
        """Validate a chunk schema and merge handles across trainer ranks.

        The data/end marker, chunk index, names, dtypes, shapes, and byte sizes
        must match on every rank. The explicit end marker makes a different
        number of chunks fail in the same collective instead of leaving one
        rank waiting forever for the next collective.
        """
        if not torch.distributed.is_initialized() or torch.distributed.get_world_size() == 1:
            return handles

        world_size = torch.distributed.get_world_size()
        payload = {
            "chunk_index": chunk_index,
            "done": done,
            "schema": schema,
            "handles": handles,
        }
        gathered: list[dict[str, Any] | None] = [None] * world_size
        torch.distributed.all_gather_object(gathered, payload)

        expected_header = (chunk_index, done, schema)
        for rank, rank_payload in enumerate(gathered):
            if not isinstance(rank_payload, dict):
                raise ValueError(f"NPU IPC rank {rank} returned an invalid chunk payload")
            header = (
                rank_payload.get("chunk_index"),
                rank_payload.get("done"),
                rank_payload.get("schema"),
            )
            if header != expected_header:
                raise ValueError(
                    "NPU IPC trainer chunk schema mismatch at "
                    f"chunk {chunk_index}: rank {rank} reported "
                    f"index={header[0]}, done={header[1]}, schema={header[2]!r}; "
                    f"local done={done}, schema={schema!r}"
                )
            rank_handles = rank_payload.get("handles")
            if not isinstance(rank_handles, list) or len(rank_handles) != len(handles):
                raise ValueError(
                    f"NPU IPC handle lists have different lengths across trainer ranks at chunk {chunk_index}"
                )

        torch.distributed.barrier()
        torch.npu.synchronize()

        if self.is_sender:
            merged: list[dict[str, tuple]] = []
            for param_idx in range(len(handles)):
                m: dict[str, tuple] = {}
                for rank_payload in gathered:
                    if rank_payload is not None:
                        m.update(rank_payload["handles"][param_idx])
                if not m:
                    raise ValueError(f"NPU IPC handle {param_idx} was empty after all-gather")
                merged.append(m)
            return merged
        return [{} for _ in handles]

    @staticmethod
    def _post_send_sync() -> None:
        """Barrier + synchronize after a send; no-op if single-NPU."""
        if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
            torch.distributed.barrier()
        torch.npu.synchronize()

    def _send_unpacked(self, source: "WeightSource") -> list[torch.Tensor]:
        """Send unpacked weights in bounded chunks within one update round.

        ``packed_buffer_size_bytes`` is also the unpacked chunk budget. A
        single tensor larger than the budget is sent alone, so peak staging is
        bounded by the budget plus the largest materialized source tensor.
        Every chunk waits for all colocated consumers before its strong tensor
        references are released.
        """

        def chunks() -> Any:
            iterator = iter(source)
            pending: tuple[str, torch.Tensor] | None = None
            exhausted = False
            while pending is not None or not exhausted:
                names: list[str] = []
                dtype_names: list[str] = []
                shapes: list[list[int]] = []
                ipc_handles: list[dict[str, tuple]] = []
                weight_refs: list[torch.Tensor] = []
                chunk_bytes = 0
                while True:
                    if pending is not None:
                        name, tensor = pending
                        pending = None
                    else:
                        try:
                            name, tensor = next(iterator)
                        except StopIteration:
                            exhausted = True
                            break

                    tensor_bytes = tensor.numel() * tensor.element_size()
                    if names and chunk_bytes + tensor_bytes > self.packed_buffer_size_bytes:
                        pending = (name, tensor)
                        break
                    if tensor_bytes > self.packed_buffer_size_bytes:
                        warnings.warn(
                            f"Tensor {name!r} has size {tensor_bytes} bytes, which "
                            f"exceeds the unpacked chunk budget "
                            f"{self.packed_buffer_size_bytes}; sending it as one chunk.",
                            stacklevel=2,
                        )
                    weight = tensor.detach().contiguous()
                    _, ipc_args = reduce_tensor(weight)
                    names.append(name)
                    dtype_names.append(str(tensor.dtype).split(".")[-1])
                    shapes.append(list(tensor.shape))
                    ipc_handles.append({self.npu_uuid: ipc_args})
                    weight_refs.append(weight)
                    chunk_bytes += tensor_bytes

                if names:
                    yield names, dtype_names, shapes, ipc_handles, weight_refs
                    # Drop the generator frame's aliases before materializing
                    # the next chunk.  The caller has already synchronized all
                    # consumers before it resumes this generator.
                    del names, dtype_names, shapes, ipc_handles, weight_refs

        chunk_iterator = iter(chunks())
        chunk_index = 0
        while True:
            done, prepared = self._run_rank_preparation(
                f"unpacked chunk {chunk_index} preparation",
                lambda: next(chunk_iterator),
            )
            if done:
                break
            assert prepared is not None
            names, dtype_names, shapes, ipc_handles, weight_refs = prepared
            schema = [
                (
                    name,
                    dtype_name,
                    tuple(shape),
                    weight.numel() * weight.element_size(),
                )
                for name, dtype_name, shape, weight in zip(names, dtype_names, shapes, weight_refs)
            ]
            merged_handles = self._all_gather_and_merge_handles(
                ipc_handles,
                schema=schema,
                chunk_index=chunk_index,
            )
            self._do_send(
                names=names,
                dtype_names=dtype_names,
                shapes=shapes,
                ipc_handles=merged_handles,
            )
            self._post_send_sync()
            # Assignment evaluates the next ``_run_rank_preparation`` call
            # before replacing these locals.  Delete every completed-chunk
            # alias now so the next lazy source materialization cannot overlap
            # with a full previous staging chunk.
            del prepared, names, dtype_names, shapes
            del ipc_handles, weight_refs, schema, merged_handles
            chunk_index += 1
        return []

    def _send_packed(self, source: "WeightSource") -> None:
        """Send weights in bounded-memory chunks (packed mode)."""
        post_iter_func: Callable = lambda item: item[1]

        def chunks() -> Any:
            # Keep ``iter(source)`` lazy so ordinary __iter__ initialization
            # failures occur inside the coordinated preparation boundary.
            yield from packed_npu_ipc_producer(
                iterator=iter(source),
                npu_uuid=self.npu_uuid,
                post_iter_func=post_iter_func,
                buffer_size_bytes=self.packed_buffer_size_bytes,
                device=self.device,
            )

        chunk_iterator = iter(chunks())
        chunk_index = 0
        while True:
            done, chunk = self._run_rank_preparation(
                f"packed chunk {chunk_index} preparation",
                lambda: next(chunk_iterator),
            )
            if done:
                break
            assert chunk is not None
            schema = [
                (name, dtype_name, tuple(shape), tensor_size)
                for name, dtype_name, shape, tensor_size in zip(
                    chunk["names"],
                    chunk["dtype_names"],
                    chunk["shapes"],
                    chunk["tensor_sizes"],
                )
            ]
            ipc_handle = self._all_gather_and_merge_handles(
                [chunk["ipc_handle"]],
                schema=schema,
                chunk_index=chunk_index,
            )[0]
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
            chunk_index += 1

    def _do_send(
        self,
        names: list[str],
        dtype_names: list[str],
        shapes: list[list[int]],
        ipc_handles: list[dict[str, tuple]] | dict[str, tuple],
        tensor_sizes: list[int] | None = None,
    ) -> None:
        """Build and send one update while every rank observes the result.

        Only the sender constructs and transmits the payload, but non-sender
        ranks must still enter ``_run_sender_rpc`` so an RPC failure reaches
        the entire trainer group before any rank enters the next barrier.
        """

        def send_update() -> None:
            update_fields: dict[str, Any] = {
                "names": names,
                "dtype_names": dtype_names,
                "shapes": shapes,
                "ipc_handles": ipc_handles,
            }
            if tensor_sizes is not None:
                update_fields["tensor_sizes"] = tensor_sizes
            update_info = NPUIPCWeightTransferUpdateInfo(**update_fields)
            self.client.update_weights(asdict(update_info))

        self._run_sender_rpc("update", send_update)

