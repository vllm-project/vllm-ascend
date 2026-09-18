# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HCCL-based weight transfer engine."""

from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, ClassVar

import torch
from typing_extensions import Self

if TYPE_CHECKING:
    from vllm.distributed.weight_transfer.base import VLLMWeightSyncClient, WeightSource

    from vllm_ascend.distributed.device_communicators.pyhccl import PyHcclCommunicator

from vllm.config import VllmConfig
from vllm.config.weight_transfer import WeightTransferConfig
from vllm.distributed.weight_transfer.base import (
    ParamMeta,
    TrainerInitInfo,
    TrainerWeightTransferEngine,
    WeightTransferEngine,
    WeightTransferUpdateInfo,
)

# HCCLWeightTransferInitInfo is re-exported here for convenience; its canonical
# home is hccl_common, shared with the initialization helpers.
from vllm_ascend.distributed.weight_transfer.hccl_common import (
    HCCLWeightTransferInitInfo,
    worker_init_process_group,
)
from vllm_ascend.distributed.weight_transfer.hccl_common import (
    trainer_init as open_trainer_endpoint,
)
from vllm_ascend.distributed.weight_transfer.packed_tensor import (
    DEFAULT_PACKED_BUFFER_SIZE_BYTES,
    DEFAULT_PACKED_NUM_BUFFERS,
    packed_broadcast_consumer,
    packed_broadcast_producer,
)

__all__ = [
    "HCCLWeightTransferInitInfo",
    "HCCLTrainerInitInfo",
    "HCCLWeightTransferUpdateInfo",
    "HCCLWeightTransferEngine",
    "HCCLTrainerWeightTransferEngine",
]


@dataclass
class HCCLWeightTransferUpdateInfo(WeightTransferUpdateInfo):
    """Per-round update info for the HCCL weight transfer backend.

    Whether the transfer is packed (and the buffer geometry) is a must-agree
    wire param carried on the init info (`HCCLTrainerInitInfo` /
    `HCCLWeightTransferInitInfo`), not here; this carries only the per-round
    parameter metadata. Keeping the wire format out of the per-round payload
    removes the chance of the sender and receiver disagreeing about it.
    """

    names: list[str]
    """Names of the parameters to transfer (e.g. ``model.layers.0.weight``)."""
    dtype_names: list[str]
    """Torch dtype names (e.g. ``bfloat16``, ``float32``) for each parameter."""
    shapes: list[list[int]]
    """Shapes of each parameter as integer lists."""

    def __post_init__(self):
        """Validate that all lists have the same length."""
        num_params = len(self.names)
        if len(self.dtype_names) != num_params:
            raise ValueError(
                f"`dtype_names` should be of the same size as `names`: "
                f"got {len(self.dtype_names)} and {len(self.names)}"
            )
        if len(self.shapes) != num_params:
            raise ValueError(
                f"`shapes` should be of the same size as `names`: got {len(self.shapes)} and {len(self.names)}"
            )


class HCCLWeightTransferEngine(WeightTransferEngine[HCCLWeightTransferInitInfo, HCCLWeightTransferUpdateInfo]):
    """
    Weight transfer engine using HCCL for communication between trainer and workers.

    This implementation uses HCCL broadcast operations to transfer weights from
    the trainer (rank 0) to all inference workers in a process group.
    """

    # Define backend-specific dataclass types
    init_info_cls = HCCLWeightTransferInitInfo
    update_info_cls = HCCLWeightTransferUpdateInfo

    def __init__(  # type: ignore[misc]
        self,
        config: WeightTransferConfig,
        vllm_config: VllmConfig,
        device: torch.device,
        model: torch.nn.Module,
    ) -> None:
        super().__init__(config, vllm_config, device, model)
        self.model_update_group: PyHcclCommunicator | None = None  # type: ignore[no-redef]
        # Set from the trainer-supplied init info at the handshake; defaults are
        # only for the (unreachable) receive-before-init case.
        self.packed = False
        self.packed_buffer_size_bytes = DEFAULT_PACKED_BUFFER_SIZE_BYTES
        self.packed_num_buffers = DEFAULT_PACKED_NUM_BUFFERS

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

    def init_transfer_engine(self, init_info: HCCLWeightTransferInitInfo) -> None:
        """
        Initialize the HCCL process group with the trainer and record the
        trainer-supplied wire params so the worker decodes exactly as the
        trainer encodes.

        Args:
            init_info: HCCL initialization info containing master address, port,
                      rank offset, world size, and the packed wire params
        """
        self.packed = init_info.packed
        self.packed_buffer_size_bytes = init_info.packed_buffer_size_bytes
        self.packed_num_buffers = init_info.packed_num_buffers
        self.model_update_group = worker_init_process_group(init_info, self.parallel_config)

    def receive_weights(
        self,
        update_info: HCCLWeightTransferUpdateInfo,
    ) -> None:
        """
        Receive weights from trainer via HCCL broadcast and load them incrementally.

        Whether to use packed broadcasting (and the buffer geometry) is read from
        `self.packed` / `self.packed_*`, set at the init handshake from the
        trainer's init info, so it is guaranteed to match how the trainer encoded.

        Args:
            update_info: HCCL update info containing parameter names, dtypes, and
                        shapes
        """
        if self.model_update_group is None:
            raise RuntimeError("HCCL weight transfer not initialized. Call init_transfer_engine() first.")

        from vllm.model_executor.model_loader.mtp_validation import (
            disable_mtp_completeness_check,
        )

        # The transfer loads weights in batches (packed) or one parameter at a
        # time (unpacked), so a model whose loader validates that every expected
        # parameter arrived would reject a single batch; the transaction as a
        # whole is what must be complete.
        with disable_mtp_completeness_check():
            if self.packed:
                # Build iterator of (name, (shape, dtype)) from update_info
                def state_dict_info_iterator():
                    for name, dtype_name, shape in zip(update_info.names, update_info.dtype_names, update_info.shapes):
                        dtype = getattr(torch, dtype_name)
                        yield (name, (shape, dtype))

                packed_broadcast_consumer(
                    iterator=state_dict_info_iterator(),
                    group=self.model_update_group,
                    src=0,
                    post_unpack_func=self.model.load_weights,
                    buffer_size_bytes=self.packed_buffer_size_bytes,
                    num_buffers=self.packed_num_buffers,
                    device=self.device,
                )
            else:
                # Use simple one-by-one broadcasting. Allocate on the worker's own
                # device and drive its stream: the receive path is not wrapped in
                # a device context by the caller, so the ambient current device is
                # not guaranteed to be this worker's device.
                stream = torch.npu.current_stream(self.device)
                for name, dtype_name, shape in zip(update_info.names, update_info.dtype_names, update_info.shapes):
                    dtype = getattr(torch, dtype_name)
                    weight = torch.empty(shape, dtype=dtype, device=self.device)
                    self.model_update_group.broadcast(weight, src=0, stream=stream)
                    self.model.load_weights([(name, weight)])
                    del weight

    def shutdown(self) -> None:
        if self.model_update_group is not None:
            # Clean up the communicator by removing the reference
            self.model_update_group = None


@dataclass
class HCCLTrainerInitInfo(TrainerInitInfo):
    """Trainer-side init info for the HCCL weight transfer backend.

    The sender opens its endpoint as HCCL rank 0, so it needs no
    ``rank_offset``; the inference workers sit at rank 1. ``world_size`` is the
    full trainer + worker HCCL group size. ``rank`` (from ``TrainerInitInfo``)
    identifies this trainer process, and rank 0 is the sender.

    ``packed`` / buffer sizes are the transfer's wire params: the sender
    propagates them to the worker at ``trainer_init`` so the two sides cannot
    disagree. ``backend`` is the factory dispatch key.
    """

    backend: ClassVar[str] = "hccl"

    master_address: str
    master_port: int
    world_size: int
    packed: bool = False
    packed_buffer_size_bytes: int = DEFAULT_PACKED_BUFFER_SIZE_BYTES
    packed_num_buffers: int = DEFAULT_PACKED_NUM_BUFFERS


class HCCLTrainerWeightTransferEngine(TrainerWeightTransferEngine[HCCLTrainerInitInfo]):
    """Trainer-side HCCL weight transfer engine.

    Mirrors upstream's ``NCCLTrainerWeightTransferEngine`` (see
    ``vllm/distributed/weight_transfer/nccl_engine.py``) on top of the Ascend
    ``PyHcclCommunicator``.

    On the sender (rank 0) it holds the HCCL communicator and drives the full
    update round trip: it starts the inference-side update, runs the
    trainer-side broadcast concurrently with it (both rendezvous inside the
    same HCCL calls) and finishes the update. Non-sender trainer ranks hold no
    communicator: they only iterate the source to stay inside the trainer-side
    collective (e.g. FSDP ``full_tensor()``) and skip the client RPCs and the
    broadcast.

    ``packed`` / buffer sizes come from ``HCCLTrainerInitInfo``; the sender
    propagates them to the worker at ``trainer_init``, so per-round payloads
    carry only parameter metadata.
    """

    init_info_cls = HCCLTrainerInitInfo

    def __init__(
        self,
        *,
        client: "VLLMWeightSyncClient",
        source: "WeightSource | None" = None,
        is_sender: bool = True,
        packed: bool = False,
        packed_buffer_size_bytes: int = DEFAULT_PACKED_BUFFER_SIZE_BYTES,
        packed_num_buffers: int = DEFAULT_PACKED_NUM_BUFFERS,
    ) -> None:
        super().__init__(client=client, source=source, is_sender=is_sender)
        self.packed = packed
        self.packed_buffer_size_bytes = packed_buffer_size_bytes
        self.packed_num_buffers = packed_num_buffers
        self.model_update_group: PyHcclCommunicator | None = None

    @classmethod
    def trainer_init(
        cls,
        init_info: HCCLTrainerInitInfo,
        *,
        client: "VLLMWeightSyncClient",
        source: "WeightSource | None" = None,
    ) -> Self:
        if source is None:
            raise ValueError("HCCL trainer weight transfer requires a WeightSource.")
        engine = cls(
            client=client,
            source=source,
            is_sender=init_info.is_sender,
            packed=init_info.packed,
            packed_buffer_size_bytes=init_info.packed_buffer_size_bytes,
            packed_num_buffers=init_info.packed_num_buffers,
        )
        if not engine.is_sender:
            # Non-sender trainer ranks aren't part of the transfer HCCL group and
            # don't drive the inference side; they only participate in the
            # trainer-side gather during send_weights.
            return engine

        # Workers sit at rank_offset 1, after the single trainer sender rank 0.
        # The packed wire params ride the worker init info, so the two sides
        # cannot disagree about the wire format.
        worker_init_info = HCCLWeightTransferInitInfo(
            master_address=init_info.master_address,
            master_port=init_info.master_port,
            rank_offset=1,
            world_size=init_info.world_size,
            packed=init_info.packed,
            packed_buffer_size_bytes=init_info.packed_buffer_size_bytes,
            packed_num_buffers=init_info.packed_num_buffers,
        )

        # The inference workers block inside init_weight_transfer_engine waiting
        # for the HCCL rendezvous, so kick that off on a side thread while we
        # open the trainer endpoint (rank 0); both sides must rendezvous
        # together.
        with ThreadPoolExecutor(max_workers=1) as exe:
            future = exe.submit(
                engine.client.init_weight_transfer_engine,
                asdict(worker_init_info),
            )
            engine.model_update_group = open_trainer_endpoint(init_info)
            future.result()  # surface any inference-side init error

        return engine

    def send_weights(self) -> None:
        assert self.source is not None  # guaranteed by trainer_init / __init__
        source = self.source

        # Metadata is declared without gathering. For Megatron it is itself a
        # collective, so every rank runs it; only the sender ships it.
        meta = source.metadata()

        if not self.is_sender:
            # Non-sender ranks only join the trainer-side gather collective.
            self._broadcast(source, meta)
            self._post_send_sync()
            return

        update_info = HCCLWeightTransferUpdateInfo(
            names=[m.name for m in meta],
            dtype_names=[str(m.dtype).split(".")[-1] for m in meta],
            shapes=[list(m.shape) for m in meta],
        )

        self.client.start_weight_update()
        # update_weights (workers receive) must run concurrently with the
        # trainer-side broadcast - both rendezvous inside the same HCCL calls.
        exe = ThreadPoolExecutor(max_workers=1)
        try:
            future = exe.submit(self.client.update_weights, asdict(update_info))
            # Cheap best-effort: if update_weights already failed (e.g. a bad
            # request rejected before any HCCL call), surface it now instead of
            # hanging in broadcast waiting for a peer that will never arrive.
            if future.done():
                future.result()
            self._broadcast(source, meta)
            future.result()  # surface inference-side errors
        finally:
            # Never wait for the RPC thread here. If the broadcast raised, the
            # worker is still blocked in the matching HCCL call and will never
            # return, so joining would turn the error into a permanent hang.
            # Let the exception out instead; the transfer group is unusable
            # either way and the caller has to tear it down.
            exe.shutdown(wait=False)
        self.client.finish_weight_update()
        self._post_send_sync()

    def _broadcast(self, source: "WeightSource", meta: list[ParamMeta]) -> None:
        """Iterate the source (materializing each tensor - a collective on all
        ranks) and, on the sender, broadcast from rank 0, packed or one-by-one.
        Non-sender ranks only replay the iteration to stay in the collective."""
        if not self.is_sender:
            for _ in source:
                pass
            return

        assert self.model_update_group is not None, "trainer_init() must be called before _broadcast()."
        pairs = self._checked_iter(source, meta)
        if self.packed:
            packed_broadcast_producer(
                iterator=pairs,
                group=self.model_update_group,
                src=0,
                post_iter_func=lambda item: item[1],
                buffer_size_bytes=self.packed_buffer_size_bytes,
                num_buffers=self.packed_num_buffers,
            )
        else:
            stream = torch.npu.current_stream()
            for _name, tensor in pairs:
                # The communicator ships `numel` elements straight from
                # `data_ptr()`, so a non-contiguous view would ship whatever
                # follows its base pointer. Keep the copy referenced until the
                # broadcast is enqueued. (The packed path linearizes itself.)
                send = tensor if tensor.is_contiguous() else tensor.contiguous()
                self.model_update_group.broadcast(send, src=0, stream=stream)

    @staticmethod
    def _checked_iter(source: "WeightSource", meta: list[ParamMeta]) -> Iterator[tuple[str, torch.Tensor]]:
        """Yield the source's pairs, checking each against what the worker was
        told to expect.

        The worker sizes its receive buffers - and in packed mode cuts its chunk
        boundaries - from the update info, which is built from ``metadata()``. If
        iteration disagrees with it, the two sides split the stream differently
        and the transfer hangs in HCCL or loads garbage. Checking here costs one
        comparison per parameter and turns that into an error naming the first
        divergent parameter. Sender-only: under pipeline parallelism a
        non-sender's yielded tensor is not meaningful.
        """
        sent = 0
        for name, tensor in source:
            if sent >= len(meta):
                raise ValueError(
                    f"WeightSource yielded more parameters than metadata() "
                    f"declared ({len(meta)}); first extra is {name!r}."
                )
            expected = meta[sent]
            if name != expected.name or tensor.dtype != expected.dtype or tuple(tensor.shape) != expected.shape:
                raise ValueError(
                    "WeightSource metadata() disagrees with iteration at index "
                    f"{sent}: declared {expected.name!r} "
                    f"{expected.dtype} {tuple(expected.shape)}, got {name!r} "
                    f"{tensor.dtype} {tuple(tensor.shape)}. Both channels must "
                    "enumerate the same parameters in the same order."
                )
            sent += 1
            yield name, tensor
        if sent != len(meta):
            raise ValueError(
                f"WeightSource yielded {sent} parameters but metadata() "
                f"declared {len(meta)}; the worker is waiting for the rest."
            )

    def _post_send_sync(self) -> None:
        """Wait for this rank's transfer work to land before returning.

        Broadcasts are only *enqueued* by `send_weights`: the unpacked path on
        the current stream, the packed path on the producer's own streams (which
        it drains itself). Waiting here lets a caller mutate parameters, or start
        the next step on another stream, as soon as `send_weights` returns,
        instead of silently depending on same-stream ordering. Every rank waits:
        a non-sender's `full_tensor()` gathers feed the sender's broadcast, so
        they must have landed before it may touch its shards.

        The engine owns this wait, exactly like upstream's trainer engines: it is
        a trainer-side concern and has no meaning on the inference worker, which
        synchronizes in `update_weights` instead.
        """
        if torch.npu.is_available():
            torch.npu.current_stream().synchronize()

    def shutdown(self) -> None:
        self.model_update_group = None
