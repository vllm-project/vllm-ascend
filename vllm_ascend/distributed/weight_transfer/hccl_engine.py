# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HCCL-based weight transfer engine."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, ClassVar, Self

import torch

if TYPE_CHECKING:
    from vllm.distributed.weight_transfer.base import VLLMWeightSyncClient, WeightSource

    from vllm_ascend.distributed.device_communicators.pyhccl import PyHcclCommunicator

from vllm.config import VllmConfig
from vllm.config.weight_transfer import WeightTransferConfig
from vllm.distributed.weight_transfer.base import (
    TrainerInitInfo,
    TrainerWeightTransferEngine,
    WeightTransferEngine,
    WeightTransferInitInfo,
    WeightTransferUpdateInfo,
)

from vllm_ascend.distributed.weight_transfer.packed_tensor import (
    DEFAULT_PACKED_BUFFER_SIZE_BYTES,
    DEFAULT_PACKED_NUM_BUFFERS,
    packed_broadcast_consumer,
    packed_broadcast_producer,
)


@dataclass
class HCCLWeightTransferInitInfo(WeightTransferInitInfo):
    """Initialization info for HCCL weight transfer backend."""

    master_address: str
    """IP address of the trainer (rank 0) for HCCL process group setup."""
    master_port: int
    """Port on the trainer for HCCL process group setup."""
    rank_offset: int
    """Offset added to each vLLM worker's rank within the HCCL group.
    Typically 1 (trainer is rank 0, workers start at rank 1)."""
    world_size: int
    """Total number of participants in the HCCL group (trainer + all workers)."""


@dataclass
class HCCLWeightTransferUpdateInfo(WeightTransferUpdateInfo):
    """Update info for HCCL weight transfer backend."""

    names: list[str]
    """Names of the parameters to transfer (e.g. ``model.layers.0.weight``)."""
    dtype_names: list[str]
    """Torch dtype names (e.g. ``bfloat16``, ``float32``) for each parameter."""
    shapes: list[list[int]]
    """Shapes of each parameter as integer lists."""
    packed: bool = False
    """Whether to use packed tensor broadcasting for efficiency.
    When True, multiple tensors are batched together before broadcasting
    to reduce HCCL communication overhead."""
    packed_buffer_size_bytes: int = DEFAULT_PACKED_BUFFER_SIZE_BYTES
    """Size in bytes for each packed tensor buffer.
    Both producer and consumer must use the same value."""
    packed_num_buffers: int = DEFAULT_PACKED_NUM_BUFFERS
    """Number of buffers for double/triple buffering during packed transfer.
    Both producer and consumer must use the same value."""

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
        Initialize HCCL process group with the trainer.

        Args:
            init_info: HCCL initialization info containing master address, port,
                      rank offset, and world size
        """

        # Calculate the global rank in the trainer-worker process group
        # Must account for data parallel to get unique ranks across all workers
        dp_rank = self.parallel_config.data_parallel_index
        world_size_per_dp = self.parallel_config.world_size  # TP * PP
        rank_within_dp = self.parallel_config.rank

        # Unique rank across all DP groups
        worker_rank = dp_rank * world_size_per_dp + rank_within_dp
        rank = worker_rank + init_info.rank_offset
        # Create stateless process group
        device = torch.accelerator.current_device_index()
        self.model_update_group = HCCLWeightTransferEngine._stateless_init_process_group(
            init_info.master_address,
            init_info.master_port,
            rank,
            init_info.world_size,
            device=device,
        )

    def receive_weights(
        self,
        update_info: HCCLWeightTransferUpdateInfo,
    ) -> None:
        """
        Receive weights from trainer via HCCL broadcast and load them incrementally.

        If update_info.packed is True, uses packed tensor broadcasting for
        efficient transfer of multiple weights in batches. Otherwise, uses simple
        one-by-one broadcasting.

        Args:
            update_info: HCCL update info containing parameter names, dtypes, shapes,
                        and packed flag
        """
        if self.model_update_group is None:
            raise RuntimeError("HCCL weight transfer not initialized. Call init_transfer_engine() first.")

        if update_info.packed:
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
                buffer_size_bytes=update_info.packed_buffer_size_bytes,
                num_buffers=update_info.packed_num_buffers,
            )
        else:
            # Use simple one-by-one broadcasting
            for name, dtype_name, shape in zip(update_info.names, update_info.dtype_names, update_info.shapes):
                dtype = getattr(torch, dtype_name)
                weight = torch.empty(shape, dtype=dtype, device="npu")
                self.model_update_group.broadcast(weight, src=0, stream=torch.npu.current_stream())
                self.model.load_weights([(name, weight)])
                del weight

    def shutdown(self) -> None:
        if self.model_update_group is not None:
            # Clean up the communicator by removing the reference
            self.model_update_group = None

    @staticmethod
    def _post_send_sync(stream: torch.npu.Stream | None = None) -> None:
        """Wait for this rank's transfer work to land before returning.

        Broadcasts are only *enqueued* by the send path: the unpacked path on
        the stream it was given, the packed path on the producer's own streams
        (which it drains itself). Waiting here lets a caller mutate parameters,
        or start the next step on another stream, as soon as the send returns,
        instead of silently depending on same-stream ordering. Non-sender ranks
        wait too: their ``full_tensor()`` gathers feed the sender's broadcast,
        so they must have landed before it may touch its shards.

        Mirrors the upstream trainer engines' ``_post_send_sync`` with the NPU
        device API.
        """
        if not torch.npu.is_available():
            return
        sender_stream = stream if stream is not None else torch.npu.current_stream()
        sender_stream.synchronize()

    @staticmethod
    def _stateless_init_process_group(master_address, master_port, rank, world_size, device):
        """
        vLLM provides `StatelessProcessGroup` to create a process group
        without considering the global process group in torch.distributed.
        It is recommended to create `StatelessProcessGroup`, and then initialize
        the data-plane communication (HCCL) between external (train processes)
        and vLLM workers.
        """
        from vllm.distributed.utils import StatelessProcessGroup

        from vllm_ascend.distributed.device_communicators.pyhccl import PyHcclCommunicator

        pg = StatelessProcessGroup.create(host=master_address, port=master_port, rank=rank, world_size=world_size)
        pyhccl = PyHcclCommunicator(pg, device=device)
        return pyhccl


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
        stream: torch.npu.Stream | None = None,
    ) -> None:
        super().__init__(client=client, source=source, is_sender=is_sender)
        self.packed = packed
        self.packed_buffer_size_bytes = packed_buffer_size_bytes
        self.packed_num_buffers = packed_num_buffers
        # Captured at construction so `_post_send_sync` waits for the stream the
        # unpacked path broadcasts on even if the caller switches streams later.
        self.stream = stream if stream is not None else torch.npu.current_stream()
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
        worker_init_info = HCCLWeightTransferInitInfo(
            master_address=init_info.master_address,
            master_port=init_info.master_port,
            rank_offset=1,
            world_size=init_info.world_size,
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
            engine.model_update_group = HCCLWeightTransferEngine._stateless_init_process_group(
                init_info.master_address,
                init_info.master_port,
                0,
                init_info.world_size,
                device=torch.accelerator.current_device_index(),
            )
            future.result()  # surface any inference-side init error

        return engine

    def send_weights(self) -> None:
        assert self.source is not None  # guaranteed by trainer_init
        source = self.source

        # Metadata is declared without gathering. For Megatron it is itself a
        # collective, so every rank runs it; only the sender ships it.
        meta = source.metadata()

        if not self.is_sender:
            # Non-sender ranks only join the trainer-side gather collective.
            for _ in source:
                pass
            self._post_send_sync()
            return

        update_info = HCCLWeightTransferUpdateInfo(
            names=[m.name for m in meta],
            dtype_names=[str(m.dtype).split(".")[-1] for m in meta],
            shapes=[list(m.shape) for m in meta],
            packed=self.packed,
            packed_buffer_size_bytes=self.packed_buffer_size_bytes,
            packed_num_buffers=self.packed_num_buffers,
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
            self._broadcast(source)
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

    def _broadcast(self, source: "WeightSource") -> None:
        """Broadcast the source from rank 0, packed or one-by-one."""
        assert self.model_update_group is not None, "trainer_init() must be called before send_weights()."
        if self.packed:
            packed_broadcast_producer(
                iterator=iter(source),
                group=self.model_update_group,
                src=0,
                post_iter_func=lambda item: item[1],
                buffer_size_bytes=self.packed_buffer_size_bytes,
                num_buffers=self.packed_num_buffers,
            )
        else:
            for _name, tensor in source:
                # The communicator ships `numel` elements straight from
                # `data_ptr()`, so a non-contiguous view would ship whatever
                # follows its base pointer. Keep the copy referenced until the
                # broadcast is enqueued. (The packed path linearizes itself.)
                send = tensor if tensor.is_contiguous() else tensor.contiguous()
                self.model_update_group.broadcast(send, src=0, stream=self.stream)

    def _post_send_sync(self) -> None:
        """Wait for this rank's broadcast to land before returning."""
        HCCLWeightTransferEngine._post_send_sync(self.stream)

    def shutdown(self) -> None:
        self.model_update_group = None
