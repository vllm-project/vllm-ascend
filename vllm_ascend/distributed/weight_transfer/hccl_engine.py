# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HCCL-based weight transfer engine."""

from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from dataclasses import asdict, dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, ClassVar

import torch
from typing_extensions import Self

if TYPE_CHECKING:
    from vllm_ascend.distributed.device_communicators.pyhccl import PyHcclCommunicator

from vllm.config import VllmConfig
from vllm.config.weight_transfer import WeightTransferConfig
from vllm.distributed.weight_transfer.base import (
    ParamMeta,
    TrainerInitInfo,
    TrainerWeightTransferEngine,
    VLLMWeightSyncClient,
    WeightSource,
    WeightTransferEngine,
    WeightTransferInitInfo,
    WeightTransferUpdateInfo,
)

from vllm_ascend.distributed.weight_transfer import hccl_common
from vllm_ascend.distributed.weight_transfer.packed_tensor import (
    DEFAULT_PACKED_BUFFER_SIZE_BYTES,
    DEFAULT_PACKED_NUM_BUFFERS,
    packed_hccl_broadcast_consumer,
    packed_hccl_broadcast_producer,
)


class _HCCLTrainerState(Enum):
    READY = auto()
    FAILED = auto()
    CLOSED = auto()


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
    packed: bool | None = None
    """Packed mode fixed at init; ``None`` means legacy per-update parsing."""
    packed_buffer_size_bytes: int | None = None
    packed_num_buffers: int | None = None

    def __post_init__(self) -> None:
        if self.packed is not None and not isinstance(self.packed, bool):
            raise ValueError("`packed` must be a boolean or None")
        if self.packed_buffer_size_bytes is not None and self.packed_buffer_size_bytes <= 0:
            raise ValueError("`packed_buffer_size_bytes` must be positive")
        if self.packed_num_buffers is not None and self.packed_num_buffers < 1:
            raise ValueError("`packed_num_buffers` must be at least 1")


@dataclass
class HCCLTrainerInitInfo(TrainerInitInfo):
    """Trainer-side HCCL handshake and fixed wire parameters."""

    backend: ClassVar[str] = "hccl"

    master_address: str
    master_port: int
    world_size: int
    rank_offset: int = 1
    packed: bool | None = None
    packed_buffer_size_bytes: int | None = None
    packed_num_buffers: int | None = None

    def __post_init__(self) -> None:
        if self.packed is not None and not isinstance(self.packed, bool):
            raise ValueError("`packed` must be a boolean or None")
        if self.packed_buffer_size_bytes is not None and self.packed_buffer_size_bytes <= 0:
            raise ValueError("`packed_buffer_size_bytes` must be positive")
        if self.packed_num_buffers is not None and self.packed_num_buffers < 1:
            raise ValueError("`packed_num_buffers` must be at least 1")


@dataclass
class HCCLTrainerSendWeightsArgs:
    """Arguments for HCCL trainer_send_weights method."""

    group: Any
    """Process group (PyHcclCommunicator) for HCCL communication."""
    src: int = 0
    """Source rank (default 0, trainer is typically rank 0)."""
    post_iter_func: Callable[[tuple[str, torch.Tensor]], torch.Tensor] | None = None
    """Optional function to apply to each (name, tensor) pair before broadcasting.
    If None, extracts just the tensor."""
    packed: bool = False
    """Whether to use packed tensor broadcasting for efficiency.
    When True, multiple tensors are batched together before broadcasting
    to reduce HCCL communication overhead."""
    stream: torch.npu.Stream | None = None
    """ACL stream to use for broadcasting if packed is False.
    If packed is True, new streams will be created for each buffer."""
    packed_buffer_size_bytes: int = DEFAULT_PACKED_BUFFER_SIZE_BYTES
    """Size in bytes for each packed tensor buffer.
    Must match the value used in HCCLWeightTransferUpdateInfo."""
    packed_num_buffers: int = DEFAULT_PACKED_NUM_BUFFERS
    """Number of buffers for double/triple buffering during packed transfer.
    Must match the value used in HCCLWeightTransferUpdateInfo."""


@dataclass
class HCCLWeightTransferUpdateInfo(WeightTransferUpdateInfo):
    """Update info for HCCL weight transfer backend."""

    names: list[str]
    """Names of the parameters to transfer (e.g. ``model.layers.0.weight``)."""
    dtype_names: list[str]
    """Torch dtype names (e.g. ``bfloat16``, ``float32``) for each parameter."""
    shapes: list[list[int]]
    """Shapes of each parameter as integer lists."""
    packed: bool | None = None
    """Whether to use packed tensor broadcasting for efficiency.
    When True, multiple tensors are batched together before broadcasting
    to reduce HCCL communication overhead."""
    packed_buffer_size_bytes: int | None = None
    """Size in bytes for each packed tensor buffer.
    Both producer and consumer must use the same value."""
    packed_num_buffers: int | None = None
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
        for name, shape in zip(self.names, self.shapes):
            if any(dimension < 0 for dimension in shape):
                raise ValueError(f"`{name}` has a negative shape: {shape}")
        if self.packed is not None and not isinstance(self.packed, bool):
            raise ValueError("`packed` must be a boolean or None")
        if self.packed_buffer_size_bytes is not None and self.packed_buffer_size_bytes <= 0:
            raise ValueError("`packed_buffer_size_bytes` must be positive")
        if self.packed_num_buffers is not None and self.packed_num_buffers < 1:
            raise ValueError("`packed_num_buffers` must be at least 1")


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
        self.packed = False
        self.packed_buffer_size_bytes = DEFAULT_PACKED_BUFFER_SIZE_BYTES
        self.packed_num_buffers = DEFAULT_PACKED_NUM_BUFFERS
        self._init_packed_explicit = False
        self._init_buffer_size_explicit = False
        self._init_num_buffers_explicit = False

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

        self.shutdown()

        if init_info.world_size < 2:
            raise ValueError("HCCL weight transfer requires at least one trainer and one worker")
        self._init_packed_explicit = init_info.packed is not None
        self._init_buffer_size_explicit = init_info.packed_buffer_size_bytes is not None
        self._init_num_buffers_explicit = init_info.packed_num_buffers is not None
        self.packed = False if init_info.packed is None else init_info.packed
        self.packed_buffer_size_bytes = (
            DEFAULT_PACKED_BUFFER_SIZE_BYTES
            if init_info.packed_buffer_size_bytes is None
            else init_info.packed_buffer_size_bytes
        )
        self.packed_num_buffers = (
            DEFAULT_PACKED_NUM_BUFFERS if init_info.packed_num_buffers is None else init_info.packed_num_buffers
        )

        self.model_update_group = hccl_common.worker_init_process_group(init_info, self.parallel_config)

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

        packed = self.packed if update_info.packed is None else update_info.packed
        init_packed_explicit = getattr(self, "_init_packed_explicit", False)
        if init_packed_explicit and update_info.packed is not None and update_info.packed != self.packed:
            raise ValueError(
                "HCCL update `packed` conflicts with the value fixed at init: "
                f"init={self.packed}, update={update_info.packed}"
            )
        init_buffer_size_explicit = getattr(self, "_init_buffer_size_explicit", False)
        if init_buffer_size_explicit:
            if (
                update_info.packed_buffer_size_bytes is not None
                and update_info.packed_buffer_size_bytes != self.packed_buffer_size_bytes
            ):
                raise ValueError(
                    "HCCL update `packed_buffer_size_bytes` conflicts with the value fixed at init: "
                    f"init={self.packed_buffer_size_bytes}, "
                    f"update={update_info.packed_buffer_size_bytes}"
                )
            buffer_size = self.packed_buffer_size_bytes
        else:
            buffer_size = (
                self.packed_buffer_size_bytes
                if update_info.packed_buffer_size_bytes is None
                else update_info.packed_buffer_size_bytes
            )

        init_num_buffers_explicit = getattr(self, "_init_num_buffers_explicit", False)
        if init_num_buffers_explicit:
            if update_info.packed_num_buffers is not None and update_info.packed_num_buffers != self.packed_num_buffers:
                raise ValueError(
                    "HCCL update `packed_num_buffers` conflicts with the value fixed at init: "
                    f"init={self.packed_num_buffers}, update={update_info.packed_num_buffers}"
                )
            num_buffers = self.packed_num_buffers
        else:
            num_buffers = (
                self.packed_num_buffers if update_info.packed_num_buffers is None else update_info.packed_num_buffers
            )

        for dtype_name in update_info.dtype_names:
            dtype = getattr(torch, dtype_name, None)
            if not isinstance(dtype, torch.dtype):
                raise ValueError(f"unknown torch dtype name {dtype_name!r}")

        from vllm.model_executor.model_loader.mtp_validation import (
            disable_mtp_completeness_check,
        )

        with torch.npu.device(self.device):
            if packed:
                # Build iterator of (name, (shape, dtype)) from update_info.
                def state_dict_info_iterator():
                    for name, dtype_name, shape in zip(
                        update_info.names,
                        update_info.dtype_names,
                        update_info.shapes,
                    ):
                        dtype = getattr(torch, dtype_name)
                        yield (name, (shape, dtype))

                def load_weights(weights):
                    with disable_mtp_completeness_check():
                        self.model.load_weights(weights)

                packed_hccl_broadcast_consumer(
                    iterator=state_dict_info_iterator(),
                    group=self.model_update_group,
                    src=0,
                    post_unpack_func=load_weights,
                    buffer_size_bytes=buffer_size,
                    num_buffers=num_buffers,
                    device=self.device,
                )
            else:
                # Use simple one-by-one broadcasting.
                for name, dtype_name, shape in zip(
                    update_info.names,
                    update_info.dtype_names,
                    update_info.shapes,
                ):
                    dtype = getattr(torch, dtype_name)
                    weight = torch.empty(shape, dtype=dtype, device=self.device)
                    self.model_update_group.broadcast(
                        weight,
                        src=0,
                        stream=torch.npu.current_stream(),
                    )
                    with disable_mtp_completeness_check():
                        self.model.load_weights([(name, weight)])
                    del weight

    def update_weights(self, update_info: dict[str, Any]) -> None:
        """Run the base update wrapper on the worker's assigned NPU."""
        with torch.npu.device(self.device):
            super().update_weights(update_info)

    def shutdown(self) -> None:
        if self.model_update_group is not None:
            self.model_update_group.close()
            self.model_update_group = None

    @staticmethod
    def trainer_send_weights(
        iterator: Iterator[tuple[str, torch.Tensor]],
        trainer_args: dict[str, Any] | HCCLTrainerSendWeightsArgs,
    ) -> None:
        """Broadcast weights from trainer to vLLM workers.

        Args:
            iterator: Iterator of model parameters. Returns (name, tensor) tuples
            trainer_args: Dictionary or HCCLTrainerSendWeightsArgs instance containing
                         HCCL-specific arguments. If a dict, should contain keys from
                         HCCLTrainerSendWeightsArgs.

        Example:
            >>> from vllm_ascend.distributed.weight_transfer.hccl_engine import (
            ...     HCCLWeightTransferEngine,
            ...     HCCLTrainerSendWeightsArgs,
            ... )
            >>> param_iter = ((n, p) for n, p in model.named_parameters())
            >>> args = HCCLTrainerSendWeightsArgs(group=group, packed=True)
            >>> HCCLWeightTransferEngine.trainer_send_weights(param_iter, args)
        """
        # Parse trainer args - accept either dict or dataclass instance
        if isinstance(trainer_args, dict):
            args = HCCLTrainerSendWeightsArgs(**trainer_args)
        else:
            args = trainer_args

        if args.post_iter_func is None:
            # Default: extract just the tensor from (name, tensor) tuple
            post_iter_func = lambda x: x[1]
        else:
            post_iter_func = args.post_iter_func

        if args.packed:
            # Use packed tensor broadcasting for efficiency
            packed_hccl_broadcast_producer(
                iterator=iterator,
                group=args.group,
                src=args.src,
                post_iter_func=post_iter_func,
                buffer_size_bytes=args.packed_buffer_size_bytes,
                num_buffers=args.packed_num_buffers,
                device=getattr(args.group, "device", None),
            )
        else:
            # Use simple one-by-one broadcasting
            for item in iterator:
                tensor = post_iter_func(item)
                args.group.broadcast(
                    tensor,
                    src=args.src,
                    stream=args.stream or torch.npu.current_stream(),
                )

    @staticmethod
    def trainer_init(
        init_info: HCCLWeightTransferInitInfo | dict,
    ) -> "PyHcclCommunicator":
        """
        Initialize HCCL process group for trainer-side weight transfer.

        The trainer is always rank 0 in the process group. Uses the current
        Ascend device (torch.accelerator.current_device_index()).

        Args:
            init_info: Either an HCCLWeightTransferInitInfo object or a dict with keys:
                - master_address: str
                - master_port: int
                - world_size: int

        Returns:
            PyHcclCommunicator for weight transfer.

        Example:
            >>> from vllm_ascend.distributed.weight_transfer.hccl_engine import (
            ...     HCCLWeightTransferEngine,
            ... )
            >>> group = HCCLWeightTransferEngine.trainer_init(
            ...     dict(
            ...         master_address=master_address,
            ...         master_port=master_port,
            ...         world_size=world_size,
            ...     ),
            ... )
        """
        if isinstance(init_info, dict):
            master_address = init_info["master_address"]
            master_port = init_info["master_port"]
            world_size = init_info["world_size"]
        else:
            # HCCLWeightTransferInitInfo object
            master_address = init_info.master_address
            master_port = init_info.master_port
            world_size = init_info.world_size

        return hccl_common.trainer_init(
            HCCLTrainerInitInfo(
                master_address=master_address,
                master_port=master_port,
                world_size=world_size,
            ),
            rank=0,
        )

    @staticmethod
    def _stateless_init_process_group(master_address, master_port, rank, world_size, device):
        """Compatibility adapter for the shared HCCL initializer."""
        return hccl_common.stateless_init_process_group(master_address, master_port, rank, world_size, device)


class HCCLTrainerWeightTransferEngine(TrainerWeightTransferEngine[HCCLTrainerInitInfo]):
    """Stateful HCCL trainer engine using vLLM's four-method client contract."""

    init_info_cls = HCCLTrainerInitInfo

    def __init__(
        self,
        *,
        client: VLLMWeightSyncClient,
        source: WeightSource,
        is_sender: bool,
        packed: bool,
        packed_buffer_size_bytes: int,
        packed_num_buffers: int,
        device: torch.device,
        group: Any,
    ) -> None:
        super().__init__(client=client, source=source, is_sender=is_sender)
        self.packed = packed
        self.packed_buffer_size_bytes = packed_buffer_size_bytes
        self.packed_num_buffers = packed_num_buffers
        self.device = device
        self.model_update_group = group
        self._state = _HCCLTrainerState.READY

    @classmethod
    def trainer_init(
        cls,
        init_info: HCCLTrainerInitInfo,
        *,
        client: VLLMWeightSyncClient,
        source: WeightSource | None = None,
    ) -> Self:
        if source is None:
            raise ValueError("HCCL trainer weight transfer requires a WeightSource.")
        if init_info.world_size < 2:
            raise ValueError("HCCL trainer weight transfer requires at least one worker")

        packed = bool(init_info.packed) if init_info.packed is not None else False
        buffer_size = (
            DEFAULT_PACKED_BUFFER_SIZE_BYTES
            if init_info.packed_buffer_size_bytes is None
            else init_info.packed_buffer_size_bytes
        )
        num_buffers = (
            DEFAULT_PACKED_NUM_BUFFERS if init_info.packed_num_buffers is None else init_info.packed_num_buffers
        )

        device_index = torch.accelerator.current_device_index()
        if device_index is None:
            raise ValueError("HCCL trainer requires an explicit current NPU device")
        device = torch.device("npu", device_index)
        worker_init_info = HCCLWeightTransferInitInfo(
            master_address=init_info.master_address,
            master_port=init_info.master_port,
            rank_offset=init_info.rank_offset,
            world_size=init_info.world_size,
            packed=packed,
            packed_buffer_size_bytes=buffer_size,
            packed_num_buffers=num_buffers,
        )
        payload = asdict(worker_init_info)

        if init_info.is_sender:
            # The worker-side init waits for the trainer-side stateless group;
            # overlap the RPC and rendezvous so neither side waits serially.
            executor = ThreadPoolExecutor(max_workers=1)
            group = None
            try:
                init_future = executor.submit(client.init_weight_transfer_engine, payload)
                group = hccl_common.stateless_init_process_group(
                    init_info.master_address,
                    init_info.master_port,
                    init_info.rank,
                    init_info.world_size,
                    device_index,
                )
                init_future.result()
            except BaseException:
                executor.shutdown(wait=False, cancel_futures=True)
                if group is not None:
                    with suppress(Exception):
                        group.close()
                raise
            else:
                executor.shutdown(wait=True)
        else:
            # Non-sender trainer ranks participate in source collectives but do
            # not join the trainer-to-worker HCCL group.  The transfer group
            # contains rank 0 (the sender) plus inference workers only.
            group = None

        return cls(
            client=client,
            source=source,
            is_sender=init_info.is_sender,
            packed=packed,
            packed_buffer_size_bytes=buffer_size,
            packed_num_buffers=num_buffers,
            device=device,
            group=group,
        )

    def _checked_iter(
        self,
        source: WeightSource,
        metadata: list[ParamMeta],
    ) -> Iterator[tuple[str, torch.Tensor]]:
        """Yield source tensors only when they match metadata element-for-element."""
        expected = iter(metadata)
        for index, (name, tensor) in enumerate(source):
            meta = next(expected, None)
            if meta is None:
                raise ValueError(f"WeightSource yielded an extra parameter at index {index}: {name!r}")
            if name != meta.name:
                raise ValueError(
                    "WeightSource iteration disagrees with metadata() at index "
                    f"{index}: expected name {meta.name!r}, got {name!r}"
                )
            if tensor.dtype != meta.dtype:
                raise ValueError(
                    f"WeightSource parameter {name!r} has dtype {tensor.dtype}, metadata declares {meta.dtype}"
                )
            if tuple(tensor.shape) != tuple(meta.shape):
                raise ValueError(
                    f"WeightSource parameter {name!r} has shape {tuple(tensor.shape)}, "
                    f"metadata declares {tuple(meta.shape)}"
                )
            yield name, tensor
        if next(expected, None) is not None:
            raise ValueError("WeightSource iteration ended before metadata()")

    def _post_send_sync(self) -> None:
        """Complete this rank's source materialization before returning."""
        with torch.npu.device(self.device):
            torch.npu.current_stream().synchronize()

    def _broadcast(
        self,
        source: WeightSource,
        metadata: list[ParamMeta],
    ) -> None:
        if not self.is_sender:
            with torch.npu.device(self.device):
                for _, tensor in source:
                    del tensor
            return

        with torch.npu.device(self.device):
            if self.model_update_group is None:
                raise RuntimeError("HCCL trainer communicator is not initialized")
            if self.packed:
                packed_hccl_broadcast_producer(
                    iterator=self._checked_iter(source, metadata),
                    group=self.model_update_group,
                    src=0,
                    post_iter_func=lambda item: item[1],
                    buffer_size_bytes=self.packed_buffer_size_bytes,
                    num_buffers=self.packed_num_buffers,
                    device=self.device,
                )
                return

            # Conservative bounded-memory implementation: do not release a
            # source tensor until the HCCL operation using its storage is
            # complete. A later optimization can replace this with a small
            # event-backed in-flight window.
            for _, source_tensor in self._checked_iter(source, metadata):
                tensor = source_tensor.detach().contiguous().to(self.device)
                self.model_update_group.broadcast(tensor, src=0, stream=torch.npu.current_stream())
                torch.npu.current_stream().synchronize()
                del tensor

    def send_weights(self) -> None:
        if self._state is _HCCLTrainerState.FAILED:
            raise RuntimeError(
                "HCCL trainer is in a failed state; create a new trainer engine before sending another weight update"
            )
        if self._state is _HCCLTrainerState.CLOSED:
            raise RuntimeError(
                "HCCL trainer is closed; create a new trainer engine before sending another weight update"
            )
        executor: ThreadPoolExecutor | None = None
        try:
            source = self.source
            metadata = source.metadata()
            names = [meta.name for meta in metadata]
            if len(names) != len(set(names)):
                raise ValueError("WeightSource.metadata() contains duplicate parameter names")
            if not self.is_sender:
                self._broadcast(source, metadata)
                self._post_send_sync()
                return

            update_info = HCCLWeightTransferUpdateInfo(
                names=[meta.name for meta in metadata],
                dtype_names=[str(meta.dtype).split(".")[-1] for meta in metadata],
                shapes=[list(meta.shape) for meta in metadata],
                packed=self.packed,
                packed_buffer_size_bytes=self.packed_buffer_size_bytes,
                packed_num_buffers=self.packed_num_buffers,
            )
            self.client.start_weight_update()
            executor = ThreadPoolExecutor(max_workers=1)
            update_future = executor.submit(self.client.update_weights, asdict(update_info))
            if update_future.done():
                update_future.result()
            self._broadcast(source, metadata)
            update_future.result()
            executor.shutdown(wait=False)
            executor = None
            self._post_send_sync()
            self.client.finish_weight_update()
        except BaseException:
            self._state = _HCCLTrainerState.FAILED
            if executor is not None:
                executor.shutdown(wait=False, cancel_futures=True)
            group, self.model_update_group = self.model_update_group, None
            if group is not None:
                with suppress(Exception):
                    group.close()
            raise

    def shutdown(self) -> None:
        if self._state is _HCCLTrainerState.CLOSED:
            return
        self._state = _HCCLTrainerState.CLOSED
        group, self.model_update_group = self.model_update_group, None
        if group is not None:
            group.close()
