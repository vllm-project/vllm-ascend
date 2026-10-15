# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HCCL-based weight transfer engine.

HCCL speaks the same wire protocol as upstream's NCCL backend: the trainer -- at
rank 0 of a `StatelessProcessGroup` group -- broadcasts a dense tensor stream,
optionally packed into fixed-size chunks, to every inference worker. The Ascend
backend therefore *inherits* the upstream engines

    vllm.distributed.weight_transfer.nccl_engine.NCCLWeightTransferEngine
    vllm.distributed.weight_transfer.nccl_engine.NCCLTrainerWeightTransferEngine

and overrides only what is device- or backend-bound:

* the communicator (`PyHcclCommunicator` instead of `PyNcclCommunicator`),
* the packed producer/consumer and the streams (`torch.npu`, not `torch.cuda`),
* the legacy per-update packed wire params older trainers still send,
* the trainer lifecycle guards that keep a broken group from being reused.

Everything else -- the `send_weights` round trip, the metadata/iteration
cross-check, the layerwise-reload lifecycle and the base `update_weights`
wrapper -- is upstream's, so upstream fixes and new wire params reach the HCCL
backend without a parallel implementation to keep in sync.
"""

from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, ClassVar

import torch
from typing_extensions import Self
from vllm.config import VllmConfig
from vllm.config.weight_transfer import WeightTransferConfig
from vllm.distributed.weight_transfer.base import (
    ParamMeta,
    VLLMWeightSyncClient,
    WeightSource,
    WeightTransferInitInfo,
)
from vllm.distributed.weight_transfer.nccl_engine import (
    NCCLTrainerInitInfo,
    NCCLTrainerWeightTransferEngine,
    NCCLWeightTransferEngine,
    NCCLWeightTransferUpdateInfo,
)

from vllm_ascend.distributed.weight_transfer import hccl_common
from vllm_ascend.distributed.weight_transfer.packed_tensor import (
    DEFAULT_PACKED_BUFFER_SIZE_BYTES,
    DEFAULT_PACKED_NUM_BUFFERS,
    packed_hccl_broadcast_consumer,
    packed_hccl_broadcast_producer,
)

if TYPE_CHECKING:
    from vllm_ascend.distributed.device_communicators.pyhccl import PyHcclCommunicator


class _HCCLTrainerState(Enum):
    READY = auto()
    FAILED = auto()
    CLOSED = auto()


def _validate_packed_params(
    packed: bool | None,
    packed_buffer_size_bytes: int | None,
    packed_num_buffers: int | None,
) -> None:
    """Validate the three-valued packed wire params shared by init and update."""
    if packed is not None and not isinstance(packed, bool):
        raise ValueError("`packed` must be a boolean or None")
    if packed_buffer_size_bytes is not None and packed_buffer_size_bytes <= 0:
        raise ValueError("`packed_buffer_size_bytes` must be positive")
    if packed_num_buffers is not None and packed_num_buffers < 1:
        raise ValueError("`packed_num_buffers` must be at least 1")


@dataclass
class HCCLWeightTransferInitInfo(WeightTransferInitInfo):
    """Initialization info for the HCCL weight transfer backend.

    Carries the subset of upstream's `NCCLWeightTransferInitInfo` that the HCCL
    data plane uses, but keeps the packed wire params three-valued: a legacy
    trainer declares them per update rather than at init, so `None` ("not fixed
    here") has to stay distinguishable from an explicit `False` or default.
    """

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
    """Packed buffer size fixed at init; ``None`` means legacy per-update parsing."""
    packed_num_buffers: int | None = None
    """Number of packed buffers fixed at init; ``None`` means per-update parsing."""

    def __post_init__(self) -> None:
        _validate_packed_params(self.packed, self.packed_buffer_size_bytes, self.packed_num_buffers)


@dataclass
class HCCLTrainerInitInfo(NCCLTrainerInitInfo):
    """Trainer-side HCCL handshake and fixed wire parameters.

    Same rendezvous and wire fields as upstream's `NCCLTrainerInitInfo` -- so a
    wire param added upstream is inherited instead of having to be mirrored --
    with the HCCL factory key and the HCCL default of sending unpacked.
    """

    backend: ClassVar[str] = "hccl"

    packed: bool = False


@dataclass
class HCCLTrainerSendWeightsArgs:
    """Arguments for the legacy `trainer_send_weights` entry point."""

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
class HCCLWeightTransferUpdateInfo(NCCLWeightTransferUpdateInfo):
    """Per-round update info for the HCCL weight transfer backend.

    Upstream's parameter metadata plus the legacy packed wire params, which
    trainers that predate the init-time handshake still send per update. Both
    are ignored once `init_transfer_engine` fixed a value.
    """

    packed: bool | None = None
    """Whether to use packed tensor broadcasting for this update."""
    packed_buffer_size_bytes: int | None = None
    """Size in bytes for each packed tensor buffer."""
    packed_num_buffers: int | None = None
    """Number of buffers for double/triple buffering during packed transfer."""

    def __post_init__(self) -> None:
        super().__post_init__()
        _validate_packed_params(self.packed, self.packed_buffer_size_bytes, self.packed_num_buffers)


class HCCLWeightTransferEngine(NCCLWeightTransferEngine):
    """Weight transfer engine using HCCL for communication between trainer and workers.

    Inherits upstream's `NCCLWeightTransferEngine` and overrides only the HCCL
    process group, the NPU data plane and the legacy per-update wire params.
    """

    # The HCCL dataclasses are not subclasses of the NCCL ones: the worker-side
    # init info keeps three-valued packed params for the legacy RPC path.
    init_info_cls = HCCLWeightTransferInitInfo  # type: ignore[assignment]
    update_info_cls = HCCLWeightTransferUpdateInfo  # type: ignore[assignment]

    def __init__(
        self,
        config: WeightTransferConfig,
        vllm_config: VllmConfig,
        device: torch.device,
        model: torch.nn.Module,
    ) -> None:
        super().__init__(config, vllm_config, device, model)
        # Which wire params the trainer fixed at init; the remaining ones may
        # still arrive per update (legacy trainers).
        self._init_packed_explicit = False
        self._init_buffer_size_explicit = False
        self._init_num_buffers_explicit = False

    def init_transfer_engine(self, init_info: HCCLWeightTransferInitInfo) -> None:  # type: ignore[override]
        """Initialize the HCCL process group with the trainer and record the
        trainer-supplied wire params so the worker decodes exactly as the
        trainer encodes.

        Args:
            init_info: HCCL initialization info containing master address, port,
                      rank offset, world size, and the packed wire params

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

        self.model_update_group = hccl_common.worker_init_process_group(  # type: ignore[assignment]
            init_info, self.parallel_config
        )

    def _wire_params(self, update_info: HCCLWeightTransferUpdateInfo) -> tuple[bool, int, int]:
        """Resolve the packed wire params for one update.

        A value fixed at init wins and rejects a conflicting update; otherwise
        the update's value (legacy trainer) or the init default applies.
        """
        if self._init_packed_explicit and update_info.packed is not None and update_info.packed != self.packed:
            raise ValueError(
                "HCCL update `packed` conflicts with the value fixed at init: "
                f"init={self.packed}, update={update_info.packed}"
            )
        packed = self.packed if update_info.packed is None else update_info.packed

        if self._init_buffer_size_explicit:
            if (
                update_info.packed_buffer_size_bytes is not None
                and update_info.packed_buffer_size_bytes != self.packed_buffer_size_bytes
            ):
                raise ValueError(
                    "HCCL update `packed_buffer_size_bytes` conflicts with the value fixed at init: "
                    f"init={self.packed_buffer_size_bytes}, update={update_info.packed_buffer_size_bytes}"
                )
            buffer_size_bytes = self.packed_buffer_size_bytes
        else:
            buffer_size_bytes = (
                self.packed_buffer_size_bytes
                if update_info.packed_buffer_size_bytes is None
                else update_info.packed_buffer_size_bytes
            )

        if self._init_num_buffers_explicit:
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

        return packed, buffer_size_bytes, num_buffers

    def receive_weights(self, update_info: HCCLWeightTransferUpdateInfo) -> None:  # type: ignore[override]
        """Receive weights from the trainer via HCCL broadcast and load them
        incrementally.

        Whether to use packed broadcasting (and the buffer geometry) is resolved
        by `_wire_params` from the init handshake and the per-update params, so
        the worker decodes exactly as the trainer encoded.

        Args:
            update_info: HCCL update info containing parameter names, dtypes,
                        and shapes

        """
        if self.model_update_group is None:
            raise RuntimeError("HCCL weight transfer not initialized. Call init_transfer_engine() first.")

        packed, buffer_size_bytes, num_buffers = self._wire_params(update_info)
        for dtype_name in update_info.dtype_names:
            if not isinstance(getattr(torch, dtype_name, None), torch.dtype):
                raise ValueError(f"unknown torch dtype name {dtype_name!r}")

        from vllm.model_executor.model_loader.mtp_validation import (
            disable_mtp_completeness_check,
        )

        # The worker is handed an explicit NPU device, so bind it before
        # touching any stream: every NCCL-path helper upstream uses resolves the
        # ambient device instead.
        with torch.npu.device(self.device), disable_mtp_completeness_check():
            if packed:

                def state_dict_info_iterator():
                    for name, dtype_name, shape in zip(
                        update_info.names,
                        update_info.dtype_names,
                        update_info.shapes,
                    ):
                        yield (name, (shape, getattr(torch, dtype_name)))

                packed_hccl_broadcast_consumer(
                    iterator=state_dict_info_iterator(),
                    group=self.model_update_group,
                    src=0,
                    post_unpack_func=self.model.load_weights,
                    buffer_size_bytes=buffer_size_bytes,
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
                    weight = torch.empty(shape, dtype=getattr(torch, dtype_name), device=self.device)
                    self.model_update_group.broadcast(
                        weight,
                        src=0,
                        stream=torch.npu.current_stream(),
                    )
                    self.model.load_weights([(name, weight)])
                    del weight

    def update_weights(self, update_info: dict[str, Any]) -> None:
        """Run the base update wrapper on the worker's assigned NPU."""
        with torch.npu.device(self.device):
            super().update_weights(update_info)

    def shutdown(self) -> None:
        # PyHcclCommunicator releases its comm through `close()`; the upstream
        # NCCL engine calls `destroy()`.
        group = getattr(self, "model_update_group", None)
        self.model_update_group = None
        if group is not None:
            group.close()

    @staticmethod
    def trainer_send_weights(
        iterator: Iterator[tuple[str, torch.Tensor]],
        trainer_args: dict[str, Any] | HCCLTrainerSendWeightsArgs,
    ) -> None:
        """Broadcast weights from a trainer that drives the RPCs itself.

        Legacy entry point kept for existing trainers; new code should build
        `HCCLTrainerWeightTransferEngine` through `WeightTransferTrainerFactory`.

        Args:
            iterator: Iterator of model parameters. Returns (name, tensor) tuples
            trainer_args: Dictionary or HCCLTrainerSendWeightsArgs instance containing
                         HCCL-specific arguments. If a dict, should contain keys from
                         HCCLTrainerSendWeightsArgs.

        """
        args = HCCLTrainerSendWeightsArgs(**trainer_args) if isinstance(trainer_args, dict) else trainer_args
        post_iter_func = (lambda item: item[1]) if args.post_iter_func is None else args.post_iter_func

        if args.packed:
            packed_hccl_broadcast_producer(
                iterator=iterator,
                group=args.group,
                src=args.src,
                post_iter_func=post_iter_func,
                buffer_size_bytes=args.packed_buffer_size_bytes,
                num_buffers=args.packed_num_buffers,
                device=getattr(args.group, "device", None),
            )
            return

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
        """Initialize the HCCL process group for a trainer-side weight transfer.

        Legacy counterpart of `HCCLTrainerWeightTransferEngine.trainer_init`; the
        trainer is always rank 0 in the process group.

        Args:
            init_info: Either an HCCLWeightTransferInitInfo object or a dict with keys:
                - master_address: str
                - master_port: int
                - world_size: int

        Returns:
            PyHcclCommunicator for weight transfer.

        """
        if isinstance(init_info, dict):
            master_address = init_info["master_address"]
            master_port = init_info["master_port"]
            world_size = init_info["world_size"]
        else:
            master_address = init_info.master_address
            master_port = init_info.master_port
            world_size = init_info.world_size

        return hccl_common.trainer_init(
            HCCLTrainerInitInfo(
                master_address=master_address,
                master_port=master_port,
                world_size=world_size,
                rank=0,
            ),
            rank=0,
        )


class HCCLTrainerWeightTransferEngine(NCCLTrainerWeightTransferEngine):
    """Trainer-side HCCL weight transfer engine.

    Inherits upstream's round trip (`send_weights`, `_checked_iter`) and
    overrides the HCCL rendezvous, the NPU data plane and the lifecycle guards
    that keep a broken transfer group from being reused.
    """

    init_info_cls = HCCLTrainerInitInfo

    def __init__(  # type: ignore[override]
        self,
        *,
        client: VLLMWeightSyncClient,
        source: WeightSource,
        is_sender: bool = True,
        packed: bool = False,
        packed_buffer_size_bytes: int = DEFAULT_PACKED_BUFFER_SIZE_BYTES,
        packed_num_buffers: int = DEFAULT_PACKED_NUM_BUFFERS,
        device: torch.device,
        group: "PyHcclCommunicator | None",
    ) -> None:
        super().__init__(
            client=client,
            source=source,
            is_sender=is_sender,
            packed=packed,
            packed_buffer_size_bytes=packed_buffer_size_bytes,
            packed_num_buffers=packed_num_buffers,
        )
        self.device = device
        self.model_update_group = group  # type: ignore[assignment]
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

        device_index = torch.accelerator.current_device_index()
        if device_index is None:
            raise ValueError("HCCL trainer requires an explicit current NPU device")
        device = torch.device("npu", device_index)
        worker_init_info = HCCLWeightTransferInitInfo(
            master_address=init_info.master_address,
            master_port=init_info.master_port,
            rank_offset=init_info.rank_offset,
            world_size=init_info.world_size,
            packed=init_info.packed,
            packed_buffer_size_bytes=init_info.packed_buffer_size_bytes,
            packed_num_buffers=init_info.packed_num_buffers,
        )

        # Only the sender joins the trainer-to-worker group and drives the
        # inference side; the other trainer ranks just join the trainer-side
        # collectives in `_broadcast`.
        group: PyHcclCommunicator | None = None
        if init_info.is_sender:
            # The worker-side init blocks inside the HCCL rendezvous, so drive
            # the RPC on a side thread while this rank opens rank 0.
            executor = ThreadPoolExecutor(max_workers=1)
            try:
                init_future = executor.submit(
                    client.init_weight_transfer_engine,
                    hccl_common.worker_init_payload(worker_init_info),
                )
                group = hccl_common.trainer_init(init_info, rank=init_info.rank)
                init_future.result()
            except BaseException:
                executor.shutdown(wait=False, cancel_futures=True)
                if group is not None:
                    with suppress(Exception):
                        group.close()
                raise
            executor.shutdown(wait=True)

        return cls(
            client=client,
            source=source,
            is_sender=init_info.is_sender,
            packed=init_info.packed,
            packed_buffer_size_bytes=init_info.packed_buffer_size_bytes,
            packed_num_buffers=init_info.packed_num_buffers,
            device=device,
            group=group,
        )

    def _require_usable(self) -> None:
        if self._state is _HCCLTrainerState.FAILED:
            raise RuntimeError(
                "HCCL trainer is in a failed state; create a new trainer engine before sending another weight update"
            )
        if self._state is _HCCLTrainerState.CLOSED:
            raise RuntimeError(
                "HCCL trainer is closed; create a new trainer engine before sending another weight update"
            )

    def _release_group(self) -> None:
        group, self.model_update_group = getattr(self, "model_update_group", None), None
        if group is not None:
            group.close()

    def send_weights(self) -> None:
        """Drive one update round trip through upstream's orchestration.

        The only addition is the lifecycle guard: a failed round trip leaves the
        HCCL group unusable, so the engine refuses further sends instead of
        hanging on a peer that will never arrive.
        """
        self._require_usable()
        try:
            super().send_weights()
        except BaseException:
            self._state = _HCCLTrainerState.FAILED
            with suppress(Exception):
                self._release_group()
            raise

    def _broadcast(self, source: WeightSource, meta: list[ParamMeta]) -> None:
        names = [param.name for param in meta]
        if len(names) != len(set(names)):
            raise ValueError("WeightSource.metadata() contains duplicate parameter names")

        if not self.is_sender:
            # Non-sender ranks only replay the source to stay in the trainer-side
            # collective (an FSDP `full_tensor()` gather, for instance), on the
            # device this engine was built for.
            with torch.npu.device(self.device):
                super()._broadcast(source, meta)
            return

        with torch.npu.device(self.device):
            if self.model_update_group is None:
                raise RuntimeError("HCCL trainer communicator is not initialized")
            if self.packed:
                packed_hccl_broadcast_producer(
                    iterator=self._checked_iter(source, meta),
                    group=self.model_update_group,
                    src=0,
                    post_iter_func=lambda item: item[1],
                    buffer_size_bytes=self.packed_buffer_size_bytes,
                    num_buffers=self.packed_num_buffers,
                    device=self.device,
                )
                return

            # PyHcclCommunicator::broadcast asserts the tensor lives on the
            # communicator's device and reads it asynchronously, so keep a
            # device-local copy alive until the broadcast has completed.
            for _, source_tensor in self._checked_iter(source, meta):
                send = source_tensor.detach().to(self.device).contiguous()
                self.model_update_group.broadcast(send, src=0, stream=torch.npu.current_stream())
                torch.npu.current_stream().synchronize()
                del send

    def _post_send_sync(self) -> None:
        """Wait for this rank's transfer work to land before returning."""
        with torch.npu.device(self.device):
            torch.npu.current_stream().synchronize()

    def shutdown(self) -> None:
        if self._state is _HCCLTrainerState.CLOSED:
            return
        self._state = _HCCLTrainerState.CLOSED
        self._release_group()
