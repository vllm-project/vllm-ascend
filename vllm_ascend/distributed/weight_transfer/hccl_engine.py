# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HCCL adapters for upstream vLLM's dense NCCL weight-transfer engines.

Reload lifecycle, ``send_weights`` orchestration, metadata checks and the worker
init payload stay upstream. Only NPU operations, the HCCL rendezvous and the
legacy per-update packed wire parameters are local. The packed producer/consumer
and the layerwise cross-chunk behaviour are unchanged.
"""

from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

import torch
from typing_extensions import Self
from vllm.distributed.weight_transfer.base import (
    ParamMeta,
    VLLMWeightSyncClient,
    WeightSource,
)
from vllm.distributed.weight_transfer.nccl_common import worker_init_payload
from vllm.distributed.weight_transfer.nccl_engine import (
    NCCLTrainerInitInfo,
    NCCLTrainerWeightTransferEngine,
    NCCLWeightTransferEngine,
    NCCLWeightTransferUpdateInfo,
)

from vllm_ascend.distributed.weight_transfer import hccl_common
from vllm_ascend.distributed.weight_transfer.hccl_common import (
    HCCLWeightTransferInitInfo,
    _validate_packed_params,
)
from vllm_ascend.distributed.weight_transfer.packed_tensor import (
    DEFAULT_PACKED_BUFFER_SIZE_BYTES,
    DEFAULT_PACKED_NUM_BUFFERS,
    packed_broadcast_consumer,
    packed_broadcast_producer,
)

if TYPE_CHECKING:
    from vllm_ascend.distributed.device_communicators.pyhccl import PyHcclCommunicator


@dataclass
class HCCLTrainerInitInfo(NCCLTrainerInitInfo):
    """Upstream trainer contract, including packed transfers by default."""

    backend: ClassVar[str] = "hccl"

    def __post_init__(self) -> None:
        _validate_packed_params(self)


@dataclass
class HCCLWeightTransferUpdateInfo(NCCLWeightTransferUpdateInfo):
    """Upstream metadata plus the legacy per-update packed wire params."""

    packed: bool | None = None
    packed_buffer_size_bytes: int | None = None
    packed_num_buffers: int | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        _validate_packed_params(self)
        for name in self.dtype_names:
            if not isinstance(getattr(torch, name, None), torch.dtype):
                raise ValueError(f"unknown torch dtype name {name!r}")


class HCCLWeightTransferEngine(NCCLWeightTransferEngine):
    """Inherit the reload lifecycle; adapt the rendezvous and the NPU data plane."""

    if TYPE_CHECKING:
        model_update_group: PyHcclCommunicator | None  # type: ignore[assignment]

    init_info_cls = HCCLWeightTransferInitInfo
    update_info_cls = HCCLWeightTransferUpdateInfo
    # Tests and alternate HCCL transports can swap the group constructor.
    _stateless_init_process_group = staticmethod(hccl_common.stateless_init_process_group)

    def init_transfer_engine(self, init_info: HCCLWeightTransferInitInfo) -> None:  # type: ignore[override]
        """Open the trainer<->worker HCCL group and remember the wire params.

        The init info stays around so `_wire_params` can tell "fixed at init"
        from "left to the update", which is what legacy trainers rely on.
        """
        self.shutdown()
        # v0.32 has no metadata cache in the parent; initialize it here for both
        # versions and never reuse metadata across transfer sessions.
        self._last_hccl_update_info: dict[torch.nn.Module, HCCLWeightTransferUpdateInfo] = {}
        if init_info.world_size < 2:
            raise ValueError("HCCL weight transfer requires at least one trainer and one worker")
        self._init_info = init_info
        # Preserve the parent's handshake attributes, resetting omitted values
        # on re-init. _wire_params still supports legacy per-update overrides.
        self.packed = init_info.packed if init_info.packed is not None else False
        self.packed_buffer_size_bytes = (
            init_info.packed_buffer_size_bytes
            if init_info.packed_buffer_size_bytes is not None
            else DEFAULT_PACKED_BUFFER_SIZE_BYTES
        )
        self.packed_num_buffers = (
            init_info.packed_num_buffers if init_info.packed_num_buffers is not None else DEFAULT_PACKED_NUM_BUFFERS
        )
        # Unique rank across data-parallel groups; the trainer occupies the
        # ranks before `rank_offset`.
        parallel = self.parallel_config
        rank = parallel.data_parallel_index * parallel.world_size + parallel.rank + init_info.rank_offset
        with torch.npu.device(self.device):
            group = self._stateless_init_process_group(
                init_info.master_address,
                init_info.master_port,
                rank,
                init_info.world_size,
                device=torch.accelerator.current_device_index(),
            )
            self.model_update_group = group

    def _wire_params(self, update: HCCLWeightTransferUpdateInfo) -> dict[str, Any]:
        """Resolve the packed wire params for one update.

        A value fixed at init wins and rejects a conflicting update; otherwise
        the update's value (legacy trainer) or the backend default applies.
        """
        params: dict[str, Any] = dict(
            packed=False,
            packed_buffer_size_bytes=DEFAULT_PACKED_BUFFER_SIZE_BYTES,
            packed_num_buffers=DEFAULT_PACKED_NUM_BUFFERS,
        )
        for name in params:
            fixed, requested = getattr(self._init_info, name), getattr(update, name)
            if fixed is not None and requested is not None and fixed != requested:
                raise ValueError(
                    f"HCCL update `{name}` conflicts with the value fixed at init: init={fixed!r}, update={requested!r}"
                )
            if fixed is not None or requested is not None:
                params[name] = fixed if fixed is not None else requested
        return params

    def receive_weights(self, update_info: HCCLWeightTransferUpdateInfo) -> None:  # type: ignore[override]
        """Receive one round over the NPU data plane and load it layerwise."""
        if self.model_update_group is None:
            raise RuntimeError("HCCL weight transfer not initialized. Call init_transfer_engine() first.")
        params = self._wire_params(update_info)
        # Upstream main omits unchanged metadata after the first update. Keep
        # the cache model-specific, while resolving this round's wire params
        # above independently of the cached parameter metadata.
        if update_info.names:
            self._last_hccl_update_info[self.model] = update_info
        else:
            update_info = self._last_hccl_update_info.get(self.model, update_info)

        from vllm.model_executor.model_loader.mtp_validation import (
            disable_mtp_completeness_check,
        )

        # The worker is handed an explicit NPU device, so bind it before touching
        # any stream: the packed helpers resolve the ambient device.
        with torch.npu.device(self.device), disable_mtp_completeness_check():
            metadata = zip(update_info.names, update_info.shapes, update_info.dtype_names)
            if params["packed"]:
                packed_broadcast_consumer(
                    iterator=((name, (shape, getattr(torch, dtype))) for name, shape, dtype in metadata),
                    group=self.model_update_group,
                    src=0,
                    post_unpack_func=self.model.load_weights,
                    buffer_size_bytes=params["packed_buffer_size_bytes"],
                    num_buffers=params["packed_num_buffers"],
                )
            else:
                for name, shape, dtype in metadata:
                    weight = torch.empty(shape, dtype=getattr(torch, dtype), device=self.device)
                    self.model_update_group.broadcast(weight, src=0, stream=torch.npu.current_stream())
                    self.model.load_weights([(name, weight)])

    def update_weights(self, update_info: dict[str, Any]) -> None:
        """Run the inherited wrapper on the worker's assigned NPU."""
        # The inherited wrapper synchronizes the ambient device at the end.
        with torch.npu.device(self.device):
            super().update_weights(update_info)

    def shutdown(self) -> None:
        # PyHcclCommunicator releases its comm through `close()`, not `destroy()`.
        group, self.model_update_group = getattr(self, "model_update_group", None), None
        if group is not None:
            group.close()


class HCCLTrainerWeightTransferEngine(NCCLTrainerWeightTransferEngine):
    """Reuse upstream's send orchestration and metadata checks, not its CUDA calls."""

    init_info_cls = HCCLTrainerInitInfo

    if TYPE_CHECKING:
        model_update_group: PyHcclCommunicator | None  # type: ignore[assignment]

    def __init__(
        self,
        *,
        client: VLLMWeightSyncClient,
        source: WeightSource,
        is_sender: bool = True,
        device: torch.device | None = None,
        packed: bool = True,
        packed_buffer_size_bytes: int = DEFAULT_PACKED_BUFFER_SIZE_BYTES,
        packed_num_buffers: int = DEFAULT_PACKED_NUM_BUFFERS,
    ) -> None:
        super().__init__(
            client=client,
            source=source,
            is_sender=is_sender,
            packed=packed,
            packed_buffer_size_bytes=packed_buffer_size_bytes,
            packed_num_buffers=packed_num_buffers,
        )
        if device is None:
            index = torch.accelerator.current_device_index()
            if index is None:
                raise ValueError("HCCL trainer requires a current NPU device")
            device = torch.device("npu", index)
        self.device = device
        self.model_update_group = None
        self._state = "ready"

    # Mirror NCCLTrainerWeightTransferEngine.trainer_init orchestration until
    # upstream exposes a hook for the endpoint opener and worker init-info type.
    @classmethod
    def trainer_init(  # type: ignore[override]
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
        engine = cls(
            client=client,
            source=source,
            is_sender=init_info.is_sender,
            packed=init_info.packed,
            packed_buffer_size_bytes=init_info.packed_buffer_size_bytes,
            packed_num_buffers=init_info.packed_num_buffers,
        )
        # Only the sender joins the trainer-to-worker group and drives the
        # inference side; the other trainer ranks just join the trainer-side
        # collectives in `_broadcast`.
        if not engine.is_sender:
            return engine
        # Upstream's `NCCLTrainerInitInfo` has no `rank_offset`: the single
        # trainer sender is rank 0, so the workers start at rank 1.
        worker_init_info = HCCLWeightTransferInitInfo(
            master_address=init_info.master_address,
            master_port=init_info.master_port,
            rank_offset=1,
            world_size=init_info.world_size,
            packed=init_info.packed,
            packed_buffer_size_bytes=init_info.packed_buffer_size_bytes,
            packed_num_buffers=init_info.packed_num_buffers,
        )
        # The worker-side init blocks inside the HCCL rendezvous, so drive the
        # RPC on a side thread while this rank opens rank 0.
        executor = ThreadPoolExecutor(max_workers=1)
        try:
            future = executor.submit(client.init_weight_transfer_engine, worker_init_payload(worker_init_info))
            engine.model_update_group = hccl_common.trainer_init(init_info, rank=init_info.rank)
            future.result()
        except BaseException:
            with suppress(Exception):
                engine.shutdown()
            raise
        finally:
            executor.shutdown(wait=False)
        return engine

    def send_weights(self) -> None:
        """Drive one round trip through upstream, refusing a dead group.

        A failed round trip leaves the HCCL group unusable, so the engine
        refuses further sends instead of hanging on a peer that never arrives.
        """
        if self._state != "ready":
            raise RuntimeError(f"HCCL trainer is {self._state}; create a new engine before sending again")
        try:
            super().send_weights()
        except BaseException:
            self._state = "failed"
            with suppress(Exception):
                self._release_group()
            raise

    def _broadcast(self, source: WeightSource, meta: list[ParamMeta]) -> None:
        with torch.npu.device(self.device):
            if not self.is_sender:
                # Non-sender ranks only replay the source to stay in the
                # trainer-side collective (an FSDP `full_tensor()` gather, say).
                super()._broadcast(source, meta)
                return
            if self.model_update_group is None:
                raise RuntimeError("HCCL trainer communicator is not initialized")
            pairs = self._checked_iter(source, meta)
            if self.packed:
                packed_broadcast_producer(
                    iterator=pairs,
                    group=self.model_update_group,
                    src=0,
                    post_iter_func=lambda item: item[1].detach().to(self.device).contiguous(),
                    buffer_size_bytes=self.packed_buffer_size_bytes,
                    num_buffers=self.packed_num_buffers,
                )
            else:
                stream = torch.npu.current_stream()
                for _name, tensor in pairs:
                    send = tensor.detach().to(self.device).contiguous()
                    self.model_update_group.broadcast(send, src=0, stream=stream)
                    # Keep the staging storage alive until HCCL has read it.
                    stream.synchronize()

    def _post_send_sync(self) -> None:
        """Wait for this rank's transfer work to land before returning."""
        if torch.npu.is_available():
            with torch.npu.device(self.device):
                torch.npu.current_stream().synchronize()

    def _release_group(self) -> None:
        # HCCL counterpart of NCCLTrainerWeightTransferEngine.shutdown: close
        # the communicator explicitly before dropping the reference.
        group, self.model_update_group = getattr(self, "model_update_group", None), None
        if group is not None:
            group.close()

    def shutdown(self) -> None:
        self._state = "closed"
        self._release_group()
