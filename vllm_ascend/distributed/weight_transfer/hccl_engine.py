# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HCCL adapters for upstream vLLM's dense NCCL weight-transfer engines.

Reload lifecycle, update payloads and metadata checks stay upstream. Only NPU
operations, HCCL rendezvous and legacy wire parameters are local. Packed helpers
and layerwise cross-chunk behavior are unchanged.
"""

from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext, suppress
from dataclasses import dataclass
from typing import Any, ClassVar

import torch
from typing_extensions import Self
from vllm.distributed.weight_transfer.base import ParamMeta, VLLMWeightSyncClient, WeightSource, WeightTransferInitInfo
from vllm.distributed.weight_transfer.nccl_common import worker_init_payload
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
    packed_broadcast_consumer,
    packed_broadcast_producer,
)


def _validate_packed_params(info: Any) -> None:
    if info.packed is not None and not isinstance(info.packed, bool):
        raise ValueError("packed must be a boolean or None")
    for name in ("packed_buffer_size_bytes", "packed_num_buffers"):
        value = getattr(info, name)
        if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value <= 0):
            raise ValueError(f"{name} must be a positive integer")


@dataclass
class HCCLWeightTransferInitInfo(WeightTransferInitInfo):
    """TCP rendezvous; omitted packed fields remain configurable per update."""

    master_address: str
    master_port: int
    rank_offset: int
    world_size: int
    packed: bool | None = None
    packed_buffer_size_bytes: int | None = None
    packed_num_buffers: int | None = None

    def __post_init__(self) -> None:
        _validate_packed_params(self)


@dataclass
class HCCLTrainerInitInfo(NCCLTrainerInitInfo):
    """Upstream trainer contract with the existing HCCL unpacked default."""

    backend: ClassVar[str] = "hccl"
    packed: bool = False

    def __post_init__(self) -> None:
        _validate_packed_params(self)


@dataclass
class HCCLTrainerSendWeightsArgs:
    """Backward-compatible arguments for the static trainer entry point."""

    group: Any
    src: int = 0
    post_iter_func: Callable[[tuple[str, torch.Tensor]], torch.Tensor] | None = None
    packed: bool = False
    stream: Any = None
    packed_buffer_size_bytes: int = DEFAULT_PACKED_BUFFER_SIZE_BYTES
    packed_num_buffers: int = DEFAULT_PACKED_NUM_BUFFERS


@dataclass
class HCCLWeightTransferUpdateInfo(NCCLWeightTransferUpdateInfo):
    """Upstream metadata plus the legacy per-update packed wire parameters."""

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
    """Inherit reload lifecycle; adapt rendezvous and receive operations."""

    init_info_cls = HCCLWeightTransferInitInfo  # type: ignore[assignment]
    update_info_cls = HCCLWeightTransferUpdateInfo
    _stateless_init_process_group = staticmethod(hccl_common.stateless_init_process_group)

    def init_transfer_engine(self, init_info: HCCLWeightTransferInitInfo) -> None:  # type: ignore[override]
        self.shutdown()
        if init_info.world_size < 2:
            raise ValueError("HCCL weight transfer requires at least one trainer and one worker")
        self._init_info = init_info
        parallel = self.parallel_config
        rank = parallel.data_parallel_index * parallel.world_size + parallel.rank + init_info.rank_offset
        with torch.npu.device(self.device):
            self.model_update_group = self._stateless_init_process_group(  # type: ignore[assignment]
                init_info.master_address,
                init_info.master_port,
                rank,
                init_info.world_size,
                device=torch.accelerator.current_device_index(),
            )

    def _wire_params(self, update: HCCLWeightTransferUpdateInfo) -> dict[str, Any]:
        params = dict(
            packed=False,
            packed_buffer_size_bytes=DEFAULT_PACKED_BUFFER_SIZE_BYTES,
            packed_num_buffers=DEFAULT_PACKED_NUM_BUFFERS,
        )
        for name in params:
            fixed, requested = getattr(self._init_info, name), getattr(update, name)
            if fixed is not None and requested is not None and fixed != requested:
                raise ValueError(f"HCCL update {name} conflicts with the value fixed at init")
            if fixed is not None or requested is not None:
                params[name] = fixed if fixed is not None else requested
        return params

    def receive_weights(self, update_info: HCCLWeightTransferUpdateInfo) -> None:
        if self.model_update_group is None:
            raise RuntimeError("HCCL weight transfer is not initialized")
        params = self._wire_params(update_info)
        from vllm.model_executor.model_loader.mtp_validation import disable_mtp_completeness_check

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
        # The inherited wrapper's final synchronization uses the ambient device.
        with torch.npu.device(self.device):
            super().update_weights(update_info)

    def shutdown(self) -> None:
        group, self.model_update_group = getattr(self, "model_update_group", None), None
        if group is not None:
            group.close()

    @staticmethod
    def trainer_init(init_info: HCCLWeightTransferInitInfo | dict) -> Any:
        """Legacy rendezvous for trainers that drive their own update RPCs."""
        return hccl_common.trainer_init(init_info)

    @staticmethod
    def trainer_send_weights(
        iterator: Iterator[tuple[str, torch.Tensor]],
        trainer_args: dict[str, Any] | HCCLTrainerSendWeightsArgs,
    ) -> None:
        args = HCCLTrainerSendWeightsArgs(**trainer_args) if isinstance(trainer_args, dict) else trainer_args
        post = args.post_iter_func or (lambda item: item[1])
        device = getattr(args.group, "device", None)
        with torch.npu.device(device) if device is not None else nullcontext():
            if args.packed:
                packed_broadcast_producer(
                    iterator=iterator,
                    group=args.group,
                    src=args.src,
                    post_iter_func=post,
                    buffer_size_bytes=args.packed_buffer_size_bytes,
                    num_buffers=args.packed_num_buffers,
                )
            else:
                for item in iterator:
                    tensor = post(item)
                    args.group.broadcast(tensor, src=args.src, stream=args.stream or torch.npu.current_stream())


class HCCLTrainerWeightTransferEngine(NCCLTrainerWeightTransferEngine):
    """Reuse upstream send orchestration and metadata checks, not CUDA calls."""

    init_info_cls = HCCLTrainerInitInfo

    def __init__(self, *, device: torch.device | None = None, packed: bool = False, **kwargs: Any) -> None:
        super().__init__(packed=packed, **kwargs)
        if device is None:
            index = torch.accelerator.current_device_index()
            if index is None:
                raise ValueError("HCCL trainer requires a current NPU device")
            device = torch.device("npu", index)
        self.device = device
        self.model_update_group: Any = None
        self._state = "ready"

    @classmethod
    def trainer_init(
        cls, init_info: HCCLTrainerInitInfo, *, client: VLLMWeightSyncClient, source: WeightSource | None = None
    ) -> Self:
        if source is None or init_info.world_size < 2:
            raise ValueError("HCCL trainer requires a WeightSource and at least one worker")
        engine = cls(
            client=client,
            source=source,
            is_sender=init_info.is_sender,
            packed=init_info.packed,
            packed_buffer_size_bytes=init_info.packed_buffer_size_bytes,
            packed_num_buffers=init_info.packed_num_buffers,
        )
        if not engine.is_sender:
            return engine
        info = HCCLWeightTransferInitInfo(
            init_info.master_address,
            init_info.master_port,
            1,
            init_info.world_size,
            init_info.packed,
            init_info.packed_buffer_size_bytes,
            init_info.packed_num_buffers,
        )
        executor = ThreadPoolExecutor(max_workers=1)
        try:
            future = executor.submit(client.init_weight_transfer_engine, worker_init_payload(info))
            engine.model_update_group = hccl_common.trainer_init(init_info, rank=0)
            future.result()
        except BaseException:
            with suppress(Exception):
                engine.shutdown()
            raise
        finally:
            executor.shutdown(wait=False)
        return engine

    def send_weights(self) -> None:
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
        if len({param.name for param in meta}) != len(meta):
            raise ValueError("WeightSource.metadata() contains duplicate parameter names")
        with torch.npu.device(self.device):
            if not self.is_sender:
                super()._broadcast(source, meta)
                return
            if self.model_update_group is None:
                raise RuntimeError("HCCL trainer communicator is not initialized")
            HCCLWeightTransferEngine.trainer_send_weights(
                self._checked_iter(source, meta),
                HCCLTrainerSendWeightsArgs(
                    group=self.model_update_group,
                    packed=self.packed,
                    post_iter_func=lambda item: item[1].detach().to(self.device).contiguous(),
                    packed_buffer_size_bytes=self.packed_buffer_size_bytes,
                    packed_num_buffers=self.packed_num_buffers,
                ),
            )

    def _post_send_sync(self) -> None:
        with torch.npu.device(self.device):
            torch.npu.current_stream().synchronize()

    def _release_group(self) -> None:
        group, self.model_update_group = self.model_update_group, None
        if group is not None:
            group.close()

    def shutdown(self) -> None:
        self._state = "closed"
        self._release_group()
