# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Ascend transport for the upstream M2N weight-transfer protocol."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

import torch
from vllm.distributed.weight_transfer.m2n_common import (
    M2NLayout,
    M2NMesh,
    M2NParamMeta,
    M2NWireParam,
    Placements,
    check_complete_plan_agreement,
    check_source_plan_agreement,
    expected_local_shape,
    source_plan_digest,
    validate_transfer_plan,
)
from vllm.distributed.weight_transfer.m2n_engine import (
    M2NWeightTransferEngine,
    M2NWeightTransferInitInfo,
    M2NWeightTransferUpdateInfo,
)
from vllm.distributed.weight_transfer.m2n_layout import M2NDestination
from vllm.distributed.weight_transfer.m2n_trainer import M2NTrainerInitInfo, M2NTrainerWeightTransferEngine
from vllm.distributed.weight_transfer.nccl_common import (
    NCCLWeightTransferInitInfo,
    stateless_init_metadata_group,
    worker_init_metadata_group,
)

from vllm_ascend.distributed.device_communicators.pyhccl import PyHcclCommunicator
from vllm_ascend.distributed.device_communicators.pyhccl_wrapper import HCCLLibrary
from vllm_ascend.distributed.weight_transfer.hccl_m2n_transport import (
    TransferPiece,
    plan_rank_transfers,
    transfer_parameter,
)

if TYPE_CHECKING:
    from vllm.distributed.utils import StatelessProcessGroup
    from vllm.distributed.weight_transfer.base import WeightSource

SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
ACL_FORMAT_ND = 2


def _validate_hccl_options(info: HCCLM2NTrainerInitInfo | HCCLM2NWeightTransferInitInfo) -> None:
    if info.nccl_unique_id_b64 is not None or info.max_cta is not None:
        raise ValueError("hccl_m2n uses TCP rendezvous and does not accept NCCL runtime options")
    if not isinstance(info.direct_write, bool):
        raise ValueError("hccl_m2n direct_write must be a boolean")


@dataclass(kw_only=True)
class HCCLM2NTrainerInitInfo(M2NTrainerInitInfo):
    backend: ClassVar[str] = "hccl_m2n"
    direct_write: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()
        _validate_hccl_options(self)


@dataclass(kw_only=True)
class HCCLM2NWeightTransferInitInfo(M2NWeightTransferInitInfo):
    direct_write: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()
        _validate_hccl_options(self)


def _require_nd(tensor: torch.Tensor) -> None:
    import torch_npu

    if tensor.device.type != "npu" or torch_npu.get_npu_format(tensor) != ACL_FORMAT_ND:
        raise ValueError("hccl_m2n requires ND tensors on NPU")


@dataclass
class _TransferSession:
    group: PyHcclCommunicator
    pieces: list[tuple[TransferPiece, ...]]


def _open_session(
    group: StatelessProcessGroup,
    device: torch.device | None,
    metas: Sequence[M2NParamMeta],
    source_digest: str,
    dst_mesh: M2NMesh,
    local_placements: list[Placements | None] | None = None,
    local_error: str | None = None,
) -> _TransferSession:
    # The first inference rank publishes placements on the metadata group.
    # This keeps placement and runtime checks ahead of native HCCL init.
    root = dst_mesh.start_rank
    placements = group.broadcast_obj(local_placements if group.rank == root else None, src=root)
    pieces = []
    if local_error is None:
        try:
            if placements is None:
                raise RuntimeError("The first inference worker could not resolve the destination layout")
            if local_placements is not None and local_placements != placements:
                raise ValueError("Inference workers resolved different hccl_m2n destination layouts")
            if device is None or device.type != "npu" or device.index is None:
                raise ValueError("hccl_m2n needs an assigned NPU device")
            for meta in metas:
                if meta.dtype not in SUPPORTED_DTYPES:
                    raise ValueError(f"hccl_m2n does not support {meta.dtype} for {meta.name!r}")
            validate_transfer_plan(metas, dst_mesh, placements)
            pieces = [
                plan_rank_transfers(meta.shape, meta.source_layout, M2NLayout(dst_mesh, placement), group.rank)
                for meta, placement in zip(metas, placements)
            ]
            HCCLLibrary().require_batch_send_recv()
        except Exception as exc:
            local_error = f"{type(exc).__name__}: {exc}"

    check_complete_plan_agreement(group, source_digest, dst_mesh, placements or [], local_error)
    assert device is not None
    communicator = PyHcclCommunicator(group, device)
    if not communicator.available or communicator.disabled:
        raise RuntimeError("HCCL M2N communicator is unavailable")
    return _TransferSession(communicator, pieces)


class HCCLM2NTrainerWeightTransferEngine(M2NTrainerWeightTransferEngine):
    """Reuse upstream orchestration and send local shards over HCCL."""

    init_info_cls = HCCLM2NTrainerInitInfo
    _failed = False

    def _worker_init_info(self, init_info: HCCLM2NTrainerInitInfo) -> dict[str, Any]:
        return {**super()._worker_init_info(init_info), "direct_write": init_info.direct_write}

    def _initialize(self, init_info: HCCLM2NTrainerInitInfo, source: WeightSource) -> None:
        self._num_trainer_ranks = init_info.num_trainer_ranks
        source_error = None
        try:
            if not source.implements_iter_local():
                raise TypeError("hccl_m2n requires a WeightSource with iter_local()")
            self._prepare_source_plan(source, self._num_trainer_ranks)
        except Exception as exc:
            source_error = f"{type(exc).__name__}: {exc}"
            self._wire_params = []
            self._source_digest = source_plan_digest([])

        executor = None
        worker_init = None
        session = None
        reservation = init_info.listen_socket
        try:
            if self.is_sender:
                executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="hccl-m2n-init")
                worker_init = executor.submit(
                    self.client.init_weight_transfer_engine, self._worker_init_info(init_info)
                )
            metadata_group = stateless_init_metadata_group(
                init_info.master_address,
                init_info.master_port,
                self.rank,
                init_info.world_size,
                listen_socket=reservation,
            )

            digest = check_source_plan_agreement(
                metadata_group, self._wire_params, expected_digest=self._source_digest, local_error=source_error
            )
            dst_mesh = M2NMesh(init_info.destination_mesh_dims, self._num_trainer_ranks)
            device = None
            device_error = None
            try:
                device = torch.device("npu", torch.npu.current_device())
            except Exception as exc:
                device_error = f"{type(exc).__name__}: {exc}"
            session = _open_session(metadata_group, device, self._metas, digest, dst_mesh, local_error=device_error)
            if worker_init is not None:
                worker_init.result()
        except BaseException:
            if session is not None:
                with suppress(Exception):
                    session.group.close()
            raise
        finally:
            if executor is not None:
                executor.shutdown(wait=False, cancel_futures=True)
            if reservation is not None:
                reservation.close()
                init_info.listen_socket = None

        assert session is not None
        self._metadata_group = metadata_group
        self.group = session.group
        self._handle = session

    @torch.no_grad()
    def _send(self) -> None:
        session: _TransferSession = self._handle
        assert self.source is not None
        device = session.group.device
        with torch.npu.device(device):
            torch.npu.synchronize(device)
            stream = torch.npu.current_stream(device)
            for (name, shard), meta, pieces in zip(self.source.iter_local(), self._metas, session.pieces, strict=True):
                if name != meta.name or shard.dtype != meta.dtype or shard.shape != expected_local_shape(meta):
                    raise ValueError(f"hccl_m2n source yielded a shard that differs from the plan for {meta.name!r}")
                if shard.device != device:
                    raise ValueError(f"hccl_m2n source shard {name!r} must be on {device}")
                _require_nd(shard)
                transfer_parameter(session.group, stream, shard, pieces, send=True)

    def send_weights(self) -> None:
        if self._failed or self._handle is None:
            raise RuntimeError("hccl_m2n trainer session is not ready")
        try:
            super().send_weights()
        except BaseException:
            self._failed = True
            raise

    def shutdown(self) -> None:
        session, self._handle = self._handle, None
        self.group = None
        self._metadata_group = None
        if session is not None:
            session.group.close()


class HCCLM2NWeightTransferEngine(M2NWeightTransferEngine):
    """Receive local shards directly or reconstruct checkpoint weights for the loader."""

    init_info_cls = HCCLM2NWeightTransferInitInfo
    _failed = False
    _received = False

    def init_transfer_engine(self, init_info: HCCLM2NWeightTransferInitInfo) -> None:
        if self._failed or self._handle is not None:
            raise RuntimeError("hccl_m2n requires a fresh worker session")
        metadata_group = worker_init_metadata_group(
            NCCLWeightTransferInitInfo(
                master_address=init_info.master_address,
                master_port=init_info.master_port,
                rank_offset=init_info.rank_offset,
                world_size=init_info.world_size,
            ),
            self.parallel_config,
        )
        wire_params: tuple[M2NWireParam, ...] = ()
        source_error = None
        try:
            wire_params = init_info.parse_wire_params()
            self._prepare_source_plan(wire_params, init_info.rank_offset)
        except Exception as exc:
            source_error = f"{type(exc).__name__}: {exc}"
        digest = check_source_plan_agreement(
            metadata_group, wire_params, expected_digest=init_info.source_digest, local_error=source_error
        )

        dst_mesh = M2NMesh(tuple(init_info.dst_mesh_dims), init_info.rank_offset)
        placements = None
        destination_error = None
        try:
            self._prepare_hccl_destinations(init_info)
            placements = [destination.placements for destination in self._parameter_destinations]
        except Exception as exc:
            destination_error = f"{type(exc).__name__}: {exc}"
        self._handle = _open_session(
            metadata_group, self.device, self._metas, digest, dst_mesh, placements, destination_error
        )

    def _prepare_hccl_destinations(self, init_info: HCCLM2NWeightTransferInitInfo) -> None:
        config = self.parallel_config
        if (
            config.pipeline_parallel_size != 1
            or config.world_size != config.tensor_parallel_size
            or config.enable_expert_parallel
            or config.enable_eplb
            or self.model_config.is_moe
            or self.model_config.quantization is not None
            or self.model_config.enable_sleep_mode
        ):
            raise ValueError("hccl_m2n currently supports dense unquantized models with TP and DP")
        expected_mesh = (config.data_parallel_size, config.tensor_parallel_size)
        if tuple(init_info.dst_mesh_dims) != expected_mesh:
            raise ValueError(f"hccl_m2n dst_mesh_dims must match (DP, TP) = {expected_mesh}")

        if init_info.direct_write:
            if any(getattr(layer, "precast_fp32_weight", False) for layer in self.model.modules()):
                raise ValueError("hccl_m2n direct_write requires weights without precomputed FP32 copies")
            self._prepare_destination_plan(init_info)
            for destination in self._parameter_destinations:
                if destination.tensor is not None:
                    _require_nd(destination.tensor)
        else:
            self._parameter_destinations = [M2NDestination(meta.name, None, None) for meta in self._metas]
            self._uses_load_weights = bool(self._metas)

    def start_weight_update(self) -> None:
        self._require_session()
        self._received = False
        try:
            with torch.npu.device(self.device):
                super().start_weight_update()
        except BaseException:
            self._failed = True
            raise

    def update_weights(self, update_info: dict[str, Any]) -> None:
        try:
            with torch.npu.device(self.device):
                super().update_weights(update_info)
        except BaseException:
            self._failed = True
            raise

    @torch.no_grad()
    def receive_weights(self, update_info: M2NWeightTransferUpdateInfo) -> None:
        session = self._require_session()
        if self._received or update_info.names != [meta.name for meta in self._metas]:
            raise ValueError("hccl_m2n requires exactly one update in the negotiated parameter order")

        from vllm.model_executor.model_loader.mtp_validation import disable_mtp_completeness_check

        stream = torch.npu.current_stream(self.device)
        transferred = 0

        def receive() -> Iterator[tuple[str, torch.Tensor]]:
            nonlocal transferred
            for meta, destination, pieces in zip(
                self._metas, self._parameter_destinations, session.pieces, strict=True
            ):
                buffer = destination.tensor
                if buffer is None:
                    buffer = torch.empty(meta.shape, dtype=meta.dtype, device=self.device)
                transfer_parameter(session.group, stream, buffer, pieces, send=False)
                transferred += 1
                if not destination.direct:
                    yield meta.name, buffer

        with disable_mtp_completeness_check():
            if self._uses_load_weights:
                self.model.load_weights(receive())
            else:
                for _ in receive():
                    pass
        if transferred != len(self._metas):
            raise RuntimeError("The model loader did not consume the complete hccl_m2n update")
        self._received = True

    def finish_weight_update(self) -> None:
        self._require_session()
        if not self._received:
            raise RuntimeError("hccl_m2n update has not been received")
        try:
            with torch.npu.device(self.device):
                super().finish_weight_update()
        except BaseException:
            self._failed = True
            raise

    def _require_session(self) -> _TransferSession:
        if self._failed or self._handle is None:
            raise RuntimeError("hccl_m2n worker session is not ready")
        return self._handle

    def shutdown(self) -> None:
        session, self._handle = self._handle, None
        self._parameter_destinations = []
        if session is not None:
            session.group.close()
