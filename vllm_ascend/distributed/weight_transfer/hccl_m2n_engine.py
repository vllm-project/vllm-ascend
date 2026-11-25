# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Ascend transport for the upstream M2N weight-transfer protocol."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing, suppress
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

import torch
from vllm.distributed.weight_transfer.m2n_common import (
    DESTINATION_SHARD_AXIS,
    REPLICATED,
    M2NLayout,
    M2NMesh,
    M2NParamMeta,
    M2NWireParam,
    Placements,
    check_complete_plan_agreement,
    check_source_plan_agreement,
    check_transferable,
    expected_local_shape,
    resolve_layout,
    source_plan_digest,
    validate_layout,
)
from vllm.distributed.weight_transfer.m2n_common import (
    SUPPORTED_DTYPES as UPSTREAM_SUPPORTED_DTYPES,
)
from vllm.distributed.weight_transfer.m2n_engine import (
    M2NWeightTransferEngine,
    M2NWeightTransferInitInfo,
    M2NWeightTransferUpdateInfo,
)
from vllm.distributed.weight_transfer.m2n_layout import M2NDestination, resolve_parameter_destinations
from vllm.distributed.weight_transfer.m2n_trainer import M2NTrainerInitInfo, M2NTrainerWeightTransferEngine
from vllm.distributed.weight_transfer.nccl_common import (
    NCCLWeightTransferInitInfo,
    stateless_init_metadata_group,
    worker_init_metadata_group,
)

from vllm_ascend.distributed.device_communicators.pyhccl import PyHcclCommunicator
from vllm_ascend.distributed.device_communicators.pyhccl_wrapper import HCCLLibrary, hcclDataTypeEnum
from vllm_ascend.distributed.weight_transfer.hccl_m2n_packed import plan_batches, transfer_packed_parameters
from vllm_ascend.distributed.weight_transfer.hccl_m2n_transport import (
    TransferPiece,
    plan_rank_transfers,
    transfer_parameter,
)

if TYPE_CHECKING:
    from vllm.distributed.utils import StatelessProcessGroup
    from vllm.distributed.weight_transfer.base import WeightSource


def _is_hccl_dtype(dtype: torch.dtype) -> bool:
    try:
        hcclDataTypeEnum.from_torch(dtype)
    except ValueError:
        return False
    return True


SUPPORTED_DTYPES: frozenset[torch.dtype] = frozenset(
    dtype for dtype in UPSTREAM_SUPPORTED_DTYPES if _is_hccl_dtype(dtype)
)
ACL_FORMAT_ND = 2

# On a failed non-packed stream drain, keep tensors alive until process exit
# because raw HCCL pointers are invisible to PyTorch's NPU allocator.
_UNSYNCHRONIZED_TENSORS: list[torch.Tensor] = []


def _validate_hccl_options(info: HCCLM2NTrainerInitInfo | HCCLM2NWeightTransferInitInfo) -> None:
    if info.nccl_unique_id_b64 is not None or info.max_cta is not None:
        raise ValueError("hccl_m2n uses TCP rendezvous and does not accept NCCL runtime options")
    if not isinstance(info.direct_write, bool):
        raise ValueError("hccl_m2n direct_write must be a boolean")
    if type(info.packed_buffer_size_bytes) is not int or info.packed_buffer_size_bytes < 0:
        raise ValueError("hccl_m2n packed_buffer_size_bytes must be a nonnegative integer")


@dataclass(kw_only=True)
class HCCLM2NTrainerInitInfo(M2NTrainerInitInfo):
    backend: ClassVar[str] = "hccl_m2n"
    direct_write: bool = True
    # Budget in logical checkpoint bytes per batch, with 0 disabling batching.
    # In packed mode the source must preserve each yielded shard until the
    # update finishes, since packing consumes several yielded shards together.
    packed_buffer_size_bytes: int = 0

    def __post_init__(self) -> None:
        super().__post_init__()
        _validate_hccl_options(self)


@dataclass(kw_only=True)
class HCCLM2NWeightTransferInitInfo(M2NWeightTransferInitInfo):
    direct_write: bool = True
    packed_buffer_size_bytes: int = 0

    def __post_init__(self) -> None:
        super().__post_init__()
        _validate_hccl_options(self)


def _is_nd(tensor: torch.Tensor) -> bool:
    import torch_npu

    return tensor.device.type == "npu" and torch_npu.get_npu_format(tensor) == ACL_FORMAT_ND


def _require_nd(tensor: torch.Tensor) -> None:
    if not _is_nd(tensor):
        raise ValueError("hccl_m2n requires ND tensors on NPU")


def _validate_hccl_transfer_plan(
    metas: Sequence[M2NParamMeta],
    dst_mesh: M2NMesh,
    dst_placements: Sequence[Placements | None],
) -> None:
    """Validate layouts using upstream checks without NCCL kernel shard caps."""
    if len(dst_placements) != len(metas):
        raise ValueError(f"destination plan has {len(dst_placements)} parameters, expected {len(metas)}")
    for meta, placement in zip(metas, dst_placements):
        if meta.dtype not in SUPPORTED_DTYPES:
            raise ValueError(f"hccl_m2n does not support {meta.dtype} for {meta.name!r}")
        check_transferable(meta.name, meta.dtype, meta.shape)
        src = resolve_layout(
            meta.source_layout.mesh,
            meta.source_layout.placements,
            f"parameter '{meta.name}' source placements",
        )
        validate_layout(*src, meta.shape, f"parameter '{meta.name}' source layout")
        dst = resolve_layout(
            dst_mesh,
            placement,
            f"parameter '{meta.name}' destination placements",
        )
        validate_layout(*dst, meta.shape, f"parameter '{meta.name}' destination layout")


@dataclass
class _TransferSession:
    group: PyHcclCommunicator
    pieces: list[tuple[TransferPiece, ...]]
    batches: tuple[range, ...] = ()
    packed_buffer_size_bytes: int = 0
    comm_stream: torch.npu.Stream | None = None

    def close(self) -> None:
        self.comm_stream = None
        self.group.close()

    def transfer(
        self,
        stream: torch.npu.Stream,
        tensors: Iterator[torch.Tensor],
        *,
        send: bool,
        indices: Sequence[int] | None = None,
        metas: Sequence[M2NParamMeta] | None = None,
        needs_host_sync: Sequence[bool] | None = None,
    ) -> Iterator[torch.Tensor]:
        if indices is None:
            selected_pieces: Sequence[tuple[TransferPiece, ...]] = self.pieces
            selected_batches: Sequence[range] = self.batches
        else:
            selected_pieces = [self.pieces[index] for index in indices]
            selected_batches = ()
            if self.packed_buffer_size_bytes and metas is not None:
                candidate = plan_batches([metas[index] for index in indices], self.packed_buffer_size_bytes)
                if any(len(batch) > 1 for batch in candidate):
                    selected_batches = candidate

        if selected_batches:
            if self.comm_stream is None:
                self.comm_stream = torch.npu.Stream(device=self.group.device)
            yield from transfer_packed_parameters(
                self.group,
                stream,
                tensors,
                selected_pieces,
                selected_batches,
                send=send,
                comm_stream=self.comm_stream,
                needs_host_sync=needs_host_sync,
            )
            return

        keepalive: list[torch.Tensor] = []
        try:
            for index, (tensor, pieces) in enumerate(zip(tensors, selected_pieces, strict=True)):
                sync_now = not send and (needs_host_sync is None or needs_host_sync[index])
                if sync_now:
                    transfer_parameter(self.group, stream, tensor, pieces, send=send, sync=True)
                    keepalive.clear()
                else:
                    transfer_parameter(
                        self.group,
                        stream,
                        tensor,
                        pieces,
                        send=send,
                        sync=False,
                        keepalive=keepalive,
                    )
                yield tensor
        finally:
            if keepalive:
                try:
                    stream.synchronize()
                except BaseException:
                    _UNSYNCHRONIZED_TENSORS.extend(keepalive)
                    raise
                keepalive.clear()


def _open_session(
    group: StatelessProcessGroup,
    device: torch.device | None,
    metas: Sequence[M2NParamMeta],
    source_digest: str,
    dst_mesh: M2NMesh,
    local_placements: list[Placements | None] | None = None,
    local_error: str | None = None,
    *,
    packed_buffer_size_bytes: int = 0,
) -> _TransferSession:
    # The first inference rank publishes placements on the metadata group.
    # This keeps placement and runtime checks ahead of native HCCL init.
    root = dst_mesh.start_rank
    placements = group.broadcast_obj(local_placements if group.rank == root else None, src=root)
    pieces = []
    batches: tuple[range, ...] = ()
    if packed_buffer_size_bytes:
        # Agree on the wire protocol as well as the upstream layouts before
        # either side opens HCCL. Preserve the digest for the original path.
        source_digest = f"{source_digest}:packed:{packed_buffer_size_bytes}"
    if local_error is None:
        try:
            if placements is None:
                raise RuntimeError("The first inference worker could not resolve the destination layout")
            if local_placements is not None and local_placements != placements:
                raise ValueError("Inference workers resolved different hccl_m2n destination layouts")
            if device is None or device.type != "npu" or device.index is None:
                raise ValueError("hccl_m2n needs an assigned NPU device")
            _validate_hccl_transfer_plan(metas, dst_mesh, placements)
            pieces = [
                plan_rank_transfers(
                    meta.shape,
                    meta.source_layout,
                    M2NLayout(dst_mesh, placement),
                    group.rank,
                    param_index=index,
                )
                for index, (meta, placement) in enumerate(zip(metas, placements))
            ]
            if packed_buffer_size_bytes:
                candidate_batches = plan_batches(metas, packed_buffer_size_bytes)
                if any(len(batch) > 1 for batch in candidate_batches):
                    batches = candidate_batches
            HCCLLibrary().require_batch_send_recv()
        except Exception as exc:
            local_error = f"{type(exc).__name__}: {exc}"

    check_complete_plan_agreement(group, source_digest, dst_mesh, placements or [], local_error)
    assert device is not None
    communicator = PyHcclCommunicator(group, device)
    if not communicator.available or communicator.disabled:
        raise RuntimeError("HCCL M2N communicator is unavailable")
    return _TransferSession(communicator, pieces, batches, packed_buffer_size_bytes=packed_buffer_size_bytes)


class HCCLM2NTrainerWeightTransferEngine(M2NTrainerWeightTransferEngine):
    """Reuse upstream orchestration and send local shards over HCCL."""

    init_info_cls = HCCLM2NTrainerInitInfo
    _failed = False

    def _worker_init_info(self, init_info: HCCLM2NTrainerInitInfo) -> dict[str, Any]:
        payload = {**super()._worker_init_info(init_info), "direct_write": init_info.direct_write}
        if init_info.packed_buffer_size_bytes:
            payload["packed_buffer_size_bytes"] = init_info.packed_buffer_size_bytes
        return payload

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
            session = _open_session(
                metadata_group,
                device,
                self._metas,
                digest,
                dst_mesh,
                local_error=device_error,
                packed_buffer_size_bytes=init_info.packed_buffer_size_bytes,
            )
            if worker_init is not None:
                worker_init.result()
        except BaseException:
            if session is not None:
                with suppress(Exception):
                    session.close()
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
        source = self.source
        device = session.group.device

        def local_shards() -> Iterator[torch.Tensor]:
            for (name, shard), meta in zip(source.iter_local(), self._metas, strict=True):
                if name != meta.name or shard.dtype != meta.dtype or shard.shape != expected_local_shape(meta):
                    raise ValueError(f"hccl_m2n source yielded a shard that differs from the plan for {meta.name!r}")
                if shard.device != device:
                    raise ValueError(f"hccl_m2n source shard {name!r} must be on {device}")
                _require_nd(shard)
                yield shard

        with torch.npu.device(device):
            torch.npu.synchronize(device)
            stream = torch.npu.current_stream(device)
            for _ in session.transfer(stream, local_shards(), send=True):
                pass

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
        self._dst_placements = []
        self._dst_mesh = None
        self._metas = []
        self._wire_params = []
        self._source_digest = ""
        self._failed = False
        if session is not None:
            session.close()


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
            metadata_group,
            self.device,
            self._metas,
            digest,
            dst_mesh,
            placements,
            destination_error,
            packed_buffer_size_bytes=init_info.packed_buffer_size_bytes,
        )

    def _prepare_hccl_destinations(self, init_info: HCCLM2NWeightTransferInitInfo) -> None:
        config = self.parallel_config
        self._dst_mesh = M2NMesh(tuple(init_info.dst_mesh_dims), init_info.rank_offset)
        expected_workers = config.data_parallel_size * config.world_size
        if self._dst_mesh.size != expected_workers:
            raise ValueError(
                f"hccl_m2n dst_mesh_dims {tuple(init_info.dst_mesh_dims)} must cover {expected_workers} workers"
            )

        expected_tp_mesh = (config.data_parallel_size, config.tensor_parallel_size)
        allow_direct = (
            init_info.direct_write
            and config.pipeline_parallel_size == 1
            and config.world_size == config.tensor_parallel_size
            and not config.enable_expert_parallel
            and not config.enable_eplb
            and not getattr(self.model_config, "is_moe", False)
            and getattr(self.model_config, "quantization", None) is None
            and not getattr(self.model_config, "enable_sleep_mode", False)
            and tuple(init_info.dst_mesh_dims) == expected_tp_mesh
        )
        shard_axis_size = self._dst_mesh.dims[DESTINATION_SHARD_AXIS]
        destinations = resolve_parameter_destinations(
            self.model,
            [m.name for m in self._metas],
            [m.dtype for m in self._metas],
            [m.shape for m in self._metas],
            num_workers=self._dst_mesh.size,
            shard_axis_size=shard_axis_size,
            allow_direct=allow_direct,
        )
        if allow_direct:
            # Demote only modules that hold precomputed FP32 weight copies or
            # non-ND parameter storage, preserving module-level reload semantics
            # without forcing unrelated layers onto the fallback path.
            precast_param_ids = {
                id(param)
                for module in self.model.modules()
                if getattr(module, "precast_fp32_weight", False)
                for param in module.parameters()
            }
            params_by_name = dict(self.model.named_parameters()) if precast_param_ids else {}
            fallback_modules = {
                destination.name.rpartition(".")[0]
                for destination in destinations
                if destination.direct
                and (
                    (destination.tensor is not None and not _is_nd(destination.tensor))
                    or (
                        destination.name in params_by_name and id(params_by_name[destination.name]) in precast_param_ids
                    )
                )
            }
            if fallback_modules:
                destinations = [
                    M2NDestination(destination.name, REPLICATED, None)
                    if destination.direct and destination.name.rpartition(".")[0] in fallback_modules
                    else destination
                    for destination in destinations
                ]

        self._parameter_destinations = destinations
        self._index = {meta.name: index for index, meta in enumerate(self._metas)}
        self._uses_load_weights = any(not destination.direct for destination in self._parameter_destinations)

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
        if not isinstance(update_info.names, list) or any(not isinstance(name, str) for name in update_info.names):
            raise TypeError("hccl_m2n update names must be a list of strings")

        if not self._index and self._metas:
            self._index = {meta.name: index for index, meta in enumerate(self._metas)}
        requested_indices: list[int] = []
        for name in update_info.names:
            index = self._index.get(name)
            if index is None:
                raise ValueError(
                    f"parameter '{name}' was not declared at init; the "
                    "trainer must send the same parameter set it announced"
                )
            requested_indices.append(index)

        from vllm.model_executor.model_loader.mtp_validation import disable_mtp_completeness_check

        is_full_order = requested_indices == list(range(len(self._metas)))
        stream = torch.npu.current_stream(self.device)
        transferred = 0
        needs_host_sync = [not self._parameter_destinations[index].direct for index in requested_indices]

        def buffers() -> Iterator[torch.Tensor]:
            for index in requested_indices:
                meta = self._metas[index]
                destination = self._parameter_destinations[index]
                buffer = destination.tensor
                if buffer is None:
                    buffer = torch.empty(meta.shape, dtype=meta.dtype, device=self.device)
                yield buffer

        received = session.transfer(
            stream,
            buffers(),
            send=False,
            indices=None if is_full_order else requested_indices,
            metas=self._metas,
            needs_host_sync=needs_host_sync,
        )

        def receive() -> Iterator[tuple[str, torch.Tensor]]:
            nonlocal transferred
            for index, buffer in zip(requested_indices, received, strict=True):
                meta = self._metas[index]
                destination = self._parameter_destinations[index]
                transferred += 1
                if not destination.direct:
                    yield meta.name, buffer

        has_fallback = any(needs_host_sync)
        with closing(received), disable_mtp_completeness_check():
            if has_fallback:
                self.model.load_weights(receive())
            else:
                for _ in receive():
                    pass
        if transferred != len(requested_indices):
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
        self._metas = []
        self._dst_mesh = None
        self._index = {}
        self._uses_load_weights = False
        self._received = False
        self._failed = False
        if session is not None:
            session.close()
