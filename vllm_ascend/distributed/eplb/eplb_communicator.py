# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""Ascend communicators for asynchronous EPLB."""

import contextlib
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed import P2POp, ProcessGroup, batch_isend_irecv
from vllm.distributed.eplb.eplb_communicator import (
    EplbCommunicator,
    TorchDistGlooStagedEplbCommunicator,
)
from vllm.distributed.utils import is_weak_contiguous
from vllm.logger import logger
from vllm.utils.gpu_sync_debug import gpu_sync_allowed
from vllm.utils.network_utils import get_ip, get_open_port, join_host_port

_HIXL_MEMORY_ALIGNMENT = 2 * 1024 * 1024
_HIXL_MAX_REGISTERED_REGIONS = 256
_TRANSFER_TIMEOUT_SECONDS = 300
_STATUS_POLL_SECONDS = 0.0005


@dataclass(frozen=True)
class _HixlTransferTiming:
    launch_ms: float
    transfer_ms: float
    confirmation_ms: float
    request_count: int
    transfer_bytes: int


class AscendGlooEplbCommunicator(TorchDistGlooStagedEplbCommunicator):
    """Gloo CPU-staging EPLB communicator for async mode on Ascend.

    Gloo uses CPU-side P2P and does not require the NCCL/HCCL buffer
    reservation collective that the upstream profile path runs. Disabling
    it also avoids passing Ascend's EplbExpertTensorList to all_gather,
    which does not implement the __torch_function__ protocol for
    distributed collectives.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._stream: torch.Stream | None = None
        self._pinned_staging_buffers: dict[tuple[torch.dtype, tuple[int, ...]], list[torch.Tensor]] = {}

    def set_stream(self, stream: torch.Stream | None) -> None:
        self._stream = stream

    def _acquire_staging_buffer(
        self,
        tensor: torch.Tensor,
        buffer_indices: dict[tuple[torch.dtype, tuple[int, ...]], int],
    ) -> torch.Tensor:
        key = tensor.dtype, tuple(tensor.shape)
        buffer_index = buffer_indices.get(key, 0)
        buffers = self._pinned_staging_buffers.setdefault(key, [])
        if buffer_index == len(buffers):
            buffers.append(torch.empty_like(tensor, device="cpu", pin_memory=True))
        buffer_indices[key] = buffer_index + 1
        return buffers[buffer_index]

    def execute(self) -> None:
        if not self._ops:
            return

        stream = self._stream
        p2p_ops: list[P2POp] = []
        recv_staging: list[tuple[torch.Tensor, torch.Tensor]] = []
        buffer_indices: dict[tuple[torch.dtype, tuple[int, ...]], int] = {}
        try:
            with stream if stream is not None else contextlib.nullcontext():
                for operation, tensor, peer_rank in self._ops:
                    cpu_tensor = self._acquire_staging_buffer(tensor, buffer_indices)
                    if operation == "send":
                        cpu_tensor.copy_(tensor, non_blocking=True)
                        p2p_ops.append(
                            P2POp(
                                dist.isend,
                                cpu_tensor,
                                group=self._cpu_group,
                                group_peer=peer_rank,
                            )
                        )
                    else:
                        p2p_ops.append(
                            P2POp(
                                dist.irecv,
                                cpu_tensor,
                                group=self._cpu_group,
                                group_peer=peer_rank,
                            )
                        )
                        recv_staging.append((tensor, cpu_tensor))
        finally:
            self._ops.clear()

        with gpu_sync_allowed():
            if stream is not None:
                stream.synchronize()
            else:
                torch.accelerator.current_stream().synchronize()

        for request in batch_isend_irecv(p2p_ops):
            request.wait()

        with stream if stream is not None else contextlib.nullcontext():
            for dst_tensor, cpu_tensor in recv_staging:
                dst_tensor.copy_(cpu_tensor, non_blocking=True)

    @property
    def needs_profile_buffer_reservation(self) -> bool:
        return False


class AscendHixlEplbCommunicator(EplbCommunicator):
    """Read expert weights directly between registered NPU allocations."""

    receiver_initiated = True

    def __init__(
        self,
        cpu_group: ProcessGroup,
        all_expert_weights: Sequence[Sequence[Any]],
        expert_buffer: Sequence[Any],
    ) -> None:
        try:
            import hixl  # type: ignore[import-not-found]
        except ImportError as error:
            raise RuntimeError(
                "HIXL EPLB requires the HIXL Python package from the CANN HIXL distribution"
            ) from error

        if not all_expert_weights or not all_expert_weights[0] or not expert_buffer:
            raise ValueError("HIXL EPLB requires expert weights and receive buffers")

        first_view = all_expert_weights[0][0]
        first_tensors = self._storage_tensors(first_view)
        first_tensor = first_tensors[0]
        if first_tensor.device.type != "npu" or first_tensor.ndim == 0 or first_tensor.shape[0] == 0:
            raise ValueError("HIXL EPLB requires non-empty NPU expert tensors")

        self._hixl = hixl
        self._cpu_group = cpu_group
        self._rank = cpu_group.rank()
        self._world_size = cpu_group.size()
        self._device = first_tensor.device
        self._num_local_experts = first_tensor.shape[0] if len(first_tensors) == 1 else len(first_tensors)
        self._engine: Any | None = None
        self._registered_handles: list[int] = []
        self._remote_engines: dict[int, str] = {}
        self._remote_send_meta: dict[int, dict[tuple[int, int], tuple[tuple[int, ...], int]]] = {}
        self._expert_to_src_row: list[dict[int, int]] | None = None
        self._layer_idx: int | None = None
        self._pending_reads: dict[int, list[tuple[int, int, int]]] = {}
        self._pending_bytes = 0

        self._validate_tensors(all_expert_weights, expert_buffer)
        self._initialize(all_expert_weights, expert_buffer)
        self._log_initialized()

    def _validate_tensors(
        self,
        all_expert_weights: Sequence[Sequence[Any]],
        expert_buffer: Sequence[Any],
    ) -> None:
        for layer_views in all_expert_weights:
            for view in layer_views:
                self._validate_view(view)
        for view in expert_buffer:
            self._validate_view(view)

    @staticmethod
    def _storage_tensors(view: Any) -> tuple[torch.Tensor, ...]:
        return (view,) if hasattr(view, "data_ptr") else tuple(view)

    def _validate_view(self, view: Any) -> None:
        tensors = self._storage_tensors(view)
        if not tensors:
            raise ValueError("HIXL EPLB does not support empty expert weight views")
        if len(tensors) not in (1, self._num_local_experts):
            raise ValueError("HIXL EPLB weight views must contain one storage or one tensor per local expert")
        for tensor in tensors:
            self._validate_storage(tensor)
        if len(tensors) == 1:
            if tensors[0].shape[0] != self._num_local_experts:
                raise ValueError("HIXL EPLB weight views must align their first dimension with local experts")
        elif any(tensor.nbytes != tensors[0].nbytes for tensor in tensors[1:]):
            raise ValueError("HIXL EPLB per-expert tensors in one weight view must have equal sizes")

    def _validate_storage(self, tensor: torch.Tensor) -> None:
        if tensor.device != self._device or tensor.ndim == 0 or not is_weak_contiguous(tensor):
            raise ValueError("HIXL EPLB tensors must share one contiguous slot-aligned NPU layout")

    def _iter_storage_tensors(self, views: Sequence[Any]) -> Iterator[torch.Tensor]:
        for view in views:
            yield from self._storage_tensors(view)

    def _initialize(
        self,
        all_expert_weights: Sequence[Sequence[Any]],
        expert_buffer: Sequence[Any],
    ) -> None:
        torch.npu.set_device(self._device)
        self._engine = self._hixl.Hixl()
        local_engine = join_host_port(get_ip(), get_open_port())
        try:
            self._check_status(
                self._engine.initialize(local_engine, {}),
                "initialize",
            )
            tensors = [
                tensor for layer_views in all_expert_weights for tensor in self._iter_storage_tensors(layer_views)
            ]
            tensors.extend(self._iter_storage_tensors(expert_buffer))
            self._register_tensor_ranges(tensors)
            self._exchange_remote_state(local_engine, all_expert_weights)
            self._connect_peers()
        except Exception:
            self._close()
            raise

    def _register_tensor_ranges(self, tensors: Sequence[torch.Tensor]) -> None:
        segments = sorted(
            {
                (int(segment["address"]), int(segment["total_size"]))
                for segment in torch.npu.memory_snapshot()
                if segment.get("device") == self._device.index
            }
        )
        ranges: list[tuple[int, int]] = []
        for tensor in tensors:
            segment = next(
                (
                    (address, size)
                    for address, size in segments
                    if address <= tensor.data_ptr() and tensor.data_ptr() + tensor.nbytes <= address + size
                ),
                None,
            )
            if segment is None:
                raise RuntimeError("HIXL EPLB could not resolve an allocator segment for every expert tensor")
            segment_address, segment_size = segment
            if segment_address % _HIXL_MEMORY_ALIGNMENT or segment_size % _HIXL_MEMORY_ALIGNMENT:
                raise RuntimeError("HIXL EPLB allocator segments must be 2 MiB aligned")
            tensor_start = tensor.data_ptr() // _HIXL_MEMORY_ALIGNMENT * _HIXL_MEMORY_ALIGNMENT
            tensor_end = (
                (tensor.data_ptr() + tensor.nbytes + _HIXL_MEMORY_ALIGNMENT - 1)
                // _HIXL_MEMORY_ALIGNMENT
                * _HIXL_MEMORY_ALIGNMENT
            )
            ranges.append((tensor_start, tensor_end))

        regions: list[tuple[int, int]] = []
        for start, end in sorted(ranges):
            if regions and start <= regions[-1][1]:
                regions[-1] = (regions[-1][0], max(end, regions[-1][1]))
            else:
                regions.append((start, end))
        if len(regions) > _HIXL_MAX_REGISTERED_REGIONS:
            raise RuntimeError(
                f"HIXL EPLB requires {len(regions)} memory registrations; "
                f"the HIXL limit is {_HIXL_MAX_REGISTERED_REGIONS}"
            )
        if self._rank == 0:
            registered_bytes = sum(end - start for start, end in regions)
            logger.info(
                "Registering %d NPU memory regions (%.2f GiB) for HIXL EPLB.",
                len(regions),
                registered_bytes / 1024**3,
            )
        for start, end in regions:
            self._register_region(start, end - start)

    def _register_region(self, address: int, size: int) -> None:
        assert self._engine is not None
        status, handle = self._engine.register_mem(
            self._hixl.MemDesc(address, size),
            self._hixl.MemType.MEM_DEVICE,
        )
        self._check_status(status, "register memory")
        self._registered_handles.append(handle)

    def _exchange_remote_state(
        self,
        local_engine: str,
        all_expert_weights: Sequence[Sequence[Any]],
    ) -> None:
        local_meta: dict[tuple[int, int], tuple[tuple[int, ...], int]] = {}
        for layer_idx, layer_views in enumerate(all_expert_weights):
            for tensor_idx, view in enumerate(layer_views):
                tensors = self._storage_tensors(view)
                if len(tensors) == 1:
                    tensor = tensors[0]
                    stride = tensor.nbytes // self._num_local_experts
                    addresses = tuple(tensor.data_ptr() + slot * stride for slot in range(self._num_local_experts))
                else:
                    stride = tensors[0].nbytes
                    addresses = tuple(tensor.data_ptr() for tensor in tensors)
                local_meta[(layer_idx, tensor_idx)] = addresses, stride

        gathered: list[tuple[str, dict[tuple[int, int], tuple[tuple[int, ...], int]]] | None] = [
            None
        ] * self._world_size
        torch.distributed.all_gather_object(
            gathered,
            (local_engine, local_meta),
            group=self._cpu_group,
        )
        for peer_rank, peer_state in enumerate(gathered):
            if peer_rank == self._rank:
                continue
            if peer_state is None or peer_state[1].keys() != local_meta.keys():
                raise RuntimeError(f"HIXL EPLB metadata mismatch with rank {peer_rank}")
            for key, (peer_addresses, peer_stride) in peer_state[1].items():
                if len(peer_addresses) != self._num_local_experts:
                    raise RuntimeError(f"HIXL EPLB expert count mismatch with rank {peer_rank} for {key}")
                if peer_stride != local_meta[key][1]:
                    raise RuntimeError(f"HIXL EPLB tensor size mismatch with rank {peer_rank} for {key}")
            self._remote_engines[peer_rank] = peer_state[0]
            self._remote_send_meta[peer_rank] = peer_state[1]

    def _connect_peers(self) -> None:
        assert self._engine is not None
        local_error: Exception | None = None
        for peer_rank, remote_engine in self._remote_engines.items():
            try:
                self._check_status(
                    self._engine.connect(remote_engine, _TRANSFER_TIMEOUT_SECONDS * 1000),
                    f"connect to rank {peer_rank}",
                )
            except Exception as error:
                local_error = local_error or error
        self._confirm_all_ranks(local_error, "connection")

    def _check_status(self, status: int, operation: str) -> None:
        if status != self._hixl.SUCCESS:
            raise RuntimeError(f"HIXL EPLB {operation} failed with status {status}")

    def set_stream(self, stream: torch.Stream | None) -> None:
        # HIXL owns its transfer streams. Binding the worker thread to this
        # device supplies the ACL context required by HIXL APIs.
        torch.npu.set_device(self._device)

    def set_transfer_context(self, old_indices: np.ndarray, layer_idx: int) -> None:
        if self._pending_reads:
            raise RuntimeError("HIXL EPLB started a layer with pending transfers")
        placement = np.asarray(old_indices).reshape(
            self._world_size,
            self._num_local_experts,
        )
        self._expert_to_src_row = [
            {int(expert_id): slot for slot, expert_id in enumerate(rank_experts) if expert_id != -1}
            for rank_experts in placement
        ]
        self._layer_idx = layer_idx

    def add_send(
        self,
        tensors: list[torch.Tensor],
        dst_rank: int,
        expert_id: int,
    ) -> None:
        # Receiver-initiated HIXL READs access pre-registered live weights.
        pass

    def add_recv(
        self,
        tensors: list[torch.Tensor],
        src_rank: int,
        expert_id: int,
    ) -> None:
        if self._expert_to_src_row is None or self._layer_idx is None:
            raise RuntimeError("set_transfer_context() must precede HIXL receives")
        src_slot = self._expert_to_src_row[src_rank][expert_id]
        peer_meta = self._remote_send_meta[src_rank]
        descriptors = self._pending_reads.setdefault(src_rank, [])
        for tensor_idx, tensor in enumerate(tensors):
            remote_addresses, remote_stride = peer_meta[(self._layer_idx, tensor_idx)]
            if tensor.nbytes != remote_stride:
                raise RuntimeError(f"HIXL EPLB receive size {tensor.nbytes} does not match remote size {remote_stride}")
            descriptors.append((tensor.data_ptr(), remote_addresses[src_slot], remote_stride))
            self._pending_bytes += remote_stride

    def execute(self) -> None:
        if self._layer_idx is None:
            raise RuntimeError("set_transfer_context() must precede HIXL execution")
        phase_started_at = time.perf_counter()
        requests: list[int] = []
        local_error: Exception | None = None
        try:
            requests = self._start_transfers()
        except Exception as error:
            local_error = error
        launch_finished_at = time.perf_counter()
        try:
            if local_error is None:
                self._wait_for_transfers(requests)
        except Exception as error:
            local_error = error
        transfer_finished_at = time.perf_counter()
        try:
            # Publish the layer only after every one-sided READ is complete.
            # The foreground can then defer an unavailable result instead of
            # waiting for transfer safety during workspace commit.
            self._confirm_all_ranks(local_error, "transfer")
        finally:
            confirmed_at = time.perf_counter()
            self.__dict__.setdefault("_eplb_hixl_phase_timings", []).append(
                _HixlTransferTiming(
                    launch_ms=(launch_finished_at - phase_started_at) * 1000,
                    transfer_ms=(transfer_finished_at - launch_finished_at) * 1000,
                    confirmation_ms=(confirmed_at - transfer_finished_at) * 1000,
                    request_count=len(requests),
                    transfer_bytes=self._pending_bytes,
                )
            )
            self._pending_reads.clear()
            self._pending_bytes = 0
            self._expert_to_src_row = None
            self._layer_idx = None

    def _confirm_all_ranks(self, local_error: Exception | None, operation: str) -> None:
        completed = torch.tensor(int(local_error is None), dtype=torch.int32)
        work = torch.distributed.all_reduce(
            completed,
            group=self._cpu_group,
            async_op=True,
        )
        work.wait(timeout=timedelta(seconds=_TRANSFER_TIMEOUT_SECONDS))
        if local_error is not None:
            raise local_error
        if completed.item() != self._world_size:
            raise RuntimeError(f"HIXL EPLB {operation} failed on another rank")

    def _start_transfers(self) -> list[int]:
        assert self._engine is not None
        requests = []
        for src_rank, descriptors in self._pending_reads.items():
            operations = [
                self._hixl.TransferOpDesc(
                    local_addr=local_addr,
                    remote_addr=remote_addr,
                    len=length,
                )
                for local_addr, remote_addr, length in descriptors
            ]
            status, request = self._engine.transfer_async(
                self._remote_engines[src_rank],
                self._hixl.TransferOp.READ,
                operations,
            )
            self._check_status(status, f"read from rank {src_rank}")
            requests.append(request)
        return requests

    def _wait_for_transfers(self, requests: list[int]) -> None:
        assert self._engine is not None
        pending = set(requests)
        deadline = time.monotonic() + _TRANSFER_TIMEOUT_SECONDS
        while pending:
            for request in tuple(pending):
                status, transfer_status = self._engine.get_transfer_status(request)
                self._check_status(status, "query transfer")
                if transfer_status == self._hixl.TransferStatus.COMPLETED:
                    pending.remove(request)
                elif transfer_status != self._hixl.TransferStatus.WAITING:
                    raise RuntimeError(f"HIXL EPLB transfer failed with state {transfer_status}")
            if pending:
                if time.monotonic() >= deadline:
                    raise TimeoutError("HIXL EPLB transfer timed out")
                time.sleep(_STATUS_POLL_SECONDS)

    @property
    def needs_profile_buffer_reservation(self) -> bool:
        return False

    def _close(self) -> None:
        engine = getattr(self, "_engine", None)
        if engine is None:
            return
        self._engine = None
        with contextlib.suppress(Exception):
            torch.npu.set_device(self._device)
        for remote_engine in getattr(self, "_remote_engines", {}).values():
            with contextlib.suppress(Exception):
                engine.disconnect(remote_engine)
        for handle in reversed(getattr(self, "_registered_handles", [])):
            with contextlib.suppress(Exception):
                engine.deregister_mem(handle)
        with contextlib.suppress(Exception):
            engine.finalize()
        self._registered_handles.clear()
        self._remote_engines.clear()
        self._remote_send_meta.clear()

    def __del__(self) -> None:
        with contextlib.suppress(Exception):
            self._close()
