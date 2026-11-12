# SPDX-License-Identifier: Apache-2.0
"""Mooncake V2 staging and landing into SFA-managed Host KV."""

import threading
import time
from collections.abc import Iterable, Iterator
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from typing import Any, NamedTuple

import torch
from vllm.logger import logger

from vllm_ascend import envs as ascend_envs
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_sfa.transfer_targets import (
    DSAPDMooncakeTransferTarget,
    MooncakeKVRegistry,
)
from vllm_ascend.distributed.kv_transfer.utils.mooncake_transfer_engine import global_te


@dataclass
class _DSAPDMooncakeStagingBank:
    """Only this bank's executor may mutate its streams and events.

    Contexts survive receives to reuse streams/events. Pending copies retain their Tensor
    owners until draining succeeds, including when an event/stream wait fails.
    The request lock also protects context replacement and final draining.
    """

    slot_base: int
    slot_count: int
    stream: Any
    copy_streams: tuple[Any, ...] = ()
    copy_events: tuple[Any, ...] = ()
    completion_event: Any = None
    completion_pending: bool = False
    inflight_owners: list[tuple[torch.Tensor, ...]] = field(default_factory=list)
    needs_stream_drain: bool = False
    pending_device: torch.device | None = None


class _DSAPDMooncakeStagingCopy(NamedTuple):
    """Immutable copy metadata shared by the planner and its bank worker."""

    destination: int
    length: int
    staging_offset: int
    target_bytes: torch.Tensor


class _DSAPDMooncakeBankBatch(NamedTuple):
    """Prepared under the request lock; address lists are read-only to executors."""

    batch_start: int
    bank_index: int
    local_staging_ptrs: list[int]
    remote_src_ptrs: list[int]
    transfer_lengths: list[int]
    copies: tuple[_DSAPDMooncakeStagingCopy, ...]


@dataclass(frozen=True)
class _DSAPDMooncakeStagingPlan:
    batches: tuple[_DSAPDMooncakeBankBatch, ...]


@dataclass(frozen=True)
class _DSAPDMooncakeBankBatchResult:
    return_code: int
    read_secs: float
    read_settle_secs: float
    land_exposed_secs: float


class MooncakeSFAStagingReader:
    """Read through a bounded HBM window into the manager's existing Host KV."""

    engine: Any
    kv_caches: dict[str, Any]
    block_len_per_addr: list[list[int]]
    kv_registry: MooncakeKVRegistry

    def __init__(self, engine: Any, host_kv: Any, block_lens: list[list[int]]) -> None:
        self.engine = engine
        self.host_kv = host_kv
        self.kv_registry = host_kv.registry
        self.kv_caches = host_kv.transfer_caches
        self.block_len_per_addr = block_lens
        self._init_dsa_pd_mooncake_staging()

    def _init_dsa_pd_mooncake_staging(self) -> None:
        self._dsa_pd_mooncake_staging_storage: torch.Tensor | None = None
        self._dsa_pd_mooncake_staging_tensor: torch.Tensor | None = None
        self._dsa_pd_mooncake_staging_bytes = 0
        self._dsa_pd_mooncake_staging_lock = threading.Lock()
        self._dsa_pd_mooncake_staging_banks: tuple[_DSAPDMooncakeStagingBank, ...] = ()
        self._pipeline_banks = ascend_envs.VLLM_ASCEND_DSA_PD_MOONCAKE_BANKS
        self._landing_streams = [torch.npu.Stream() for _ in range(max(1, self._pipeline_banks))]
        self._landing_stream = self._landing_streams[0]
        self._pipeline_landing_enabled = ascend_envs.VLLM_ASCEND_DSA_PD_MOONCAKE_PIPELINE
        # Host-only KV still needs a registered HBM region before ADXL connects.
        self._ensure_dsa_pd_mooncake_staging_tensor()

    @staticmethod
    def _iter_kv_cache_tensors(kv_cache_tuple: Any) -> Iterator[torch.Tensor]:
        if isinstance(kv_cache_tuple, torch.Tensor):
            yield kv_cache_tuple
            return
        if isinstance(kv_cache_tuple, (list, tuple)):
            for item in kv_cache_tuple:
                if isinstance(item, torch.Tensor):
                    yield item

    def _get_dsa_pd_mooncake_staging_device(self) -> torch.device | str:
        for kv_cache_tuple in self.kv_caches.values():
            for tensor in self._iter_kv_cache_tensors(kv_cache_tuple):
                if tensor.device.type != "cpu":
                    return tensor.device
        return "npu"

    def _get_dsa_pd_mooncake_staging_chunk_bytes(self) -> int:
        block_bytes = [
            int(block_len)
            for per_layer_block_lens in self.block_len_per_addr
            for block_len in per_layer_block_lens
            if int(block_len) > 0
        ]
        alignment = 2 * 1024 * 1024
        chunk_bytes = max(max(block_bytes, default=alignment), alignment)
        return ((chunk_bytes + alignment - 1) // alignment) * alignment

    def _get_dsa_pd_mooncake_staging_window(self) -> int:
        return ascend_envs.VLLM_ASCEND_DSA_PD_MOONCAKE_STAGING_WINDOW

    def _ensure_dsa_pd_mooncake_staging_tensor(self) -> torch.Tensor:
        chunk_bytes = self._get_dsa_pd_mooncake_staging_chunk_bytes()
        staging_window = self._get_dsa_pd_mooncake_staging_window()
        required_bytes = chunk_bytes * staging_window
        if self._dsa_pd_mooncake_staging_tensor is not None and self._dsa_pd_mooncake_staging_bytes >= required_bytes:
            return self._dsa_pd_mooncake_staging_tensor

        # Preserve the old allocation if a previous request could not drain it.
        self._drain_dsa_pd_mooncake_staging_banks(getattr(self, "_dsa_pd_mooncake_staging_banks", ()))
        alignment = 2 * 1024 * 1024
        device = self._get_dsa_pd_mooncake_staging_device()
        storage = torch.empty((required_bytes + alignment,), dtype=torch.int8, device=device)
        offset = (-int(storage.data_ptr())) % alignment
        staging = storage[offset : offset + required_bytes]
        if staging.data_ptr() % alignment != 0:
            raise RuntimeError("DSA PD Mooncake NPU staging tensor is not 2MB aligned.")

        global_te.register_buffer([int(staging.data_ptr())], [required_bytes])
        self._dsa_pd_mooncake_staging_storage = storage
        self._dsa_pd_mooncake_staging_tensor = staging
        self._dsa_pd_mooncake_staging_bytes = required_bytes
        logger.info(
            "Allocated Mooncake NPU staging buffer: bytes=%s chunk_bytes=%s window=%s device=%s",
            required_bytes,
            chunk_bytes,
            staging_window,
            staging.device,
        )
        return staging

    def _get_dsa_pd_mooncake_staging_banks(
        self, bank_count: int, bank_window: int, pipeline_active: bool
    ) -> tuple[_DSAPDMooncakeStagingBank, ...]:
        streams = self._landing_streams[:bank_count] if pipeline_active else [self._landing_stream]
        if len(streams) != bank_count:
            raise RuntimeError("DSA Mooncake staging bank count exceeds its allocated landing streams.")
        previous = getattr(self, "_dsa_pd_mooncake_staging_banks", ())
        # A failed event wait can leave an old context pending. Finish it before
        # starting another request or dropping the stream ownership.
        self._drain_dsa_pd_mooncake_staging_banks(previous)
        copy_stream_count = (
            min(ascend_envs.VLLM_ASCEND_DSA_PD_MOONCAKE_COPY_STREAMS_PER_BANK, bank_window)
            if pipeline_active and not ascend_envs.VLLM_ASCEND_DSA_PD_MOONCAKE_READ_PROBE
            else 1
        )
        if len(previous) == bank_count and all(
            bank.slot_count == bank_window and bank.stream is stream for bank, stream in zip(previous, streams)
        ):
            for bank in previous:
                self._configure_dsa_pd_mooncake_copy_streams(bank, copy_stream_count)
            return previous
        banks = []
        for index, stream in enumerate(streams):
            banks.append(
                _DSAPDMooncakeStagingBank(
                    slot_base=index * bank_window,
                    slot_count=bank_window,
                    stream=stream,
                )
            )
        self._dsa_pd_mooncake_staging_banks = tuple(banks)
        for bank in banks:
            self._configure_dsa_pd_mooncake_copy_streams(bank, copy_stream_count)
        return self._dsa_pd_mooncake_staging_banks

    @staticmethod
    def _configure_dsa_pd_mooncake_copy_streams(bank: _DSAPDMooncakeStagingBank, count: int) -> None:
        # The caller drained the old layout before changing its stream owners.
        streams = bank.copy_streams or (bank.stream,)
        bank.copy_streams = streams[:count] + tuple(torch.npu.Stream() for _ in range(max(0, count - len(streams))))
        events = bank.copy_events
        bank.copy_events = events[: count - 1] + tuple(
            torch.npu.Event() for _ in range(max(0, count - 1 - len(events)))
        )

    def _prepare_dsa_pd_mooncake_bank_plan(
        self,
        staged_transfers: list[tuple[int, int, int]],
        staged_targets: dict[int, tuple[torch.Tensor, int]],
        staging: torch.Tensor,
        staging_chunk_bytes: int,
        banks: tuple[_DSAPDMooncakeStagingBank, ...],
    ) -> _DSAPDMooncakeStagingPlan:
        """Split Host ranges, merge copies, and reject conflicting destinations."""
        staging_ptr = int(staging.data_ptr())
        staging_alignment = 2 * 1024 * 1024
        bank_window = banks[0].slot_count
        batches: list[_DSAPDMooncakeBankBatch] = []
        intervals = []
        transfers = iter(staged_transfers)
        # A Host range can continue in another batch without materializing its pieces.
        remaining = 0
        batch_start = 0
        while True:
            bank_index = len(batches) % len(banks)
            bank = banks[bank_index]
            local_staging_ptrs: list[int] = []
            remote_src_ptrs: list[int] = []
            transfer_lengths: list[int] = []
            copy_plan: list[tuple[int, int, int, torch.Tensor, int]] = []
            for position in range(bank_window):
                while remaining == 0:
                    transfer = next(transfers, None)
                    if transfer is None:
                        break
                    local_dst, remote_src, remaining = transfer
                    target_bytes, target_offset = staged_targets[local_dst]
                    remote_page_offset = remote_src % staging_alignment
                if remaining == 0:
                    break
                length = min(remaining, staging_chunk_bytes - remote_page_offset)
                staging_offset = (bank.slot_base + position) * staging_chunk_bytes + remote_page_offset
                local_staging_ptrs.append(staging_ptr + staging_offset)
                remote_src_ptrs.append(remote_src)
                transfer_lengths.append(length)
                previous = copy_plan[-1] if copy_plan else None
                if (
                    previous is not None
                    and previous[0] + previous[1] == local_dst
                    and previous[2] + previous[1] == staging_offset
                ):
                    copy_plan[-1] = (previous[0], previous[1] + length, previous[2], previous[3], previous[4])
                else:
                    copy_plan.append((local_dst, length, staging_offset, target_bytes, target_offset))
                remaining -= length
                local_dst += length
                remote_src += length
                target_offset += length
                # A nonfinal piece fills its slot; the next source page is aligned.
                remote_page_offset = 0
            if not local_staging_ptrs:
                break
            copies = []
            for destination, length, staging_offset, target_bytes, target_offset in copy_plan:
                if target_offset + length > target_bytes.numel():
                    raise RuntimeError("Coalesced DSA Mooncake copy crosses a Host KV tensor boundary.")
                copies.append(_DSAPDMooncakeStagingCopy(destination, length, staging_offset, target_bytes))
                intervals.append((destination, destination + length))
            batches.append(
                _DSAPDMooncakeBankBatch(
                    batch_start,
                    bank_index,
                    local_staging_ptrs,
                    remote_src_ptrs,
                    transfer_lengths,
                    tuple(copies),
                )
            )
            batch_start += len(local_staging_ptrs)
        previous_end = -1
        for start, end in sorted(intervals):
            if start < previous_end:
                raise ValueError("Overlapping Mooncake Host destination ranges are not supported.")
            previous_end = end
        return _DSAPDMooncakeStagingPlan(tuple(batches))

    def _execute_dsa_pd_mooncake_bank_batch(
        self,
        session_id: str,
        bank: _DSAPDMooncakeStagingBank,
        batch: _DSAPDMooncakeBankBatch,
        staging: torch.Tensor,
        pipeline_active: bool,
        remote_request_id: str,
    ) -> _DSAPDMooncakeBankBatchResult:
        """Reclaim, READ, then land one batch without reacquiring the request lock."""
        land_exposed_secs = 0.0
        if pipeline_active:
            reclaim_started = time.perf_counter()
            if bank.completion_pending:
                bank.completion_event.synchronize()
                bank.completion_pending = False
            bank.inflight_owners.clear()
            land_exposed_secs += time.perf_counter() - reclaim_started
        ret, read_secs, read_settle_secs = self._read_staging_batch(session_id, batch, remote_request_id)
        if ret < 0:
            return _DSAPDMooncakeBankBatchResult(ret, read_secs, read_settle_secs, land_exposed_secs)
        if pipeline_active:
            bank.inflight_owners.append((staging, *(copy.target_bytes for copy in batch.copies)))
            bank.needs_stream_drain = True
            try:
                streams = bank.copy_streams or (bank.stream,)
                count = min(len(streams), max(1, len(batch.copies)))
                for index, stream in enumerate(streams[:count]):
                    with torch.npu.stream(stream):
                        self.host_kv.land(batch.copies[index::count], staging, bank.inflight_owners)
                        if index:
                            bank.copy_events[index - 1].record(stream)
                # Join auxiliary streams without blocking this Host worker.
                # The bank event then protects every source slice and target.
                for event in bank.copy_events[: count - 1]:
                    bank.stream.wait_event(event)
                if bank.completion_event is None:
                    bank.completion_event = torch.npu.Event()
                bank.completion_event.record(bank.stream)
                bank.completion_pending = True
                bank.needs_stream_drain = False
            except Exception:
                self._synchronize_dsa_pd_mooncake_bank_streams(bank)
                bank.completion_pending = False
                bank.completion_event = None
                bank.copy_events = tuple(torch.npu.Event() for _ in bank.copy_events)
                bank.needs_stream_drain = False
                raise
        else:
            land_started = time.perf_counter()
            # Synchronous landing uses the caller's current stream, rather
            # than the bank streams. Retain its owners until a device wait
            # succeeds, even if landing or synchronization raises.
            bank.inflight_owners.append((staging, *(copy.target_bytes for copy in batch.copies)))
            bank.pending_device = staging.device
            try:
                self.host_kv.land(batch.copies, staging, bank.inflight_owners)
            finally:
                torch.npu.synchronize(bank.pending_device)
                bank.pending_device = None
                bank.inflight_owners.clear()
            land_exposed_secs += time.perf_counter() - land_started
        return _DSAPDMooncakeBankBatchResult(ret, read_secs, read_settle_secs, land_exposed_secs)

    @staticmethod
    def _synchronize_dsa_pd_mooncake_bank_streams(bank: _DSAPDMooncakeStagingBank) -> None:
        first_error = None
        for stream in bank.copy_streams or (bank.stream,):
            try:
                stream.synchronize()
            except Exception as error:
                if first_error is None:
                    first_error = error
        if first_error is not None:
            raise first_error

    def _drain_dsa_pd_mooncake_staging_banks(self, banks: tuple[_DSAPDMooncakeStagingBank, ...]) -> None:
        first_error = None
        for bank in banks:
            bank_error = None
            try:
                if bank.completion_pending:
                    bank.completion_event.synchronize()
                    bank.completion_pending = False
            except Exception as error:
                bank_error = error
            if bank_error is not None or bank.needs_stream_drain:
                try:
                    self._synchronize_dsa_pd_mooncake_bank_streams(bank)
                    bank.completion_pending = False
                    # A failed record/wait may leave the event unusable. Drop
                    # it only after the stream confirms its copies are done.
                    bank.completion_event = None
                    bank.copy_events = tuple(torch.npu.Event() for _ in bank.copy_events)
                    bank.needs_stream_drain = False
                except Exception as error:
                    if bank_error is None:
                        bank_error = error
            if bank.pending_device is not None:
                try:
                    torch.npu.synchronize(bank.pending_device)
                    bank.pending_device = None
                except Exception as error:
                    if bank_error is None:
                        bank_error = error
            if not bank.needs_stream_drain and not bank.completion_pending and bank.pending_device is None:
                bank.inflight_owners.clear()
            # A failed wait on one bank must not skip the other banks.
            if first_error is None and bank_error is not None:
                first_error = bank_error
        if first_error is not None:
            raise first_error

    def drain_pending_landing(self) -> None:
        """Confirm all submitted Host copies have stopped before releasing KV."""
        with self._dsa_pd_mooncake_staging_lock:
            self._drain_dsa_pd_mooncake_staging_banks(self._dsa_pd_mooncake_staging_banks)

    def _get_dsa_pd_mooncake_bank_workers(self, banks: tuple[_DSAPDMooncakeStagingBank, ...]) -> int:
        """Use one worker per effective bank; keep a single bank on the caller."""
        return len(banks) if len(banks) > 1 else 0

    def _execute_dsa_pd_mooncake_bank_workers(
        self,
        session_id: str,
        plan: _DSAPDMooncakeStagingPlan,
        banks: tuple[_DSAPDMooncakeStagingBank, ...],
        staging: torch.Tensor,
        remote_request_id: str,
        workers: int,
    ) -> list[tuple[_DSAPDMooncakeBankBatch, _DSAPDMooncakeBankBatchResult]]:
        """Run each bank's queue in one task under the caller's request lock.

        A task keeps reclaim -> READ -> copy order within its bank and advances
        without a coordinator round trip between batches.
        Device setup is thread-local; the staging allocation and all target
        validation are complete before the pool starts. The pool is joined on
        every exit, leaving final stream/event draining to the coordinator.
        """
        bank_batches: list[list[_DSAPDMooncakeBankBatch]] = [[] for _ in banks]
        for batch in plan.batches:
            bank_batches[batch.bank_index].append(batch)
        stopped = threading.Event()
        completed = {}
        failures: dict[int, BaseException] = {}
        pending: dict[Future, int] = {}

        def execute(
            batches: list[_DSAPDMooncakeBankBatch],
        ) -> tuple[
            list[tuple[_DSAPDMooncakeBankBatch, _DSAPDMooncakeBankBatchResult]],
            tuple[int, BaseException] | None,
        ]:
            results: list[tuple[_DSAPDMooncakeBankBatch, _DSAPDMooncakeBankBatchResult]] = []
            for batch in batches:
                if stopped.is_set():
                    break
                try:
                    result = self._execute_dsa_pd_mooncake_bank_batch(
                        session_id, banks[batch.bank_index], batch, staging, True, remote_request_id
                    )
                except BaseException as error:
                    stopped.set()
                    return results, (batch.batch_start, error)
                results.append((batch, result))
                if result.return_code < 0:
                    stopped.set()
                    break
            return results, None

        pool = ThreadPoolExecutor(
            max_workers=min(workers, len(banks)),
            thread_name_prefix="dsa-mooncake-bank",
            initializer=torch.npu.set_device,
            initargs=(staging.device,),
        )
        try:
            for batches in bank_batches:
                if stopped.is_set():
                    break
                if batches:
                    pending[pool.submit(execute, batches)] = batches[0].batch_start
            while pending:
                finished, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in sorted(finished, key=lambda value: pending[value]):
                    batch_start = pending.pop(future)
                    if future.cancelled():
                        continue
                    try:
                        results, failure = future.result()
                    except BaseException as error:
                        failures[batch_start] = error
                        stopped.set()
                    else:
                        for batch, result in results:
                            completed[batch.batch_start] = (batch, result)
                        if failure is not None:
                            failures[failure[0]] = failure[1]
                if stopped.is_set():
                    for future in pending:
                        future.cancel()
        except BaseException:
            stopped.set()
            for future in pending:
                future.cancel()
            raise
        finally:
            # Running READs are allowed to finish. No bank state is drained or
            # made available to another request while its host task can use it.
            pool.shutdown(wait=True, cancel_futures=True)
        if failures:
            raise failures[min(failures)]
        return [completed[index] for index in sorted(completed)]

    def batch_transfer_sync_read(
        self,
        session_id: str,
        local_destinations: list[int],
        remote_sources: list[int],
        lengths: list[int],
        request_id: str,
        *,
        transfer_targets: list[DSAPDMooncakeTransferTarget] | None = None,
    ) -> int:
        if not (len(local_destinations) == len(remote_sources) == len(lengths)):
            raise ValueError("Mooncake READ address and length lists must have equal sizes.")
        if transfer_targets is not None:
            if len(transfer_targets) != len(local_destinations):
                raise ValueError("Mooncake transfer target count must match address count.")
            requires_staging = any(target.offloaded for target in transfer_targets)
        else:
            requires_staging = any(
                self.kv_registry.resolve_offloaded_tensor(local_dst, length) is not None
                for local_dst, length in zip(local_destinations, lengths)
            )
        target_kwargs = {} if transfer_targets is None else {"transfer_targets": transfer_targets}
        if not requires_staging:
            return self._batch_transfer_sync_read_with_optional_staging_unlocked(
                session_id, local_destinations, remote_sources, lengths, request_id, **target_kwargs
            )

        # The staging slots are shared by all peer handlers on this worker.
        # Keep ownership across both the Mooncake read and the NPU-to-offload
        # copy so another request cannot overwrite a slot still being consumed.
        with self._dsa_pd_mooncake_staging_lock:
            return self._batch_transfer_sync_read_with_optional_staging_unlocked(
                session_id, local_destinations, remote_sources, lengths, request_id, **target_kwargs
            )

    def _batch_transfer_sync_read_with_optional_staging_unlocked(
        self,
        session_id: str,
        local_destinations: list[int],
        remote_sources: list[int],
        lengths: list[int],
        remote_request_id: str,
        *,
        transfer_targets: list[DSAPDMooncakeTransferTarget] | None = None,
    ) -> int:
        # The upstream address builder retains each local KV tensor's identity.
        # Validate the layout once under the staging lock, then use its targets
        # directly instead of rediscovering Host/HBM from every destination.
        resolved_targets = (
            self.kv_registry.prepare_transfer_targets(transfer_targets, local_destinations, lengths)
            if transfer_targets is not None
            else None
        )
        direct_local_destinations: list[int] = []
        direct_remote_sources: list[int] = []
        direct_lengths: list[int] = []
        staged_transfers: list[tuple[int, int, int]] = []
        # Request-local plans reuse byte views owned by the KV registry.
        # The registry drops its views on KV Tensor destruction or reset.
        staged_targets: dict[int, tuple[torch.Tensor, int]] = {}
        staging_chunk_bytes = self._get_dsa_pd_mooncake_staging_chunk_bytes()

        for position, (local_dst, remote_src, length) in enumerate(zip(local_destinations, remote_sources, lengths)):
            resolved = (
                resolved_targets[position]
                if resolved_targets is not None
                else self.kv_registry.resolve_offloaded_byte_tensor(local_dst, length)
            )
            if resolved is None:
                if resolved_targets is None and self.kv_registry.get_register_location(local_dst, length) == "cpu":
                    raise RuntimeError("Missing DSA PD Mooncake offloaded KV tensor for a Host READ destination.")
                direct_local_destinations.append(local_dst)
                direct_remote_sources.append(remote_src)
                direct_lengths.append(length)
                continue
            target_tensor, target_offset = resolved
            local_dst = int(local_dst)
            remote_src = int(remote_src)
            if length > 0:
                staged_transfers.append((local_dst, remote_src, int(length)))
                staged_targets[local_dst] = (target_tensor, target_offset)

        if not staged_transfers:
            if direct_local_destinations:
                ret = self.engine.batch_transfer_sync_read(
                    session_id, direct_local_destinations, direct_remote_sources, direct_lengths
                )
                if ret < 0:
                    return ret
            return 0

        staging = self._ensure_dsa_pd_mooncake_staging_tensor()
        staging_window = self._get_dsa_pd_mooncake_staging_window()
        # Pipeline mode: WINDOW keeps meaning the total staging capacity and
        # the buffer is split into BANKS banks of window/BANKS slots that
        # consecutive reads round-robin between (the network batch shrinks
        # accordingly; the staging footprint is identical with the pipeline
        # on or off). Flipping to a fresh bank needs no wait at all, so one
        # read's landing hides under the next reads; coming back to a bank
        # BANKS reads later is guarded by a just-in-time bank-event wait. The
        # final drain keeps the external contract: returning 0 means every
        # window has landed and is visible.
        pipeline_active = self._pipeline_landing_enabled and staging_window >= 2
        if self._pipeline_landing_enabled and not pipeline_active:
            logger.warning(
                "VLLM_ASCEND_DSA_PD_MOONCAKE_PIPELINE needs a staging window >= 2 "
                "(got %s); falling back to the synchronous landing path.",
                staging_window,
            )
        banks = self._pipeline_banks if pipeline_active else 1
        if pipeline_active and banks > staging_window:
            logger.warning(
                "VLLM_ASCEND_DSA_PD_MOONCAKE_BANKS=%s exceeds the staging window (%s); clamping banks to %s.",
                banks,
                staging_window,
                staging_window,
            )
            banks = staging_window
        # Trailing slots stay idle when the window is not divisible by the
        # bank count; every bank keeps bank_window contiguous slots.
        bank_window = (staging_window // banks) if pipeline_active else staging_window
        bank_contexts = self._get_dsa_pd_mooncake_staging_banks(banks, bank_window, pipeline_active)
        timing = self._start_staging_receive_timing()
        workers = 0
        try:
            plan = self._prepare_dsa_pd_mooncake_bank_plan(
                staged_transfers, staged_targets, staging, staging_chunk_bytes, bank_contexts
            )
            # Validate every Host destination before a mixed receive writes HBM.
            if direct_local_destinations:
                ret = self.engine.batch_transfer_sync_read(
                    session_id, direct_local_destinations, direct_remote_sources, direct_lengths
                )
                if ret < 0:
                    return ret
            if pipeline_active and not ascend_envs.VLLM_ASCEND_DSA_PD_MOONCAKE_READ_PROBE:
                workers = self._get_dsa_pd_mooncake_bank_workers(bank_contexts)
            batch_results: Iterable[tuple[_DSAPDMooncakeBankBatch, _DSAPDMooncakeBankBatchResult]]
            if workers:
                batch_results = self._execute_dsa_pd_mooncake_bank_workers(
                    session_id, plan, bank_contexts, staging, remote_request_id, workers
                )
            else:
                # Keep the synchronous control and device-wide probe on the caller.
                def serial_batches() -> Iterator[tuple[_DSAPDMooncakeBankBatch, _DSAPDMooncakeBankBatchResult]]:
                    for batch in plan.batches:
                        yield (
                            batch,
                            self._execute_dsa_pd_mooncake_bank_batch(
                                session_id,
                                bank_contexts[batch.bank_index],
                                batch,
                                staging,
                                pipeline_active,
                                remote_request_id,
                            ),
                        )

                batch_results = serial_batches()
            for batch, result in batch_results:
                self._record_staging_batch_timing(timing, batch, result)
                if result.return_code < 0:
                    return result.return_code
        finally:
            if pipeline_active:
                self._drain_staging_receive(timing, bank_contexts)
        self._log_staging_receive(
            timing,
            remote_request_id,
            pipeline_active=pipeline_active,
            chunk_count=int(timing["plan_entries"]),
            banks=bank_contexts,
            workers=workers,
        )
        return 0

    def _read_staging_batch(
        self, session_id: str, batch: _DSAPDMooncakeBankBatch, remote_request_id: str
    ) -> tuple[int, float, float]:
        """Measure READ and its optional debug probe before any landing starts."""
        read_started = time.perf_counter()
        ret = self.engine.batch_transfer_sync_read(
            session_id, batch.local_staging_ptrs, batch.remote_src_ptrs, batch.transfer_lengths
        )
        read_secs = time.perf_counter() - read_started
        read_settle_secs = 0.0
        if ret >= 0 and ascend_envs.VLLM_ASCEND_DSA_PD_MOONCAKE_READ_PROBE:
            settle_started = time.perf_counter()
            torch.npu.synchronize()
            read_settle_secs = time.perf_counter() - settle_started
        if ret < 0:
            logger.error(
                "Staged Mooncake batch read failed: request=%s "
                "batch_start=%s batch_size=%s first_local=%s first_remote=%s first_len=%s ret=%s",
                remote_request_id,
                batch.batch_start,
                len(batch.transfer_lengths),
                batch.local_staging_ptrs[0],
                batch.remote_src_ptrs[0],
                batch.transfer_lengths[0],
                ret,
            )
        return ret, read_secs, read_settle_secs

    @staticmethod
    def _start_staging_receive_timing() -> dict[str, float | int]:
        # Use Host clocks only: timing events can block on a busy device.
        return {
            "total_started": time.perf_counter(),
            "read_secs": 0.0,
            "read_settle_secs": 0.0,
            "land_exposed_secs": 0.0,
            "batches": 0,
            "plan_entries": 0,
            "copy_calls": 0,
        }

    @staticmethod
    def _record_staging_batch_timing(
        timing: dict[str, float | int], batch: _DSAPDMooncakeBankBatch, result: _DSAPDMooncakeBankBatchResult
    ) -> None:
        timing["read_secs"] += result.read_secs
        timing["read_settle_secs"] += result.read_settle_secs
        timing["land_exposed_secs"] += result.land_exposed_secs
        if result.return_code >= 0:
            timing["plan_entries"] += len(batch.transfer_lengths)
            timing["copy_calls"] += len(batch.copies)
            timing["batches"] += 1

    def _drain_staging_receive(
        self, timing: dict[str, float | int], banks: tuple[_DSAPDMooncakeStagingBank, ...]
    ) -> None:
        drain_started = time.perf_counter()
        self._drain_dsa_pd_mooncake_staging_banks(banks)
        timing["land_exposed_secs"] += time.perf_counter() - drain_started

    @staticmethod
    def _log_staging_receive(
        timing: dict[str, float | int],
        request_id: str,
        *,
        pipeline_active: bool,
        chunk_count: int,
        banks: tuple[_DSAPDMooncakeStagingBank, ...],
        workers: int,
    ) -> None:
        # Report network time separately from exposed landing/wait time.
        # Plan entries count raw chunks; copies count coalesced descriptors.
        logger.info(
            "Staged DSA PD mooncake receive timing: request=%s mode=%s batches=%s chunks=%s "
            "copies=%s plan=%s total_s=%.3f read_s=%.3f read_settle_s=%.3f land_exposed_s=%.3f bank_workers=%s "
            "copy_streams_per_bank=%s",
            request_id,
            "pipeline" if pipeline_active else "sync",
            timing["batches"],
            chunk_count,
            timing["copy_calls"],
            timing["plan_entries"],
            time.perf_counter() - timing["total_started"],
            timing["read_secs"],
            timing["read_settle_secs"],
            timing["land_exposed_secs"],
            workers,
            len(banks[0].copy_streams),
        )

    def get_transfer_targets(
        self, layer_names: list[str], base_addresses: list[list[int]]
    ) -> list[list[DSAPDMooncakeTransferTarget]] | None:
        """Bind local layer/cache metadata once; never serialize these handles."""
        if len(layer_names) != len(base_addresses):
            raise ValueError("Mooncake local KV layer names and address metadata differ.")
        layers = []
        for layer_name, addresses in zip(layer_names, base_addresses):
            tensors = list(self._iter_kv_cache_tensors(self.kv_caches[layer_name]))
            if [int(tensor.data_ptr()) for tensor in tensors] != addresses:
                raise ValueError("Mooncake local KV tensor identity differs from address metadata.")
            layers.append(tensors)
        return self.kv_registry.create_transfer_targets(layers)
