# SPDX-License-Identifier: Apache-2.0
"""DSA Host KV landing shared by the Mooncake pull connectors."""

import threading
import time
from collections.abc import Iterator
from typing import Any

import torch
from vllm.logger import logger

from vllm_ascend import envs as ascend_envs
from vllm_ascend.distributed.kv_transfer.utils.mooncake_transfer_engine import global_te
from vllm_ascend.kv_offload.dsa_pd_mooncake import (
    get_dsa_pd_mooncake_kv_register_location,
    get_dsa_pd_mooncake_offloaded_kv_tensor,
)


class DSAPDMooncakeStagingMixin:
    """Own a bounded staging window until every Host KV copy has landed."""

    engine: Any
    kv_caches: dict[str, Any]
    block_len_per_addr: list[list[int]]

    def _init_dsa_pd_mooncake_staging(self) -> None:
        self._dsa_pd_mooncake_staging_storage: torch.Tensor | None = None
        self._dsa_pd_mooncake_staging_tensor: torch.Tensor | None = None
        self._dsa_pd_mooncake_staging_bytes = 0
        self._dsa_pd_mooncake_staging_lock = threading.Lock()
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

    def _copy_dsa_pd_mooncake_staging_to_offloaded_kv(self, dst: int, length: int, staging_offset: int = 0) -> None:
        resolved = get_dsa_pd_mooncake_offloaded_kv_tensor(dst, length)
        if resolved is None:
            raise RuntimeError(f"Missing DSA PD Mooncake offloaded KV tensor for dst={dst} length={length}.")
        target_tensor, offset = resolved
        staging = self._ensure_dsa_pd_mooncake_staging_tensor()
        target_bytes = target_tensor.view(torch.int8).reshape(-1)
        target_bytes.narrow(0, offset, length).copy_(
            staging[staging_offset : staging_offset + length], non_blocking=False
        )

    @staticmethod
    def _drain_dsa_pd_mooncake_slot_events(pending_slot_events: list[torch.npu.Event | None]) -> None:
        for slot, pending_event in enumerate(pending_slot_events):
            if pending_event is not None:
                pending_event.synchronize()
                pending_slot_events[slot] = None

    @staticmethod
    def _coalesce_dsa_pd_mooncake_copy_plan(copy_plan: list[tuple[int, int, int]]) -> list[tuple[int, int, int]]:
        """Merge adjacent plan entries that are contiguous on both sides.

        Consecutive pieces of one layer run are contiguous in the offloaded
        KV pool; when their remote page offsets are zero they are contiguous
        in staging too (consecutive slots), so the whole run lands as one
        copy instead of one copy per 2 MiB piece. Non-contiguous neighbours
        stay separate entries.
        """
        merged: list[tuple[int, int, int]] = []
        for dst, length, staging_offset in copy_plan:
            if merged:
                prev_dst, prev_length, prev_offset = merged[-1]
                if prev_dst + prev_length == dst and prev_offset + prev_length == staging_offset:
                    merged[-1] = (prev_dst, prev_length + length, prev_offset)
                    continue
            merged.append((dst, length, staging_offset))
        return merged

    def _batch_transfer_sync_read_with_optional_staging(
        self,
        session_id: str,
        src_list: list[int],
        dst_list: list[int],
        length_list: list[int],
        remote_request_id: str,
    ) -> int:
        if not (len(src_list) == len(dst_list) == len(length_list)):
            raise ValueError("Mooncake READ address and length lists must have equal sizes.")
        requires_staging = any(
            get_dsa_pd_mooncake_offloaded_kv_tensor(local_dst, length) is not None
            for local_dst, length in zip(src_list, length_list)
        )
        if not requires_staging:
            return self._batch_transfer_sync_read_with_optional_staging_unlocked(
                session_id, src_list, dst_list, length_list, remote_request_id
            )

        # The staging slots are shared by all peer handlers on this worker.
        # Keep ownership across both the Mooncake read and the NPU-to-offload
        # copy so another request cannot overwrite a slot still being consumed.
        with self._dsa_pd_mooncake_staging_lock:
            return self._batch_transfer_sync_read_with_optional_staging_unlocked(
                session_id, src_list, dst_list, length_list, remote_request_id
            )

    def _batch_transfer_sync_read_with_optional_staging_unlocked(
        self,
        session_id: str,
        src_list: list[int],
        dst_list: list[int],
        length_list: list[int],
        remote_request_id: str,
    ) -> int:
        direct_src_list: list[int] = []
        direct_dst_list: list[int] = []
        direct_length_list: list[int] = []
        staged_transfers: list[tuple[int, int, int]] = []
        staging_chunk_bytes = self._get_dsa_pd_mooncake_staging_chunk_bytes()
        staging_alignment = 2 * 1024 * 1024

        for local_dst, remote_src, length in zip(src_list, dst_list, length_list):
            if get_dsa_pd_mooncake_offloaded_kv_tensor(local_dst, length) is None:
                if get_dsa_pd_mooncake_kv_register_location(local_dst, length) == "cpu":
                    raise RuntimeError("Missing DSA PD Mooncake offloaded KV tensor for a Host READ destination.")
                direct_src_list.append(local_dst)
                direct_dst_list.append(remote_src)
                direct_length_list.append(length)
                continue
            remaining = int(length)
            offset = 0
            while remaining > 0:
                remote_page_offset = (int(remote_src) + offset) % staging_alignment
                piece_capacity = staging_chunk_bytes - remote_page_offset
                piece_len = min(remaining, piece_capacity)
                staged_transfers.append((int(local_dst) + offset, int(remote_src) + offset, piece_len))
                offset += piece_len
                remaining -= piece_len

        if direct_src_list:
            ret = self.engine.batch_transfer_sync_read(
                session_id,
                direct_src_list,
                direct_dst_list,
                direct_length_list,
            )
            if ret < 0:
                return ret

        if not staged_transfers:
            return 0

        staging = self._ensure_dsa_pd_mooncake_staging_tensor()
        staging_ptr = int(staging.data_ptr())
        staging_window = self._get_dsa_pd_mooncake_staging_window()
        # Pipeline mode: WINDOW keeps meaning the total staging capacity and
        # the buffer is split into BANKS banks of window/BANKS slots that
        # consecutive reads round-robin between (the network batch shrinks
        # accordingly; the staging footprint is identical with the pipeline
        # on or off). Flipping to a fresh bank needs no wait at all, so one
        # read's landing hides under the next reads; coming back to a bank
        # BANKS reads later is guarded by a just-in-time per-slot wait. The
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
        pending_slot_events: list[torch.npu.Event | None] = [None] * (
            banks * bank_window if pipeline_active else staging_window
        )
        # Per-receive timing (one INFO line per request): read_s is the
        # network stage, land_exposed_s the landing time spent on the
        # critical path (waits plus, in sync modes, the copies themselves).
        # copies/plan bracket the coalescing attribution: plan counts raw
        # staging-window entries, copies the actual copy_ calls after
        # adjacent runs were merged.
        # Device-side landing duration is deliberately NOT measured with
        # enable_timing events: elapsed_time() can block for a long time on
        # a busy device (observed ~75s stalls), which would poison the very
        # receive it instruments.
        timing_total_start = time.perf_counter()
        timing_read_secs = 0.0
        timing_read_settle_secs = 0.0
        timing_land_exposed_secs = 0.0
        timing_batches = 0
        timing_plan_entries = 0
        timing_copy_calls = 0
        try:
            for batch_start in range(0, len(staged_transfers), bank_window):
                batch = staged_transfers[batch_start : batch_start + bank_window]
                bank_index = (batch_start // bank_window) % banks if pipeline_active else 0
                bank_slot_base = bank_index * bank_window if pipeline_active else 0
                landing_stream = self._landing_streams[bank_index] if pipeline_active else self._landing_stream
                local_staging_ptrs = []
                remote_src_ptrs = []
                transfer_lengths = []
                copy_plan = []
                for position, (local_dst, remote_src, length) in enumerate(batch):
                    slot = bank_slot_base + position
                    staging_base_offset = slot * staging_chunk_bytes
                    staging_offset = staging_base_offset + int(remote_src) % (2 * 1024 * 1024)
                    local_staging_ptrs.append(staging_ptr + staging_offset)
                    remote_src_ptrs.append(remote_src)
                    transfer_lengths.append(length)
                    copy_plan.append((local_dst, length, staging_offset))
                timing_plan_entries += len(copy_plan)
                copy_plan = self._coalesce_dsa_pd_mooncake_copy_plan(copy_plan)
                timing_copy_calls += len(copy_plan)
                if pipeline_active:
                    reclaim_started = time.perf_counter()
                    for slot in range(bank_slot_base, bank_slot_base + len(batch)):
                        pending_event = pending_slot_events[slot]
                        if pending_event is not None:
                            pending_event.synchronize()
                            pending_slot_events[slot] = None
                    timing_land_exposed_secs += time.perf_counter() - reclaim_started
                read_started = time.perf_counter()
                ret = self.engine.batch_transfer_sync_read(
                    session_id, local_staging_ptrs, remote_src_ptrs, transfer_lengths
                )
                timing_read_secs += time.perf_counter() - read_started
                if ret >= 0 and ascend_envs.VLLM_ASCEND_DSA_PD_MOONCAKE_READ_PROBE:
                    # Timing-probe runs only: reveals whether read_s was the full
                    # network transfer or just its submission (see envs.py).
                    settle_started = time.perf_counter()
                    torch.npu.synchronize()
                    timing_read_settle_secs += time.perf_counter() - settle_started
                if ret < 0:
                    logger.error(
                        "Staged Mooncake batch read failed: request=%s "
                        "batch_start=%s batch_size=%s first_local=%s first_remote=%s first_len=%s ret=%s",
                        remote_request_id,
                        batch_start,
                        len(batch),
                        local_staging_ptrs[0] if local_staging_ptrs else None,
                        remote_src_ptrs[0] if remote_src_ptrs else None,
                        transfer_lengths[0] if transfer_lengths else None,
                        ret,
                    )
                    return ret
                if pipeline_active:
                    try:
                        with torch.npu.stream(landing_stream):
                            for local_dst, length, staging_offset in copy_plan:
                                self._copy_dsa_pd_mooncake_staging_to_offloaded_kv(local_dst, length, staging_offset)
                            for slot in range(bank_slot_base, bank_slot_base + len(batch)):
                                landing_event = torch.npu.Event()
                                landing_event.record(landing_stream)
                                pending_slot_events[slot] = landing_event
                    except Exception:
                        landing_stream.synchronize()
                        raise
                    timing_batches += 1
                else:
                    # Synchronous fallback (kill switch off, or window < 2):
                    # device-wide sync keeps the legacy v0 contract.
                    land_started = time.perf_counter()
                    try:
                        for local_dst, length, staging_offset in copy_plan:
                            self._copy_dsa_pd_mooncake_staging_to_offloaded_kv(local_dst, length, staging_offset)
                    finally:
                        torch.npu.synchronize()
                    timing_land_exposed_secs += time.perf_counter() - land_started
                    timing_batches += 1

        finally:
            if pipeline_active:
                drain_started = time.perf_counter()
                self._drain_dsa_pd_mooncake_slot_events(pending_slot_events)
                timing_land_exposed_secs += time.perf_counter() - drain_started
        logger.info(
            "Staged DSA PD mooncake receive timing: request=%s mode=%s batches=%s chunks=%s "
            "copies=%s plan=%s total_s=%.3f read_s=%.3f read_settle_s=%.3f land_exposed_s=%.3f",
            remote_request_id,
            "pipeline" if pipeline_active else "sync",
            timing_batches,
            len(staged_transfers),
            timing_copy_calls,
            timing_plan_entries,
            time.perf_counter() - timing_total_start,
            timing_read_secs,
            timing_read_settle_secs,
            timing_land_exposed_secs,
        )
        return 0


class DSAPDMooncakeStagingReader(DSAPDMooncakeStagingMixin):
    """V2 adapter; local destinations are the first address list in READ."""

    def __init__(
        self,
        engine: Any,
        kv_caches: dict[str, Any],
        block_len_per_addr: list[list[int]],
    ) -> None:
        self.engine = engine
        self.kv_caches = kv_caches
        self.block_len_per_addr = block_len_per_addr
        self._init_dsa_pd_mooncake_staging()

    def batch_transfer_sync_read(
        self,
        session_id: str,
        local_destinations: list[int],
        remote_sources: list[int],
        lengths: list[int],
        request_id: str,
    ) -> int:
        return self._batch_transfer_sync_read_with_optional_staging(
            session_id, local_destinations, remote_sources, lengths, request_id
        )
