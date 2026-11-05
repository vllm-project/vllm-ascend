# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import contextlib
import math
import threading
import time
from bisect import bisect_left, bisect_right
from collections.abc import Iterator
from itertools import accumulate
from typing import Any
from uuid import uuid4

import torch
from vllm.logger import logger
from vllm.utils.network_utils import get_ip, join_host_port

from vllm_ascend.model_loader.rfork.tensor_layout import (
    build_structural_digest,
    collect_transferable_tensors,
    find_non_npu_state_tensors,
    log_tensor_layout_summary,
    read_npu_format,
)
from vllm_ascend.model_loader.rfork.timing import StageTimer
from vllm_ascend.model_loader.rfork.types import SeedTransferInfo

MAX_TRANSFER_CHUNK_BYTES = 1024**3
MAX_TRANSFER_CHUNK_SEGMENTS = 512
MAX_MEMORY_REGISTRATION_BATCH_ITEMS = 4096
MAX_TRANSFER_ENGINE_FINALIZE_ATTEMPTS = 10
TRANSFER_ENGINE_FINALIZE_RETRY_INTERVAL_SEC = 1.0
MAX_TRANSFER_ENGINE_RPC_PORT = 65535


def _select_weight_blocks(
    memory_snapshot: list[dict[str, Any]],
    tensor_ranges: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    """Find overlapping allocator blocks with a prefix-max tensor interval index."""
    if not tensor_ranges:
        return []
    ranges = sorted(set(tensor_ranges))
    starts = [start for start, _ in ranges]
    max_ends = list(accumulate((end for _, end in ranges), max))
    active_blocks: set[tuple[int, int]] = set()
    for segment in memory_snapshot:
        for block in segment.get("blocks", []):
            address = block.get("address", -1)
            size = block.get("size", -1)
            if not is_positive_int(address) or not is_positive_int(size) or block.get("state") != "active_allocated":
                continue
            index = bisect_left(starts, address + size) - 1
            if index >= 0 and max_ends[index] > address:
                active_blocks.add((address, size))

    merged: list[tuple[int, int]] = []
    for address, size in sorted(active_blocks):
        if not merged or merged[-1][0] + merged[-1][1] < address:
            merged.append((address, size))
        else:
            start, length = merged[-1]
            merged[-1] = (start, max(start + length, address + size) - start)
    return merged


def iter_transfer_chunks(
    weight_names: list[str],
    seed_ptrs: list[int],
    client_ptrs: list[int],
    lengths: list[int],
) -> Iterator[tuple[list[str], list[int], list[int], list[int]]]:
    if not (len(weight_names) == len(seed_ptrs) == len(client_ptrs) == len(lengths)):
        raise ValueError("RFork transfer lists must have equal lengths")
    if not all(is_positive_int(length) for length in lengths):
        raise ValueError("RFork transfer segment length must be a positive integer")

    chunk_names: list[str] = []
    chunk_seed_ptrs: list[int] = []
    chunk_client_ptrs: list[int] = []
    chunk_lengths: list[int] = []
    chunk_bytes = 0
    for name, seed_ptr, client_ptr, length in zip(weight_names, seed_ptrs, client_ptrs, lengths, strict=True):
        offset = 0
        while offset < length:
            segment_length = min(MAX_TRANSFER_CHUNK_BYTES, length - offset)
            if chunk_lengths and (
                chunk_bytes + segment_length > MAX_TRANSFER_CHUNK_BYTES
                or len(chunk_lengths) >= MAX_TRANSFER_CHUNK_SEGMENTS
            ):
                yield chunk_names, chunk_seed_ptrs, chunk_client_ptrs, chunk_lengths
                chunk_names, chunk_seed_ptrs, chunk_client_ptrs, chunk_lengths = [], [], [], []
                chunk_bytes = 0

            chunk_names.append(name)
            chunk_seed_ptrs.append(seed_ptr + offset)
            chunk_client_ptrs.append(client_ptr + offset)
            chunk_lengths.append(segment_length)
            chunk_bytes += segment_length
            offset += segment_length

    if chunk_lengths:
        yield chunk_names, chunk_seed_ptrs, chunk_client_ptrs, chunk_lengths


def _tensor_storage_owner(tensor: torch.Tensor) -> Any:
    """Retain the storage object used by native memory registration."""
    untyped_storage = getattr(tensor, "untyped_storage", None)
    if callable(untyped_storage):
        try:
            return untyped_storage()
        except Exception:
            pass
    storage = getattr(tensor, "storage", None)
    if callable(storage):
        try:
            return storage()
        except Exception:
            pass
    raise RuntimeError("RFork could not retain the original storage owner for a registered tensor")


def is_positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


class RForkTransferBackend:
    """Own one YuanRong TransferEngine and its registered model memory.

    Thread safety: All public methods must be called with external synchronization.
    The owning RForkSession serializes access. Internal _lifecycle_lock protects
    only TransferEngine finalization and memory unregistration batching.

    registered_memory_addresses is modified during unregistration batching under
    _lifecycle_lock; readers must hold the lock or ensure no concurrent unregistration.
    """

    def __init__(self, tp_rank: int | None = None) -> None:
        self.tp_rank = tp_rank
        self.transfer_engine: Any | None = None
        self.transfer_session_id: str | None = None
        self.weight_manifest: dict[str, int] | None = None
        self.weight_formats: dict[str, int] | None = None
        self.registered_structural_digest: str | None = None
        self.registered_memory_addresses: list[int] = []
        self._registered_tensors: list[tuple[str, torch.Tensor]] | None = None
        self._registered_transferable_storages: list[Any] | None = None
        self._memory_registration_cls: Any | None = None
        self._not_ready_error_code: Any | None = None
        self._not_found_error_code: Any | None = None
        self._lifecycle_lock = threading.RLock()

    def _initialize_transfer_engine(self, timing: StageTimer | None = None) -> None:
        try:
            from yr.datasystem import (  # type: ignore[import-not-found]
                ErrorCode,
                MemoryRegistration,
                TransferEngine,
            )
        except ImportError as exc:
            raise ImportError(
                "RFork requires the YuanRong TransferEngine, MemoryRegistration, and ErrorCode APIs."
            ) from exc

        engine = TransferEngine()
        host = get_ip()
        # Port 0 lets the engine bind and hold a port itself (OS-assigned, or within
        # YuanRong's YR_TE_RPC_PORT_MIN/MAX range); the real port is read back below.
        endpoint = join_host_port(host, 0)
        device_name = f"npu:{torch.npu.current_device()}"
        if timing is not None:
            timing.mark("engine_initialize")
        try:
            result = engine.initialize(endpoint, "ascend", device_name)
        finally:
            if timing is not None:
                timing.mark("engine_init_overhead")
        if result.is_error():
            raise RuntimeError(
                f"YuanRong TransferEngine initialize({endpoint!r}, 'ascend', {device_name!r}) failed: "
                f"{result.to_string()}"
            )
        get_rpc_port = getattr(engine, "get_rpc_port", None)
        if not callable(get_rpc_port):
            self._finalize_failed_initialization(engine)
            raise RuntimeError("RFork requires YuanRong TransferEngine with get_rpc_port support.")
        try:
            rpc_port = get_rpc_port()
        except Exception as exc:
            self._finalize_failed_initialization(engine)
            raise RuntimeError(
                f"YuanRong TransferEngine get_rpc_port() raised for endpoint {endpoint!r}: {exc}"
            ) from exc
        if not isinstance(rpc_port, int) or not (0 < rpc_port <= MAX_TRANSFER_ENGINE_RPC_PORT):
            self._finalize_failed_initialization(engine)
            raise RuntimeError(
                f"YuanRong TransferEngine returned an invalid RPC port {rpc_port!r} for endpoint {endpoint!r}."
            )
        try:
            session_id = join_host_port(host, rpc_port)
            not_ready_error_code = ErrorCode.kNotReady
            not_found_error_code = getattr(ErrorCode, "kNotFound", None)
        except Exception:
            self._finalize_failed_initialization(engine)
            raise
        self.transfer_session_id = session_id
        self._memory_registration_cls = MemoryRegistration
        self._not_ready_error_code = not_ready_error_code
        self._not_found_error_code = not_found_error_code
        self.transfer_engine = engine

    @staticmethod
    def _finalize_failed_initialization(engine: Any) -> None:
        with contextlib.suppress(Exception):
            engine.finalize()

    def _engine(self) -> Any:
        if self.transfer_engine is None:
            raise RuntimeError("TransferEngine is not initialized.")
        return self.transfer_engine

    def _clear_registration_state(self) -> None:
        self.weight_manifest = None
        self.weight_formats = None
        self.registered_structural_digest = None
        self.registered_memory_addresses = []
        self._registered_tensors = None
        self._registered_transferable_storages = None

    def validate_final_layout(self, model, *, peer_session_id: str | None = None) -> None:
        """Verify the final layout against the full inventory captured at registration."""
        with self._lifecycle_lock:
            tensors = collect_transferable_tensors(model)
            formats = {}
            for name, tensor in tensors:
                fmt = read_npu_format(tensor)
                if fmt is None:
                    raise RuntimeError(f"RFork could not read the final NPU storage format for {name}")
                formats[name] = fmt
            digest = build_structural_digest(tensors, known_formats=formats)
            if not self.registered_structural_digest or digest != self.registered_structural_digest:
                raise RuntimeError("RFork final tensor structure differs from registration")
            try:
                log_tensor_layout_summary(
                    tensors,
                    stage="receiver_after_transfer_finalize",
                    session_id=self.transfer_session_id,
                    peer_session_id=peer_session_id,
                    known_formats=formats,
                )
            except Exception:
                logger.exception("RFork could not log the final tensor layout summary.")

    def register_memory_region(
        self,
        model,
        *,
        purpose: str = "unknown",
        identity=None,
    ) -> bool:
        timing = StageTimer("lock_wait")
        outcome = "exception"
        session_id = None
        try:
            with self._lifecycle_lock:
                try:
                    timing.mark("engine_init_overhead")
                    if self.transfer_engine is None:
                        self._initialize_transfer_engine(timing)
                    timing.mark("device_check")
                    registered = self._register_memory_region_locked(model, timing)
                    outcome = "success" if registered else "failed"
                    return registered
                finally:
                    session_id = self.transfer_session_id
                    timing.mark("lock_release")
        finally:
            total_s, stages_s = timing.finish()
            logger.info(
                "RFork registration timing: scope=register_memory_region relation=nested registration_id=%s "
                "purpose=%s model_kind=%s global_rank=%s tp_rank=%s session=%s outcome=%s total_s=%.6f stages_s=%s",
                uuid4().hex,
                purpose,
                "unknown" if identity is None else ("draft" if identity.is_draft_model else "main"),
                getattr(identity, "global_rank", "unknown"),
                self.tp_rank,
                session_id,
                outcome,
                total_s,
                stages_s,
            )

    def _register_memory_region_locked(
        self,
        model,
        timing: StageTimer,
    ) -> bool:
        transfer_engine = self._engine()
        start_reg_mr_time = time.perf_counter()

        non_npu_state = find_non_npu_state_tensors(model)
        if non_npu_state:
            logger.error(
                "RFork does not support mixed-device model state; non-NPU tensors include: %s",
                non_npu_state[:10],
            )
            return False

        timing.mark("stale_cleanup")
        if self.registered_memory_addresses:
            stale_block_count = len(self.registered_memory_addresses)
            if not self._unregister_weight_blocks(transfer_engine):
                return False
            logger.info(
                "Retried unregister for %d addresses left registered by a failed reset.",
                stale_block_count,
            )

        timing.mark("collect")
        transferable_tensors = collect_transferable_tensors(model)
        if not transferable_tensors:
            logger.error("RFork refuses to register an empty transferable tensor manifest.")
            return False

        timing.mark("manifest")
        weight_mr_dict: dict[str, int] = {}
        weight_format_dict: dict[str, int] = {}
        tensor_ranges: list[tuple[int, int]] = []
        transferable_storages: list[Any] = []
        for name, weight in transferable_tensors:
            transferable_storages.append(_tensor_storage_owner(weight))
            weight_ptr = weight.data_ptr()
            weight_numel = weight.numel()
            weight_size = weight.element_size()
            if not all(is_positive_int(value) for value in (weight_ptr, weight_numel, weight_size)):
                logger.error("RFork found an invalid tensor manifest entry for %s", name)
                return False
            weight_mr_dict[name] = weight_ptr
            weight_format = read_npu_format(weight)
            if weight_format is None:
                logger.error("RFork could not read the NPU storage format for %s", name)
                return False
            weight_format_dict[name] = weight_format
            tensor_ranges.append((weight_ptr, weight_ptr + weight_numel * weight_size))

        timing.mark("digest")
        structural_digest = build_structural_digest(transferable_tensors, known_formats=weight_format_dict)

        timing.mark("memory_snapshot")
        try:
            memory_snapshot = torch.npu.memory.memory_snapshot()
        except Exception as exc:
            logger.error("Failed to snapshot NPU memory for RFork registration: %s", exc)
            return False

        timing.mark("block_plan")
        merged_blocks = _select_weight_blocks(memory_snapshot, tensor_ranges)
        backing_starts = [start for start, _ in merged_blocks]

        logical_registrations: list[tuple[int, int, int, int]] = []
        for tensor_start, tensor_end in sorted(set(tensor_ranges)):
            index = bisect_right(backing_starts, tensor_start) - 1
            backing_block = merged_blocks[index] if index >= 0 else None
            if backing_block is not None and tensor_end > backing_block[0] + backing_block[1]:
                backing_block = None
            if backing_block is None:
                logger.error(
                    "RFork tensor range [%d, %d) is not fully covered by a registration block",
                    tensor_start,
                    tensor_end,
                )
                return False
            backing_start, backing_size = backing_block
            if (
                logical_registrations
                and logical_registrations[-1][2:] == (backing_start, backing_size)
                and tensor_start <= logical_registrations[-1][0] + logical_registrations[-1][1]
            ):
                logical_start, logical_size, _, _ = logical_registrations[-1]
                logical_registrations[-1] = (
                    logical_start,
                    max(logical_start + logical_size, tensor_end) - logical_start,
                    backing_start,
                    backing_size,
                )
            else:
                logical_registrations.append((tensor_start, tensor_end - tensor_start, backing_start, backing_size))

        timing.mark("native_register_overhead")
        # Keep backing storage alive until every attempted registration is released.
        self._registered_transferable_storages = transferable_storages
        memory_registration_cls = self._memory_registration_cls
        batch_register_memory_ex = getattr(transfer_engine, "batch_register_memory_ex", None)
        if memory_registration_cls is None or not callable(batch_register_memory_ex):
            logger.error(
                "RFork requires YuanRong TransferEngine with MemoryRegistration and batch_register_memory_ex support."
            )
            self._clear_registration_state()
            return False

        for batch_start in range(0, len(logical_registrations), MAX_MEMORY_REGISTRATION_BATCH_ITEMS):
            registration_batch = logical_registrations[batch_start : batch_start + MAX_MEMORY_REGISTRATION_BATCH_ITEMS]
            attempted_addresses = [registration[0] for registration in registration_batch]
            self.registered_memory_addresses.extend(attempted_addresses)
            try:
                registrations = [memory_registration_cls(*registration) for registration in registration_batch]
                timing.mark("native_calls")
                try:
                    result = batch_register_memory_ex(registrations)
                finally:
                    timing.mark("native_register_overhead")
            except Exception as exc:
                logger.error(
                    "TransferEngine memory registration raised for batch of %d logical regions: %s",
                    len(registration_batch),
                    exc,
                )
                result = None

            if result is None or result.is_error():
                if result is not None:
                    logger.error(
                        "TransferEngine memory registration failed for batch of %d logical regions, ret: %s",
                        len(registration_batch),
                        result.to_string(),
                    )
                # Retain every attempted address and owner until unregistration confirms absence.
                timing.mark("registration_rollback")
                if not self._unregister_weight_blocks(transfer_engine):
                    logger.error("RFork registration rollback is incomplete; preserving registration state for retry.")
                return False

        self.weight_manifest = weight_mr_dict
        self.weight_formats = weight_format_dict
        self._registered_tensors = transferable_tensors
        self.registered_structural_digest = structural_digest
        logger.debug(
            "register_memory_region time: %.4fs, weights: %d",
            time.perf_counter() - start_reg_mr_time,
            len(weight_mr_dict),
        )
        timing.mark("layout_summary")
        log_tensor_layout_summary(
            transferable_tensors,
            stage="registration",
            session_id=self.transfer_session_id,
            known_formats=weight_format_dict,
        )
        return True

    def _unregister_weight_blocks(self, transfer_engine: Any) -> bool:
        while self.registered_memory_addresses:
            address_batch = self.registered_memory_addresses[:MAX_MEMORY_REGISTRATION_BATCH_ITEMS]
            try:
                result = transfer_engine.batch_unregister_memory(address_batch)
            except Exception as exc:
                logger.error("batch_unregister_memory raised for batch of %d regions: %s", len(address_batch), exc)
                return False
            if result is None or result.is_error():
                if result is not None and self._is_not_found_result(result):
                    # A missing address prevents the batch from being fully released.
                    unresolved = self._unregister_addresses_individually(transfer_engine, address_batch)
                    self.registered_memory_addresses[: len(address_batch)] = unresolved
                    if not unresolved:
                        continue
                    logger.error("batch_unregister_memory left %d regions registered.", len(unresolved))
                elif result is not None:
                    logger.error(
                        "batch_unregister_memory failed for batch of %d regions, ret: %s",
                        len(address_batch),
                        result.to_string(),
                    )
                return False
            del self.registered_memory_addresses[: len(address_batch)]

        self._clear_registration_state()
        return True

    def _is_not_found_result(self, result: Any) -> bool:
        get_code = getattr(result, "get_code", None)
        code = get_code() if callable(get_code) else None
        return self._not_found_error_code is not None and code == self._not_found_error_code

    def _unregister_addresses_individually(self, transfer_engine: Any, addresses: list[int]) -> list[int]:
        unresolved: list[int] = []
        for address in addresses:
            try:
                result = transfer_engine.batch_unregister_memory([address])
            except Exception as exc:
                logger.error("batch_unregister_memory raised for address %d: %s", address, exc)
                unresolved.append(address)
                continue
            if result is None or (result.is_error() and not self._is_not_found_result(result)):
                unresolved.append(address)
                if result is not None:
                    logger.error(
                        "batch_unregister_memory failed for address %d, ret: %s",
                        address,
                        result.to_string(),
                    )
        return unresolved

    def unregister_memory_region(self) -> bool:
        with self._lifecycle_lock:
            return self._unregister_memory_region_locked()

    def _unregister_memory_region_locked(self) -> bool:
        start_unreg_mr_time = time.perf_counter()
        if not self.registered_memory_addresses:
            self._clear_registration_state()
            logger.debug("unregister_memory_region skipped because no blocks are registered.")
            return True
        transfer_engine = self._engine()
        if not self._unregister_weight_blocks(transfer_engine):
            return False
        logger.debug(
            "unregister_memory_region time: %.4fs",
            time.perf_counter() - start_unreg_mr_time,
        )
        return True

    def finalize_transfer_engine(
        self,
        max_attempts: int = MAX_TRANSFER_ENGINE_FINALIZE_ATTEMPTS,
        retry_interval_sec: float = TRANSFER_ENGINE_FINALIZE_RETRY_INTERVAL_SEC,
    ) -> bool:
        if not isinstance(max_attempts, int) or isinstance(max_attempts, bool) or max_attempts <= 0:
            raise ValueError("RFork TransferEngine finalize max_attempts must be a positive integer")
        if (
            isinstance(retry_interval_sec, bool)
            or not isinstance(retry_interval_sec, (int, float))
            or not math.isfinite(float(retry_interval_sec))
            or retry_interval_sec < 0
        ):
            raise ValueError("RFork TransferEngine finalize retry_interval_sec must be a non-negative number")

        with self._lifecycle_lock:
            if self.transfer_engine is None:
                return True
            transfer_engine = self._engine()
            for attempt in range(1, max_attempts + 1):
                try:
                    result = transfer_engine.finalize()
                except Exception as exc:
                    logger.error(
                        "TransferEngine finalize raised on attempt %d/%d: %s",
                        attempt,
                        max_attempts,
                        exc,
                    )
                    return False
                if not result.is_error():
                    self._clear_registration_state()
                    self.transfer_session_id = None
                    self.transfer_engine = None
                    return True

                logger.warning(
                    "TransferEngine finalize failed on attempt %d/%d: %s",
                    attempt,
                    max_attempts,
                    result.to_string(),
                )
                get_code = getattr(result, "get_code", None)
                if (
                    self._not_ready_error_code is None
                    or not callable(get_code)
                    or get_code() != self._not_ready_error_code
                ):
                    return False
                if attempt < max_attempts:
                    time.sleep(float(retry_interval_sec))
            return False

    def read_weights_from_seed(
        self,
        seed_info: SeedTransferInfo,
    ) -> bool:
        with self._lifecycle_lock:
            return self._read_weights_from_seed_locked(seed_info)

    def _read_weights_from_seed_locked(
        self,
        seed_info: SeedTransferInfo,
    ) -> bool:
        transferable_tensors = self._registered_tensors
        if transferable_tensors is None:
            logger.error("RFork cannot read seed weights without a destination tensor inventory.")
            return False

        weight_names: list[str] = []
        seed_ptr_list: list[int] = []
        client_ptr_list: list[int] = []
        client_len_list: list[int] = []
        # Keep reads per name; address-only merging is unsafe when remote/local offsets differ.
        for name, tensor in transferable_tensors:
            seed_ptr = seed_info.weights.get(name)
            if not is_positive_int(seed_ptr):
                logger.error("RFork seed returned a missing or invalid address for %s", name)
                return False
            weight_names.append(name)
            seed_ptr_list.append(seed_ptr)
            client_ptr_list.append(tensor.data_ptr())
            client_len_list.append(tensor.numel() * tensor.element_size())

        log_tensor_layout_summary(
            transferable_tensors,
            stage="receiver_before_read",
            session_id=self.transfer_session_id,
            peer_session_id=seed_info.session_id,
            known_formats=self.weight_formats,
        )

        chunks = list(iter_transfer_chunks(weight_names, seed_ptr_list, client_ptr_list, client_len_list))
        total_bytes = sum(client_len_list)
        total_gib = total_bytes / (1024**3)
        transfer_start = time.perf_counter()
        logger.debug(
            "transfer weights starts, weights: %d, chunks: %d, total bytes: %.2f GiB",
            len(client_len_list),
            len(chunks),
            total_gib,
        )
        for index, (chunk_names, chunk_seed_ptrs, chunk_client_ptrs, chunk_lengths) in enumerate(chunks, 1):
            chunk_start = time.perf_counter()
            result = self._engine().batch_transfer_sync_read(
                seed_info.session_id,
                chunk_client_ptrs,
                chunk_seed_ptrs,
                chunk_lengths,
            )
            if result.is_error():
                logger.error(
                    "Failed to transfer weights chunk %d/%d, first: %s, last: %s, ret=%s",
                    index,
                    len(chunks),
                    chunk_names[0],
                    chunk_names[-1],
                    result.to_string(),
                )
                return False
            logger.debug(
                "transfer weights chunk %d/%d done, time: %.4fs",
                index,
                len(chunks),
                time.perf_counter() - chunk_start,
            )
        transfer_elapsed = time.perf_counter() - transfer_start
        throughput_gib_s = total_gib / transfer_elapsed if transfer_elapsed > 0 else 0.0
        log_transfer = logger.info if self.tp_rank == 0 else logger.debug
        log_transfer(
            "RFork weight transfer completed: tp_rank=%s, weights=%d, chunks=%d, bytes=%.2f GiB, "
            "elapsed=%.4fs, throughput=%.2f GiB/s",
            self.tp_rank,
            len(client_len_list),
            len(chunks),
            total_gib,
            transfer_elapsed,
            throughput_gib_s,
        )
        return True


__all__ = [
    "MAX_MEMORY_REGISTRATION_BATCH_ITEMS",
    "MAX_TRANSFER_CHUNK_BYTES",
    "MAX_TRANSFER_CHUNK_SEGMENTS",
    "MAX_TRANSFER_ENGINE_FINALIZE_ATTEMPTS",
    "RForkTransferBackend",
]
