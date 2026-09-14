# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import math
import threading
import time
from bisect import bisect_left, bisect_right
from collections.abc import Iterator, Mapping
from itertools import accumulate
from typing import Any

import torch
from vllm.logger import logger
from vllm.utils.network_utils import get_ip, get_open_port, join_host_port

from vllm_ascend.model_loader.rfork.manifest import (
    is_positive_int,
    normalize_dtype_name,
    read_npu_format,
    update_registered_weight_info,
    validate_weight_manifest,
)
from vllm_ascend.model_loader.rfork.tensor_layout import (
    collect_transferable_tensors,
    find_non_npu_state_tensors,
    is_transferable_tensor,
    reshape_tensor_to_seed_shape,
    validate_transferable_tensor_layout,
)
from vllm_ascend.model_loader.rfork.types import SeedTransferInfo

MAX_TRANSFER_CHUNK_BYTES = 1024**3
MAX_TRANSFER_CHUNK_SEGMENTS = 512
MAX_MEMORY_REGISTRATION_BATCH_ITEMS = 4096
MAX_TRANSFER_ENGINE_FINALIZE_ATTEMPTS = 10
TRANSFER_ENGINE_FINALIZE_RETRY_INTERVAL_SEC = 1.0


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


def _is_tensor_in_blocks(tensor: torch.Tensor, blocks: list[tuple[int, int]]) -> bool:
    tensor_range = (tensor.data_ptr(), tensor.numel() * tensor.element_size())
    return not _subtract_weight_blocks([tensor_range], blocks)


def _split_tensors_by_excluded_blocks(
    transferable_tensors: list[tuple[str, torch.Tensor]],
    excluded_blocks: list[tuple[int, int]],
) -> tuple[list[tuple[str, torch.Tensor]], list[str]]:
    if not excluded_blocks:
        return list(transferable_tensors), []

    kept_tensors: list[tuple[str, torch.Tensor]] = []
    excluded_names: list[str] = []
    for name, tensor in transferable_tensors:
        if _is_tensor_in_blocks(tensor, excluded_blocks):
            excluded_names.append(name)
        else:
            kept_tensors.append((name, tensor))
    return kept_tensors, excluded_names


def _subtract_weight_blocks(
    weight_blocks: list[tuple[int, int]],
    excluded_blocks: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    if not weight_blocks or not excluded_blocks:
        return list(weight_blocks)

    sorted_excluded = sorted(excluded_blocks)
    residual_blocks: list[tuple[int, int]] = []
    for address, size in weight_blocks:
        block_end = address + size
        cursor = address
        for excluded_address, excluded_size in sorted_excluded:
            excluded_end = excluded_address + excluded_size
            if excluded_end <= cursor:
                continue
            if excluded_address >= block_end:
                break
            if excluded_address > cursor:
                residual_blocks.append((cursor, excluded_address - cursor))
            cursor = max(cursor, excluded_end)
            if cursor >= block_end:
                break
        if cursor < block_end:
            residual_blocks.append((cursor, block_end - cursor))
    return residual_blocks


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


def _append_unique_addresses(addresses: list[int], new_addresses: list[int]) -> None:
    known = set(addresses)
    for address in new_addresses:
        if address not in known:
            addresses.append(address)
            known.add(address)


def _validate_transferable_tensor_layouts(transferable_tensors: list[tuple[str, torch.Tensor]]) -> None:
    for name, tensor in transferable_tensors:
        validate_transferable_tensor_layout(name, tensor)


class RForkTransferBackend:
    """Own one YuanRong TransferEngine and its registered model memory."""

    def __init__(self) -> None:
        self.transfer_engine: Any | None = None
        self.transfer_session_id: str | None = None
        self.weight_manifest: dict[str, Any] | None = None
        self.weight_formats: dict[str, int] | None = None
        self.registered_weight_blocks: list[tuple[int, int]] = []
        self.registered_memory_addresses: list[int] = []
        self.excluded_weight_blocks: list[tuple[int, int]] = []
        self._registered_transferable_tensors: list[tuple[str, torch.Tensor]] | None = None
        self._registered_transferable_storages: list[Any] | None = None
        self._all_transferable_tensors_excluded = False
        self._memory_registration_cls: Any | None = None
        self._not_ready_error_code: Any | None = None
        self._not_found_error_code: Any | None = None
        self._lifecycle_lock = threading.RLock()
        self._is_initialized = False

    def _initialize_transfer_engine(self) -> None:
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
        endpoint = join_host_port(get_ip(), get_open_port())
        device_name = f"npu:{torch.npu.current_device()}"
        result = engine.initialize(endpoint, "ascend", device_name)
        if result.is_error():
            raise RuntimeError(
                f"YuanRong TransferEngine initialize({endpoint!r}, 'ascend', {device_name!r}) failed: "
                f"{result.to_string()}"
            )
        self.transfer_engine = engine
        self.transfer_session_id = endpoint
        self._memory_registration_cls = MemoryRegistration
        self._not_ready_error_code = ErrorCode.kNotReady
        self._not_found_error_code = getattr(ErrorCode, "kNotFound", None)
        self._is_initialized = True

    def _engine(self) -> Any:
        if self.transfer_engine is None:
            raise RuntimeError("TransferEngine is not initialized.")
        return self.transfer_engine

    def _clear_registration_state(self) -> None:
        self.weight_manifest = None
        self.weight_formats = None
        self.registered_weight_blocks = []
        self.registered_memory_addresses = []
        self._registered_transferable_tensors = None
        self._registered_transferable_storages = None
        self._all_transferable_tensors_excluded = False

    def snapshot_registered_weight_blocks(self) -> list[tuple[int, int]]:
        """Return a consistent copy for draft exclusion; this does not pin its lifetime."""
        with self._lifecycle_lock:
            return list(self.registered_weight_blocks)

    def register_memory_region(
        self,
        model,
        processed_layout: bool,
        exclude_blocks: list[tuple[int, int]] | None = None,
    ) -> bool:
        with self._lifecycle_lock:
            # Initialize native resources on the NPU caller thread only when registration is required.
            if self.transfer_engine is None:
                self._initialize_transfer_engine()
            return self._register_memory_region_locked(model, processed_layout, exclude_blocks)

    def can_reuse_shared_weights(self, model, processed_layout: bool, exclude_blocks: list[tuple[int, int]]) -> bool:
        """Whether every live weight is already owned by the loaded target model."""
        if not exclude_blocks or find_non_npu_state_tensors(model):
            return False
        transferable_tensors = list(collect_transferable_tensors(model, processed_layout))
        _validate_transferable_tensor_layouts(transferable_tensors)
        independent, shared_names = _split_tensors_by_excluded_blocks(transferable_tensors, exclude_blocks)
        return bool(shared_names) and not independent

    def _register_memory_region_locked(
        self,
        model,
        processed_layout: bool,
        exclude_blocks: list[tuple[int, int]] | None = None,
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

        if self.registered_weight_blocks or self.registered_memory_addresses:
            stale_block_count = len(self.registered_weight_blocks)
            if not self._unregister_weight_blocks(transfer_engine):
                return False
            logger.info(
                "Retried unregister for %d blocks left registered by a failed reset.",
                stale_block_count,
            )

        excluded_blocks = list(exclude_blocks) if exclude_blocks else []
        self.excluded_weight_blocks = excluded_blocks
        all_transferable_tensors = list(collect_transferable_tensors(model, processed_layout))
        _validate_transferable_tensor_layouts(all_transferable_tensors)
        transferable_tensors, excluded_names = _split_tensors_by_excluded_blocks(
            all_transferable_tensors,
            excluded_blocks,
        )
        if excluded_names:
            logger.debug(
                "Skipping %d weights shared with the target model (already registered), e.g. %s",
                len(excluded_names),
                ", ".join(excluded_names[:3]),
            )

        if not transferable_tensors and not excluded_names:
            logger.error("RFork refuses to register an empty transferable tensor manifest.")
            return False

        transferable_names = [name for name, _ in transferable_tensors]
        if any(not isinstance(name, str) or not name for name in transferable_names) or len(transferable_names) != len(
            set(transferable_names)
        ):
            logger.error("RFork refuses a manifest with duplicate or empty tensor names.")
            return False

        weight_mr_dict: dict[str, Any] = {}
        weight_format_dict: dict[str, int] = {}
        tensor_ranges: list[tuple[int, int]] = []
        transferable_storages: list[Any] = []
        for name, weight in transferable_tensors:
            if not is_transferable_tensor(weight):
                logger.error("RFork found an invalid transferable tensor entry: %r", name)
                return False
            transferable_storages.append(_tensor_storage_owner(weight))
            weight_ptr = weight.data_ptr()
            weight_numel = weight.numel()
            weight_size = weight.element_size()
            if not all(is_positive_int(value) for value in (weight_ptr, weight_numel, weight_size)):
                logger.error("RFork found an invalid tensor manifest entry for %s", name)
                return False
            weight_shape = tuple(weight.shape)
            weight_mr_dict[name] = (
                weight_ptr,
                weight_numel,
                weight_size,
                weight_shape,
                normalize_dtype_name(weight.dtype),
            )
            weight_format = read_npu_format(weight)
            if weight_format is None:
                logger.error("RFork could not read the NPU storage format for %s", name)
                return False
            weight_format_dict[name] = weight_format
            tensor_ranges.append((weight_ptr, weight_ptr + weight_numel * weight_size))

        try:
            memory_snapshot = torch.npu.memory.memory_snapshot()
        except Exception as exc:
            logger.error("Failed to snapshot NPU memory for RFork registration: %s", exc)
            return False

        merged_blocks = _select_weight_blocks(memory_snapshot, tensor_ranges)
        merged_blocks = _subtract_weight_blocks(merged_blocks, excluded_blocks)
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

        registered_memory_addresses: list[int] = []
        # Retain provisional metadata and owners before registration so failures remain recoverable.
        self.weight_manifest = weight_mr_dict
        self.weight_formats = weight_format_dict
        self.registered_weight_blocks = list(merged_blocks)
        self.registered_memory_addresses = []
        self._registered_transferable_tensors = transferable_tensors
        self._registered_transferable_storages = transferable_storages
        self._all_transferable_tensors_excluded = bool(not transferable_tensors and excluded_names)
        if logical_registrations:
            memory_registration_cls = self._memory_registration_cls
            batch_register_memory_ex = getattr(transfer_engine, "batch_register_memory_ex", None)
            if memory_registration_cls is None or not callable(batch_register_memory_ex):
                logger.error(
                    "RFork requires YuanRong TransferEngine with MemoryRegistration and "
                    "batch_register_memory_ex support."
                )
                self._clear_registration_state()
                return False

            for batch_start in range(0, len(logical_registrations), MAX_MEMORY_REGISTRATION_BATCH_ITEMS):
                registration_batch = logical_registrations[
                    batch_start : batch_start + MAX_MEMORY_REGISTRATION_BATCH_ITEMS
                ]
                attempted_addresses = [registration[0] for registration in registration_batch]
                _append_unique_addresses(self.registered_memory_addresses, attempted_addresses)
                try:
                    registrations = [memory_registration_cls(*registration) for registration in registration_batch]
                    result = batch_register_memory_ex(registrations)
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
                    if self.registered_memory_addresses and not self._unregister_weight_blocks(transfer_engine):
                        logger.error(
                            "RFork registration rollback is incomplete; preserving registration state for retry."
                        )
                    return False

                _append_unique_addresses(registered_memory_addresses, attempted_addresses)

        # Publish confirmed addresses while retaining the manifest and owners for active transfers.
        self.registered_memory_addresses = list(registered_memory_addresses)
        logger.debug(
            "register_memory_region time: %.4fs, weights: %d",
            time.perf_counter() - start_reg_mr_time,
            len(weight_mr_dict),
        )
        return True

    def _unregister_weight_blocks(self, transfer_engine: Any) -> bool:
        remaining_addresses = list(dict.fromkeys(self.registered_memory_addresses))
        if not remaining_addresses:
            self._clear_registration_state()
            return True

        while remaining_addresses:
            address_batch = remaining_addresses[:MAX_MEMORY_REGISTRATION_BATCH_ITEMS]
            try:
                result = transfer_engine.batch_unregister_memory(address_batch)
            except Exception as exc:
                self.registered_memory_addresses = list(remaining_addresses)
                logger.error(
                    "batch_unregister_memory raised for batch of %d regions: %s",
                    len(address_batch),
                    exc,
                )
                return False
            if result is None or result.is_error():
                if result is not None and self._is_not_found_result(result):
                    # Retry per address because batch unregistration may stop at the first absent region.
                    unresolved_addresses = self._unregister_addresses_individually(transfer_engine, address_batch)
                    del remaining_addresses[: len(address_batch)]
                    remaining_addresses = unresolved_addresses + remaining_addresses
                    self.registered_memory_addresses = list(remaining_addresses)
                    if unresolved_addresses:
                        logger.error(
                            "batch_unregister_memory left %d regions registered after a not-found result.",
                            len(unresolved_addresses),
                        )
                        return False
                    continue
                self.registered_memory_addresses = list(remaining_addresses)
                if result is not None:
                    logger.error(
                        "batch_unregister_memory failed for batch of %d regions, ret: %s",
                        len(address_batch),
                        result.to_string(),
                    )
                return False
            del remaining_addresses[: len(address_batch)]
            self.registered_memory_addresses = list(remaining_addresses)

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
        if not self.registered_weight_blocks and not self.registered_memory_addresses:
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
            if not self._is_initialized:
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
                    self._is_initialized = False
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
        model,
        seed_info: SeedTransferInfo,
        processed_layout: bool,
    ) -> bool:
        with self._lifecycle_lock:
            return self._read_weights_from_seed_locked(model, seed_info, processed_layout)

    def _read_weights_from_seed_locked(
        self,
        model,
        seed_info: SeedTransferInfo,
        processed_layout: bool,
    ) -> bool:
        if (
            not isinstance(seed_info.session_id, str)
            or not seed_info.session_id
            or not isinstance(seed_info.weights, Mapping)
        ):
            logger.error("RFork seed returned an invalid session or weight manifest.")
            return False

        transferable_tensors = self._registered_transferable_tensors
        if transferable_tensors is None:
            transferable_tensors = list(collect_transferable_tensors(model, processed_layout))
        _validate_transferable_tensor_layouts(transferable_tensors)
        if not transferable_tensors:
            if self._all_transferable_tensors_excluded and self.weight_manifest == {} and not seed_info.weights:
                logger.debug("RFork seed transfer has no local weights because all tensors are shared.")
                return True
            logger.error("RFork refuses to transfer an empty local tensor manifest.")
            return False

        local_names = [name for name, _ in transferable_tensors]
        local_name_set = set(local_names)
        if (
            len(local_names) != len(local_name_set)
            or not local_name_set
            or any(not isinstance(name, str) or not name for name in local_name_set)
        ):
            logger.error("RFork local tensor manifest has duplicate or empty names.")
            return False

        remote_names = list(seed_info.weights)
        remote_name_set = set(remote_names)
        if (
            len(remote_names) != len(remote_name_set)
            or not remote_name_set
            or any(not isinstance(name, str) or not name for name in remote_name_set)
        ):
            logger.error("RFork remote tensor manifest has duplicate or empty names.")
            return False

        excluded_blocks = self.excluded_weight_blocks
        local_only = local_name_set - remote_name_set
        remote_only = remote_name_set - local_name_set
        skipped_shared_names: set[str] = set()
        if local_only:
            for name, tensor in transferable_tensors:
                if name in local_only and _is_tensor_in_blocks(tensor, excluded_blocks):
                    logger.debug(
                        "Skip RFork weight %s shared with the target model: not present in the seed manifest.",
                        name,
                    )
                    skipped_shared_names.add(name)
            if local_only - skipped_shared_names:
                logger.error(
                    "RFork manifest names differ: local_only=%s, remote_only=%s",
                    sorted(local_only - skipped_shared_names, key=str),
                    sorted(remote_only, key=str),
                )
                return False
        if remote_only:
            full_model_transferable_tensors = list(collect_transferable_tensors(model, processed_layout))
            _validate_transferable_tensor_layouts(full_model_transferable_tensors)
            full_model_tensors = dict(full_model_transferable_tensors)
            for name in remote_only:
                tensor = full_model_tensors.get(name)
                if tensor is not None and _is_tensor_in_blocks(tensor, excluded_blocks):
                    skipped_shared_names.add(name)
            if remote_only - skipped_shared_names:
                logger.error(
                    "RFork manifest names differ: local_only=%s, remote_only=%s",
                    sorted(local_only, key=str),
                    sorted(remote_only - skipped_shared_names, key=str),
                )
                return False

        parsed_remote = validate_weight_manifest(seed_info, transferable_tensors, skipped_shared_names)
        if parsed_remote is None:
            return False

        reshape_events: list[tuple[str, tuple[int, ...], tuple[int, ...]]] = []
        for name, tensor in transferable_tensors:
            if name in skipped_shared_names:
                continue
            if not reshape_tensor_to_seed_shape(name, tensor, parsed_remote[name][3], reshape_events):
                return False
            update_registered_weight_info(
                self.weight_manifest,
                name,
                tensor,
            )

        # Revalidate cached tensors immediately before constructing raw byte reads.
        _validate_transferable_tensor_layouts(transferable_tensors)

        weight_names: list[str] = []
        seed_ptr_list: list[int] = []
        client_ptr_list: list[int] = []
        client_len_list: list[int] = []
        # Keep reads per name; address-only merging is unsafe when remote/local offsets differ.
        for name, tensor in transferable_tensors:
            if name in skipped_shared_names:
                continue
            weight_names.append(name)
            seed_ptr_list.append(parsed_remote[name][0])
            client_ptr_list.append(tensor.data_ptr())
            client_len_list.append(tensor.numel() * tensor.element_size())

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
        logger.debug(
            "RFork weight transfer completed: weights=%d, chunks=%d, bytes=%.2f GiB, "
            "elapsed=%.4fs, throughput=%.2f GiB/s",
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
