# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""NPU tensor I/O using upstream Store lookup, read grouping and poison bookkeeping."""

from __future__ import annotations

import threading
from collections.abc import Callable, Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from functools import partial
from typing import Any

import torch
from vllm.distributed.ec_transfer.ec_connector.mooncake_store_embedding.data import (
    TensorSpec,
)
from vllm.distributed.ec_transfer.ec_connector.mooncake_store_embedding.store_client import (
    _MOONCAKE_TENSOR_HEADER,
    _SAFE_PUT_REJECTIONS,
    EmbeddingStoreError,
    EmbeddingStoreOperationError,
    MooncakeEmbeddingStoreClient,
    _batch_get_completed_safely,
    _encode_mooncake_tensor_metadata,
    _validate_mooncake_tensor_metadata,
)
from vllm.logger import logger
from vllm.utils.math_utils import round_up

from vllm_ascend.distributed.ec_transfer.ec_connector.mooncake.memory import (
    AscendProducerMemoryPool,
    AscendStagedSources,
)

_PROCESS_LIFETIME_STORE_OWNERS: list[tuple[Any, Any]] = []


class AscendMooncakeEmbeddingStoreClient(MooncakeEmbeddingStoreClient):
    """Reuse upstream control/codec logic; override NPU I/O and fixed-pool lifetime."""

    def __init__(
        self,
        store: Any,
        staging_pool: AscendProducerMemoryPool,
        replicate_config: Any | None = None,
        *,
        read_buffer_bytes: int = 128 * 1024**2,
        use_fabric_mem: bool = False,
        use_store_independent_te: bool = False,
    ) -> None:
        """Bind Store I/O to the producer slab and initialize completion tracking."""
        super().__init__(
            store,
            replicate_config=replicate_config,
            read_buffer_bytes=read_buffer_bytes,
        )
        if use_fabric_mem and use_store_independent_te:
            raise ValueError("FabricMem and an independently configured Store are different topologies")
        self._independent = use_store_independent_te
        self._bound = False
        self._staging_pool = staging_pool
        self._closed = False
        self._copy_stream: Any | None = None
        # Mooncake's Python Store GET is synchronous and its concurrent-call
        # contract is not exposed. Serialize foreground fallback and optional
        # background GETs while still overlapping them with model work.
        self._get_io_lock = threading.RLock()
        # Completion waits must never run on the model thread.  Copies on the
        # private stream are ordered, so one worker can retire their leases in
        # submission order without serializing the copies themselves.
        self._get_completion_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="ec-mooncake-get-completion",
        )
        # Called synchronously after validating a whole GET chunk, before
        # submitting its cache copies. One callback preserves native P2P
        # batching instead of scheduling one executor task per Store object.
        self.on_get_payloads: Callable[[dict[str, torch.Tensor]], list[Future[None]]] | None = None

    def load_tensors(
        self,
        expected: Mapping[str, TensorSpec],
        device: torch.device | str,
    ) -> dict[str, torch.Tensor]:
        """Load expected tensors while serializing access to native GET state."""
        with self._get_io_lock:
            loaded = super().load_tensors(expected, device)
            return loaded

    def batch_exists(self, keys: list[str]) -> list[bool]:
        """Check object presence under the shared GET I/O lock."""
        with self._get_io_lock:
            return super().batch_exists(keys)

    @contextmanager
    def get_payload_callback(
        self,
        callback: Callable[[dict[str, torch.Tensor]], list[Future[None]]] | None,
    ) -> Any:
        """Bind the direct-send callback for one GET under the serialized I/O lock."""
        with self._get_io_lock:
            previous = self.on_get_payloads
            self.on_get_payloads = callback
            try:
                yield
            finally:
                self.on_get_payloads = previous

    def load_tensors_ready(
        self,
        expected: Mapping[str, TensorSpec],
        device: torch.device | str,
    ) -> dict[str, torch.Tensor]:
        """Load on a background thread and return only when targets are ready."""
        with self._get_io_lock:
            loaded = super().load_tensors(expected, device)
            copy_stream = self._copy_stream
            if copy_stream is not None:
                torch.npu.set_device(device)
                copy_stream.synchronize()
            return loaded

    def _check_healthy(self) -> None:
        """Reject operations on a poisoned or closed Store client."""
        super()._check_healthy()
        if self._closed:
            raise EmbeddingStoreError("Ascend Mooncake Store client is closed")

    def _poison(self, owners: list[Any]) -> None:
        """Retain uncertain I/O owners and prevent producer slab reuse."""
        super()._poison(owners)
        self._staging_pool.poison()

    def bind_staging_buffer(self, tensor: torch.Tensor) -> None:
        """Bind the fixed EC slab and register it with an independent Store."""
        self._check_healthy()
        if tensor is not self._staging_pool.tensor:
            raise ValueError("Store must bind the worker's fixed staging pool")
        if self._bound:
            return
        if self._independent:
            status = self.store.register_buffer(tensor.data_ptr(), tensor.nbytes)
            if status != 0:
                raise EmbeddingStoreOperationError(f"Failed to register Store staging pool: {status}")
        self._bound = True

    def put_tensor(self, pool_key: str, tensor: torch.Tensor) -> None:
        """Publish one tensor through the staged PUT path."""
        self.put_tensors([(pool_key, tensor)])

    def put_tensors(self, items: list[tuple[str, torch.Tensor]]) -> None:
        """Publish whole objects from the fixed staging pool."""
        self._check_healthy()
        if not items:
            return
        tensors = [tensor for _, tensor in items]
        device = tensors[0].device
        if any(t.device != device or t.device.type != "npu" or not t.is_contiguous() for t in tensors):
            raise EmbeddingStoreOperationError("Store PUT requires contiguous tensors on one NPU")
        torch.npu.set_device(device)
        torch.npu.synchronize(device)
        staged = self._staging_pool.stage_with_headers(tensors, [self.prepare_put_header(t) for t in tensors])
        if staged is None:
            raise EmbeddingStoreOperationError("Store PUT batch exceeds the fixed staging pool")
        try:
            for (key, _), tensor, header in zip(items, staged.tensors, staged.headers, strict=True):
                assert header is not None
                self.put_prepared_tensor(
                    key, [header.data_ptr(), tensor.data_ptr()], [header.nbytes, tensor.nbytes], [staged]
                )
        finally:
            if not self._poisoned:
                self._staging_pool.release(staged)

    def prepare_put_header(self, tensor: torch.Tensor) -> torch.Tensor:
        """Build the small CPU header copied alongside the staged payload."""
        metadata = _encode_mooncake_tensor_metadata(tensor)
        return torch.frombuffer(bytearray(metadata), dtype=torch.uint8)

    def put_prepared_tensor(
        self,
        pool_key: str,
        addresses: list[int],
        sizes: list[int],
        owners: list[Any],
    ) -> None:
        """PUT buffers within the fixed pool; never register transient ranges."""
        [result] = self.put_prepared_tensors([(pool_key, addresses, sizes, owners)])
        if result < 0:
            raise EmbeddingStoreOperationError(f"failed to put embedding tensor for {pool_key}")
        logger.info("Stored encoder output in Mooncake Store")

    def put_prepared_tensors(
        self,
        items: list[tuple[str, list[int], list[int], list[Any]]],
    ) -> list[int]:
        """Publish one staged encoder batch with one native Store call."""
        self._check_healthy()
        if not items:
            return []
        if not self._bound:
            raise EmbeddingStoreOperationError("Store staging pool is not bound")
        owners = [owner for _, _, _, item_owners in items for owner in item_owners]
        try:
            results = self.store.batch_put_from_multi_buffers(
                [pool_key for pool_key, _, _, _ in items],
                [addresses for _, addresses, _, _ in items],
                [sizes for _, _, sizes, _ in items],
                self.replicate_config,
            )
            if (
                not isinstance(results, list)
                or len(results) != len(items)
                or any(
                    type(result) is not int or (result < 0 and result not in _SAFE_PUT_REJECTIONS) for result in results
                )
            ):
                raise EmbeddingStoreError("Mooncake I/O completion is unconfirmed; retaining buffers")
        except EmbeddingStoreOperationError:
            raise
        except BaseException as error:
            self._poison(owners)
            raise EmbeddingStoreError("Mooncake I/O completion is unconfirmed; retaining buffers") from error
        return results

    def _load_pool_chunk(self, expected: dict[str, TensorSpec], device: torch.device) -> dict[str, torch.Tensor]:
        """Read objects into leased slab regions and retain all active readers."""
        self._check_healthy()
        if not self._bound:
            raise EmbeddingStoreOperationError("Store staging pool is not bound")
        if device.type != "npu":
            raise EmbeddingStoreOperationError("Ascend embedding target must be on an NPU")
        torch.npu.set_device(device)
        header_size = _MOONCAKE_TENSOR_HEADER.size
        sizes = [header_size + spec.nbytes for spec in expected.values()]
        # Neither model-thread fallback nor async GET may wait for staging
        # work that the model still needs to bind; unavailable hits are computed.
        lease = self._staging_pool.reserve_bytes(sizes, device, wait=False)
        if lease is None:
            return {}
        confirmed = False
        copying = False
        targets: list[torch.Tensor] = []
        sends: list[Future[None]] = []
        copy_event: Any | None = None
        try:
            addresses = [buffer.data_ptr() for buffer in lease.tensors]
            results = self.store.batch_get_into(list(expected), addresses, sizes)
            if not _batch_get_completed_safely(results, sizes):
                raise EmbeddingStoreError("Unsafe batch GET results")
            confirmed = True
            loaded: dict[str, torch.Tensor] = {}
            payloads: dict[str, torch.Tensor] = {}
            hits = [
                (key, spec, buffer)
                for (key, spec), buffer, size, result in zip(
                    expected.items(),
                    lease.tensors,
                    sizes,
                    results,
                    strict=True,
                )
                if result == size
            ]
            metadata_blob = b""
            if hits:
                # Headers live in separate aligned NPU regions. Moving each
                # tiny header to CPU independently introduces one device/host
                # synchronization per object. Gather a complete GET chunk on
                # the private copy stream and synchronize only that stream.
                copying = True
                metadata_blob = self._copy_metadata_to_cpu([buffer for _, _, buffer in hits], header_size, device)
                copying = False
            for index, (key, spec, buffer) in enumerate(hits):
                metadata = metadata_blob[index * header_size : (index + 1) * header_size]
                try:
                    _validate_mooncake_tensor_metadata(key, metadata, spec)
                except EmbeddingStoreOperationError as error:
                    logger.warning("Store GET skipped for %s: %s", key, error)
                    continue
                dtype = getattr(torch, spec.dtype.removeprefix("torch."))
                payload = buffer.narrow(0, header_size, spec.nbytes).view(dtype).view(spec.shape)
                target = torch.empty(spec.shape, dtype=dtype, device=device)
                targets.append(target)
                payloads[key] = payload
                loaded[key] = target
            if payloads and self.on_get_payloads is not None:
                copying = True
                sends.extend(self.on_get_payloads(payloads))
                copying = False
            if loaded:
                copying = True
                copy_event = self._copy_loaded_payloads(loaded, payloads, device)
                copying = False
            if loaded:
                logger.info("Loaded %d encoder output(s) from Mooncake Store", len(loaded))
            return loaded
        except BaseException as error:
            if not confirmed or copying:
                self._poison([lease, *targets])
                raise EmbeddingStoreError("Store GET/copy completion is unconfirmed; retaining staging") from error
            raise
        finally:
            if not self._poisoned:
                if copy_event is None:
                    self._release_get_after_sends(lease, sends)
                else:
                    self._defer_get_release(lease, copy_event, sends, targets, device)

    def _get_copy_stream(self, device: torch.device) -> Any:
        """Lazily create the private stream used by serialized GET copies."""
        copy_stream = self._copy_stream
        if copy_stream is None:
            copy_stream = torch.npu.Stream(device=device)
            self._copy_stream = copy_stream
        return copy_stream

    def _copy_metadata_to_cpu(
        self,
        buffers: list[torch.Tensor],
        header_size: int,
        device: torch.device,
    ) -> bytes:
        """Gather Store headers into one CPU buffer for validation."""
        headers = [buffer.narrow(0, 0, header_size) for buffer in buffers]
        if buffers[0].device.type != "npu":
            return torch.stack(headers).cpu().numpy().tobytes()
        with torch.npu.stream(self._get_copy_stream(device)):
            return torch.stack(headers).cpu().numpy().tobytes()

    def _copy_loaded_payloads(
        self,
        loaded: dict[str, torch.Tensor],
        payloads: dict[str, torch.Tensor],
        device: torch.device,
    ) -> Any | None:
        """Enqueue cache copies without synchronizing the model stream.

        CPU tensors occur only in mock tests and complete synchronously.  On
        NPU, a private stream isolates Store copies from model work.  The model
        stream waits on the recorded event device-side, while a completion
        worker retains the source lease until both the event and direct sends
        finish.
        """
        if not loaded or next(iter(loaded.values())).device.type != "npu":
            for key, target in loaded.items():
                target.copy_(payloads[key], non_blocking=True)
            return None

        copy_stream = self._get_copy_stream(device)
        compute_stream = torch.npu.current_stream(device)
        with torch.npu.stream(copy_stream):
            for key, target in loaded.items():
                target.record_stream(copy_stream)
                target.copy_(payloads[key], non_blocking=True)
            try:
                event = torch.npu.Event()
                event.record(copy_stream)
                compute_stream.wait_event(event)
            except BaseException:
                # If an Event cannot establish the lifetime/order boundary,
                # drain only the private copy stream.  This preserves safety
                # without synchronizing unrelated model-stream work.
                copy_stream.synchronize()
                return None
        return event

    def _defer_get_release(
        self,
        lease: AscendStagedSources,
        copy_event: Any,
        sends: list[Future[None]],
        targets: list[torch.Tensor],
        device: torch.device,
    ) -> None:
        """Retain a GET slab until its copy Event and P2P reads complete."""

        def complete() -> None:
            """Wait for cache copies, then release the slab after P2P readers settle."""
            try:
                torch.npu.set_device(device)
                copy_event.synchronize()
            except BaseException as error:
                # The copy may have reached native code.  Do not infer
                # retirement after an unknown completion state.
                self._poison([lease, *targets])
                logger.error("Store GET completion is unconfirmed; retaining staging: %s", error)
                return
            if not self._poisoned:
                # Do not block the ordered Event worker behind a slow P2P
                # Future. Once the copy is settled, normal callbacks can
                # release this lease as soon as its own sends finish.
                self._release_get_after_sends(lease, sends)

        try:
            self._get_completion_executor.submit(complete)
        except BaseException as error:
            self._poison([lease, *targets])
            raise EmbeddingStoreError("Could not track Store GET completion; retaining staging") from error

    def _release_get_after_sends(self, lease: AscendStagedSources, sends: list[Future[None]]) -> None:
        """Cache copies are settled; any failed P2P reader poisons this GET."""
        self._staging_pool.release_after(lease, sends, retain=partial(self._poison, [lease]))

    def close(self) -> None:
        """Drain copy tracking and preserve native owners of bound EC storage."""
        if self._closed:
            return
        # Worker shutdown drains the P2P executor before closing Store, so all
        # send Futures observed by these completion jobs can reach a terminal
        # state.  Joining here also settles every copy Event before teardown.
        self._get_completion_executor.shutdown(wait=True, cancel_futures=False)
        self._check_healthy()

        if self._bound:
            # Shared and independent registrations both borrow the EC slab.
            # Keep the native Store (including its Host segments) alive until
            # the EC owner exits; closing either side is not a retirement fence.
            _PROCESS_LIFETIME_STORE_OWNERS.append((self.store, self._staging_pool))
            self._closed = True
            return

        try:
            status = self.store.close()
            if status != 0:
                raise EmbeddingStoreError(f"failed to close embedding Mooncake Store: {status}")
        except BaseException:
            self._poison([self._staging_pool])
            raise
        self._closed = True

    def _load_chunk(self, expected: dict[str, TensorSpec], device: torch.device) -> dict[str, torch.Tensor]:
        """Add only the aligned pool bound to upstream read-byte grouping."""
        loaded: dict[str, torch.Tensor] = {}
        chunk: dict[str, TensorSpec] = {}
        allocated_bytes = 0
        for key, spec in expected.items():
            allocated = round_up(_MOONCAKE_TENSOR_HEADER.size + spec.nbytes, self._staging_pool._allocator.alignment)
            if allocated > self._staging_pool.capacity:
                continue
            if chunk and allocated_bytes + allocated > self._staging_pool.capacity:
                loaded.update(self._load_pool_chunk(chunk, device))
                chunk, allocated_bytes = {}, 0
            chunk[key] = spec
            allocated_bytes += allocated
        if chunk:
            loaded.update(self._load_pool_chunk(chunk, device))
        return loaded
