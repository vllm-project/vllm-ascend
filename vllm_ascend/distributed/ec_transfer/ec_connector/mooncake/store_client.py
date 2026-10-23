# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""NPU tensor I/O using upstream Store lookup, read grouping and poison bookkeeping."""

from __future__ import annotations

import threading
from collections.abc import Callable
from concurrent.futures import Future
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
from vllm_ascend.distributed.ec_transfer.ec_connector.mooncake.metrics import ECMetrics

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
        super().__init__(
            store,
            replicate_config=replicate_config,
            read_buffer_bytes=read_buffer_bytes,
        )
        if use_fabric_mem and use_store_independent_te:
            raise ValueError("FabricMem and independent RoCE are different Store topologies")
        self._independent = use_store_independent_te
        self._bound = False
        self._staging_pool = staging_pool
        self._closed = False
        self.metrics = ECMetrics("store_get")
        # Called synchronously after GET validation, before submitting cache copy.
        self.on_get_payload: Callable[[str, torch.Tensor], Future[None] | None] | None = None

    def _check_healthy(self) -> None:
        super()._check_healthy()
        if self._closed:
            raise EmbeddingStoreError("Ascend Mooncake Store client is closed")

    def _poison(self, owners: list[Any]) -> None:
        super()._poison(owners)
        self._staging_pool.poison()

    def bind_staging_buffer(self, tensor: torch.Tensor) -> None:
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

    def _validate_buffers(self, addresses: list[int], sizes: list[int]) -> None:
        pool = self._staging_pool.tensor
        if not self._bound or pool is None:
            raise EmbeddingStoreOperationError("Store staging pool is not bound")
        start, end = pool.data_ptr(), pool.data_ptr() + pool.nbytes
        if len(addresses) != len(sizes) or any(
            size <= 0 or address < start or address + size > end for address, size in zip(addresses, sizes, strict=True)
        ):
            raise EmbeddingStoreOperationError("Store I/O must use the fixed staging pool")

    def put_tensor(self, pool_key: str, tensor: torch.Tensor) -> None:
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
        self._check_healthy()
        self._validate_buffers(addresses, sizes)
        try:
            [result] = self.store.batch_put_from_multi_buffers(
                [pool_key],
                [addresses],
                [sizes],
                self.replicate_config,
            )
            if type(result) is not int or (result < 0 and result not in _SAFE_PUT_REJECTIONS):
                raise EmbeddingStoreError("Mooncake I/O completion is unconfirmed; retaining buffers")
        except EmbeddingStoreOperationError:
            raise
        except BaseException as error:
            self._poison(owners)
            raise EmbeddingStoreError("Mooncake I/O completion is unconfirmed; retaining buffers") from error
        if result < 0:
            raise EmbeddingStoreOperationError(f"failed to put embedding tensor for {pool_key}")
        logger.info("Stored encoder output in Mooncake Store")

    def _load_pool_chunk(self, expected: dict[str, TensorSpec], device: torch.device) -> dict[str, torch.Tensor]:
        self._check_healthy()
        if device.type != "npu":
            raise EmbeddingStoreOperationError("Ascend embedding target must be on an NPU")
        torch.npu.set_device(device)
        header_size = _MOONCAKE_TENSOR_HEADER.size
        sizes = [header_size + spec.nbytes for spec in expected.values()]
        # GET runs on the model thread. Never wait here for work that the same
        # thread may still need to bind/dispatch; unavailable hits are computed.
        self.metrics.add(
            get_candidate_objects=len(expected), get_candidate_payload_bytes=sum(s.nbytes for s in expected.values())
        )
        lease = self._staging_pool.reserve_bytes(sizes, device, wait=False)
        if lease is None:
            self.metrics.add(
                get_pool_unavailable_objects=len(expected),
                get_pool_unavailable_payload_bytes=sum(s.nbytes for s in expected.values()),
            )
            self.metrics.emit()
            return {}
        confirmed = False
        copying = False
        targets: list[torch.Tensor] = []
        sends: list[Future[None]] = []
        try:
            addresses = [buffer.data_ptr() for buffer in lease.tensors]
            self._validate_buffers(addresses, sizes)
            results = self.store.batch_get_into(list(expected), addresses, sizes)
            if not _batch_get_completed_safely(results, sizes):
                raise EmbeddingStoreError("Unsafe batch GET results")
            confirmed = True
            loaded: dict[str, torch.Tensor] = {}
            for (key, spec), buffer, size, result in zip(
                expected.items(),
                lease.tensors,
                sizes,
                results,
                strict=True,
            ):
                if result != size:
                    continue
                copying = True
                metadata = bytes(buffer.narrow(0, 0, header_size).cpu().tolist())
                copying = False
                try:
                    _validate_mooncake_tensor_metadata(key, metadata, spec)
                except EmbeddingStoreOperationError as error:
                    logger.warning("Store GET skipped for %s: %s", key, error)
                    continue
                dtype = getattr(torch, spec.dtype.removeprefix("torch."))
                payload = buffer.narrow(0, header_size, spec.nbytes).view(dtype).view(spec.shape)
                target = torch.empty(spec.shape, dtype=dtype, device=device)
                targets.append(target)
                if self.on_get_payload is not None:
                    copying = True
                    future = self.on_get_payload(key, payload)
                    if future is not None:
                        sends.append(future)
                copying = True
                target.copy_(payload, non_blocking=True)
                torch.npu.current_stream(device).synchronize()
                copying = False
                loaded[key] = target
                self.metrics.add(get_loaded_objects=1, get_loaded_payload_bytes=spec.nbytes)
            if loaded:
                logger.info("Loaded %d encoder output(s) from Mooncake Store", len(loaded))
            self.metrics.emit()
            return loaded
        except BaseException as error:
            if not confirmed or copying:
                self._poison([lease, *targets])
                raise EmbeddingStoreError("Store GET/copy completion is unconfirmed; retaining staging") from error
            raise
        finally:
            if not self._poisoned:
                self._release_get_after_sends(lease, sends)

    def _release_get_after_sends(self, lease: AscendStagedSources, sends: list[Future[None]]) -> None:
        """Cache copies are settled; P2P tasks hold the remaining read leases."""
        if not sends:
            self._staging_pool.release(lease)
            return
        lock = threading.Lock()
        remaining = len(sends)

        def completed(future: Future[None]) -> None:
            nonlocal remaining
            try:
                error = future.exception()
            except BaseException as exc:
                error = exc
            if error is not None:
                # A failed send may have submitted native work. Keep the whole
                # GET chunk until process exit rather than infer retirement.
                self._poison([lease])
            with lock:
                remaining -= 1
                last = remaining == 0
            if last and not self._poisoned:
                self._staging_pool.release(lease)

        for future in sends:
            future.add_done_callback(completed)

    def close(self) -> None:
        if self._closed:
            return
        self.metrics.emit(force=True)
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
