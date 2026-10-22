# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
# SPDX-License-Identifier: Apache-2.0
"""Ascend tensor I/O for the cross-encoder Mooncake Store."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from vllm.distributed.ec_transfer.ec_connector.mooncake_store_embedding.data import TensorSpec
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

from vllm_ascend.distributed.ec_transfer.ec_connector.mooncake.memory import AscendProducerMemoryPool
from vllm_ascend.distributed.ec_transfer.ec_connector.mooncake.store_memory import (
    AscendStoreMemory,
    AscendStoreTopology,
)
from vllm_ascend.distributed.ec_transfer.ec_connector.mooncake.transfer import AscendMooncakeTransfer

# Retain the native Store and its borrowed EC slab together until process exit.
# Shared topology borrows one registration; independent topology adds a Store
# registration. Retention is a lifetime policy, not transport validation.
_PROCESS_LIFETIME_STORE_OWNERS: list[tuple[Any, Any]] = []


class AscendMooncakeEmbeddingStoreClient:
    """Ascend data plane with upstream object/lookup and poison semantics.

    The upstream client is composed only for control and lifetime bookkeeping.
    Its tensor registration, CPU buffers and close implementation are never used.
    """

    def __init__(
        self,
        store: Any,
        staging_pool: AscendProducerMemoryPool,
        registration_transfer: AscendMooncakeTransfer,
        replicate_config: Any | None = None,
        *,
        read_buffer_bytes: int = 128 * 1024**2,
        use_fabric_mem: bool = False,
        use_store_independent_te: bool = False,
    ) -> None:
        self._control = MooncakeEmbeddingStoreClient(
            store,
            replicate_config=replicate_config,
            read_buffer_bytes=read_buffer_bytes,
        )
        if use_fabric_mem and use_store_independent_te:
            raise ValueError("FabricMem and independent RoCE are different Store topologies")
        self.store = store
        self.replicate_config = replicate_config
        self._read_buffer_bytes = read_buffer_bytes
        topology = (
            AscendStoreTopology.FABRIC_MEM
            if use_fabric_mem
            else AscendStoreTopology.INDEPENDENT
            if use_store_independent_te
            else AscendStoreTopology.SHARED
        )
        self._memory = AscendStoreMemory(store, registration_transfer, topology)
        self._staging_pool = staging_pool
        self._use_fabric_mem = use_fabric_mem
        self._closed = False

    @property
    def uses_store_independent_te(self) -> bool:
        return self._memory.topology is AscendStoreTopology.INDEPENDENT

    @property
    def _poisoned(self) -> bool:
        return self._control._poisoned

    @property
    def _unsafe_owners(self) -> list[Any]:
        return self._control._unsafe_owners

    def _check_healthy(self) -> None:
        self._control._check_healthy()
        if self._closed:
            raise EmbeddingStoreError("Ascend Mooncake Store client is closed")

    def _poison(self, owners: list[Any]) -> None:
        # Upstream retains poisoned controls until process exit. Include this
        # client so the native Store, registration policy and pool survive too.
        self._control._poison([self, *owners])
        self._staging_pool.poison()

    def exists(self, pool_key: str) -> bool:
        return self.batch_exists([pool_key])[0]

    def batch_exists(self, keys: list[str]) -> list[bool]:
        self._check_healthy()
        return self._control.batch_exists(keys)

    def bind_staging_buffer(self, tensor: torch.Tensor) -> None:
        self._check_healthy()
        self._memory.bind_staging(tensor)

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
        self._memory.validate_buffers(addresses, sizes)
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

    def retain_fallback_owners(self, owners: list[Any]) -> None:
        """Poison owners whose submitted transfer completion is uncertain."""
        self._poison(owners)

    def _load_chunk(self, expected: dict[str, TensorSpec], device: torch.device) -> dict[str, torch.Tensor]:
        self._check_healthy()
        if device.type != "npu":
            raise EmbeddingStoreOperationError("Ascend embedding target must be on an NPU")
        torch.npu.set_device(device)
        header_size = _MOONCAKE_TENSOR_HEADER.size
        sizes = [header_size + spec.nbytes for spec in expected.values()]
        # GET runs on the model thread. Never wait here for work that the same
        # thread may still need to bind/dispatch; unavailable hits are computed.
        lease = self._staging_pool.reserve_bytes(sizes, device, wait=False)
        if lease is None:
            return {}
        confirmed = False
        copying = False
        targets: list[torch.Tensor] = []
        try:
            addresses = [buffer.data_ptr() for buffer in lease.tensors]
            self._memory.validate_buffers(addresses, sizes)
            results = self.store.batch_get_into(list(expected), addresses, sizes)
            if not _batch_get_completed_safely(results, sizes):
                raise EmbeddingStoreError("Unsafe batch GET results")
            confirmed = True
            loaded: dict[str, torch.Tensor] = {}
            for (key, spec), buffer, region, size, result in zip(
                expected.items(),
                lease.tensors,
                list(lease.regions),
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
                copying = True
                target.copy_(payload, non_blocking=True)
                torch.npu.current_stream(device).synchronize()
                copying = False
                # Cache owns independent HBM. Retain the completed staging copy
                # opportunistically so the immediate P2P can take its lease.
                self._staging_pool.cache_loaded(target, payload, buffer.narrow(0, 0, header_size), region)
                lease.regions.remove(region)
                loaded[key] = target
            return loaded
        except BaseException as error:
            if not confirmed or copying:
                self._poison([lease, *targets])
                raise EmbeddingStoreError("Store GET/copy completion is unconfirmed; retaining staging") from error
            raise
        finally:
            if not self._poisoned:
                self._staging_pool.release(lease)

    def close(self) -> None:
        if self._closed:
            return
        self._check_healthy()

        if self._memory.staging_owner is not None:
            # Shared and independent registrations both borrow the EC slab.
            # Keep the native Store (including its Host segments) alive until
            # the EC owner exits; closing either side is not a retirement fence.
            _PROCESS_LIFETIME_STORE_OWNERS.append((self.store, self._memory.staging_owner))
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

    def load_tensors(
        self,
        expected: Mapping[str, TensorSpec],
        device: torch.device | str,
    ) -> dict[str, torch.Tensor]:
        self._check_healthy()
        device = torch.device(device)
        loaded: dict[str, torch.Tensor] = {}
        chunk: dict[str, TensorSpec] = {}
        chunk_bytes = 0
        allocated_bytes = 0
        alignment = self._staging_pool._allocator.alignment
        for key, spec in expected.items():
            size = _MOONCAKE_TENSOR_HEADER.size + spec.nbytes
            allocated = round_up(size, alignment)
            if size > self._read_buffer_bytes or allocated > self._staging_pool.capacity:
                continue
            if chunk and (
                chunk_bytes + size > self._read_buffer_bytes
                or allocated_bytes + allocated > self._staging_pool.capacity
            ):
                loaded.update(self._load_chunk(chunk, device))
                chunk, chunk_bytes, allocated_bytes = {}, 0, 0
            chunk[key] = spec
            chunk_bytes += size
            allocated_bytes += allocated
        if chunk:
            loaded.update(self._load_chunk(chunk, device))
        return loaded
