# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
# SPDX-License-Identifier: Apache-2.0
"""Ascend setup for the cross-encoder Mooncake Store client."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from typing import Any

import torch
from vllm.distributed.ec_transfer.ec_connector.mooncake_store_embedding.data import (
    TensorSpec,
)
from vllm.distributed.ec_transfer.ec_connector.mooncake_store_embedding.store_client import (
    _SAFE_PUT_REJECTIONS,
    EmbeddingStoreError,
    EmbeddingStoreOperationError,
    MooncakeEmbeddingStoreClient,
    _encode_mooncake_tensor_metadata,
)
from vllm.distributed.mooncake_store import DEFAULT_TENANT_ID, MooncakeStoreConfig
from vllm.logger import init_logger
from vllm.utils.network_utils import get_ip

from vllm_ascend.distributed.ec_transfer.ec_connector.mooncake.bounce import (
    ASCEND_DIRECT_MEMORY_ALIGNMENT,
    AscendBounceArena,
    _plan_source,
    _RegistrationRangePlan,
)
from vllm_ascend.distributed.ec_transfer.ec_connector.mooncake.transfer import (
    AscendMooncakeTransfer,
)

logger = init_logger(__name__)


def ensure_mooncake_store_imported() -> None:
    """Load Store bindings before the TransferEngine bindings.

    Mooncake's Python extensions share TransferEngine type registration. The
    Store extension must be loaded before ``mooncake.engine`` creates the
    process-wide engine that is later passed to ``MooncakeDistributedStore``.
    """
    try:
        import mooncake.store  # noqa: F401  # type: ignore
    except ImportError as error:
        raise ImportError("Install mooncake with Store support to enable cross_encoder_cache.") from error


class AscendMooncakeEmbeddingStoreClient(MooncakeEmbeddingStoreClient):
    """Use the Ascend producer arena and the worker-owned TransferEngine."""

    def __init__(
        self,
        store: Any,
        bounce_arena: AscendBounceArena,
        registration_transfer: AscendMooncakeTransfer,
        replicate_config: Any | None = None,
        *,
        read_buffer_bytes: int = 128 * 1024**2,
    ) -> None:
        super().__init__(
            store,
            replicate_config=replicate_config,
            read_buffer_bytes=read_buffer_bytes,
        )
        self._bounce_arena = bounce_arena
        self._registration_transfer = registration_transfer
        self._put_header_tensor: torch.Tensor | None = None

    def _ensure_bounce_available(self) -> torch.Tensor:
        bounce = self._bounce_arena.tensor
        if bounce is None:
            raise EmbeddingStoreOperationError("Ascend Mooncake bounce arena is unavailable")

        address = bounce.data_ptr()
        if address % ASCEND_DIRECT_MEMORY_ALIGNMENT != 0:
            raise RuntimeError("Ascend Mooncake bounce arena is not 2 MiB aligned")

        return bounce

    @contextmanager
    def _registered_io(
        self,
        buffers: list[tuple[Any, int, int]],
    ) -> Iterator[Callable[[int], None]]:
        """Register direct ranges through the worker's Mooncake transfer."""
        self._check_healthy()
        addresses: list[int] = []
        submitted = False
        confirmed = False

        def confirm(result: int) -> None:
            nonlocal confirmed
            if type(result) is not int or (result < 0 and result not in _SAFE_PUT_REJECTIONS):
                raise EmbeddingStoreError("Mooncake I/O completion is unconfirmed; retaining buffers")
            confirmed = True

        ranges = tuple(_RegistrationRangePlan(addr, size, (owner,)) for owner, addr, size in buffers)
        owners = [owner for owner, _, _ in buffers]
        try:
            try:
                addresses = self._registration_transfer.acquire_registration_ranges(ranges)
            except RuntimeError as error:
                raise EmbeddingStoreOperationError(f"Failed to register embedding buffer: {error}") from error
            submitted = True
            yield confirm
        except BaseException as error:
            if submitted and not confirmed:
                self._poison(owners)
                raise EmbeddingStoreError("Mooncake I/O completion is unconfirmed; retaining buffers") from error
            raise
        finally:
            if addresses and (not submitted or confirmed):
                try:
                    released = self._registration_transfer.release_registration_ranges(addresses)
                    if not released:
                        raise EmbeddingStoreError("could not confirm embedding buffer unregistration")
                except BaseException:
                    self._poison(owners)
                    raise

    def put_tensor(self, pool_key: str, tensor: torch.Tensor) -> None:
        self._check_healthy()
        if tensor.device.type != "npu":
            raise EmbeddingStoreOperationError("embedding tensor must be on an NPU")
        if not tensor.is_contiguous():
            raise EmbeddingStoreOperationError("embedding tensor must be contiguous")

        # Store publication runs on a background CPU thread. Wait for NPU
        # producer work before Mooncake reads a direct range or a prefix copied
        # through the private bounce-copy stream.
        torch.npu.set_device(tensor.device)
        torch.npu.synchronize(tensor.device)
        metadata = _encode_mooncake_tensor_metadata(tensor)
        # The single publisher reuses this header only after PUT completion.
        # Ascend HIXL must see device-registerable host memory here: adding an
        # ordinary ctypes allocation to the shared engine's registrations can
        # break the first lazy HCCS channel preparation. Pinned host memory is
        # system DRAM, not device memory.
        if self._put_header_tensor is None:
            header = torch.empty(
                len(metadata),
                dtype=torch.uint8,
                device="cpu",
                pin_memory=True,
            )
            ret = self.store.register_buffer(header.data_ptr(), len(metadata))
            if ret != 0:
                raise EmbeddingStoreOperationError(f"Failed to register embedding header: {ret}")
            self._put_header_tensor = header

        header = self._put_header_tensor
        assert header is not None
        if header.numel() != len(metadata):
            raise EmbeddingStoreOperationError("Embedding header size changed")
        header.copy_(torch.frombuffer(bytearray(metadata), dtype=torch.uint8))
        header_ptr = header.data_ptr()
        source = _plan_source(tensor)
        lease = None
        retain_lease = False

        try:
            addresses = [header_ptr]
            sizes = [len(metadata)]

            if source.prefix_nbytes > 0:
                self._ensure_bounce_available()
                lease = self._bounce_arena.acquire(source.prefix_nbytes)
                assert lease is not None
                bounce_address = self._bounce_arena.copy(
                    lease,
                    [(tensor, 0, source.prefix_nbytes)],
                )
                addresses.append(bounce_address)
                sizes.append(source.prefix_nbytes)

            direct_buffers = []
            if source.registration_address is not None:
                direct_buffers.append(
                    (
                        tensor,
                        source.registration_address,
                        source.registration_nbytes,
                    )
                )
            if source.direct_address is not None:
                addresses.append(source.direct_address)
                sizes.append(source.direct_nbytes)

            with self._registered_io(direct_buffers) as confirm:
                [result] = self.store.batch_put_from_multi_buffers(
                    [pool_key],
                    [addresses],
                    [sizes],
                    self.replicate_config,
                )
                confirm(result)
                if result < 0:
                    raise EmbeddingStoreOperationError(f"failed to put embedding tensor for {pool_key}")
            logger.info("Stored encoder output in Mooncake Store")
        except BaseException:
            if lease is not None and self._poisoned:
                self._poison([self._bounce_arena, tensor, lease])
                retain_lease = True
            raise
        finally:
            if lease is not None and not retain_lease:
                self._bounce_arena.release(lease)

    def close(self) -> None:
        self._check_healthy()

        header = self._put_header_tensor
        if header is not None:
            try:
                self._unregister_buffer(header.data_ptr())
            except BaseException:
                # Keep the allocation alive after a failed unregister so its
                # address cannot be reused while the engine still owns it.
                self._poison([header])
                raise
            self._put_header_tensor = None

        # The base header remains unused, so the base close only tears down the
        # Store and its read buffer. The shared TransferEngine remains worker-owned.
        super().close()

    def load_tensors(
        self,
        expected: Mapping[str, TensorSpec],
        device: torch.device | str,
    ) -> dict[str, torch.Tensor]:
        loaded = super().load_tensors(expected, device)
        if loaded:
            logger.info("Loaded %d encoder output(s) from Mooncake Store", len(loaded))
        return loaded


def _setup_store_with_engine(
    store: Any,
    config: MooncakeStoreConfig,
    local_hostname: str,
    engine: Any,
) -> None:
    setup_kwargs: dict[str, Any] = {"engine": engine}
    if config.tenant_id != DEFAULT_TENANT_ID:
        setup_kwargs["tenant_id"] = config.tenant_id
    ret = store.setup(
        local_hostname,
        config.metadata_server,
        config.global_segment_size,
        config.local_buffer_size,
        config.protocol,
        config.device_name,
        config.master_server_address,
        **setup_kwargs,
    )
    if ret != 0:
        raise RuntimeError("Initialize MooncakeDistributedStore failed.")


def create_ascend_mooncake_embedding_store_client(
    bounce_arena: AscendBounceArena,
    registration_transfer: AscendMooncakeTransfer,
    read_buffer_bytes: int = 128 * 1024**2,  # 128 MiB
) -> AscendMooncakeEmbeddingStoreClient:
    """Create an embedding Store client using the worker's TransferEngine."""
    config = MooncakeStoreConfig.load_from_config()
    if config.protocol != "ascend":
        raise ValueError("Ascend cross_encoder_cache requires Mooncake Store protocol='ascend'")
    if config.enable_offload:
        raise ValueError("cross_encoder_cache supports a RAM Store; disable enable_offload")

    try:
        from mooncake.store import (  # type: ignore
            MooncakeDistributedStore,
            ObjectDataType,
            ReplicateConfig,
        )
    except ImportError as error:
        raise ImportError("Install mooncake with Store support to enable cross_encoder_cache.") from error

    from vllm.distributed.kv_transfer.kv_connector.v1.mooncake import rdma_utils

    # TransferEngine is the Python-facing adapter. Store setup expects the
    # shared_ptr-backed native engine exposed by the adapter.
    engine = registration_transfer._ensure_engine().get_engine()
    store = MooncakeDistributedStore()
    local_hostname = rdma_utils.get_requester_local_hostname(get_ip())
    _setup_store_with_engine(store, config, local_hostname, engine)

    replicate_config = ReplicateConfig()
    replicate_config.data_type = ObjectDataType.TENSOR
    logger.info(
        "Initialized embedding Mooncake store with external TransferEngine "
        "mode=%s global_segment_size=%d local_buffer_size=%d",
        config.mode,
        config.global_segment_size,
        config.local_buffer_size,
    )
    return AscendMooncakeEmbeddingStoreClient(
        store,
        bounce_arena,
        registration_transfer,
        replicate_config=replicate_config,
        read_buffer_bytes=read_buffer_bytes,
    )
