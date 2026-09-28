# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
# SPDX-License-Identifier: Apache-2.0
"""Ascend setup for the cross-encoder Mooncake Store client."""

from __future__ import annotations

import ctypes
from typing import Any

import torch
from vllm.distributed.ec_transfer.ec_connector.mooncake_store_embedding.store_client import (
    EmbeddingStoreOperationError,
    MooncakeEmbeddingStoreClient,
    _encode_mooncake_tensor_metadata,
    create_mooncake_embedding_store_client,
)
from vllm.distributed.mooncake_store import MooncakeStoreConfig

from vllm_ascend.distributed.ec_transfer.ec_connector.mooncake.bounce import (
    ASCEND_DIRECT_MEMORY_ALIGNMENT,
    AscendBounceArena,
    _plan_source,
)


class AscendMooncakeEmbeddingStoreClient(MooncakeEmbeddingStoreClient):
    """Use the Ascend producer arena with a Store-owned TransferEngine."""

    def __init__(
        self,
        store: Any,
        bounce_arena: AscendBounceArena,
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
        self._store_bounce_tensor: torch.Tensor | None = None

    def _ensure_bounce_registered(self) -> torch.Tensor:
        bounce = self._bounce_arena.tensor
        if bounce is None:
            raise EmbeddingStoreOperationError("Ascend Mooncake bounce arena is unavailable")

        address = bounce.data_ptr()
        if address % ASCEND_DIRECT_MEMORY_ALIGNMENT != 0:
            raise RuntimeError("Ascend Mooncake bounce arena is not 2 MiB aligned")

        with self._lifetime_lock:
            registered = self._store_bounce_tensor
            if registered is not None:
                return registered

            ret = self.store.register_buffer(address, bounce.nbytes)
            if ret != 0:
                raise EmbeddingStoreOperationError(f"Failed to register embedding bounce arena: {ret}")
            self._store_bounce_tensor = bounce

        return bounce

    def put_tensor(self, pool_key: str, tensor: torch.Tensor) -> None:
        self._check_healthy()
        if tensor.device.type != "npu":
            raise EmbeddingStoreOperationError("embedding tensor must be on an NPU")
        if not tensor.is_contiguous():
            raise EmbeddingStoreOperationError("embedding tensor must be contiguous")

        metadata = _encode_mooncake_tensor_metadata(tensor)
        # The single publisher reuses this header only after PUT completion.
        if self._put_header is None:
            header = ctypes.create_string_buffer(len(metadata))
            ret = self.store.register_buffer(ctypes.addressof(header), len(metadata))
            if ret != 0:
                raise EmbeddingStoreOperationError(f"Failed to register embedding header: {ret}")
            self._put_header = header

        header_ptr = ctypes.addressof(self._put_header)
        ctypes.memmove(header_ptr, metadata, len(metadata))
        source = _plan_source(tensor)
        lease = None
        retain_lease = False

        try:
            addresses = [header_ptr]
            sizes = [len(metadata)]

            if source.prefix_nbytes > 0:
                self._ensure_bounce_registered()
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
        except BaseException:
            if lease is not None and self._poisoned:
                self._poison([self._bounce_arena, tensor, lease])
                retain_lease = True
            raise
        finally:
            if lease is not None and not retain_lease:
                self._bounce_arena.release(lease)

    def close(self) -> None:
        """Unregister Engine B's bounce arena view before closing the Store."""
        self._check_healthy()
        bounce = self._store_bounce_tensor
        if bounce is not None:
            try:
                self._unregister_buffer(bounce.data_ptr())
            except BaseException:
                self._poison([self._bounce_arena, bounce])
                raise
            self._store_bounce_tensor = None

        super().close()


def create_ascend_mooncake_embedding_store_client(
    bounce_arena: AscendBounceArena,
    read_buffer_bytes: int = 128 * 1024**2,  # 128 MiB
) -> AscendMooncakeEmbeddingStoreClient:
    """Create an embedding Store client with a Store-owned TransferEngine."""
    config = MooncakeStoreConfig.load_from_config()
    if config.protocol != "ascend":
        raise ValueError("Ascend cross_encoder_cache requires Mooncake Store protocol='ascend'")

    # The upstream factory does not pass engine=, so Store creates Engine B
    # instead of reusing the process-wide global_te (Engine A) used by P2P.
    client = create_mooncake_embedding_store_client(
        read_buffer_bytes=read_buffer_bytes,
    )
    return AscendMooncakeEmbeddingStoreClient(
        client.store,
        bounce_arena,
        replicate_config=client.replicate_config,
        read_buffer_bytes=read_buffer_bytes,
    )
