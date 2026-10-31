# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Store engine setup, deferred publication and asynchronous ownership."""

from __future__ import annotations

import json
import os
import traceback
from concurrent.futures import Future
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any

import torch
from vllm.distributed.ec_transfer.ec_connector.mooncake_store_embedding.backend import (
    MooncakeEmbeddingStoreBackend,
    _StoreSave,
)
from vllm.distributed.ec_transfer.ec_connector.mooncake_store_embedding.data import (
    make_embedding_key,
)
from vllm.distributed.ec_transfer.ec_connector.mooncake_store_embedding.store_client import (
    EmbeddingStoreError,
    EmbeddingStoreOperationError,
)
from vllm.distributed.mooncake_store import DEFAULT_TENANT_ID, MooncakeStoreConfig
from vllm.logger import logger
from vllm.utils.network_utils import get_ip

from vllm_ascend.distributed.ec_transfer.ec_connector.mooncake.memory import AscendProducerMemoryPool
from vllm_ascend.distributed.ec_transfer.ec_connector.mooncake.store_client import AscendMooncakeEmbeddingStoreClient
from vllm_ascend.distributed.ec_transfer.ec_connector.mooncake.transfer import AscendMooncakeTransfer


def use_store_independent_te() -> bool:
    """Select the KV-Pool-compatible independent Store topology."""
    if os.getenv("ASCEND_ENABLE_USE_FABRIC_MEM", "0") == "1":
        return False
    raw = os.getenv("ASCEND_GLOBAL_RESOURCE_CONFIG")
    if not raw:
        return False
    try:
        config = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError("ASCEND_GLOBAL_RESOURCE_CONFIG must be a JSON object") from error
    if not isinstance(config, dict):
        raise ValueError("ASCEND_GLOBAL_RESOURCE_CONFIG must be a JSON object")
    return True


def ensure_mooncake_store_imported() -> None:
    """Load Store bindings before the TransferEngine bindings.

    Mooncake's Python extensions share TransferEngine type registration. The
    Store extension must be loaded before ``mooncake.engine`` creates the
    process-wide engine that is later passed to ``MooncakeDistributedStore``.
    """
    try:
        import mooncake.store  # type: ignore[import-not-found]  # noqa: F401
    except ImportError as error:
        raise ImportError("Install mooncake with Store support to enable cross_encoder_cache.") from error


def _setup_store_engine(
    store: Any,
    config: MooncakeStoreConfig,
    local_hostname: str,
    topology: str,
    transfer_engine: Any | None = None,
) -> None:
    """Attach an EC engine or let Store create an engine with no local buffer."""
    setup_kwargs: dict[str, Any] = {}
    local_buffer_size = 0
    if transfer_engine is not None:
        setup_kwargs["engine"] = transfer_engine.get_engine()
        local_hostname = f"{local_hostname}:{transfer_engine.get_rpc_port()}"
        local_buffer_size = config.local_buffer_size
    if config.tenant_id != DEFAULT_TENANT_ID:
        setup_kwargs["tenant_id"] = config.tenant_id
    ret = store.setup(
        local_hostname,
        config.metadata_server,
        config.global_segment_size,
        local_buffer_size,
        config.protocol,
        config.device_name,
        config.master_server_address,
        **setup_kwargs,
    )
    if ret != 0:
        if transfer_engine is not None:
            raise RuntimeError("Initialize MooncakeDistributedStore failed.")
        raise RuntimeError(f"Initialize {topology} MooncakeDistributedStore failed: {ret}")


def create_ascend_mooncake_embedding_store_client(
    staging_pool: AscendProducerMemoryPool,
    registration_transfer: AscendMooncakeTransfer,
    read_buffer_bytes: int = 128 * 1024**2,  # 128 MiB
) -> AscendMooncakeEmbeddingStoreClient:
    """Create an embedding Store using the selected Ascend engine topology."""
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

    store = MooncakeDistributedStore()
    local_hostname = rdma_utils.get_requester_local_hostname(get_ip())
    use_fabric_mem = os.getenv("ASCEND_ENABLE_USE_FABRIC_MEM", "0") == "1"
    use_store_independent = use_store_independent_te()
    if not use_fabric_mem and os.getenv("ASCEND_BUFFER_POOL"):
        raise ValueError("Ascend cross_encoder_cache does not support ASCEND_BUFFER_POOL")
    if use_fabric_mem:
        # Mooncake only tags a TransferEngine as Store-owned and reads the
        # FabricMem setting while Store initializes that engine. Passing the
        # already initialized EC P2P engine would keep store=false and
        # use_fabric_mem=false. As in Ascend's KV Store backend, the effective
        # local buffer is zero in this mode; Store owns only its FabricMem
        # global segment. The EC staging slab remains registered solely with
        # the worker-owned P2P engine and is not registered again here.
        _setup_store_engine(store, config, local_hostname, "FabricMem")
        engine_topology = "Store-owned FabricMem TransferEngine"
        effective_local_buffer_size = 0
    elif use_store_independent:
        _setup_store_engine(store, config, local_hostname, "independent")
        engine_topology = "Store-owned independently configured TransferEngine"
        effective_local_buffer_size = 0
    else:
        transfer_engine = registration_transfer._ensure_engine()
        _setup_store_engine(store, config, local_hostname, "shared", transfer_engine)
        engine_topology = "shared EC/Store RoCE TransferEngine"
        effective_local_buffer_size = config.local_buffer_size

    replicate_config = ReplicateConfig()
    replicate_config.data_type = ObjectDataType.TENSOR
    logger.info(
        "Initialized embedding Mooncake store with %s mode=%s global_segment_size=%d local_buffer_size=%d",
        engine_topology,
        config.mode,
        config.global_segment_size,
        effective_local_buffer_size,
    )
    return AscendMooncakeEmbeddingStoreClient(
        store,
        staging_pool,
        replicate_config=replicate_config,
        read_buffer_bytes=read_buffer_bytes,
        use_fabric_mem=use_fabric_mem,
        use_store_independent_te=use_store_independent,
    )


def _record_store_ready_event(tensor: torch.Tensor) -> torch.Event | None:
    """Record when an NPU output is ready for background publication."""
    if tensor.device.type != "npu":
        return None
    event = torch.npu.Event()
    event.record(torch.npu.current_stream(tensor.device))
    return event


class _SaveState(Enum):
    """Track deferred publication from admission through submission."""

    DEFERRED = auto()
    CHECKING = auto()
    PREPARED = auto()
    SUBMITTED = auto()


@dataclass
class _DeferredSave(_StoreSave):
    """Retain an admitted output and its publication state."""

    identifier: str
    pool_key: str
    state: _SaveState = _SaveState.DEFERRED
    future: Future[Any] | None = None


@dataclass(frozen=True)
class PreparedStoreSave:
    """Identify an admitted output selected for staged publication."""

    identifier: str
    pool_key: str
    tensor: torch.Tensor
    save: _DeferredSave


@dataclass(frozen=True)
class PreparedStorePut:
    """Pair a publication ticket with staged addresses and their owners."""

    ticket: PreparedStoreSave
    addresses: list[int]
    sizes: list[int]
    owners: list[Any]


class AscendDeferredStoreBackend(MooncakeEmbeddingStoreBackend):
    """Reserve publisher budget now and choose the PUT memory path later."""

    store_client: AscendMooncakeEmbeddingStoreClient
    _closed: bool
    _fatal_error: BaseException | None

    def save_output(self, identifier: str, tensor: torch.Tensor) -> None:
        """Admit a selected encoder output for deferred Store publication."""
        if identifier in self._step_candidates:
            self.reap()
            self._admit(identifier, tensor)

    def set_step_candidates(self, identifiers: set[str]) -> None:
        """Select immutable outputs publishable by the current model step."""
        self.reap()
        self._step_candidates = identifiers

    def _admit(self, identifier: str, tensor: torch.Tensor) -> bool:
        """Reserve item and shared-storage budgets for a contiguous output."""
        if not tensor.is_contiguous():
            return False
        storage = tensor.untyped_storage()
        storage_ptr = storage.data_ptr()
        with self._pending_lock:
            if self._fatal_error is not None:
                raise EmbeddingStoreError("Store publisher has failed") from self._fatal_error
            if self._closed or identifier in self._pending:
                return False
            _, ref_count = self._pending_storages.get(storage_ptr, (storage, 0))
            budget_bytes = 0 if ref_count else storage.nbytes()
            if (
                len(self._pending) >= self.max_pending_items
                or self._pending_bytes + budget_bytes > self.max_pending_bytes
            ):
                return False
            save = _DeferredSave(
                identifier=identifier,
                pool_key=make_embedding_key(self.namespace, identifier),
                tensor=tensor,
                ready_event=_record_store_ready_event(tensor),
                storage_ptr=storage_ptr,
            )
            self._pending[identifier] = (None, save)  # type: ignore[assignment]
            self._pending_storages[storage_ptr] = (storage, ref_count + 1)
            self._pending_bytes += budget_bytes
        return True

    def prepare_saves(self, identifiers: list[str]) -> dict[str, PreparedStoreSave]:
        """Run one immutable-key preflight for an admitted encoder batch."""
        selected: list[tuple[str, _DeferredSave]] = []
        with self._pending_lock:
            if self._fatal_error is not None:
                raise EmbeddingStoreError("Store publisher has failed") from self._fatal_error
            for identifier in dict.fromkeys(identifiers):
                item = self._pending.get(identifier)
                if item is None:
                    continue
                save = item[1]
                assert isinstance(save, _DeferredSave)
                if save.state is _SaveState.DEFERRED:
                    selected.append((identifier, save))
            if not selected:
                return {}
            future = self._executor.submit(
                self.store_client.batch_exists,
                [save.pool_key for _, save in selected],
            )
            for _, save in selected:
                save.state = _SaveState.CHECKING
                save.future = future

        try:
            exists = future.result()
        except (EmbeddingStoreOperationError, OSError) as error:
            logger.warning(
                "Store PUT preflight failed for identifiers=%s: %s",
                [identifier for identifier, _ in selected],
                error,
            )
            with self._pending_lock:
                for identifier, save in selected:
                    self._finalize_locked(identifier, save)
            return {}
        except BaseException as error:
            traceback.clear_frames(error.__traceback__)
            with self._pending_lock:
                if self._fatal_error is None:
                    self._fatal_error = error
            return {}

        if not isinstance(exists, list) or len(exists) != len(selected):
            lookup_error = RuntimeError("Mooncake Store returned an unexpected number of lookup results")
            with self._pending_lock:
                if self._fatal_error is None:
                    self._fatal_error = lookup_error
            return {}

        prepared = {}
        with self._pending_lock:
            for (identifier, save), already_exists in zip(selected, exists, strict=True):
                if already_exists:
                    self._finalize_locked(identifier, save)
                    continue
                tensor = save.tensor
                assert tensor is not None
                save.state = _SaveState.PREPARED
                save.future = None
                prepared[identifier] = PreparedStoreSave(identifier, save.pool_key, tensor, save)
        return prepared

    def submit_prepared_batch(
        self,
        store_puts: list[PreparedStorePut],
    ) -> Future[None]:
        """Submit all prepared objects through one publisher/native batch."""
        if not store_puts:
            raise ValueError("prepared Store PUT batch must not be empty")
        with self._pending_lock:
            for store_put in store_puts:
                ticket = store_put.ticket
                current = self._pending.get(ticket.identifier)
                if current is None or current[1] is not ticket.save:
                    raise RuntimeError("prepared Store save is no longer pending")
                if ticket.save.state is not _SaveState.PREPARED:
                    raise RuntimeError("prepared Store save was already submitted")
            future = self._executor.submit(
                self._put_prepared_batch,
                store_puts,
            )
            for store_put in store_puts:
                store_put.ticket.save.future = future
                store_put.ticket.save.state = _SaveState.SUBMITTED
            return future

    def _put_prepared_batch(
        self,
        store_puts: list[PreparedStorePut],
    ) -> None:
        """Publish staged objects and release their source-save ownership."""
        try:
            results = self.store_client.put_prepared_tensors(
                [
                    (
                        store_put.ticket.pool_key,
                        store_put.addresses,
                        store_put.sizes,
                        store_put.owners,
                    )
                    for store_put in store_puts
                ]
            )
            for store_put, result in zip(store_puts, results, strict=True):
                if result < 0:
                    logger.warning(
                        "Store PUT rejected for %s: result=%d",
                        store_put.ticket.identifier,
                        result,
                    )
                else:
                    logger.info("Stored encoder output in Mooncake Store")
        except BaseException as error:
            traceback.clear_frames(error.__traceback__)
            raise
        finally:
            for store_put in store_puts:
                store_put.ticket.save.release()

    def abort_prepared(self, ticket: PreparedStoreSave) -> None:
        """Return budgets for a prepared ticket that was never submitted."""
        with self._pending_lock:
            current = self._pending.get(ticket.identifier)
            if current is not None and current[1] is ticket.save and ticket.save.state is _SaveState.PREPARED:
                self._finalize_locked(ticket.identifier, ticket.save)

    def _finalize_locked(self, identifier: str, save: _DeferredSave) -> None:
        """Remove a pending save and release its shared-storage budget reference."""
        current = self._pending.get(identifier)
        if current is None or current[1] is not save:
            return
        del self._pending[identifier]
        storage, ref_count = self._pending_storages[save.storage_ptr]
        if ref_count == 1:
            del self._pending_storages[save.storage_ptr]
            self._pending_bytes -= storage.nbytes()
        else:
            self._pending_storages[save.storage_ptr] = (storage, ref_count - 1)
        save.release()

    def reap(self) -> None:
        """Finalize completed publications and surface fatal publisher failures."""
        with self._pending_lock:
            for identifier, (_, raw_save) in list(self._pending.items()):
                save = raw_save
                assert isinstance(save, _DeferredSave)
                if save.state is not _SaveState.SUBMITTED or save.future is None or not save.future.done():
                    continue
                try:
                    save.future.result()
                except (EmbeddingStoreOperationError, OSError) as error:
                    logger.warning("Store PUT failed for %s: %s", identifier, error)
                    error.__traceback__ = None
                except BaseException as error:
                    if self._fatal_error is None:
                        self._fatal_error = error
                    continue
                self._finalize_locked(identifier, save)
            fatal = self._fatal_error
        if fatal is not None:
            raise fatal

    def shutdown(self) -> None:
        """Drain deferred publications and stop the publisher before Store closure."""
        with self._pending_lock:
            if self._closed:
                return
            self._closed = True
            for _, raw_save in list(self._pending.values()):
                save = raw_save
                assert isinstance(save, _DeferredSave)
                if save.state is _SaveState.DEFERRED:
                    save.future = self._executor.submit(self._save, save.pool_key, save)
                    save.state = _SaveState.SUBMITTED
                elif save.state is _SaveState.PREPARED:
                    self._finalize_locked(save.identifier, save)
        self._executor.shutdown(wait=True, cancel_futures=False)
        self.reap()
        self.store_client.close()
