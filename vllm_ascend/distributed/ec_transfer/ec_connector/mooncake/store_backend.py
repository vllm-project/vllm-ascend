# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
# SPDX-License-Identifier: Apache-2.0
"""Deferred Store publication for the shared-engine Ascend fallback path."""

from __future__ import annotations

import traceback
from concurrent.futures import Future
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any

import torch
from vllm.distributed.ec_transfer.ec_connector.mooncake_store_embedding.backend import (
    MooncakeEmbeddingStoreBackend,
    _StoreSave,
    _record_tensor_ready_event,
)
from vllm.distributed.ec_transfer.ec_connector.mooncake_store_embedding.data import (
    make_embedding_key,
)
from vllm.distributed.ec_transfer.ec_connector.mooncake_store_embedding.store_client import (
    EmbeddingStoreError,
    EmbeddingStoreOperationError,
)
from vllm.logger import init_logger

from vllm_ascend.distributed.ec_transfer.ec_connector.mooncake.store_client import (
    AscendMooncakeEmbeddingStoreClient,
)

logger = init_logger(__name__)


class _SaveState(Enum):
    DEFERRED = auto()
    CHECKING = auto()
    PREPARED = auto()
    SUBMITTED = auto()


@dataclass
class _DeferredSave(_StoreSave):
    identifier: str
    pool_key: str
    state: _SaveState = _SaveState.DEFERRED
    future: Future[None] | Future[bool] | None = None


@dataclass(frozen=True)
class PreparedStoreSave:
    identifier: str
    pool_key: str
    tensor: torch.Tensor
    save: _DeferredSave


@dataclass(frozen=True)
class PreparedStorePut:
    ticket: PreparedStoreSave
    addresses: list[int]
    sizes: list[int]
    owners: list[Any]


class AscendDeferredStoreBackend(MooncakeEmbeddingStoreBackend):
    """Reserve publisher budget now and choose the PUT memory path later."""

    store_client: AscendMooncakeEmbeddingStoreClient

    def save_output(self, identifier: str, tensor: torch.Tensor) -> None:
        if identifier in self._step_candidates:
            self.reap()
            self._admit(identifier, tensor)

    def _admit(self, identifier: str, tensor: torch.Tensor) -> bool:
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
                ready_event=_record_tensor_ready_event(tensor),
                storage_ptr=storage_ptr,
            )
            self._pending[identifier] = (None, save)  # type: ignore[assignment]
            self._pending_storages[storage_ptr] = (storage, ref_count + 1)
            self._pending_bytes += budget_bytes
        return True

    def _submit_original_locked(self, save: _DeferredSave) -> None:
        if save.state is not _SaveState.DEFERRED:
            return
        save.state = _SaveState.SUBMITTED
        save.future = self._executor.submit(self._save, save.pool_key, save)

    def submit_original(self, identifier: str) -> bool:
        with self._pending_lock:
            item = self._pending.get(identifier)
            if item is None:
                return False
            save = item[1]
            assert isinstance(save, _DeferredSave)
            self._submit_original_locked(save)
            return save.state is _SaveState.SUBMITTED

    def prepare_fallback(self, identifier: str) -> PreparedStoreSave | None:
        with self._pending_lock:
            item = self._pending.get(identifier)
            if item is None:
                return None
            save = item[1]
            assert isinstance(save, _DeferredSave)
            if save.state is not _SaveState.DEFERRED:
                return None
            save.state = _SaveState.CHECKING
            save.future = self._executor.submit(self.store_client.exists, save.pool_key)
            future = save.future

        try:
            exists = future.result()
        except (EmbeddingStoreOperationError, OSError) as error:
            logger.warning("Store PUT preflight failed for %s: %s", identifier, error)
            with self._pending_lock:
                self._finalize_locked(identifier, save)
            return None
        except BaseException as error:
            traceback.clear_frames(error.__traceback__)
            with self._pending_lock:
                if self._fatal_error is None:
                    self._fatal_error = error
            return None

        if exists:
            with self._pending_lock:
                self._finalize_locked(identifier, save)
            return None
        tensor = save.tensor
        assert tensor is not None
        save.state = _SaveState.PREPARED
        save.future = None
        return PreparedStoreSave(identifier, save.pool_key, tensor, save)

    def submit_prepared(
        self,
        store_put: PreparedStorePut,
    ) -> Future[None]:
        ticket = store_put.ticket
        with self._pending_lock:
            current = self._pending.get(ticket.identifier)
            if current is None or current[1] is not ticket.save:
                raise RuntimeError("prepared Store save is no longer pending")
            if ticket.save.state is not _SaveState.PREPARED:
                raise RuntimeError("prepared Store save was already submitted")
            ticket.save.state = _SaveState.SUBMITTED
            future = self._executor.submit(
                self._put_prepared,
                store_put,
            )
            ticket.save.future = future
            return future

    def _put_prepared(
        self,
        store_put: PreparedStorePut,
    ) -> None:
        ticket = store_put.ticket
        try:
            self.store_client.put_prepared_tensor(
                ticket.pool_key,
                store_put.addresses,
                store_put.sizes,
                store_put.owners,
            )
        except BaseException as error:
            traceback.clear_frames(error.__traceback__)
            raise
        finally:
            ticket.save.release()

    def abort_prepared(self, ticket: PreparedStoreSave) -> None:
        with self._pending_lock:
            current = self._pending.get(ticket.identifier)
            if (
                current is not None
                and current[1] is ticket.save
                and ticket.save.state is _SaveState.PREPARED
            ):
                self._finalize_locked(ticket.identifier, ticket.save)

    def _finalize_locked(self, identifier: str, save: _DeferredSave) -> None:
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
        with self._pending_lock:
            for identifier, (_, raw_save) in list(self._pending.items()):
                save = raw_save
                assert isinstance(save, _DeferredSave)
                if (
                    save.state is not _SaveState.SUBMITTED
                    or save.future is None
                    or not save.future.done()
                ):
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
        with self._pending_lock:
            if self._closed:
                return
            self._closed = True
            for _, raw_save in list(self._pending.values()):
                save = raw_save
                assert isinstance(save, _DeferredSave)
                if save.state is _SaveState.DEFERRED:
                    self._submit_original_locked(save)
                elif save.state is _SaveState.PREPARED:
                    self._finalize_locked(save.identifier, save)
        self._executor.shutdown(wait=True, cancel_futures=False)
        self.reap()
        self.store_client.close()
