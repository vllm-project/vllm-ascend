"""Select final Store transfers admitted for Backend submission."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Protocol

from ..invocation import StoreTransfer
from ..representation import BindingBatch

ObjectPresenceObserver = Callable[[list[str]], tuple[int, ...]]


class StoreAdmission(Protocol):
    """Select final Store bindings that may be submitted to the Backend."""

    def select(
        self, transfers: list[StoreTransfer], observe_presence: ObjectPresenceObserver
    ) -> list[StoreTransfer]: ...


class UnconditionalStoreAdmission:
    """Admit every Store binding when the Backend safely overwrites objects."""

    @staticmethod
    def select(transfers: list[StoreTransfer], observe_presence: ObjectPresenceObserver) -> list[StoreTransfer]:
        del observe_presence
        return transfers


class BackendExistenceStoreAdmission:
    """Admit only Store bindings whose final Backend object does not exist."""

    @staticmethod
    def select(transfers: list[StoreTransfer], observe_presence: ObjectPresenceObserver) -> list[StoreTransfer]:
        keys = _unique_object_keys(transfers)
        if not keys:
            return transfers
        presence = observe_presence(list(keys))
        if len(presence) != len(keys):
            raise RuntimeError(f"Store exists returned {len(presence)} results for {len(keys)} keys")
        if any(value not in (0, 1) for value in presence):
            raise RuntimeError("Store exists returned states other than 0 or 1")
        admitted_keys = {key for key, value in zip(keys, presence, strict=True) if value != 1}
        if len(admitted_keys) == len(keys):
            return transfers
        return [_select_transfer_bindings(transfer, admitted_keys) for transfer in transfers]


def _unique_object_keys(transfers: Sequence[StoreTransfer]) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            binding.remote_object.key
            for transfer in transfers
            for batch in transfer.batches
            for binding in batch.bindings
        )
    )


def _select_transfer_bindings(transfer: StoreTransfer, admitted_keys: set[str]) -> StoreTransfer:
    batches = tuple(
        BindingBatch(
            batch.group_id,
            tuple(binding for binding in batch.bindings if binding.remote_object.key in admitted_keys),
        )
        for batch in transfer.batches
    )
    return StoreTransfer(transfer.request_id, batches, transfer.store_job_id)
