"""Select final Store transfers admitted for Backend submission."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from ..values.evidence import RemoteObjectObservation
from ..values.representation import BindingBatch, RemoteKVObject
from ..values.selection import StoreTransfer


class StoreAdmission(Protocol):
    """Select final Store bindings that may be submitted to the Backend."""

    def observation_targets(self, transfers: Sequence[StoreTransfer]) -> tuple[RemoteKVObject, ...]: ...

    def select(
        self,
        transfers: list[StoreTransfer],
        observations: Sequence[RemoteObjectObservation],
    ) -> list[StoreTransfer]: ...


class UnconditionalStoreAdmission:
    """Admit every Store binding when the Backend safely overwrites objects."""

    @staticmethod
    def observation_targets(transfers: Sequence[StoreTransfer]) -> tuple[RemoteKVObject, ...]:
        del transfers
        return ()

    @staticmethod
    def select(transfers: list[StoreTransfer], observations: Sequence[RemoteObjectObservation]) -> list[StoreTransfer]:
        if observations:
            raise ValueError("Unconditional Store admission does not consume Backend observations")
        return transfers


class BackendExistenceStoreAdmission:
    """Admit only Store bindings whose final Backend object does not exist."""

    @staticmethod
    def observation_targets(transfers: Sequence[StoreTransfer]) -> tuple[RemoteKVObject, ...]:
        return _unique_remote_objects(transfers)

    @staticmethod
    def select(transfers: list[StoreTransfer], observations: Sequence[RemoteObjectObservation]) -> list[StoreTransfer]:
        remote_objects = _unique_remote_objects(transfers)
        if not remote_objects:
            return transfers
        expected_keys = tuple(remote_object.key for remote_object in remote_objects)
        observed_keys = tuple(observation.remote_object.key for observation in observations)
        if observed_keys != expected_keys:
            raise RuntimeError(f"Store observations {observed_keys} do not match candidates {expected_keys}")
        admitted_keys = {observation.remote_object.key for observation in observations if not observation.readable}
        if len(admitted_keys) == len(expected_keys):
            return transfers
        return [_select_transfer_bindings(transfer, admitted_keys) for transfer in transfers]


def _unique_remote_objects(transfers: Sequence[StoreTransfer]) -> tuple[RemoteKVObject, ...]:
    objects_by_key = {
        binding.remote_object.key: binding.remote_object
        for transfer in transfers
        for batch in transfer.batches
        for binding in batch.bindings
    }
    return tuple(objects_by_key.values())


def _select_transfer_bindings(transfer: StoreTransfer, admitted_keys: set[str]) -> StoreTransfer:
    batches = tuple(
        BindingBatch(
            batch.group_id,
            tuple(binding for binding in batch.bindings if binding.remote_object.key in admitted_keys),
        )
        for batch in transfer.batches
    )
    return StoreTransfer(transfer.request_id, batches, transfer.store_job_id)
