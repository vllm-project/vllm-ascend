"""Narrow Backend capabilities consumed by AscendStore v1 Worker I/O."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, TypeAlias

from .registration import Registration

AddressRows: TypeAlias = list[list[int]]
SizeRows: TypeAlias = list[list[int]]
OffsetRows: TypeAlias = list[list[int]]
ResultCodes: TypeAlias = tuple[int, ...]


@dataclass(frozen=True, slots=True)
class GVARegion:
    """One readable remote object exposed without native SDK metadata types."""

    address: int
    size: int


class KVStoreBackend(Protocol):
    """Lifecycle and whole-object operations required by every v1 Backend."""

    requires_exists_before_put: bool

    def set_device(self) -> None: ...

    def register_buffer(self, addresses: list[int], sizes: list[int]) -> Registration: ...

    def close(self) -> None: ...

    def exists(self, keys: list[str]) -> tuple[bool, ...]: ...

    def load(self, keys: list[str], addresses: AddressRows, sizes: SizeRows) -> ResultCodes: ...

    def store(self, keys: list[str], addresses: AddressRows, sizes: SizeRows) -> ResultCodes: ...


class KeyRangeBackend(KVStoreBackend, Protocol):
    """Block-key sessions that copy selected byte ranges."""

    def validate_key_range_support(self) -> None: ...

    def start_key_range_load(self, keys: list[str]) -> ResultCodes: ...

    def copy_key_range_load(
        self,
        keys: list[str],
        addresses: AddressRows,
        sizes: SizeRows,
        source_offsets: OffsetRows,
    ) -> ResultCodes: ...

    def finish_key_range_load(self, keys: list[str]) -> None: ...

    def start_key_range_store(self, keys: list[str], object_sizes: list[int]) -> ResultCodes: ...

    def copy_key_range_store(
        self,
        keys: list[str],
        addresses: AddressRows,
        sizes: SizeRows,
        destination_offsets: OffsetRows,
    ) -> ResultCodes: ...

    def commit_key_range_store(self, keys: list[str]) -> ResultCodes: ...

    def revoke_key_range_store(self, keys: list[str]) -> ResultCodes: ...


class GVABackend(KVStoreBackend, Protocol):
    """Global-address sessions without exposing native key-info or directions."""

    def validate_gva_support(self) -> None: ...

    def query_gva_regions(self, keys: list[str]) -> tuple[GVARegion | None, ...]: ...

    def add_gva_leases(self, keys: list[str]) -> ResultCodes: ...

    def remove_gva_leases(self, keys: list[str]) -> None: ...

    def allocate_gva(self, keys: list[str], object_sizes: list[int]) -> tuple[int, ...]: ...

    def copy_from_gva(
        self,
        remote_addresses: list[int],
        local_addresses: list[int],
        sizes: list[int],
    ) -> int: ...

    def copy_to_gva(
        self,
        remote_addresses: list[int],
        local_addresses: list[int],
        sizes: list[int],
    ) -> int: ...

    def publish_gva(self, keys: list[str]) -> ResultCodes: ...
