"""Select Store keys that still require a Backend write."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Protocol

ObjectPresenceObserver = Callable[[list[str]], tuple[int, ...]]


class MissingFilter(Protocol):
    """Select remote keys that still require a Backend write."""

    def select_missing_keys(self, keys: Sequence[str], observe_presence: ObjectPresenceObserver) -> tuple[str, ...]: ...


class IdentityMissingFilter:
    """Preserve every Store key when the Backend overwrites existing keys safely."""

    @staticmethod
    def select_missing_keys(keys: Sequence[str], observe_presence: ObjectPresenceObserver) -> tuple[str, ...]:
        del observe_presence
        return tuple(keys)


class BackendExistenceMissingFilter:
    """Remove Store keys that already exist in the Backend."""

    @staticmethod
    def select_missing_keys(keys: Sequence[str], observe_presence: ObjectPresenceObserver) -> tuple[str, ...]:
        presence = observe_presence(list(keys))
        if len(presence) != len(keys):
            raise RuntimeError(f"Store exists returned {len(presence)} results for {len(keys)} keys")
        if any(value not in (0, 1) for value in presence):
            raise RuntimeError("Store exists returned states other than 0 or 1")
        return tuple(key for key, value in zip(keys, presence, strict=True) if value != 1)
