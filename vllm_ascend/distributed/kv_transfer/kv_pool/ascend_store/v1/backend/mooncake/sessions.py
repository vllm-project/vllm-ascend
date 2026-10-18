"""Own Mooncake object sessions without Worker or KV-layout dependencies."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..results import integer_results
from ..validation import validate_session_open

if TYPE_CHECKING:
    from .backend import MooncakeBackend


class MooncakeSessions:
    """Track opened or uncertain native sessions until publication or cleanup succeeds.

    One serial Load executor and one serial Store executor own their respective
    dictionaries. The caller drains both before close; this owner is not a fence.
    Failed opens lose ownership only after an explicit per-key rejection. An
    exception can follow partial acquisition, so attempted keys remain cleanable.
    """

    def __init__(self, backend: MooncakeBackend) -> None:
        backend.validate_key_range_support()
        self._backend = backend
        self._load_keys: dict[str, None] = {}
        self._store_keys: dict[str, bool] = {}

    @property
    def has_load_sessions(self) -> bool:
        return bool(self._load_keys)

    @property
    def pending_store_keys(self) -> tuple[str, ...]:
        return tuple(self._store_keys)

    # ====================
    # Read sessions
    # ====================

    def start_load(self, keys: list[str], object_sizes: list[int]) -> tuple[int, ...]:
        validate_session_open(keys, object_sizes, self._load_keys)
        if not keys:
            return ()
        self._load_keys.update((key, None) for key in keys)
        codes = integer_results("KeyRange start Load", keys, self._backend.start_key_range_load(keys))
        for key, code in zip(keys, codes, strict=True):
            if code != 0:
                del self._load_keys[key]
        return codes

    def finish_load(self, keys: list[str] | None = None) -> None:
        owned = (
            list(self._load_keys)
            if keys is None
            else list(dict.fromkeys(key for key in keys if key in self._load_keys))
        )
        if owned:
            self._backend.finish_key_range_load(owned)
            for key in owned:
                del self._load_keys[key]

    # ====================
    # Write sessions
    # ====================

    def start_store(self, keys: list[str], object_sizes: list[int]) -> tuple[int, ...]:
        validate_session_open(keys, object_sizes, self._store_keys)
        if not keys:
            return ()
        self._store_keys.update((key, False) for key in keys)
        codes = integer_results("KeyRange start Store", keys, self._backend.start_key_range_store(keys, object_sizes))
        for key, code in zip(keys, codes, strict=True):
            if code == 0:
                self._store_keys[key] = True
            else:
                del self._store_keys[key]
        return codes

    def commit_store(self, keys: list[str]) -> tuple[int, ...]:
        if not keys:
            return ()
        if len(set(keys)) != len(keys):
            raise ValueError("Commit keys must be unique")
        if any(not self._store_keys.get(key) for key in keys):
            raise RuntimeError("KeyRange commit has no confirmed owning Store session")
        codes = integer_results("KeyRange commit", keys, self._backend.commit_key_range_store(keys))
        for key, code in zip(keys, codes, strict=True):
            if code == 0:
                del self._store_keys[key]
        return codes

    def revoke_store(self, keys: list[str]) -> tuple[int, ...]:
        owned = list(dict.fromkeys(key for key in keys if key in self._store_keys))
        if not owned:
            return (0,) * len(keys)
        # Once abandoned, an unresolved write may be cleaned up but never committed.
        self._store_keys.update((key, False) for key in owned)
        codes = integer_results("KeyRange revoke", owned, self._backend.revoke_key_range_store(owned))
        by_key = dict(zip(owned, codes, strict=True))
        for key, code in by_key.items():
            if code == 0:
                del self._store_keys[key]
        return tuple(by_key.get(key, 0) for key in keys)

    def close(self) -> None:
        """Retry outstanding cleanup after copies stop, retaining failed ownership."""
        self.finish_load()
        self.revoke_store(list(self._store_keys))
        if self._store_keys:
            raise RuntimeError(f"KeyRange Store cleanup is unconfirmed for {tuple(self._store_keys)}")
