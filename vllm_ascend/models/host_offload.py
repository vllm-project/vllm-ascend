# SPDX-License-Identifier: Apache-2.0
"""Track host-offload owners before a partially loaded model is published."""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Protocol


class HostOffloadOwner(Protocol):
    def close_host_offload(self) -> None: ...


# Scoped to synchronous model construction in the current execution context.
# Reset on every exit; concurrent threads and nested loads have separate lists.
_ACTIVE_HOST_OFFLOAD_OWNERS: ContextVar[list[HostOffloadOwner] | None] = ContextVar(
    "active_host_offload_owners", default=None
)


def register_host_offload_owner(owner: HostOffloadOwner) -> bool:
    owners = _ACTIVE_HOST_OFFLOAD_OWNERS.get()
    if owners is not None:
        owners.append(owner)
        return True
    return False


@contextmanager
def track_host_offload_owners() -> Iterator[list[HostOffloadOwner]]:
    owners: list[HostOffloadOwner] = []
    token = _ACTIVE_HOST_OFFLOAD_OWNERS.set(owners)
    try:
        yield owners
    finally:
        _ACTIVE_HOST_OFFLOAD_OWNERS.reset(token)
