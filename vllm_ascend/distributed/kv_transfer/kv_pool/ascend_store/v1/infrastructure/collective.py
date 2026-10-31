"""An isolated CPU communication group with caller-owned lifetime.

Participants enter collective operations in the same order. Independent
concurrent operation sequences need separate groups. Close only after all
operations using this group have finished.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping
from datetime import timedelta
from typing import Any, Generic, TypeVar

import torch.distributed as dist

COLLECTIVE_TIMEOUT_SECONDS = 60

Payload = TypeVar("Payload")
ItemId = TypeVar("ItemId", bound=Hashable)
Identity = TypeVar("Identity")


class CollectiveGroup:
    def __init__(self, ranks: tuple[int, ...]) -> None:
        self._ranks = tuple(sorted(ranks))
        self._group = dist.new_group(
            ranks=list(self._ranks),
            backend="gloo",
            timeout=timedelta(seconds=COLLECTIVE_TIMEOUT_SECONDS),
            use_local_synchronization=True,
        )
        self._closed = False

    def all_gather(self, value: Payload) -> dict[int, Payload]:
        if self._closed:
            raise RuntimeError("Collective group is closed")
        results: list[Any] = [None] * len(self._ranks)
        dist.all_gather_object(results, value, group=self._group)
        return dict(zip(self._ranks, results, strict=True))

    def close(self) -> None:
        if not self._closed:
            dist.destroy_process_group(self._group)
            self._closed = True


class GroupCompletion(Generic[ItemId, Identity]):
    """Return local items completed with the same identity by every required participant.

    Identities are opaque values supporting equality. All members of the
    collective enter each round in the same order, including empty reports.
    """

    def __init__(self, required_participants: tuple[int, ...], collective: CollectiveGroup) -> None:
        if not required_participants:
            raise ValueError("Group completion requires participants")
        self._required_participants, self._collective = required_participants, collective

    def resolve(self, local_completed: Mapping[ItemId, Identity]) -> set[ItemId]:
        reports = self._collective.all_gather(local_completed)
        required = tuple(reports[rank] for rank in self._required_participants)
        return {
            item_id
            for item_id, identity in local_completed.items()
            if all(item_id in report and report[item_id] == identity for report in required)
        }
