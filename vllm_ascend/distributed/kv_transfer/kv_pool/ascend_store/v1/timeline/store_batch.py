"""Store completion fence shared by concrete Store timelines."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

from ..worker.transfer.evidence import StoreCompletion


@dataclass(slots=True)
class StoreBatch:
    """The exact completion fence for one submitted Store invocation."""

    completed: threading.Event = field(default_factory=threading.Event)
    completions: list[StoreCompletion] = field(default_factory=list)
