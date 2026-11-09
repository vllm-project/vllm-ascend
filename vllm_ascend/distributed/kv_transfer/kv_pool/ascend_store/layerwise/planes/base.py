# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""The layerwise data-plane interface implemented by every backend plane."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import ReqMeta


class LayerwiseDataPlane(ABC):
    """Step-level protocol for one layerwise backend data plane.

    The layerwise worker delegates the plane-specific parts of a transfer
    step to this interface and keeps the shared machinery (key construction,
    store I/O helpers, session tracking) on itself. Implementations hold a
    worker reference and must resolve worker helpers at call time instead of
    caching bound methods, so swapping a helper keeps affecting the whole
    step.
    """

    @abstractmethod
    def prepare_sessions(self, requests: list["ReqMeta"]) -> dict[int, list["ReqMeta"]]:
        """Open pool sessions for the step, before the per-layer save loop.

        Returns the per-group request views the save/load loops consume; an
        empty dict means the loops keep the requests unchanged.
        """

    @abstractmethod
    def prepare_load_gvas(self, requests: list["ReqMeta"]) -> None:
        """Prepare load transfer targets once the save loop is planned."""

    @abstractmethod
    def alloc_save_gvas(self, requests: list["ReqMeta"]) -> None:
        """Allocate save transfer targets once the save loop is planned."""

    @abstractmethod
    def finish_current_load_sessions(self) -> None:
        """Release the step's pool-side load sessions."""
