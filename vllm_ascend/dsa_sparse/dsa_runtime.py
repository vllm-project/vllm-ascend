"""Process-local DSA worker runtime binding.

Each DP rank is a separate worker process and each TP worker owns its own
resident/DRAM tensors.  Attention layers therefore resolve only the manager
installed in their current process; no state is shared across DP or TP ranks.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm_ascend.dsa_sparse.dsa_sparse import DSASparseV1

_WORKER_MANAGER: "DSASparseV1 | None" = None


def set_dsa_worker_manager(manager: "DSASparseV1 | None") -> None:
    global _WORKER_MANAGER
    _WORKER_MANAGER = manager


def get_dsa_worker_manager() -> "DSASparseV1 | None":
    return _WORKER_MANAGER
