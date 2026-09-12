"""Private interpreter entry point for an engine-owned KV transfer process."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def run_worker(endpoint: str, parent_fd: int, service_factory=None) -> int:
    """Build the child transport server around one KV transfer service."""
    from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mp.server import TransferServer

    if service_factory is None:
        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.mp.service import TransferService

        service_factory = TransferService
    return TransferServer(endpoint, parent_fd, service_factory).run()


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[6]))
    exit_code = run_worker(sys.argv[1], int(sys.argv[2]))
    os._exit(exit_code)
