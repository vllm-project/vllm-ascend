# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Cold, version-checked page-descriptor optimization for Ascend HiSparse.

The common worker has no backend descriptor-construction hook. Replace only
that CPU preparation method, not DMA submission, events, retirement or page
ownership. Remove this wrapper when common HiSparse supplies the vectorized
implementation or a descriptor-builder hook. Unsupported source versions and
other devices keep the original implementation.
"""

import hashlib
import inspect
from functools import wraps
from pathlib import Path

from vllm_ascend.distributed.kv_transfer.hisparse.page_dma_descriptors import enqueue_transfers

# Reviewed common worker files: PR62592's paired prefetch revision and the
# frozen integration runtime used for the measured descriptor optimization.
_WORKER_SOURCE_SHA256 = frozenset(
    {
        "81067d36662399fa5fc517bb3f334b78d48dd35247d5f405fcbd3e5982875197",
        "1953b3e582502d16df38bba4a5966e0b98492feb67b65cdac7f2b0397ca4b229",
    }
)


def install() -> bool:
    """Install once before serving; False leaves an unreviewed owner intact."""
    from vllm.distributed.kv_transfer.kv_connector.v1.hisparse.worker import HiSparseConnectorWorker

    original = HiSparseConnectorWorker._enqueue_transfers
    if getattr(original, "_ascend_vector_page_descriptors", False):
        return True
    # Do not bypass somebody else's wrapper by unwrapping its source.
    if hasattr(original, "__wrapped__"):
        return False
    try:
        source = inspect.getsourcefile(original)
        digest = None if source is None else hashlib.sha256(Path(source).read_bytes()).hexdigest()
    except (OSError, TypeError):
        return False
    if digest not in _WORKER_SOURCE_SHA256:
        return False

    @wraps(original)
    def wrapped(worker, transfers):
        handles = worker.cache_handles
        if handles and handles[0].runtime.device.type == "npu" and enqueue_transfers(worker, transfers):
            return
        return original(worker, transfers)

    wrapped._ascend_vector_page_descriptors = True
    wrapped._ascend_original = original
    HiSparseConnectorWorker._enqueue_transfers = wrapped
    return True
