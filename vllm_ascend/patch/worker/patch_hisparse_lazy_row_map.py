# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Detached candidate: discard an unused CPU row map after its original wait.

Do not remove/stagger the D2H copy, its synchronization, or any DMA/event. The
frozen finish_mirror_phase calls resolve only after the staging event completes.
Only a certified all-lazy current target with no pending draft-row consumer can
discard this map. Page spills continue to use the original page descriptors.
"""

import inspect
from pathlib import Path

COMMON_WORKER_SHA256 = "1953b3e582502d16df38bba4a5966e0b98492feb67b65cdac7f2b0397ca4b229"
FLAG = "_ascend_certified_lazy_mtp_mirror"


def can_discard(worker, state):
    if getattr(worker, "_slot_mapping_staging", None) is not state or state.num_tokens <= 0:
        return False
    if getattr(worker, "_draft_mirror_pending", None) is not False:
        return False
    if (
        getattr(worker, "_per_layer_mirrored", None) != set()
        or getattr(worker, "_submitted_mirror_layers", None) != set()
    ):
        return False
    active = [handle for handle in worker.cache_handles if handle.num_actual_tokens != 0]
    return bool(active) and all(
        handle.runtime.device.type == "npu"
        and handle.runtime.eager_host_mirror is False
        and getattr(handle, FLAG, None) is True
        and handle.decode_batch is False
        and handle.host_mirror_required is False
        for handle in active
    )


def resolve(worker, state, original):
    if not can_discard(worker, state):
        return original(worker, state)
    # The caller already synchronized the original event. No pending draft
    # rows can consume this map, and lazy enqueue still invalidates device KV.
    worker._set_row_mirrors(())
    state.num_tokens = 0


def install():
    """Explicit activation, after the reviewed lazy-MTP mirror policy."""
    from functools import wraps

    from vllm.distributed.kv_transfer.kv_connector.v1.hisparse.worker import HiSparseConnectorWorker as Worker

    from vllm_ascend.distributed.kv_transfer.hisparse.optimization_profile import verify_core_source

    original = Worker._resolve_row_mirrors
    if getattr(original, "_ascend_discard_unused_lazy_row_map", False):
        from .patch_hisparse_unused_staging import install as install_unused_staging

        install_unused_staging()
        from .patch_hisparse_block_table_views import install as install_block_table_views

        install_block_table_views()
        return
    path = Path(inspect.getsourcefile(original))
    verify_core_source(path, COMMON_WORKER_SHA256)
    if not getattr(Worker._enqueue_host_mirror, "_ascend_certified_lazy_mtp_mirror_patch", False):
        raise RuntimeError("Install the qualified lazy-MTP mirror policy first")
    source = inspect.getsource(Worker._finish_mirror_phase)
    if not source.index("state.event.synchronize()") < source.index("self._resolve_row_mirrors(state)"):
        raise RuntimeError("Row-map discard requires the original completed staging event")

    @wraps(original)
    def wrapped(worker, state):
        return resolve(worker, state, original)

    wrapped._ascend_discard_unused_lazy_row_map = True
    wrapped._ascend_original = original
    Worker._resolve_row_mirrors = wrapped

    from .patch_hisparse_unused_staging import install as install_unused_staging

    install_unused_staging()
    from .patch_hisparse_block_table_views import install as install_block_table_views

    install_block_table_views()
