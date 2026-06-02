# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""
Platform patch: Fix PP-cross-node handshake metadata key collision.

Problem
-------
When the P node uses PP=N across N machines, each PP stage's workers report
handshake metadata with *only* tp_rank (0..TP-1) as the dict key.  When
``vllm/v1/engine/core.py`` merges all workers' dicts via ``content.update()``,
later PP stages' entries overwrite earlier ones.  Only the last PP stage's
metadata survives.

Consequence: the D node's ``_get_remote_host_info_by_port()`` looks up the
correct *device_index* (``pp_rank * tp_size + tp_rank`` = port offset), but
``multi_nodes_meta_mapping`` maps it to the wrong machine's IP.  RDMA reads
fetch KV cache data from the wrong P-node machine, and the decode output is
garbled after the first token.

Fix
---
Include ``pp_rank`` in the metadata key so that every PP-stage rank tuple
produces a globally unique key::

    old key = tp_rank                     (collisions across PP stages)
    new key = pp_rank * tp_size + tp_rank  (globally unique)

The handshake port on the producer follows:
``handshake_port = kv_port + pp_rank * tp_size + tp_rank``.

The consumer's lookup does:
``rank = str(handshake_port - kv_port) = str(pp_rank * tp_size + tp_rank)``.

Therefore the new key matches the lookup exactly.  Non-PP cases (PP=1) give
``pp_rank=0 → key = tp_rank``, preserving the old behaviour.
"""

import vllm.v1.worker.gpu_worker as _gpu_worker_module
from vllm.distributed.kv_transfer import get_kv_transfer_group, has_kv_transfer_group
from vllm.distributed.parallel_state import get_pp_group, get_tp_group


def _patched_get_kv_connector_handshake_metadata(self):
    if not has_kv_transfer_group():
        return None

    connector = get_kv_transfer_group()
    if (metadata := connector.get_handshake_metadata()) is None:
        return None

    tp_group = get_tp_group()
    tp_rank = tp_group.rank_in_group
    tp_size = tp_group.world_size
    pp_rank = get_pp_group().rank_in_group
    return {pp_rank * tp_size + tp_rank: metadata}


# Replace the method on the Worker class so that all GPU workers
# (including those in subprocesses) pick up the fix.
_gpu_worker_module.Worker.get_kv_connector_handshake_metadata = (
    _patched_get_kv_connector_handshake_metadata
)
