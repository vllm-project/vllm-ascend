# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from functools import cache

import torch

# Workspace and tuning limit, not the physical core count. The LI/LD metadata
# ABI reserves 36 Cube / 72 Vector records; these kernels use at most 32/64.
INDEXER_MAX_WORKERS = 32


@cache
def get_indexer_worker_count(device: torch.device) -> int:
    """Keep the indexer's cross-core barrier within one resident wave.

    Cache host-side device properties during warmup. No tensor readback or
    device synchronization belongs in the token path.
    """
    cores = int(torch.npu.get_device_properties(device).cube_core_num)
    if cores <= 0:
        raise RuntimeError("The NPU must report a positive Cube core count")
    return min(INDEXER_MAX_WORKERS, cores)
