# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Optional, process-local evidence for temporal cache reuse."""

import json
import os
import time

from vllm.logger import logger

from vllm_ascend.ascend_config import get_ascend_config


def offload_event(event: str, **fields) -> None:
    try:
        if not get_ascend_config().kv_pool_range_debug:
            return
        payload = dict(event=event, pid=os.getpid(), t_ns=time.monotonic_ns(), **fields)
        logger.info("DSV4_OFFLOAD %s", json.dumps(payload, separators=(",", ":")))
    except Exception:
        # Diagnostics must never change transfer/failure semantics.
        pass
