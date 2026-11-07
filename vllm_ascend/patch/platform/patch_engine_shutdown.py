# SPDX-License-Identifier: Apache-2.0
"""Keep immediate request abort distinct from NPU resource cleanup."""

import vllm.v1.engine.utils as engine_utils
from vllm.platforms import current_platform

ASCEND_ENGINE_PROCESS_SHUTDOWN_TIMEOUT_S = 15.0

# Older supported vLLM versions may not expose this process-manager hook.
_ORIGINAL_SHUTDOWN_TIMEOUT = getattr(engine_utils, "get_engine_process_shutdown_timeout", None)


def get_engine_process_shutdown_timeout(request_timeout, process_timeout):
    if current_platform.device_type == "npu" and request_timeout == 0 and process_timeout in (None, 0):
        return ASCEND_ENGINE_PROCESS_SHUTDOWN_TIMEOUT_S
    assert _ORIGINAL_SHUTDOWN_TIMEOUT is not None
    return _ORIGINAL_SHUTDOWN_TIMEOUT(request_timeout, process_timeout)


if _ORIGINAL_SHUTDOWN_TIMEOUT is not None:
    engine_utils.get_engine_process_shutdown_timeout = get_engine_process_shutdown_timeout
