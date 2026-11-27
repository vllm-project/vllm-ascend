# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Allow NPU profiling while inheriting the upstream Worker constructor.

The upstream validator has no NPU entry in its profiler support table. Rebind
both the defining module and GPUWorker's cached import before parent construction.
Remove this patch in favor of NPUWorker._validate_profiler_config once
https://github.com/vllm-project/vllm/pull/65565 is available in supported vLLM
versions. Elastic EP construction uses the upstream implementation unchanged.
"""

import vllm.profiler.wrapper as profiler_wrapper
import vllm.v1.worker.gpu_worker as gpu_worker
from vllm.config.profiler import ProfilerConfig
from vllm.platforms import current_platform

_ORIGINAL_VALIDATE_WORKER_PROFILER_CONFIG = profiler_wrapper.validate_worker_profiler_config


def validate_worker_profiler_config(profiler_config: ProfilerConfig | None) -> None:
    if current_platform.device_type != "npu":
        return _ORIGINAL_VALIDATE_WORKER_PROFILER_CONFIG(profiler_config)
    if profiler_config is not None and profiler_config.profiler not in (None, "torch"):
        raise ValueError("Ascend supports only the torch NPU profiler.")


profiler_wrapper.validate_worker_profiler_config = validate_worker_profiler_config
gpu_worker.validate_worker_profiler_config = validate_worker_profiler_config
