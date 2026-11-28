# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.worker.gpu.async_utils import AsyncOutput

from vllm_ascend.worker.sentinel.npu_worker_sentinel import fault_barrier_wrapper


class AscendAsyncOutput(AsyncOutput):
    """Async output with Ascend fault-tolerance support."""

    # Device errors can also surface in get_output. Use the fault barrier
    # to keep the worker alive for recovery.
    @fault_barrier_wrapper
    def get_output(self) -> ModelRunnerOutput:
        return super().get_output()
