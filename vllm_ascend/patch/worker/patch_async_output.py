# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.worker.gpu.async_utils import AsyncOutput

from vllm_ascend.worker.sentinel.npu_worker_sentinel import fault_barrier_wrapper


class AscendAsyncOutput(AsyncOutput):
    """AsyncOutput whose ``get_output`` runs behind the FT fault barrier.

    HCCL faults (e.g. after a peer faults) escaping ``get_output`` can make
    the destructor/teardown that follows abort at the C level and kill the
    process; with FT the worker must survive to recover, so the barrier
    quarantines it (stop_device) before any teardown executes.
    """

    @fault_barrier_wrapper
    def get_output(self) -> ModelRunnerOutput:
        return super().get_output()
