# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# This patch modifies WorkerProc.enqueue_output to handle recovery from exceptions.
# It is controlled by the VLLM_ASCEND_ENABLE_RECOVERY environment variable.
import msgspec.msgpack
from typing import Any

from vllm.logger import logger
from vllm.v1.executor.multiproc_executor import WorkerProc
from vllm.v1.outputs import AsyncModelRunnerOutput

from vllm_ascend.recovery.types import ExceptionInfo


def enqueue_output(self, output: Any):
    """Prepares output from the worker and enqueues it to the
    worker_response_mq. If the output is an Exception, it is
    converted to a FAILURE response.
    """
    if isinstance(output, AsyncModelRunnerOutput):
        try:
            output = output.get_output()
        except Exception as e:
            logger.error("[WorkerProc] Enqueue_output detected exception, send to WorkerMonitor")
            self.worker.worker.exception_occur = True
            if not self.worker.worker.in_recovery:
                self.worker.worker.in_recovery = True
                exception_info = ExceptionInfo(
                    exception_type=type(e).__name__,
                    exception_msg=str(e),
                )
                exception_encode = msgspec.msgpack.encode(exception_info)
                self.worker.worker_input_socket.send(exception_encode)
            output = e
    if isinstance(output, Exception):
        result = (WorkerProc.ResponseStatus.FAILURE, str(output))
    else:
        result = (WorkerProc.ResponseStatus.SUCCESS, output)
    if (response_mq := self.worker_response_mq) is not None:
        response_mq.enqueue(result)


WorkerProc.enqueue_output = enqueue_output