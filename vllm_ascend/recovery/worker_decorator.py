import functools

import msgspec.msgpack
import torch
import torch_npu

from vllm.distributed.parallel_state import get_dp_group, get_pp_group, get_world_group
from vllm.logger import logger
from vllm_ascend.recovery.types import ExceptionInfo


def fault_recovery_decorator():
    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            if self.exception_occur or self.in_recovery:
                logger.info(f"[WorkerDecorator] Func {func.__name__} called in recovery phase, return None")
                return None
            else:
                try:
                    output = func(self, *args, **kwargs)
                    return output
                except Exception as e:
                    self.exception_occur = True
                    if self.in_recovery:
                        logger.info(f"[WorkerDecorator] Func {func.__name__} caught exception in recovery phase.Don't send error to worker monitor")
                        raise e
                    else:
                        logger.error(f"[WorkerDecorator] Func {func.__name__} occurred exception: {e}")
                        self.in_recovery = True

                        exception_info = ExceptionInfo(
                            exception_type=type(e).__name__,
                            exception_msg=str(e),
                        )
                        exception_encode = msgspec.msgpack.encode(exception_info)
                        self.worker_input_socket.send(exception_encode)
                        raise e
        return wrapper
    return decorator
