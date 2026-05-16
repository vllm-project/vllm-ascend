import functools

from vllm.logger import init_logger
from vllm_ascend.recovery.types import ExceptionInfo

logger = init_logger(__name__)


def fault_recovery_decorator():
    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            if self.in_recovery:
                return None
            else:
                try:
                    output = func(self, *args, **kwargs)
                    return output
                except Exception as e:
                    logger.error(f"Func {func.__name__} occurred exception: {e}")
                    self.in_recovery = True
                    
                    exception_info = ExceptionInfo(
                        exception_type=type(e).__name__,
                        message=str(e),
                    )
                    self.worker_input_socket.send(exception_info)
                    raise e
        return wrapper
    return decorator
