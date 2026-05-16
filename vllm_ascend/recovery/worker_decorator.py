import functools
import msgspec.msgpack

from vllm.logger import logger
from vllm_ascend.recovery.types import ExceptionInfo

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
                    exception_encode = msgspec.msgpack.encode(exception_info)
                    self.worker_input_socket.send(exception_encode)
                    raise e
        return wrapper
    return decorator
