import torch
import yaml
import torch_npu

from typing import Optional
from abc import ABC, abstractmethod
from vllm.logger import logger
from vllm_ascend.worker.common import RecoveryStatus,FaultStatus
from vllm_ascend.worker.recovery_context import RecoveryContext

force_stop_error = ["force stop"]
network_error = [
    "suspect remote error",
    "hccl op retry failed"
]


class RecoveryHandler(ABC):

    def __init__(self):
        self.next_handler = None

    @abstractmethod
    def can_handle(self, ctx:RecoveryContext) -> bool:
        pass

    @abstractmethod
    def recover(self, ctx:RecoveryContext) -> torch.Tensor:
        """Specific recovery function"""
        pass


class ForceStopHandler(RecoveryHandler):

    def can_handle(self, ctx:RecoveryContext) -> bool:
        error_str = str(ctx.exception).lower()
        for error in force_stop_error:
            if error in error_str:
                return True
        return False

    def recover(self, ctx:RecoveryContext) -> RecoveryStatus:

        """Force stop needs no extra recovery"""
        return RecoveryStatus.SUCCESS

class NetworkHandler(RecoveryHandler):

    def can_handle(self, ctx:RecoveryContext) -> bool:
        error_str = str(ctx.exception).lower()
        for error in network_error:
            if error in error_str:
                ctx.fault_queue.put_nowait(FaultStatus.NETWORK_ERR)
                return True
        return False

    def recover(self, ctx:RecoveryContext) -> RecoveryStatus:
        """Network needs no extra recovery"""
        return RecoveryStatus.SUCCESS

class RecoveryHandlerManager:
    def __init__(self):
        self.handlers = []

    def register_handler(self,handler:RecoveryHandler):
        self.handlers.append(handler)

    def find_handler(self,ctx:RecoveryContext) -> Optional[RecoveryHandler]:
        for handler in self.handlers:
            if handler.can_handle(ctx):
                return handler
        logger.warning("Can't find corresponding handler,assuming,maybe a non-target failure scenario.")
        return None
