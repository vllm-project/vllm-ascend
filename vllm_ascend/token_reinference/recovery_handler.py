from abc import ABC, abstractmethod

import torch
from vllm.logger import logger

from vllm_ascend.token_reinference.common import FaultStatus, RecoveryStatus
from vllm_ascend.token_reinference.recovery_context import RecoveryContext

force_stop_error = ["force stop"]
network_error = ["suspect remote error", "hccl op retry failed"]


class RecoveryHandler(ABC):
    def __init__(self):
        self.next_handler = None

    @abstractmethod
    def can_handle(self, ctx: RecoveryContext) -> bool:
        pass

    @abstractmethod
    def recover(self, ctx: RecoveryContext) -> torch.Tensor:
        """Specific recovery function"""
        pass


class ForceStopHandler(RecoveryHandler):
    def can_handle(self, ctx: RecoveryContext) -> bool:
        error_str = str(ctx.exception).lower()
        return any(error in error_str for error in force_stop_error)

    def recover(self, ctx: RecoveryContext) -> RecoveryStatus:
        """Force stop needs no extra recovery"""
        return RecoveryStatus.SUCCESS


class NetworkHandler(RecoveryHandler):
    def can_handle(self, ctx: RecoveryContext) -> bool:
        error_str = str(ctx.exception).lower()
        for error in network_error:
            if error in error_str:
                ctx.fault_queue.put_nowait(FaultStatus.NETWORK_ERR)
                return True
        return False

    def recover(self, ctx: RecoveryContext) -> RecoveryStatus:
        """Network needs no extra recovery"""
        return RecoveryStatus.SUCCESS


class RecoveryHandlerManager:
    def __init__(self):
        self.handlers = []

    def register_handler(self, handler: RecoveryHandler):
        self.handlers.append(handler)

    def find_handler(self, ctx: RecoveryContext) -> RecoveryHandler | None:
        for handler in self.handlers:
            if handler.can_handle(ctx):
                return handler
        logger.warning("Can't find corresponding handler,assuming maybe a non-target failure scenario.")
        return None
