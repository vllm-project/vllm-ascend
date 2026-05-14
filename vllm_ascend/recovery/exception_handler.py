from abc import ABC, abstractmethod
from enum import Enum

from vllm.config import CUDAGraphMode, VllmConfig
from vllm_ascend.recovery.types import ExceptionInfo, RecoveryAction, RecoveryStep, RecoveryPlan

class ExceptionHandler(ABC):
    
    @abstractmethod
    def can_handle(self, exception: Exception) -> bool:
        pass

    @abstractmethod
    def generate_plan(self, exception: Exception) -> RecoveryPlan:
        pass

class NetworkExceptionHandler(ExceptionHandler):
    error_code = ["507057"]

    def can_handle(self, exception: ExceptionInfo) -> bool:
        exc_str = exception.message
        for code in self.error_code:
            if code in exc_str:
                return True
        return False

    def generate_plan(self, exception: ExceptionInfo, vllm_config: VllmConfig) -> RecoveryPlan:
        """
        网络链路故障(比如灵衢l1-l2链路故障)底层可自愈，无需额外的恢复手段，恢复计划中应包含：
        1. npu故障恢复阶段: stop_device -> restart_device -> reinit_process_group
        """
        stop_device = RecoveryAction(
            name="stop_device"
        )
        
        cudagraph_mode = vllm_config.compilation_config.cudagraph_mode
        rebuild_all_resources = cudagraph_mode == CUDAGraphMode.FULL
        
        restart_device = RecoveryAction(
            name="restart_device",
            config={
                "rebuild_all_resources": rebuild_all_resources
            }
        )
        # TODO:这里可能涉及根据代际来判断是否需要rebuild_link
        reinit_process_group = RecoveryAction(
            name="reinit_process_group",
            config={
                "group": None,
                "rebuild_link": False
            }
        )
        
        npu_recover_step = RecoveryStep(
            name="npu_recover",
            executor="worker",
            actions=[stop_device, restart_device, reinit_process_group]
        )

        """
        npu侧的状态清理完毕后,需要清理host侧的缓存信息:
        2. abort出错batch及后续batch的缓存信息
        """
        clean_action = RecoveryAction(
            name="clean_cache"
        )
        
        clean_step = RecoveryStep(
            name="clean_cache",
            executor="enginecore",
            actions=[clean_action],
        )
        """
        封装为最终的recovery plan并返回,包含恢复网络链路故障的全部动作
        """
        network_recover_plan = RecoveryPlan(
            name="network_recover_play",
            steps=[npu_recover_step, clean_step]
        )
        return network_recover_plan
        

class ExceptionHandlerFactory:
    def __init__(self) -> None:
        self.handlers: list[ExceptionHandler] = []

    def _register_handler(self, handler: ExceptionHandler) -> None:
        self.handlers.append(handler)

    def get_handler(self, exception: Exception) -> ExceptionHandler:
        for handler in self.handlers:
            if handler.can_handle(exception):
                return handler
        return None
