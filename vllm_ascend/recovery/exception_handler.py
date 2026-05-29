from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

from vllm.config import CUDAGraphMode, VllmConfig
from vllm_ascend.recovery.types import ExceptionInfo, RecoveryAction, RecoveryStep, RecoveryPlan

class ExceptionHandler(ABC):
    
    @abstractmethod
    def can_handle(self, exception: ExceptionInfo) -> bool:
        pass

    @abstractmethod
    def generate_plan(self, exception: ExceptionInfo, vllm_config: VllmConfig) -> RecoveryPlan:
        pass

class NetworkExceptionHandler(ExceptionHandler):
    error_code = ["507057"]

    def can_handle(self, exception: ExceptionInfo) -> bool:
        exc_str = exception.exception_msg
        for code in self.error_code:
            if code in exc_str:
                return True
        return False

    def generate_plan(self, exception: ExceptionInfo, vllm_config: VllmConfig) -> RecoveryPlan:
        """
        网络链路故障(比如灵衢l1-l2链路故障)底层可自愈，无需额外的恢复手段，恢复计划中应包含：
        1. npu故障恢复阶段: stop_device -> restart_device -> reinit_process_group
        """
        config = dict[str, Any]()
        recovery_begin = RecoveryAction(
            name="recovery_begin"
        )
        stop_device = RecoveryAction(
            name="stop_device"
        )
        cudagraph_mode = vllm_config.compilation_config.cudagraph_mode
        rebuild_all_resources = cudagraph_mode == CUDAGraphMode.FULL
        
        restart_device = RecoveryAction(
            name="restart_device",
        )
        config["rebuild_all_resources"] = rebuild_all_resources
        reinit_process_group = RecoveryAction(
            name="reinit_process_group",
        )
        config["group"]=None
        config["rebuild_link"]=False
        npu_recover_step = RecoveryStep(
            name="npu_recover",
            target="worker",
            #actions=[recovery_begin, stop_device, restart_device, reinit_process_group],
            actions=[recovery_begin, stop_device],
            timeout_s=60
        )

        """
        npu侧的状态清理完毕后,需要清理host侧的缓存信息:
        2. abort出错batch及后续batch的缓存信息
        """
        clean_action = RecoveryAction(
            name="clear_requests"
        )
        
        clean_step = RecoveryStep(
            name="clear_requests",
            target="engine_core",
            actions=[clean_action],
            timeout_s=5
        )
        """
        封装为最终的recovery plan并返回,包含恢复网络链路故障的全部动作
        """
        clean_action_worker = RecoveryAction(
            name="clean_cache"
        )
        clean_step_worker = RecoveryStep(
            name="clean_cache_worker",
            target="worker",
            actions=[clean_action_worker],
            timeout_s=60
        )


        network_recover_plan = RecoveryPlan(
            name="network_recover_plan",
            steps=[npu_recover_step, clean_step, clean_step_worker],
            cfg=config,
            timeout_s=300
        )
        return network_recover_plan
        
class ExceptionHandlerFactory:
    def __init__(self) -> None:
        self.handlers: list[ExceptionHandler] = []

    def _register_handler(self, handler: ExceptionHandler) -> None:
        self.handlers.append(handler)

    def get_handler(self, exception: ExceptionInfo) -> ExceptionHandler:
        for handler in self.handlers:
            if handler.can_handle(exception):
                return handler
        return None
