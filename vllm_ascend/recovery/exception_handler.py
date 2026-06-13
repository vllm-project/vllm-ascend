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
        Network link failure recovery plan:
        1. device_recovery_step: recovery_begin -> stop_device -> restart_device -> reinit_process_group
        2. clean_engine_cache_step: label_dirty_requests -> clean_batch_queue -> recompute_dirty_requests
        3. clean_worker_cache_step: worker_clean_dirty_requests_cache
        4. re_capture_graph: worker_rebuild_cpu_group -> worker_recapture_graph
        """
        config = dict[str, Any]()

        # Step 1: Device recovery
        config["rebuild_all_resources"] = False
        config["group"] = None
        config["rebuild_link"] = False

        device_recovery_step = RecoveryStep(
            name="device_recovery",
            target="worker",
            actions=[
                RecoveryAction(name="recovery_begin"),
                RecoveryAction(name="stop_device"),
                RecoveryAction(name="restart_device"),
                RecoveryAction(name="reinit_process_group"),
            ],
            timeout_s=60
        )

        # Step 2: Clean engine cache
        clean_engine_cache_step = RecoveryStep(
            name="clean_engine_cache",
            target="engine_core",
            actions=[
                RecoveryAction(name="label_dirty_requests"),
                RecoveryAction(name="clean_batch_queue"),
                RecoveryAction(name="recompute_dirty_requests"),
            ],
            timeout_s=60
        )

        # Step 3: Clean worker cache
        clean_worker_cache_step = RecoveryStep(
            name="clean_worker_cache",
            target="worker",
            actions=[
                RecoveryAction(name="worker_clean_dirty_requests_cache"),
            ],
            timeout_s=60
        )

        # Step 4: Re-capture graph
        re_capture_graph_step = RecoveryStep(
            name="re_capture_graph",
            target="worker",
            actions=[
                RecoveryAction(name="worker_rebuild_cpu_group"),
                RecoveryAction(name="worker_recapture_graph"),
            ],
            timeout_s=60
        )

        network_recover_plan = RecoveryPlan(
            name="network_recover_plan",
            steps=[
                device_recovery_step,
                clean_engine_cache_step,
                clean_worker_cache_step,
                re_capture_graph_step,
            ],
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
