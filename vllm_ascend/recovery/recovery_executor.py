from typing import Callable, Dict, List, Optional, Any, Type
from vllm_ascend.recovery.types import RecoveryPlan, RecoveryStep, RecoveryAction

import torch_npu


class RecoveryExecutor:
    """
    RecoveyStep的解包执行器,根据RecoveyStep包含的action执行对应的动作
    同时负责解包RecoveryPlan，根据executor的类型筛选出当前组件应该执行的RecoveryStep
    """
    
    def __init__(self, component_type: str) -> None:
        """
        Args:
            component_type: 组件类型，如 "worker" 或 "enginecore"
        """
        self._component_type = component_type.lower()
        self._action_handlers: Dict[str, Callable[..., bool]] = {}
    
    def register_handler(self, action_name: str, action_func: Callable[..., bool]) -> None:
        """
        注册自定义 action_func
        Args:
            action_name: action 名称(如 "stop_device"),与exception_handler生成Action时一致
            action_func: 处理函数
        """
        self._action_handlers[action_name] = action_func
    
    def register_handlers(self, handlers: Dict[str, Callable[..., bool]]) -> None:
        """批量注册 action handlers"""
        self._action_handlers.update(handlers)
    
    def filter_steps_by_executor(self, plan: RecoveryPlan) -> List[RecoveryStep]:
        """
        过滤RecoveryStep,仅保留当前executor对应的Step
        """
        return [step for step in plan.steps if step.executor.lower() == self._component_type]
    
    def execute_plan(self, plan: RecoveryPlan) -> bool:
        steps_to_execute = self.filter_steps_by_executor(plan)
        
        if not steps_to_execute:
            return True
        
        for step in steps_to_execute:
            if not self.execute_step(step):
                return False
        
        return True
    
    def execute_step(self, step: RecoveryStep) -> bool:
        for action in step.actions:
            if not self.execute_action(action):
                return False
        return True
    
    def execute_action(self, action: RecoveryAction) -> bool:
        handler = self._action_handlers.get(action.name)
        
        if not handler:
            raise NotImplementedError(
                f"Action '{action.name}' not registered for component '{self._component_type}'"
            )
        
        try:
            return handler(**action.config)
        except Exception as e:
            raise RuntimeError(f"Failed to execute action '{action.name}': {str(e)}") from e