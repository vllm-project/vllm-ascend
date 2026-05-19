from enum import Enum
from typing import Any, Tuple

from vllm_ascend.recovery.actions import get_engine_core_action, get_worker_action


class StepTarget(str, Enum):
    ENGINE_CORE = "engine_core"
    WORKER = "worker"


class RecoveryAction:
    def __init__(self, name: str, target: StepTarget) -> None:
        self.name = name
        self.target = target

    def execute(self, executer: Any, cfg: dict) -> Tuple[dict, bool]:
        if self.target == StepTarget.ENGINE_CORE:
            handler = get_engine_core_action(self.name)
        else:
            handler = get_worker_action(self.name)
        return handler(executer, cfg)


class RecoveryStep:
    def __init__(
        self,
        name: str,
        target: StepTarget,
        actions: list[RecoveryAction],
        timeout_s: int = 5,
    ) -> None:
        self.name = name
        self.target = target
        self.actions = actions
        self.timeout_s = timeout_s

    def execute(self, executer: Any, cfg: dict) -> Tuple[dict, bool]:
        for action in self.actions:
            cfg, success = action.execute(executer, cfg)
            if not success:
                return cfg, False
        return cfg, True


class StepResult:
    def __init__(
        self,
        step_name: str,
        target: StepTarget,
        success: bool,
        cfg: dict,
    ) -> None:
        self.step_name = step_name
        self.target = target
        self.success = success
        self.cfg = cfg


class RecveryPlan:
    def __init__(
        self,
        name: str,
        steps: list[RecoveryStep],
        cfg: dict,
        timeout_s: int = 30,
    ) -> None:
        self.name = name
        self.steps = steps
        self.cfg = cfg
        self.timeout_s = timeout_s


class ExceptionInfo:
    def __init__(self, exception_type: str, exception_msg: str) -> None:
        self.exception_type = exception_type
        self.exception_msg = exception_msg


class FaultReport:
    def __init__(
        self,
        worker_rank: int,
        engine_index: int,
        exp: ExceptionInfo,
        plan: RecveryPlan,
    ) -> None:
        self.worker_rank = worker_rank
        self.engine_index = engine_index
        self.exp = exp
        self.plan = plan


class WorkerStepDispatch:
    def __init__(self, step: "RecoveryStep", cfg: dict) -> None:
        self.step = step
        self.cfg = cfg


class WorkerStepResult:
    def __init__(
        self,
        worker_rank: int,
        step_name: str,
        success: bool,
        cfg: dict,
    ) -> None:
        self.worker_rank = worker_rank
        self.step_name = step_name
        self.success = success
        self.cfg = cfg


class RecoveryPlanResult:
    def __init__(
        self,
        plan_name: str,
        engine_index: int,
        success: bool,
        step_results: list[StepResult],
    ) -> None:
        self.plan_name = plan_name
        self.engine_index = engine_index
        self.success = success
        self.step_results = step_results


class RecoveryComplete:
    def __init__(
        self,
        plan_name: str,
        success: bool,
        current_wave: int,
    ) -> None:
        self.plan_name = plan_name
        self.success = success
        self.current_wave = current_wave
