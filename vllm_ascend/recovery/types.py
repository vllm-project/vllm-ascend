from enum import Enum
from typing import Any, Tuple

import msgspec
from vllm_ascend.recovery.actions import get_engine_core_action, get_worker_action

FUTURE_TIMEOUT_SECONDS = 45


class StepTarget(str, Enum):
    ENGINE_CORE = "engine_core"
    WORKER = "worker"


class RecoveryAction(msgspec.Struct):
    name: str

    def execute(self, executer: Any, cfg: dict, target: str) -> Tuple[dict, bool]:
        if target == StepTarget.ENGINE_CORE.value:
            handler = get_engine_core_action(self.name)
        else:
            handler = get_worker_action(self.name)
        return handler(executer, cfg)


class RecoveryStep(msgspec.Struct):
    name: str
    target: str
    timeout_s: int = 5
    actions: list[RecoveryAction] = msgspec.field(default_factory=list)
    

    def execute(self, executer: Any, cfg: dict) -> Tuple[dict, bool]:
        for action in self.actions:
            cfg, success = action.execute(executer, cfg, self.target)
            if not success:
                return cfg, False
        return cfg, True


class StepResult(msgspec.Struct):
    step_name: str
    success: bool
    worker_rank: int
    cfg: dict = msgspec.field(default_factory=dict)
    


class RecoveryPlan(msgspec.Struct):
    name: str
    timeout_s: int
    steps: list[RecoveryStep] = msgspec.field(default_factory=list)
    cfg: dict = msgspec.field(default_factory=dict)
    


class ExceptionInfo(msgspec.Struct):
    exception_type: str
    exception_msg: str


class FaultReport(msgspec.Struct):
    worker_rank: int
    engine_index: int
    exp: ExceptionInfo
    plan: RecoveryPlan 


class WorkerStepDispatch(msgspec.Struct):
    step: RecoveryStep
    cfg: dict = msgspec.field(default_factory=dict)


class RecoveryPlanResult(msgspec.Struct):
    plan_name: str
    engine_index: int
    success: bool
    step_results: list[StepResult] = msgspec.field(default_factory=list)


class RecoveryComplete(msgspec.Struct):
    plan_name: str
    success: bool
    current_wave: int

class NetworkCheck(msgspec.Struct):
    engine_index: int