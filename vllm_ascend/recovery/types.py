from enum import Enum
from typing import Any, Tuple

import msgspec
from vllm_ascend.recovery.actions import get_engine_core_action, get_worker_action


class StepTarget(str, Enum):
    ENGINE_CORE = "engine_core"
    WORKER = "worker"


class RecoveryAction(msgspec.Struct):
    name: str
    target: str

    def execute(self, executer: Any, cfg: dict) -> Tuple[dict, bool]:
        if self.target == StepTarget.ENGINE_CORE.value:
            handler = get_engine_core_action(self.name)
        else:
            handler = get_worker_action(self.name)
        return handler(executer, cfg)


class RecoveryStep(msgspec.Struct):
    name: str
    target: str
    actions: list = []
    timeout_s: int = 5

    def execute(self, executer: Any, cfg: dict) -> Tuple[dict, bool]:
        for action_data in self.actions:
            action = RecoveryAction(
                name=action_data[0], target=action_data[1]
            )
            cfg, success = action.execute(executer, cfg)
            if not success:
                return cfg, False
        return cfg, True


class StepResult(msgspec.Struct):
    step_name: str
    target: str
    success: bool
    cfg: dict
    worker_rank: int = -1


class RecveryPlan(msgspec.Struct):
    name: str
    steps: list = []
    cfg: dict = {}
    timeout_s: int = 30


class ExceptionInfo(msgspec.Struct):
    exception_type: str = ""
    exception_msg: str = ""


class FaultReport(msgspec.Struct):
    worker_rank: int = 0
    engine_index: int = 0
    exp: ExceptionInfo = ExceptionInfo()
    plan: RecveryPlan = RecveryPlan()


class WorkerStepDispatch(msgspec.Struct):
    step: RecoveryStep = RecoveryStep()
    cfg: dict = {}


class RecoveryPlanResult(msgspec.Struct):
    plan_name: str = ""
    engine_index: int = 0
    success: bool = False
    step_results: list = []


class RecoveryComplete(msgspec.Struct):
    plan_name: str = ""
    success: bool = False
    current_wave: int = 0
