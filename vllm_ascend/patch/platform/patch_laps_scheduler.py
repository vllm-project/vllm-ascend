# mypy: ignore-errors

from vllm.v1.core.sched import scheduler as scheduler_module
from vllm.v1.core.sched.scheduler import Scheduler
from vllm_ascend.core.laps_scheduler import LAPSSchedulerMixin


BaseScheduler = Scheduler


class LAPSScheduler(LAPSSchedulerMixin, BaseScheduler):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._init_laps_waiting_queue()


scheduler_module.Scheduler = LAPSScheduler
