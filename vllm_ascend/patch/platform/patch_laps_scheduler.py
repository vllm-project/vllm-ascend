# mypy: ignore-errors

import vllm
from vllm_ascend.core.laps_scheduler import LAPSSchedulerMixin


BaseScheduler = vllm.v1.core.sched.scheduler.Scheduler


class LAPSScheduler(LAPSSchedulerMixin, BaseScheduler):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._init_laps_waiting_queue()


vllm.v1.core.sched.scheduler.Scheduler = LAPSScheduler
