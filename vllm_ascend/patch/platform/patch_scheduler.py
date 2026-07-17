from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.scheduler import Scheduler

from vllm_ascend.core.scheduler_diagnostics import print_scheduler_summary

_ORIGINAL_SCHEDULE = Scheduler.schedule
_ORIGINAL_ASYNC_SCHEDULE = AsyncScheduler.schedule


def _schedule_with_summary(self, *args, **kwargs):
    scheduler_output = _ORIGINAL_SCHEDULE(self, *args, **kwargs)
    print_scheduler_summary(self, scheduler_output)
    return scheduler_output


def _async_schedule_with_summary(self, *args, **kwargs):
    scheduler_output = _ORIGINAL_ASYNC_SCHEDULE(self, *args, **kwargs)
    print_scheduler_summary(self, scheduler_output)
    return scheduler_output


Scheduler.schedule = _schedule_with_summary
AsyncScheduler.schedule = _async_schedule_with_summary
