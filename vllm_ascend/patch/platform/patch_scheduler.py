from vllm.v1.core.sched.scheduler import Scheduler

from vllm_ascend.core.scheduler_diagnostics import print_scheduler_summary

_ORIGINAL_SCHEDULE = Scheduler.schedule


def _schedule_with_summary(self, *args, **kwargs):
    scheduler_output = _ORIGINAL_SCHEDULE(self, *args, **kwargs)
    print_scheduler_summary(self)
    return scheduler_output


Scheduler.schedule = _schedule_with_summary
