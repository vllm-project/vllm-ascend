# mypy: ignore-errors

from vllm.v1.core.sched import scheduler as scheduler_module
from vllm_ascend.core.laps_scheduler import LAPSScheduler


scheduler_module.Scheduler = LAPSScheduler
