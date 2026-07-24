#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""Add opt-in summaries to the standard synchronous and async schedulers."""

from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.scheduler import Scheduler

from vllm_ascend.core.scheduler_diagnostics import (
    diagnostics_enabled,
    print_scheduler_summary,
)

_ORIGINAL_SCHEDULE = Scheduler.schedule
_ORIGINAL_ASYNC_SCHEDULE = AsyncScheduler.schedule


def _schedule_with_diagnostics(original_schedule, self, *args, **kwargs):
    scheduler_output = original_schedule(self, *args, **kwargs)
    if diagnostics_enabled(self.vllm_config):
        print_scheduler_summary(self, scheduler_output)
    return scheduler_output


def _scheduler_schedule_with_diagnostics(self, *args, **kwargs):
    return _schedule_with_diagnostics(_ORIGINAL_SCHEDULE, self, *args, **kwargs)


def _async_scheduler_schedule_with_diagnostics(self, *args, **kwargs):
    return _schedule_with_diagnostics(_ORIGINAL_ASYNC_SCHEDULE, self, *args, **kwargs)


Scheduler.schedule = _scheduler_schedule_with_diagnostics
AsyncScheduler.schedule = _async_scheduler_schedule_with_diagnostics
