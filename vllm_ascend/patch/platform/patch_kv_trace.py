# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in scheduler attachment; upstream vLLM owns KVCacheManager construction.

Attach after base initialization so stock, asynchronous and Ascend scheduler
subclasses share the same observer. No patch is installed when tracing is off.
An upstream KVCacheManager observer API could replace this narrow attachment.
"""

import functools

from vllm_ascend.debug.kv_trace import KVTrace
from vllm_ascend.debug.kv_trace_manager import attach_manager_trace


def install_scheduler_trace(scheduler_cls):
    if getattr(scheduler_cls.__init__, "_ascend_kv_trace", False):
        return
    original_init = scheduler_cls.__init__

    @functools.wraps(original_init)
    def initialize(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        trace = KVTrace.from_env("scheduler", self.vllm_config)
        if trace is not None:
            attach_manager_trace(self.kv_cache_manager, trace)
            self.kv_cache_manager._ascend_kv_trace.wrap_schedule(self)

    initialize._ascend_kv_trace = True
    scheduler_cls.__init__ = initialize


def apply_patch():
    from vllm.v1.core.sched.scheduler import Scheduler

    install_scheduler_trace(Scheduler)
