# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Scheduler/executor output tracking retained from the validated HBM path.

Recovery uses the original FT flow without additional scheduler finalization.
The process-local weak registries pair components belonging to the same engine.
"""

from collections import deque
from weakref import WeakValueDictionary, ref

from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.executor.multiproc_executor import MultiprocExecutor

from vllm_ascend.core.recompute_scheduler import AsyncRecomputeScheduler, RecomputeScheduler

_SCHEDULERS = WeakValueDictionary()
_EXECUTORS = WeakValueDictionary()


def _engine_key(config):
    # EngineCore passes the same config object to both components. Its
    # post-handshake __post_init__ regenerates instance_id, and scale-down may
    # change the DP rank, so neither field is a stable registry key.
    # Each live component owns this config; weak registry values disappear
    # before its object identity can be reused by another engine.
    return id(config)


class HbmRecoverySchedulerMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        _SCHEDULERS[_engine_key(self.vllm_config)] = self

    def update_from_output(self, scheduler_output, model_runner_output):
        result = super().update_from_output(scheduler_output, model_runner_output)
        executor = _EXECUTORS.get(_engine_key(self.vllm_config))
        if executor is not None:
            executor.forget_consumed_output(model_runner_output)
        return result


class HbmRecoveryScheduler(HbmRecoverySchedulerMixin, Scheduler):
    pass


class HbmRecoveryAsyncScheduler(HbmRecoverySchedulerMixin, AsyncScheduler):
    pass


class HbmRecoveryRecomputeScheduler(HbmRecoverySchedulerMixin, RecomputeScheduler):
    pass


class HbmRecoveryAsyncRecomputeScheduler(HbmRecoverySchedulerMixin, AsyncRecomputeScheduler):
    pass


class HbmRecoveryExecutor(MultiprocExecutor):
    def __init__(self, *args, **kwargs):
        self._pending_hbm_outputs = deque()
        super().__init__(*args, **kwargs)
        _EXECUTORS[_engine_key(self.vllm_config)] = self

    def collective_rpc(self, method, timeout=None, args=(), kwargs=None, **options):
        result = super().collective_rpc(method, timeout=timeout, args=args, kwargs=kwargs, **options)
        if isinstance(method, str) and method in ("execute_model", "sample_tokens") and options.get("non_block"):
            # Weak refs retain only futures still held by EngineCore's batch
            # queue. Scheduler consumption retires successful results explicitly.
            self._pending_hbm_outputs = deque(item for item in self._pending_hbm_outputs if item() is not None)
            self._pending_hbm_outputs.append(ref(result))
        return result

    def forget_consumed_output(self, output):
        pending = deque()
        for item in self._pending_hbm_outputs:
            future = item()
            if future is None:
                continue
            if future.done() and future.exception() is None and future.result() is output:
                continue
            pending.append(item)
        self._pending_hbm_outputs = pending
