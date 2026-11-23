# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Evict UCE-cleared prefix pages before the original FT recovery vote."""

from weakref import WeakValueDictionary

from vllm.logger import logger
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.executor.multiproc_executor import MultiprocExecutor

from vllm_ascend.core.recompute_scheduler import AsyncRecomputeScheduler, RecomputeScheduler

# Components in an EngineCore share a live config object. Weak values avoid
# retaining old schedulers; config identity survives DP rank/config revalidation.
_SCHEDULERS = WeakValueDictionary()


def get_hbm_cache_plan(worker):
    """Read the completed DP-local plan without touching connector state."""
    return worker.worker_sentinel.retry_cache_plan


class HbmRecoverySchedulerMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        _SCHEDULERS[id(self.vllm_config)] = self

    def evict_hbm_prefix_blocks(self, plan):
        """Remove every hash alias of cleared pages, leaving references and transfers alone."""
        blocks = set(plan["blocks"])
        pool = self.kv_cache_manager.block_pool
        if any(type(block) is not int or not 0 <= block < len(pool.blocks) for block in blocks):
            raise RuntimeError("Invalid block IDs in HBM prefix eviction plan")
        self.kv_cache_manager.evict_blocks(blocks)
        logger.info(
            "[FT] HBM prefix cache evicted: request_id=%s blocks=%d",
            plan["request_id"],
            len(blocks),
        )


class HbmRecoveryScheduler(HbmRecoverySchedulerMixin, Scheduler):
    pass


class HbmRecoveryAsyncScheduler(HbmRecoverySchedulerMixin, AsyncScheduler):
    pass


class HbmRecoveryRecomputeScheduler(HbmRecoverySchedulerMixin, RecomputeScheduler):
    pass


class HbmRecoveryAsyncRecomputeScheduler(HbmRecoverySchedulerMixin, AsyncRecomputeScheduler):
    pass


class HbmRecoveryExecutor(MultiprocExecutor):
    def collective_rpc(self, method, timeout=None, args=(), kwargs=None, **options):
        """Evict HBM-cleared prefix pages after worker recovery, before the FT vote."""
        result = super().collective_rpc(method, timeout=timeout, args=args, kwargs=kwargs, **options)
        if method != "handle_ft_command" or not args or args[0].instruction != "retry":
            return result
        request = args[0]
        plans = super().collective_rpc(get_hbm_cache_plan, timeout=timeout)
        if not plans or not isinstance(plans[0], dict) or any(plan != plans[0] for plan in plans):
            raise RuntimeError("Missing or inconsistent HBM cache plans")
        plan = plans[0]
        if plan.get("request_id") != request.request_id:
            raise RuntimeError("Stale HBM cache plan")
        if type(plan.get("hbm_detected")) is not bool:
            raise RuntimeError("Missing HBM classification in cache plan")
        if not plan["hbm_detected"]:
            return result
        # Prefix eviction alone cannot repair KV still used by retained requests.
        if not request.params.get("abort", False):
            raise RuntimeError("HBM prefix recovery requires retry with abort=True")
        scheduler = _SCHEDULERS.get(id(self.vllm_config))
        if scheduler is None:
            raise RuntimeError("Missing scheduler for HBM prefix eviction")
        scheduler.evict_hbm_prefix_blocks(plan)
        return result
