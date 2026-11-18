# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""HBM recovery through the scheduler/executor class extension points.

No EngineCore or sentinel methods are replaced. The executor completes local
cleanup before its recovery RPC returns, so failures enter the original FT vote.
The process-local weak registries pair components belonging to the same engine;
no worker IPC, background mutation, or global class monkeypatch is involved.
"""

from collections import deque
from weakref import WeakValueDictionary, ref

from vllm.logger import logger
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.executor.multiproc_executor import MultiprocExecutor
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import RequestStatus

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


def collect_kv_transfer_recovery_output(worker, finished_req_ids):
    """Poll completion after worker recovery without launching compute or I/O."""
    from vllm.distributed.kv_transfer import get_kv_transfer_group
    from vllm.v1.outputs import KVConnectorOutput

    connector = get_kv_transfer_group()
    results = connector.get_transfer_results(finished_req_ids)
    return ModelRunnerOutput.with_kv_conn_output_only(
        KVConnectorOutput(
            finished_sending=results.finished_sending,
            finished_recving=results.finished_recving,
            failed_recving=results.failed_recving,
            invalid_block_ids=connector.get_block_ids_with_load_errors(),
            kv_connector_worker_meta=connector.build_connector_worker_meta(),
        )
    )


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

    def release_recovery_blocks(self, allow_preempted=False):
        """Release compute fences only after all old worker steps have stopped."""
        if self.running or any(
            not request.is_finished()
            and not (
                allow_preempted
                and request.status in (RequestStatus.PREEMPTED, RequestStatus.WAITING)
                and not request.num_in_flight_tokens
                and not any(self.kv_cache_manager.get_block_ids(request.request_id))
            )
            for request in self.requests.values()
        ):
            raise RuntimeError("Cannot release recovery blocks with active requests; retry with abort=True")
        pool = self.kv_cache_manager.block_pool
        while self.deferred_frees:
            _, blocks = self.deferred_frees.popleft()
            pool.free_blocks(reversed(blocks))
        self.processed_step_seq = self.sched_step_seq

    def finish_hbm_recovery(self, plan):
        """Invalidate cleared storage in the scheduler before the FT vote."""
        blocks = set(plan["blocks"])
        pool = self.kv_cache_manager.block_pool
        if any(type(block) is not int or not 0 <= block < len(pool.blocks) for block in blocks):
            raise RuntimeError("Invalid block IDs in Worker KV recovery plan")
        resume = self.parallel_config.fault_tolerance_config.request_failure_policy == "recompute"
        if resume and plan.get("hbm_detected") is False and not blocks:
            self.release_recovery_blocks(allow_preempted=True)
            return
        # EngineCore owns abort outputs. Never silently drop retained requests
        # here: HBM with recompute requires the original retry(abort=True) path.
        self.release_recovery_blocks()
        if not self.reset_connector_cache():
            raise RuntimeError("Failed to reset connector after retry")
        if any(block.ref_cnt > 0 and not block.is_null for block in pool.blocks):
            raise RuntimeError("Outstanding request/transfer allocations after retry abort")
        self.prev_step_scheduled_req_ids.clear()
        # Upstream on_fault already evicts the failed step's partial writes.
        # Only bytes actually cleared by HBM recovery need additional eviction.
        self.kv_cache_manager.evict_blocks(blocks)
        logger.info(
            "[FT] VA scheduler recovery complete: request_id=%s evicted_blocks=%d",
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
        elif method == "handle_ft_command" and args and args[0].instruction == "retry":
            self.finish_hbm_recovery(args[0])
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

    def drain_recovery_outputs(self, scheduler):
        # The completed recovery RPC has drained earlier responses in upstream
        # FutureWrapper. Feed only KV completions, never stale token outputs.
        pending, self._pending_hbm_outputs = self._pending_hbm_outputs, deque()
        for item in pending:
            future = item()
            if future is None:
                continue
            try:
                output = future.result()
            except Exception:
                # Upstream executor salvages KV completions from failed RPCs;
                # the aggregator consumes those with the fresh poll below.
                continue
            kv_output = output.kv_connector_output if output is not None else None
            if kv_output is not None and not kv_output.is_empty():
                scheduler.update_from_output(
                    SchedulerOutput.make_empty(), ModelRunnerOutput.with_kv_conn_output_only(kv_output)
                )
                # Future.result() returns this same cached object. The original
                # sentinel's later batch drain must not deliver it twice.
                output.kv_connector_output = None

    def finish_hbm_recovery(self, request):
        scheduler = _SCHEDULERS.get(_engine_key(self.vllm_config))
        if scheduler is None:
            raise RuntimeError("Missing VA recovery scheduler")
        plans = super().collective_rpc("get_kv_cache_recovery_plan")
        if not plans or plans[0] is None or any(plan != plans[0] for plan in plans):
            raise RuntimeError("Inconsistent Worker KV recovery plans")
        plan = plans[0]
        if plan["request_id"] != request.request_id:
            raise RuntimeError("Stale Worker KV recovery plan")
        self.drain_recovery_outputs(scheduler)
        if scheduler.connector is not None:
            if self.kv_output_aggregator is None:
                raise RuntimeError("Missing KV output aggregator during recovery")
            outputs = super().collective_rpc(
                collect_kv_transfer_recovery_output, args=(set(scheduler.finished_req_ids),)
            )
            output = self.kv_output_aggregator.aggregate(outputs)
            if (
                output is not None
                and output.kv_connector_output is not None
                and not output.kv_connector_output.is_empty()
            ):
                scheduler.update_from_output(SchedulerOutput.make_empty(), output)
        scheduler.finish_hbm_recovery(plan)
