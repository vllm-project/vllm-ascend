import time
from typing import Any, Callable, Tuple

from vllm.logger import logger
from vllm.v1.engine import EngineCoreEventType
from vllm.v1.engine.core import EngineCore
from vllm.v1.request import RequestStatus
from vllm_ascend.worker.worker import NPUWorker

ActionFunc = Callable[[Any, dict], Tuple[dict, bool]]

_ENGINE_CORE_ACTIONS: dict[str, ActionFunc] = {}
_WORKER_ACTIONS: dict[str, ActionFunc] = {}


def engine_core_action(name: str):
    def wrapper(func: ActionFunc):
        _ENGINE_CORE_ACTIONS[name] = func
        return func
    return wrapper


def worker_action(name: str):
    def wrapper(func: ActionFunc):
        _WORKER_ACTIONS[name] = func
        return func
    return wrapper


def get_engine_core_action(name: str) -> ActionFunc:
    func = _ENGINE_CORE_ACTIONS.get(name)
    if func is None:
        raise ValueError(f"Unknown engine_core action: {name}")
    return func


def get_worker_action(name: str) -> ActionFunc:
    func = _WORKER_ACTIONS.get(name)
    if func is None:
        raise ValueError(f"Unknown worker action: {name}")
    return func


@engine_core_action("clear_requests")
def clear_requests(executer: EngineCore, cfg: dict) -> Tuple[dict, bool]:
    if executer.batch_queue is not None:
        while executer.batch_queue:
            future, _, _ = executer.batch_queue.pop()
            try:
                future.result()
            except Exception:
                pass
        executer.batch_queue.clear()
        logger.info("[RAS] clear_requests: batch_queue drained")

    scheduler = executer.scheduler
    timestamp = time.monotonic()
    running_req_ids = [req.request_id for req in scheduler.running]
    while scheduler.running:
        request = scheduler.running.pop()
        scheduler.kv_cache_manager.free(request)
        scheduler.encoder_cache_manager.free(request)
        request.prompt_token_ids = request._all_token_ids.copy()
        request._output_token_ids = []
        request.num_prompt_tokens = len(request._all_token_ids)
        request.num_computed_tokens = 0
        request.status = RequestStatus.PREEMPTED
        if request.spec_token_ids:
            request.spec_token_ids = []
        request.num_preemptions += 1
        request.num_output_placeholders = 0
        request.discard_latest_async_tokens = True
        if scheduler.log_stats:
            request.record_event(EngineCoreEventType.PREEMPTED, timestamp)
        scheduler.waiting.prepend_request(request)
    scheduler.prev_step_scheduled_req_ids.clear()
    logger.info(
        "[RAS] clear_requests: %d requests preempted", len(running_req_ids)
    )

    cfg["aborted_req_ids"] = running_req_ids
    return cfg, True


@worker_action("stop_device")
def stop_device(executer: NPUWorker, cfg: dict) -> Tuple[dict, bool]:
    # TODO: add stop device logic
    return cfg, True


@worker_action("abort_requests")
def abort_requests(executer: NPUWorker, cfg: dict) -> Tuple[dict, bool]:
    # TODO: add worker abort logic
    return cfg, True
