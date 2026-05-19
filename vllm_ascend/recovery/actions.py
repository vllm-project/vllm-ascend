import time
from typing import Any, Callable, Tuple

from vllm.logger import logger
from vllm.v1.engine import EngineCoreEventType
from vllm.v1.request import RequestStatus

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
def clear_requests(executer: Any, cfg: dict) -> Tuple[dict, bool]:
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
def _stop_device(executor: Any, cfg: dict | None) -> bool:
        try:
            stop_result = torch_npu.npu.stop_device(executor.local_rank)
            if stop_result == 0:
                logger.info("stop_device executed successfully")
                return cfg, True
            else:
                logger.error(f"stop_device failed with result: {stop_result}")
                return cfg, False
        except Exception as e:
            logger.error(f"stop_device executed failed with exception: {e}")
            return cfg, False

@worker_action("restart_device")
def _restart_device(executor: Any, context:dict | None) -> bool:
    try:
        ctx = context or {}
        torch_npu.npu.restart_device(
            torch.npu.current_device(), rebuild_all_resources=ctx.get("rebuild_all_resources", False)
        )
        return cfg, True
    except Exception as e:
        logger.error(f"restart_device executed failed with exception: {e}")
        return cfg, False

@worker_action("reinit_process_group")
def _reinit_process_group(executor: Any, context:dict | None) -> bool:
    try:
        ctx = context or {}
        torch.distributed.reinit_process_group(
            group=ctx.get("group", None), rebuild_link=ctx.get("rebuild_link", True)
        )
        return cfg, True
    except Exception as e:
        logger.error(f"reinit_process_group executed failed with exception: {e}")
        return cfg, False

@worker_action("clean_cache")
def _clean_cache(executor: Any, context:dict | None) -> bool:
    try:
        ctx = context or {}
        abort_list = context.get("abort_list", [])
        model_runner = executor._worker.model_runner
        for req_id in abort_list:
            model_runner.requests.pop(req_id, None)
            model_runner.num_prompt_logprobs.pop(req_id, None)
            model_runner.input_batch.remove_request(req_id)
        return cfg, True
    except Exception as e:
        logger.error(f"worker clean_cached failed with exception: {e}")
        return cfg, False

@worker_action("recovery_finished")
def _recovery_finished(executor: Any, context:dict | None) -> bool:
    executor.in_recovery = False
    return cfg, True

@worker_action("recovery_begin")
def _recovery_begin(executor: Any, context:dict | None) -> bool:
    executor.in_recovery = True
    return cfg, True