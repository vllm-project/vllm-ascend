import time
import torch
import torch_npu
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
    # if executer.batch_queue is not None:
    #     while executer.batch_queue:
    #         future, _, _ = executer.batch_queue.pop()
    #         try:
    #             future.result()
    #         except Exception:
    #             pass
    #     executer.batch_queue.clear()
    #     logger.info("[RAS] clear_requests: batch_queue drained")

    scheduler = executer.scheduler
    # timestamp = time.monotonic()
    running_req_ids = [req.request_id for req in scheduler.running]
    # while scheduler.running:
    #     request = scheduler.running.pop()
    #     scheduler.kv_cache_manager.free(request)
    #     scheduler.encoder_cache_manager.free(request)
    #     request.prompt_token_ids = request._all_token_ids.copy()
    #     request._output_token_ids = []
    #     request.num_prompt_tokens = len(request._all_token_ids)
    #     request.num_computed_tokens = 0
    #     request.status = RequestStatus.PREEMPTED
    #     if request.spec_token_ids:
    #         request.spec_token_ids = []
    #     request.num_preemptions += 1
    #     request.num_output_placeholders = 0
    #     request.discard_latest_async_tokens = True
    #     if scheduler.log_stats:
    #         request.record_event(EngineCoreEventType.PREEMPTED, timestamp)
    #     scheduler.waiting.prepend_request(request)
    # scheduler.prev_step_scheduled_req_ids.clear()
    logger.info(
        "[RAS] clear_requests: %d requests preempted", len(running_req_ids)
    )

    cfg["abort_list"] = running_req_ids
    return cfg, True

@worker_action("stop_device")
def _stop_device(executor: Any, cfg: dict | None) -> bool:
        try:
            stop_result = torch_npu.npu.stop_device(executor.local_rank)
            if stop_result == 0:
                logger.info("stop_device executed successfully")
                retry = 0
                while not executor.exception_occur:
                    time.sleep(1)
                    retry += 1
                    if retry > 20:
                        logger.error("stop_device retry 20 times, still not occur exception")
                        break
                if not executor.exception_occur:
                    logger.warning("exception not occur after stop_device, worker may passed fucntion call")
                executor.device_stopped = True
                return cfg, True
            else:
                logger.error(f"stop_device failed with result: {stop_result}")
                return cfg, False
        except Exception as e:
            logger.error(f"stop_device executed failed with exception: {e}")
            return cfg, False

@worker_action("restart_device")
def _restart_device(executor: Any, cfg:dict | None) -> bool:
    try:
        cfg = cfg or {}
        torch_npu.npu.restart_device(
            torch.npu.current_device(), rebuild_all_resources=cfg.get("rebuild_all_resources", False)
        )
        return cfg, True
    except Exception as e:
        logger.error(f"restart_device executed failed with exception: {e}")
        return cfg, False

@worker_action("reinit_process_group")
def _reinit_process_group(executor: Any, cfg:dict | None) -> bool:
    try:
        cfg = cfg or {}
        torch.distributed.reinit_process_group(
            group=cfg.get("group", None), rebuild_link=cfg.get("rebuild_link", True)
        )
        return cfg, True
    except Exception as e:
        logger.error(f"reinit_process_group executed failed with exception: {e}")
        return cfg, False

@worker_action("clean_cache")
def _clean_cache(executor: Any, cfg:dict | None) -> bool:
    try:
        cfg = cfg or {}
        abort_list = cfg.get("abort_list", [])
        model_runner = executor.model_runner
        for req_id in abort_list:
            model_runner.requests.pop(req_id, None)
            model_runner.num_prompt_logprobs.pop(req_id, None)
            model_runner.input_batch.remove_request(req_id)
        model_runner.input_batch.condense()
        model_runner.input_batch.refresh_metadata()
        return cfg, True
    except Exception as e:
        logger.error(f"worker clean_cached failed with exception: {e}")
        return cfg, False

@worker_action("recovery_finished")
def _recovery_finished(executor: Any, cfg:dict | None) -> bool:
    executor.in_recovery = False
    executor.exception_occur = False
    return cfg, True

@worker_action("recovery_begin")
def _recovery_begin(executor: Any, cfg:dict | None) -> bool:
    executor.in_recovery = True
    return cfg, True